"""Memory-value experiment on BEAM-1M, 4 arms. Implements sections 1-2 of its pre-registration.

  PM1_BEAM_SET=1m PM1_GPU=0 PM1_RERANK_QWEN=http://127.0.0.1:18162 (no PM1_RERANK_GTE)
  mv_arms.py export <run> <convs comma> [samples=2]    -> data/mv_<run>/ (meta, per-question scored pools, answer prompts)
  mv_arms.py make-answers <run> [budget=2000] [size=40]   ONE shuffled batch (seed 17) of all arms x samples -> labeling inbox
  (collect/refix: beam_answer_tasks.py collect|refix mv_<run>_ans_<budget> <labeler>)
  mv_arms.py make-judge <run> [budget=2000] [labeler=gpt] [size=40]   ONE shuffled blind batch (seed 17), opaque ids j#####
  mv_arms.py report <run> [labeler=gpt]      primary comparisons (prereg s3) first; everything else printed under SECONDARY

Implementation decisions where the prereg is silent (documented here, reported to the decision maker, NOT in the prereg):
  * unit = one turn (episode); candidate pool = first 48 distinct turns of the BM25+dense RRF list (k_first=100 each);
    every doc shown to the reranker is cut to its first 2,500 chars; docs go in batches of 24.
  * fill rule (all arms, all budgets): walk the ordered list; a turn costs est_tokens(text)+16 (pm1.recall.est_tokens, the
    production counter); if it does not fit it is SKIPPED (counted) and the walk continues; no truncation of turns.
  * context = turns grouped by session (sessions by first date), chronological inside a session, line format
    '[<occurred_at> . <session> . <role>] <text>' (production format). No directives / ledger / tree / planning blocks.
  * U: reranked pool split by role, user turns first (by score) then assistant turns (by score); if the question matches
    the prereg regex, the order is exactly base's.
  * N: notes = teacher notes store (note table: path, content, evidence, updated), items searched by BM25 (TreeIndex.bm25,
    tokens>0) + e5 dense -> RRF(k=60) -> Qwen on the top 100 (2,500-char heads) -> ALL 100 notes expanded, in reranked order,
    to the turns they cite (dedup) -> weighted RRF (k=60) of [base reranked pool, expanded note turns] with w=1.0 on the
    notes list -> fill.
  * NC: half the budget filled with reranked notes (text '[t12,t15 . <day>] <content>', reranked order, same fill rule),
    the other half with base's raw turns (same fill rule, no dedup against the notes' pointers).
  * a rerank doc that fails after batch retry x2 and doc-by-doc (full, 1500, 800 chars) gets score None, sorts last, and is
    counted in meta (rerank_failed); nothing falls back silently. Every score is stored with its model name.
"""
from __future__ import annotations

import json
import os
import random
import re
import sqlite3
import statistics
import sys
import time
from collections import Counter, defaultdict
from pathlib import Path

import numpy as np

HERE = Path(__file__).resolve().parent
sys.path.insert(0, str(HERE))
sys.path.insert(0, str(HERE.parent / 'src'))
os.environ.setdefault('PM1_BEAM_SET', '1m')

DATA = HERE / 'data'
LAB = Path(os.environ.get('PM1_LABEL_DIR', str(Path.home() / 'pm1-labeling')))
NOTES_DIR = Path(os.environ.get('PM1_MV_NOTES', str(Path.home() / 'pm1-ext/stores/notes_1m')))
ARMS = tuple(os.environ.get('MV_ARMS', 'base,U,N,NC').split(','))   # amendment 3: the extension runs base,U only
NOTES_ON = 'N' in ARMS or 'NC' in ARMS
FACTS_ON = 'UF' in ARMS                  # U + the fact index (pm1.facts)
FACTS_DIR = Path(os.environ.get('PM1_MV_FACTS', str(Path.home() / '.cache/pm1-eval/beam1m_facts')))
BUDGETS = (2000, 6000)
DEPTH, K_FIRST, NOTE_DEPTH, W_NOTES, OVERHEAD, CUT, BATCH = 48, 100, 100, 1.0, 16, 2500, 24
SHUFFLE_SEED, BOOT_SEED, BOOT_N = 17, 2026, 4000
EXC = re.compile(r"\byou(?:'ve| have)? (?:said|told|suggested|recommended|mentioned|gave|provided|wrote)\b"
                 r"|\byour (?:answer|suggestion|recommendation|advice)\b", re.I)
JUDGE = """You grade answers to questions about a user's past conversations against a VERIFIED gold answer.
For each item return score 1 (correct and complete: matches the gold's key facts; extra correct detail is fine),
0.5 (partially correct: some key facts right, some missing or wrong), or 0 (wrong, contradicts the gold, or abstains
when the gold has an answer). For abstention questions (gold says it was never mentioned), score 1 only if the answer
says it is not known / not mentioned. Output one line per id: {"id": "...", "response": {"score": 1|0.5|0, "reason": "<short>"}}"""
CATS = ['abstention', 'contradiction_resolution', 'event_ordering', 'information_extraction', 'instruction_following',
        'knowledge_update', 'multi_session_reasoning', 'preference_following', 'summarization', 'temporal_reasoning']


# ---------- reranker: Qwen only, model name per score, failures counted ----------
class WrongModel(RuntimeError):
    pass


class Qwen:
    NAME = 'qwen'      # kept for callers; the recorded name is the one the server reports
    """llama.cpp /v1/rerank. The served model name of every response is recorded per score (a non-Qwen server is
    refused), and a document the server leaves out of its answer is a counted failure, not a -inf score (T027 M2, M3)."""

    def __init__(self, url):
        self.url = url.rstrip('/')
        self.stat = Counter()
        self.name = None

    def f(self, q, docs):
        import urllib.request
        req = urllib.request.Request(self.url + '/v1/rerank', headers={'Content-Type': 'application/json'},
                                     data=json.dumps({'query': q, 'documents': list(docs), 'top_n': len(docs)}).encode())
        with urllib.request.urlopen(req, timeout=120.0) as r:
            body = json.loads(r.read())
        name = str(body.get('model') or 'unknown')
        if 'qwen' not in name.lower():
            raise WrongModel(f'reranker at {self.url} reports model {name!r}: the prereg allows Qwen only')
        self.name = name
        self.stat[f'model:{name}'] += 1
        sc = [None] * len(docs)
        for x in body['results']:
            sc[x['index']] = float(x['relevance_score'])
        return sc

    def score(self, q, docs):
        """-> (scores or None per doc, [model name per doc], n_failed). docs already cut."""
        out, fails = [], 0
        for c in range(0, len(docs), BATCH):
            part, sc = docs[c:c + BATCH], None
            for attempt in range(2):
                try:
                    sc = self.f(q, part)
                    self.stat['batch_ok'] += 1
                    left_out = sum(1 for x in sc if x is None)      # the server did not score these: failures
                    self.stat['left_out'] += left_out
                    fails += left_out
                    break
                except WrongModel:
                    raise
                except Exception:  # noqa: BLE001
                    self.stat['batch_error'] += 1
                    time.sleep(5 * (attempt + 1))
            if sc is None:
                sc = []
                for d in part:
                    s1 = None
                    for cut in (len(d), 1500, 800):
                        try:
                            s1 = self.f(q, [d[:cut]])[0]
                            if s1 is None:
                                raise ValueError('document left out of the response')
                            self.stat['single_ok' if cut == len(d) else f'single_cut{cut}'] += 1
                            break
                        except WrongModel:
                            raise
                        except Exception:  # noqa: BLE001
                            self.stat['single_error'] += 1
                            time.sleep(2)
                    if s1 is None:
                        fails += 1
                    sc.append(s1)
            out += sc
        return out, [self.name] * len(docs), fails


def order_by(ids, scores):
    """descending score, ties keep input order, failed (None) last"""
    pos = {e: i for i, e in enumerate(ids)}
    return sorted(ids, key=lambda e: (scores[e] is None, -(scores[e] if scores[e] is not None else 0), pos[e]))


def uniq(seq):
    seen, out = set(), []
    for x in seq:
        if x not in seen:
            seen.add(x)
            out.append(x)
    return out


def wrrf(a, b, w, k=60):
    sc, first = {}, {}
    for r, t in enumerate(a):
        sc[t] = sc.get(t, 0.0) + 1.0 / (k + r + 1)
        first.setdefault(t, len(first))
    for r, t in enumerate(b):
        sc[t] = sc.get(t, 0.0) + w / (k + r + 1)
        first.setdefault(t, len(first))
    return sorted(sc, key=lambda t: (-sc[t], first[t]))


# ---------- filling and rendering ----------
def fill(order, budget, cost_of):
    used, sel, skipped = 0, [], 0
    for e in order:
        c = cost_of(e)
        if used + c > budget:
            skipped += 1
            continue
        used += c
        sel.append(e)
    return sel, used, skipped


def render_turns(sel, ep):
    by = defaultdict(list)
    for e in sel:
        by[ep(e)['session']].append(e)
    lines = []
    for sess in sorted(by, key=lambda s: min(str(ep(e)['occurred_at']) for e in by[s])):
        for e in sorted(by[sess], key=lambda e: ep(e)['seq']):
            r = ep(e)
            lines.append(f"[{r['occurred_at']} · {sess} · {r['role']}] {r['text']}")
    return lines


def pointers(ev):
    ids = [int(m.group(1)) for x in ev or [] if (m := re.fullmatch(r't?(\d+)', str(x).strip()))]
    return ids


def load_notes(conv):
    p = NOTES_DIR / f'conv{conv}.sqlite'
    if not p.exists():
        raise FileNotFoundError(f'no teacher notes for conv {conv}: {p}')
    con = sqlite3.connect(f'file:{p}?mode=ro&immutable=1', uri=True)
    items = []
    for path, content, ev, up in con.execute('SELECT path, content, evidence, updated FROM note'):
        items.append({'layer': 'note', 'id': path, 'text': f'{path}\n{content}', 'content': content,
                      'evidence': json.loads(ev or '[]'), 'day': str(up)[:10]})
    return items


def run_conv(conv, qwen, samples, outdir, qfilter=None):
    from beam_official.prompts import answer_generation_for_rag
    from beam_qa import _recaller, questions
    from pm1.lang import detect  # noqa: F401
    from pm1.recall import est_tokens, rrf
    from pm1.tree_index import TreeIndex
    store, rec = _recaller(conv)
    rec.k_first = K_FIRST
    from beam_cfg import STORES
    raw = sqlite3.connect(f'file:{STORES / f"conv{conv}.sqlite"}?mode=ro&immutable=1', uri=True)
    cache = {}

    def ep(e):
        if e not in cache:
            r = raw.execute('SELECT session, seq, role, occurred_at, text FROM episodes WHERE id=?', (e,)).fetchone()
            cache[e] = dict(zip(('session', 'seq', 'role', 'occurred_at', 'text'), r)) if r else None
        return cache[e]
    notes = load_notes(conv) if NOTES_ON else []
    fx = None
    if FACTS_ON:                       # the conversation's fact index (eval/beam1m_facts.py), same dense model as recall
        from pm1 import Store
        from pm1.facts import FactIndex
        fstore = Store(str(FACTS_DIR / f'conv{conv}.sqlite'))
        fx = FactIndex(fstore, rec.dense)
        while fx.embed_pending(256):
            pass
    nidx = TreeIndex(notes, rec.dense) if NOTES_ON else None
    qs = questions(conv)
    meta_f = open(outdir / f'meta_conv{conv}.jsonl', 'w')
    prom = {(a, b, s): open(outdir / f'prompts_conv{conv}_{a}_{b}_s{s}.jsonl', 'w')
            for a in ARMS for b in BUDGETS for s in range(1, samples + 1)}
    for cat in CATS:
        for i, qd in enumerate(qs[cat]):
            q, qid = qd['question'], f'{conv}|{cat}|{i}'
            if qfilter and qid not in qfilter:
                continue
            t0 = time.perf_counter()
            sp, _, st = rec.rank('beam', conv, q, 'rrf')
            if st.get('dense_error'):       # prereg s1: first stage = BM25 + dense; BM25-only is not that (T027 M1)
                raise RuntimeError(f'{qid}: dense retrieval was skipped ({st["dense_error"]}); rerun this run')
            rows = store.spans('beam', conv, sp)
            cand = uniq(rows[s]['episode_id'] for s in sp if s in rows)[:DEPTH]
            t_first = (time.perf_counter() - t0) * 1e3
            t1 = time.perf_counter()
            docs = [ep(e)['text'][:CUT] for e in cand]
            sc, models, fails = qwen.score(q, docs)
            score = dict(zip(cand, sc))
            base_order = order_by(cand, score)
            t_base = t_first + (time.perf_counter() - t1) * 1e3
            # notes side (only when an N / NC arm is asked for)
            t2 = time.perf_counter()
            nfails, nmodels, nscore, head, note_order, note_turns = 0, [], {}, [], [], []
            if NOTES_ON:
                qv = rec._qvec(q)
                bm = nidx.bm25(q)
                lex = [int(j) for j in np.argsort(-np.asarray(bm)) if bm[j] > 0]
                den = [int(j) for j in np.argsort(-(nidx.vec @ qv))]
                head = rrf([lex, den])[:NOTE_DEPTH]
                ndocs = [notes[j]['text'][:CUT] for j in head]
                nsc, nmodels, nfails = qwen.score(q, ndocs)
                nscore = dict(zip(head, nsc))
                note_order = order_by(head, nscore)
                note_turns = uniq(t for j in note_order for t in pointers(notes[j]['evidence']) if ep(t))
            t_note = (time.perf_counter() - t2) * 1e3
            cost = lambda e: est_tokens(ep(e)['text']) + OVERHEAD  # noqa: E731
            exc = bool(EXC.search(q))
            U_order = base_order if exc else ([e for e in base_order if ep(e)['role'] == 'user'] +
                                              [e for e in base_order if ep(e)['role'] != 'user'])
            N_order = wrrf(base_order, note_turns, W_NOTES) if NOTES_ON else []
            UF_orders = {}
            if FACTS_ON:                 # pm1 recall's rule: <3000 -> 8 facts after position 2; >=3000 -> 16 after 5 + hop 16
                qv = rec._qvec(q)
                for B in BUDGETS:
                    k, pos, hop = (16, 5, 16) if B >= 3000 else (8, 2, 0)
                    x = [e for e in fx.episodes_for('beam', str(conv), qv, k, hop=hop, question=q)
                         if ep(e) and e not in U_order[:pos]]
                    xs = set(x)
                    UF_orders[B] = U_order[:pos] + x + [e for e in U_order[pos:] if e not in xs]
            m = {'id': qid, 'conv': conv, 'type': cat, 'regex_exception': exc,
                 'cand': cand, 'cand_scores': [score[e] for e in cand], 'rerank_models': sorted(set(models)),
                 'rerank_failed': fails, 'note_rerank_failed': nfails, 'note_models': sorted(set(nmodels)),
                 'notes_top': [[notes[j]['id'], nscore[j]] for j in note_order[:20]], 'n_note_turns': len(note_turns),
                 'ms': {'base': round(t_base, 1), 'note_side': round(t_note, 1)}, 'budgets': {}}
            for B in BUDGETS:
                ctx = {}
                sel, used, sk = fill(base_order, B, cost)
                ctx['base'] = (render_turns(sel, ep), sel, used, sk)
                sel, used, sk = fill(U_order, B, cost)
                ctx['U'] = (render_turns(sel, ep), sel, used, sk)
                if FACTS_ON:
                    sel, used, sk = fill(UF_orders[B], B, cost)
                    ctx['UF'] = (render_turns(sel, ep), sel, used, sk)
                if NOTES_ON:
                    sel, used, sk = fill(N_order, B, cost)
                    ctx['N'] = (render_turns(sel, ep), sel, used, sk)
                # NC: half notes, half raw
                ncost = lambda j: est_tokens(f"[{','.join('t%d' % x for x in pointers(notes[j]['evidence']))} · {notes[j]['day']}] "
                                              f"{notes[j]['content']}") + OVERHEAD  # noqa: E731
                if NOTES_ON:
                    nsel, nused, nsk = fill(note_order, B // 2, ncost)
                    rsel, rused, rsk = fill(base_order, B - B // 2, cost)
                    nlines = [f"[{','.join('t%d' % x for x in pointers(notes[j]['evidence']))} · {notes[j]['day']}] {notes[j]['content']}"
                              for j in nsel]
                    lines = (['Memory notes (each cites the original turns it came from):'] + nlines + ['Original turns:'] if nlines else []) \
                        + render_turns(rsel, ep)
                    ctx['NC'] = (lines, rsel, nused + rused, nsk + rsk)
                m['budgets'][B] = {}
                for a in ARMS:
                    lines, sel, used, sk = ctx[a]
                    m['budgets'][B][a] = {'turn_ids': sel, 'tokens': used, 'skipped_too_big': sk,
                                          **({'note_ids': [notes[j]['id'] for j in nsel], 'note_tokens': nused} if a == 'NC' else {})}
                    prompt = answer_generation_for_rag.replace('<context>', '\n\n'.join(lines)).replace('<question>', q)
                    for s in range(1, samples + 1):
                        prom[(a, B, s)].write(json.dumps({'id': qid, 'prompt': prompt}, ensure_ascii=False) + '\n')
            meta_f.write(json.dumps(m, ensure_ascii=False) + '\n')
            meta_f.flush()
            print(f'conv {conv} {cat}[{i}] base {t_base:.0f} ms, notes {t_note:.0f} ms, failed {fails}/{nfails}', flush=True)
    for f in list(prom.values()) + [meta_f]:
        f.close()
    store.close()


def export(run, convs, samples=2):
    samples = int(samples)
    if not os.environ.get('PM1_RERANK_QWEN') or os.environ.get('PM1_RERANK_GTE'):
        raise SystemExit('need PM1_RERANK_QWEN and no PM1_RERANK_GTE')
    if os.environ.get('PM1_BEAM_SET') != '1m':
        raise SystemExit('PM1_BEAM_SET must be 1m')
    if os.environ.get('PM1_BEAM_DENSE', 'onnx') != 'onnx':      # the beam1m stores only hold e5-small-onnx vectors (T027 M1b)
        raise SystemExit('PM1_BEAM_DENSE must be onnx: the BEAM-1M stores have no vectors for another dense model')
    out = DATA / f'mv_{run}'
    if (out / 'env.json').exists():                             # prereg s6: one directory per run, never overwritten (M4)
        raise SystemExit(f'{out} exists: a run is exported once, with all its convs; use a new run id')
    out.mkdir(parents=True, exist_ok=True)
    import subprocess
    git = lambda *a: subprocess.run(['git', '-C', str(HERE), *a], capture_output=True, text=True).stdout.strip()
    (out / 'env.json').write_text(json.dumps({k: v for k, v in os.environ.items() if k.startswith('PM1_')} |
                                             {'notes_dir': str(NOTES_DIR), 'time': time.strftime('%F %T'),
                                              'git_commit': git('rev-parse', 'HEAD'),
                                              'git_dirty': bool(git('status', '--porcelain', '--', str(HERE.parent)))}, indent=1))
    qwen = Qwen(os.environ['PM1_RERANK_QWEN'])
    qf = set(json.loads(os.environ['MV_ONLY'])) if os.environ.get('MV_ONLY') else None   # smoke tests only
    for conv in convs.split(','):
        run_conv(conv, qwen, samples, out, qf)
    (out / 'rerank_stats.json').write_text(json.dumps(dict(qwen.stat)))
    print('rerank stats', dict(qwen.stat))
    # merge per-conv files into the per-(arm,budget,sample) prompt files that beam_answer_tasks understands
    for a in ARMS:
        for B in BUDGETS:
            for s in range(1, samples + 1):
                rows = []
                for c in convs.split(','):
                    rows += (out / f'prompts_conv{c}_{a}_{B}_s{s}.jsonl').read_text().splitlines()
                (DATA / f'beam_answer_prompts_mv_{run}_{a}_{B}_s{s}.jsonl').write_text('\n'.join(rows) + ('\n' if rows else ''))


def make_answers(run, budget=2000, size=40):
    from beam_answer_tasks import SPEC
    budget, size = int(budget), int(size)
    items = []
    for a in ARMS:
        for f in sorted(DATA.glob(f'beam_answer_prompts_mv_{run}_{a}_{budget}_s*.jsonl')):
            tag = f.stem.replace('beam_answer_prompts_', '')
            items += [(tag, json.loads(l)) for l in f.read_text().splitlines() if l.strip()]
    random.Random(SHUFFLE_SEED).shuffle(items)
    prefix, mp = f'mv_{run}_ans_{budget}', {}
    for k in range(0, len(items), size):
        task = f'{prefix}_{k // size + 1:03d}'
        rows = []
        for n, (t, r) in enumerate(items[k:k + size]):
            hid = f'q{k + n:05d}'
            mp[hid] = [t, r['id']]
            rows.append({'id': hid, 'prompt': r['prompt']})
        (LAB / 'inbox' / f'{task}.jsonl').write_text(''.join(json.dumps(o) + '\n' for o in rows), encoding='utf-8')
        (LAB / 'inbox' / f'{task}.md').write_text(SPEC.format(task=task), encoding='utf-8')
    (DATA / f'{prefix}.map.json').write_text(json.dumps(mp))
    print(f'{prefix}: {len(items)} items in {(len(items) + size - 1) // size} tasks')


def gold_of(qid):
    from beam_qa import questions
    conv, cat, i = qid.split('|')
    r = questions(conv)[cat][int(i)]
    rub = r['rubric']
    return cat, r['question'], '\n'.join(f'- {x}' for x in rub) if isinstance(rub, list) else str(rub)


def make_judge(run, budget=2000, labeler='gpt', size=40):
    budget, size = int(budget), int(size)
    items, mp = [], {}
    for f in sorted(g for g in DATA.glob(f'beam_answers_{labeler}_mv_{run}_*_{budget}_s*.jsonl')
                    if re.fullmatch(rf'beam_answers_{labeler}_mv_{re.escape(run)}_(?:base|U|N|NC|UF)_{budget}_s\d+', g.stem)):
        tag = f.stem.replace(f'beam_answers_{labeler}_', '')
        for r in map(json.loads, f.read_text().splitlines()):
            cat, qq, gold = gold_of(r['id'])
            items.append((tag, r['id'], f"TYPE: {cat}\nQUESTION: {qq}\nGOLD:\n{gold}\nANSWER: {r.get('response', '')}"))
    random.Random(SHUFFLE_SEED).shuffle(items)
    for k in range(0, len(items), size):
        part = f'pm1_mv_{run}_judge_{budget}_{k // size + 1:03d}'
        rows = []
        for n, (tag, qid, p) in enumerate(items[k:k + size]):
            h = f'j{k + n:05d}'
            mp[h] = [tag, qid]
            rows.append({'id': h, 'prompt': p})
        (LAB / 'inbox' / f'{part}.jsonl').write_text(''.join(json.dumps(x, ensure_ascii=True) + '\n' for x in rows))
        (LAB / 'inbox' / f'{part}.md').write_text(f'# {part} — grade answers against gold\n\n{JUDGE}\n\nEach line of {part}.jsonl has an '
                                                  '`id` and a `prompt` (TYPE, QUESTION, GOLD, ANSWER).\n')
    (DATA / f'pm1_mv_{run}_judge_{budget}.map.json').write_text(json.dumps(mp))
    print(len(items), 'judge items')


def scores(run, budget, labeler):
    mp = json.loads((DATA / f'pm1_mv_{run}_judge_{budget}.map.json').read_text())
    sc = {}
    for f in sorted((LAB / 'outbox' / labeler).glob(f'pm1_mv_{run}_judge_{budget}_*.out.jsonl'), key=lambda p: ('_fix_' in p.name, p.name)):
        for l in open(f):
            if not l.strip():
                continue
            try:
                r = json.loads(l)
            except ValueError:      # a judge line with a missing brace / quote: the id and score are still plain text in it
                m = re.search(r'"id":\s*"(j\d+)".*?"score":\s*([01](?:\.5)?)\b', l)
                if not m:
                    raise
                print(f'  recovered malformed judge line {m.group(1)} from {f.name}', file=sys.stderr)
                r = {'id': m.group(1), 'response': {'score': m.group(2)}}
            v = r.get('response')
            v = json.loads(v) if isinstance(v, str) else (v or {})
            try:
                sc[r['id']] = float(v.get('score'))
            except (TypeError, ValueError, AttributeError):
                continue
    out = defaultdict(lambda: defaultdict(list))
    for h, s in sc.items():
        tag, qid = mp[h]
        arm = re.match(rf'mv_{run}_(\w+?)_{budget}_s\d+$', tag).group(1)
        out[arm][qid].append(s)
    return {a: {q: sum(v) / len(v) for q, v in d.items()} for a, d in out.items()}


def boot(diffs):
    rng = random.Random(BOOT_SEED)
    n = len(diffs)
    bs = sorted(sum(diffs[rng.randrange(n)] for _ in range(n)) / n for _ in range(BOOT_N))
    p = min(1.0, 2 * min((sum(b <= 0 for b in bs) + 1) / (BOOT_N + 1), (sum(b >= 0 for b in bs) + 1) / (BOOT_N + 1)))
    return statistics.mean(diffs), bs[int(BOOT_N * .025)], bs[int(BOOT_N * .975) - 1], p


def report(run, labeler='gpt'):
    meta = {}
    for f in (DATA / f'mv_{run}').glob('meta_conv*.jsonl'):
        for l in open(f):
            m = json.loads(l)
            meta[m['id']] = m
    M = scores(run, 2000, labeler)
    ids = sorted(set.intersection(*[set(M[a]) for a in ARMS]))
    print(f'questions with all 4 arms scored (2000 tokens): {len(ids)}')
    comps = [('U-base', 'U', 'base'), ('N-U', 'N', 'U'), ('NC-U', 'NC', 'U')]
    res = []
    for name, a, b in comps:
        d = [M[a][q] - M[b][q] for q in ids]
        res.append((name, *boot(d)))
    order = sorted(range(3), key=lambda i: res[i][4])
    holm, run_max = [None] * 3, 0.0
    for rank, i in enumerate(order):
        run_max = max(run_max, min(1.0, res[i][4] * (3 - rank)))
        holm[i] = run_max
    print('PRIMARY (prereg s3; 2K budget; paired by question; bootstrap 4000 seed 2026; Holm over 3)')
    for (name, mean, lo, hi, p), hp in zip(res, holm):
        print(f'  {name}: {mean:+.4f} [{lo:+.4f}, {hi:+.4f}]  raw p={p:.4f}  Holm p={hp:.4f}  '
              f'SUCCESS={"yes" if hp < 0.05 and mean >= 0.03 else "no"}')
    # guardrail (prereg s3, amended 2026-10-20 before any answer): no question
    # of the 200 matches the "you said ..." regex, so the regex subset is empty; the guardrail is U-base per BEAM
    # question type, a type whose point estimate is below -0.05 fails it; plus the LME single-session-assistant check (s5)
    guard = [q for q in ids if meta.get(q, {}).get('regex_exception')]
    print(f'GUARDRAIL (a) U-base on questions matching the prereg regex (U==base there by construction): n={len(guard)}')
    print('GUARDRAIL (b) U-base per question type (fails below -0.05):')
    for cat in CATS:
        sub = [q for q in ids if q.split('|')[1] == cat]
        if sub:
            d = statistics.mean(M['U'][q] - M['base'][q] for q in sub)
            print(f'  {cat} n={len(sub)} {d:+.3f}' + ('   <-- GUARDRAIL FAILED' if d < -0.05 else ''))
    print('SECONDARY (report only, no decision)')
    for a in ARMS:
        print(f'  mean {a}: {statistics.mean(M[a][q] for q in ids):.4f}')
    for cat in CATS:
        sub = [q for q in ids if q.split('|')[1] == cat]
        if sub:
            print(f'  {cat} n={len(sub)} ' + ' '.join(f'{a}={statistics.mean(M[a][q] for q in sub):.3f}' for a in ARMS))
    rf = sum(m['rerank_failed'] + m['note_rerank_failed'] for m in meta.values())
    print(f'  rerank failures (docs): {rf}; questions: {len(meta)}')
    for a in ARMS:
        tok = [m['budgets']['2000'][a]['tokens'] for m in meta.values()]
        print(f'  {a}: mean context tokens {statistics.mean(tok):.0f}')
    for k in ('base', 'note_side'):
        v = sorted(m['ms'][k] for m in meta.values())
        print(f'  latency {k}: p50 {v[len(v) // 2]:.0f} ms, p90 {v[int(len(v) * .9)]:.0f} ms')


def report_facts(run, labeler='composer'):
    """Fact index on BEAM-1M: UF - U at 2K (primary: success = mean >= +0.03 and CI low > 0; guardrail: no
    question type below -0.05) and at 6K (secondary), paired by question, bootstrap 4000, seed 2026."""
    meta = {}
    for f in (DATA / f'mv_{run}').glob('meta_conv*.jsonl'):
        for l in open(f):
            m = json.loads(l)
            meta[m['id']] = m
    for budget in (2000, 6000):
        M = scores(run, budget, labeler)
        ids = sorted(set(M.get('UF', {})) & set(M.get('U', {})))
        if not ids:
            print(f'{budget}: no questions scored in both arms')
            continue
        mean, lo, hi, p = boot([M['UF'][q] - M['U'][q] for q in ids])
        u, uf = statistics.mean(M['U'][q] for q in ids), statistics.mean(M['UF'][q] for q in ids)
        tag = 'PRIMARY' if budget == 2000 else 'secondary'
        print(f'{tag} {budget}: n {len(ids)}  U {u:.4f}  UF {uf:.4f}  UF-U {mean:+.4f} [{lo:+.4f}, {hi:+.4f}] p={p:.4f}'
              + (f'  SUCCESS={"yes" if mean >= 0.03 and lo > 0 else "no"}' if budget == 2000 else ''))
        by = defaultdict(list)
        for q in ids:
            by[(meta.get(q) or {}).get('type') or (meta.get(q) or {}).get('category') or '?'].append(M['UF'][q] - M['U'][q])
        for t, d in sorted(by.items()):
            flag = '  GUARDRAIL' if budget == 2000 and statistics.mean(d) < -0.05 else ''
            print(f'    {t:28s} n {len(d):3d}  UF-U {statistics.mean(d):+.3f}{flag}')


def report_ext(ext, labeler='composer'):
    """Amendment 3 (the one extension the prereg allows): U-base over r1's 200 questions plus the extension run's new
    questions (base and U only). Holm over 3 as before: N-U and NC-U keep their r1 p-values (they were not extended)."""
    A, B = scores('r1', 2000, labeler), scores(ext, 2000, labeler)
    d1 = [A['U'][q] - A['base'][q] for q in sorted(set(A['U']) & set(A['base']))]
    d2 = [B['U'][q] - B['base'][q] for q in sorted(set(B['U']) & set(B['base']))]
    ids1 = sorted(set(A['U']) & set(A['base']))
    ids = sorted(set(B['U']) & set(B['base']))
    print(f'U-base questions: r1 {len(d1)} + {ext} {len(d2)} = {len(d1) + len(d2)}')
    mean, lo, hi, pU = boot(d1 + d2)
    pn, pnc = 0.2244, 0.8813                      # r1 raw p of N-U and NC-U (report r1)
    holm = min(1.0, pU * 3) if pU <= min(pn, pnc) else None
    if holm is None:
        order = sorted([('U', pU), ('N', pn), ('NC', pnc)], key=lambda t: t[1])
        run_max, hp = 0.0, {}
        for rank, (k, pv) in enumerate(order):
            run_max = max(run_max, min(1.0, pv * (3 - rank)))
            hp[k] = run_max
        holm = hp['U']
    print(f'U-base: {mean:+.4f} [{lo:+.4f}, {hi:+.4f}]  raw p={pU:.4f}  Holm p={holm:.4f}  '
          f'SUCCESS={"yes" if holm < 0.05 and mean >= 0.03 else "no"}')
    by = defaultdict(list)
    for q in ids1:
        by[q.split('|')[1]].append(A['U'][q] - A['base'][q])
    for q in ids:
        by[q.split('|')[1]].append(B['U'][q] - B['base'][q])
    print('GUARDRAIL U-base per question type (fails below -0.05):')
    for cat in CATS:
        if by[cat]:
            m = statistics.mean(by[cat])
            print(f'  {cat} n={len(by[cat])} {m:+.3f}' + ('   <-- GUARDRAIL FAILED' if m < -0.05 else ''))
    conv = defaultdict(list)
    for q in ids1:
        conv[q.split('|')[0]].append(A['U'][q] - A['base'][q])
    for q in ids:
        conv[q.split('|')[0]].append(B['U'][q] - B['base'][q])
    cm = [statistics.mean(v) for v in conv.values()]
    print(f'SECONDARY per-conversation: {len(cm)} conversations, mean of conv means {statistics.mean(cm):+.4f}, '
          f'conv-level bootstrap {boot(cm)[:3]}, conversations with U>base: {sum(x > 0 for x in cm)}')


if __name__ == '__main__':
    cmd, *a = sys.argv[1:]
    {'export': export, 'make-answers': make_answers, 'make-judge': make_judge, 'report': report, 'report-ext': report_ext,
     'report-facts': report_facts}[cmd](*a)
