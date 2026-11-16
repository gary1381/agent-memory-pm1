"""Fact route on BEAM-1M (an experiment, not adopted). pm1's U context at
6,000 tokens (the data/mv_f1 export of mv_arms.py: conv 1-10, 200 questions, 48-turn reranked pool) stays as it is; after
it come the source turns of the luna facts (beam1m_facts.py) the 0.6B rates highest for the question (cosine top --topf,
then the 0.6B), up to +X tokens. Control: U's own next turns from its pool with the same X. Gold: source_chat_ids (seq).

  PM1_BEAM_SET=1m PM1_GPU=0 OMP_NUM_THREADS=2 ../.venv/bin/python beam_factroute.py evidence
  ... beam_factroute.py export bfr1       contexts for U6 and UFR (+2,000) -> competitors/runs/bfr1/contexts.jsonl
  (answers: competitors .venv/bin/python harness/answer.py --systems U6,UFR with CMP_RUN=bfr1)
  ... beam_factroute.py judge bfr1        composer-2.5, BEAM rubric (mv_arms.JUDGE), blind and shuffled -> judged.jsonl
  ... beam_factroute.py judge bfc1 --cursor grok-4.7   one item per call through competitors harness/cursor_run.py, for
                                         answers composer-2.5 wrote (luna's weekly quota ran out 11-02; amendment 1)
  ... beam_factroute.py report bfr        three batches bfr1..3, per question mean, paired bootstrap
"""
from __future__ import annotations

import argparse
import hashlib
import json
import os
import pickle
import random
import sqlite3
import statistics
import sys
from collections import defaultdict
from pathlib import Path

import numpy as np

EV = Path(__file__).resolve().parent
sys.path.insert(0, str(EV.parent / 'src'))
sys.path.insert(0, str(EV))
HEAD, EXTRA, ANSWER_EXTRA = 6000, (1000, 2000, 4000), 2000
RERANK = {'PM1_RERANK_QWEN_VLLM': 'http://127.0.0.1:18166', 'PM1_RERANK_PROFILE': 'wait', 'PM1_QUALITY_FLOOR': 'wait',
          'PM1_JUDGE': 'off'}
SCORES = Path.home() / '.cache/pm1-eval/beam1m_factroute_scores.pkl'
RUNS = Path(os.environ.get('PM1_EVAL_RUNS', str(Path.home() / '.cache/pm1-eval/runs')))
MORE = 'More memories (oldest to newest):'


def boot(d, n=4000):
    rng = random.Random(2026)
    m = sorted(sum(rng.choices(d, k=len(d))) / len(d) for _ in range(n))
    return m[int(n * .025)], m[int(n * .975)]


def routes(topf):
    """Per question of conv 1-10: (meta row, question dict, gold episode ids, U head at 6K, {arm: turns after the head in
    that arm's order}, episode lookup)."""
    os.environ.update(RERANK)
    from beam_cfg import STORES
    from beam_coverage import gold_ids
    from beam_qa import questions
    from mv_arms import FACTS_DIR, order_by, uniq
    from pm1.embed import FastE5
    from pm1.rerank import router_from_env
    score = router_from_env().rerankers[0].score
    cache = pickle.load(open(SCORES, 'rb')) if SCORES.exists() else {}

    def ce(q, docs):
        keys = [hashlib.sha1(f'{q}\0{d}'.encode()).digest()[:12] for d in docs]
        miss = [i for i, k in enumerate(keys) if k not in cache]
        if miss:
            for i, s in zip(miss, score(q, [docs[i] for i in miss])):
                cache[keys[i]] = s
        return [cache[k] for k in keys]

    dense = FastE5(gpu=False)
    for conv in range(1, 11):
        meta = [json.loads(x) for x in open(EV / 'data' / 'mv_f1' / f'meta_conv{conv}.jsonl')]
        qs = questions(str(conv))
        raw = sqlite3.connect(f'file:{STORES / f"conv{conv}.sqlite"}?mode=ro&immutable=1', uri=True)
        ep = {r[0]: dict(zip(('session', 'seq', 'role', 'occurred_at', 'text'), r[1:]))
              for r in raw.execute('SELECT id, session, seq, role, occurred_at, text FROM episodes')}
        by_seq = {v['seq']: k for k, v in ep.items()}
        fcon = sqlite3.connect(f'file:{FACTS_DIR / f"conv{conv}.sqlite"}?mode=ro', uri=True)
        rows = fcon.execute("SELECT f.text, f.episodes, v.vec FROM facts f JOIN fact_vectors v ON v.fact_id = f.id "
                            "AND v.model = 'e5-small-onnx'").fetchall()
        ftext = [r[0] for r in rows]
        feps = [json.loads(r[1]) for r in rows]
        fmat = np.vstack([np.frombuffer(r[2], dtype=np.float16) for r in rows]).astype(np.float32)
        qd = {m['id']: qs[m['type']][int(m['id'].split('|')[2])] for m in meta}
        qv = dict(zip([m['id'] for m in meta], dense.embed([qd[m['id']]['question'] for m in meta], 'query')))
        for m in meta:
            q = qd[m['id']]['question']
            sc = dict(zip(m['cand'], m['cand_scores']))
            base = order_by(m['cand'], sc)
            u = base if m['regex_exception'] else ([e for e in base if ep[e]['role'] == 'user'] +
                                                    [e for e in base if ep[e]['role'] != 'user'])
            head = m['budgets'][str(HEAD)]['U']['turn_ids']
            hs = set(head)
            top = [int(j) for j in np.argsort(-(fmat @ qv[m['id']]))[:topf]]
            fs = ce(q, [ftext[j] for j in top])
            orders = {'ctl': [e for e in u if e not in hs],
                      'frv': uniq(e for j in top for e in feps[j] if e in ep and e not in hs),
                      'frce': uniq(e for j in order_by(top, dict(zip(top, fs))) for e in feps[j] if e in ep and e not in hs)}
            yield m, qd[m['id']], {by_seq[s] for s in gold_ids(qd[m['id']]) if s in by_seq}, head, orders, ep
        pickle.dump(cache, open(SCORES, 'wb'))
        print(f'conv{conv} done', flush=True)


def evidence(topf):
    from mv_arms import OVERHEAD, fill
    from pm1.recall import est_tokens
    res = defaultdict(list)                            # (arm, x) -> [(question type, full evidence, gold share)]
    for m, qd, gold, head, orders, ep in routes(topf):
        if not gold:
            continue
        cost = lambda e: est_tokens(ep[e]['text']) + OVERHEAD  # noqa: E731
        ev = lambda sel: (int(gold <= set(sel)), len(gold & set(sel)) / len(gold))  # noqa: E731
        res[('head', 0)].append((m['type'], *ev(head)))
        for x in EXTRA:
            for arm, order in orders.items():
                res[(arm, x)].append((m['type'], *ev(head + fill(order, x, cost)[0])))
    h = res[('head', 0)]
    print(f'{len(h)} questions with gold; U head at {HEAD}: full {statistics.mean(r[1] for r in h):.3f}, '
          f'gold share {statistics.mean(r[2] for r in h):.3f}')
    for x in EXTRA:
        c = res[('ctl', x)]
        print(f'+{x}: control full {statistics.mean(r[1] for r in c):.3f}, share {statistics.mean(r[2] for r in c):.3f}')
        for arm in ('frv', 'frce'):
            r = res[(arm, x)]
            for k, name in ((1, 'full'), (2, 'share')):
                d = [p[k] - q[k] for p, q in zip(r, c)]
                lo, hi = boot(d)
                print(f'  {arm:5s} {name:5s} {statistics.mean(p[k] for p in r):.3f}  vs control {statistics.mean(d) * 100:+.2f} '
                      f'[{lo * 100:+.2f},{hi * 100:+.2f}]', end='' if k == 1 else '\n')
        types = sorted({r[0] for r in c})
        print('  share by type (control -> frce): ' + ', '.join(
            f"{t} {statistics.mean(r[2] for r in c if r[0] == t):.2f}->"
            f"{statistics.mean(r[2] for r in res[('frce', x)] if r[0] == t):.2f}" for t in types))


def export(run, topf):
    """U6: exactly the mv_f1 U prompt at 6,000; UFR: the same context, then a block of up to 2,000 tokens of fact-routed
    turns (`frce` order, chronological within the block). Ids 'arm|6000|conv:type:i'."""
    from beam_official.prompts import answer_generation_for_rag
    from mv_arms import OVERHEAD, fill, render_turns
    from pm1.recall import est_tokens
    exported = {}
    for f in (EV / 'data' / 'mv_f1').glob('prompts_conv*_U_6000_s1.jsonl'):
        for x in open(f):
            r = json.loads(x)
            exported[r['id']] = r['prompt']
    out = RUNS / run
    out.mkdir(parents=True, exist_ok=True)
    rows, strata, same = [], {}, 0
    for m, qd, gold, head, orders, ep in routes(topf):
        look = ep.__getitem__
        cost = lambda e: est_tokens(ep[e]['text']) + OVERHEAD  # noqa: E731
        extra = fill(orders['frce'], ANSWER_EXTRA, cost)[0]
        q = qd['question']
        lines = render_turns(head, look)
        u = answer_generation_for_rag.replace('<context>', '\n\n'.join(lines)).replace('<question>', q)
        same += u == exported.get(m['id'])
        more = lines + ([MORE] + render_turns(extra, look) if extra else [])
        f = answer_generation_for_rag.replace('<context>', '\n\n'.join(more)).replace('<question>', q)
        qid = m['id'].replace('|', ':')
        for arm, p in (('U6', u), ('UFR', f)):
            rows.append({'id': f'{arm}|{HEAD}|{qid}', 'qid': qid, 'system': arm, 'budget': HEAD, 'prompt': p,
                         'n_extra': len(extra) if arm == 'UFR' else 0})
        strata[qid] = {'type': m['type'], 'n_gold': len(gold), 'n_extra': len(extra),
                       'share': [len(gold & set(head)) / len(gold), len(gold & set(head + extra)) / len(gold)] if gold else None}
    with open(out / 'contexts.jsonl', 'w') as fh:
        fh.writelines(json.dumps(r, ensure_ascii=False) + '\n' for r in rows)
    json.dump(strata, open(out / 'strata.json', 'w'))
    print(f'{len(strata)} questions, U6 prompt identical to the mv_f1 export: {same}, '
          f'mean extra turns {statistics.mean(s["n_extra"] for s in strata.values()):.1f}', flush=True)


def judge(run, batch=200, cursor=None):
    from mv_arms import JUDGE, gold_of
    from pm1 import judge as J
    out = RUNS / run
    answers = {r['id']: r for r in map(json.loads, open(out / 'answers.jsonl'))}
    path = out / 'judged.jsonl'
    have = {r['id'] for r in map(json.loads, open(path)) if r['score'] is not None} if path.exists() else set()
    ids = sorted(i for i in answers if i not in have)
    random.Random(2026).shuffle(ids)
    if cursor:
        return judge_cursor(out, answers, ids, cursor)
    J.LABELERS = (('composer', 'grok:composer-2.5'),)
    labels = {'1': 'correct and complete under the rules in the item', '0.5': 'partially correct under the rules',
              '0': 'wrong, contradicts the gold, or abstains when the gold has an answer'}
    with open(path, 'a') as fh:
        for b in range(0, len(ids), batch):
            chunk = ids[b:b + batch]
            items = []
            for n, i in enumerate(chunk):
                cat, qq, gold = gold_of(i.split('|')[2].replace(':', '|'))
                items.append((f'j{b + n:05d}', f"{JUDGE}\nTYPE: {cat}\nQUESTION: {qq}\nGOLD:\n{gold}\n"
                                               f"ANSWER: {answers[i]['answer'] or '(no answer)'}"))
            v = J.run('beam_factroute', items, 'Score the ANSWER against the GOLD with the rules above: 1, 0.5 or 0.',
                      labels) or {}
            for (jid, _), i in zip(items, chunk):
                lab = (v.get(jid) or {}).get('composer')
                fh.write(json.dumps({'id': i, 'score': float(lab) if lab in labels else None}) + '\n')
            fh.flush()
            print(f'batch {b // batch}: {len(chunk)} items, {sum(1 for j, _ in items if (v.get(j) or {}).get("composer"))} '
                  f'labelled', flush=True)


def judge_cursor(out, answers, ids, model):
    """One item per call (opaque id, shuffled), the same rubric; the score is read from the reply."""
    import re
    import subprocess
    from mv_arms import JUDGE, gold_of
    src, got = out / 'judge_items.jsonl', out / 'judge_out.jsonl'
    key = {f'j{n:05d}': i for n, i in enumerate(ids)}
    with open(src, 'w') as fh:
        for jid, i in key.items():
            cat, qq, gold = gold_of(i.split('|')[2].replace(':', '|'))
            fh.write(json.dumps({'id': jid, 'prompt': f"{JUDGE}\nid: {jid}\nTYPE: {cat}\nQUESTION: {qq}\nGOLD:\n{gold}\n"
                                                      f"ANSWER: {answers[i]['answer'] or '(no answer)'}"}) + '\n')
    got.unlink(missing_ok=True)
    cmp = RUNS.parent
    subprocess.run([os.environ.get('PM1_LABELER_PYTHON', 'python3'), 'harness/cursor_run.py', 'items',
                    '--model', model, '--src', str(src), '--out', str(got)], cwd=cmp, check=True)
    n = 0
    with open(out / 'judged.jsonl', 'a') as fh:
        for r in map(json.loads, open(got)):
            m = re.findall(r'"score"\s*:\s*"?(1|0\.5|0)\b', r['text'] or '')
            fh.write(json.dumps({'id': key[r['id']], 'score': float(m[-1]) if m else None, 'judge': model}) + '\n')
            n += bool(m)
    print(f'{len(key)} items, {n} scored by {model}', flush=True)


def report(prefix):
    per = []
    for b in (1, 2, 3):
        s = {}
        for r in map(json.loads, open(RUNS / f'{prefix}{b}' / 'judged.jsonl')):
            if r['score'] is not None:
                s[r['id']] = r['score']
        per.append(s)
    strata = json.load(open(RUNS / f'{prefix}1' / 'strata.json'))
    m = {}
    for arm in ('U6', 'UFR'):
        m[arm] = {q: statistics.mean(s[f'{arm}|{HEAD}|{q}'] for s in per) for q in strata
                  if all(f'{arm}|{HEAD}|{q}' in s for s in per)}
    qs = sorted(set(m['U6']) & set(m['UFR']))
    d = [m['UFR'][q] - m['U6'][q] for q in qs]
    lo, hi = boot(d, 10000)
    print(f"{len(qs)} questions, U6 {statistics.mean(m['U6'][q] for q in qs):.3f}, UFR {statistics.mean(m['UFR'][q] for q in qs):.3f}"
          f"  UFR - U6 {statistics.mean(d):+.3f} [{lo:+.3f},{hi:+.3f}]  batches "
          + ' '.join(f"{statistics.mean(s[f'UFR|{HEAD}|{q}'] - s[f'U6|{HEAD}|{q}'] for q in qs if f'UFR|{HEAD}|{q}' in s and f'U6|{HEAD}|{q}' in s):+.3f}"
                     for s in per))
    gain = [q for q in qs if strata[q]['share'] and strata[q]['share'][1] > strata[q]['share'][0]]
    print(f"evidence share rose on {len(gain)}: U6 {statistics.mean(m['U6'][q] for q in gain):.3f} -> UFR "
          f"{statistics.mean(m['UFR'][q] for q in gain):.3f}; other {len(qs) - len(gain)}: "
          f"{statistics.mean(m['UFR'][q] - m['U6'][q] for q in qs if q not in set(gain)):+.3f}")
    for t in sorted({strata[q]['type'] for q in qs}):
        tq = [q for q in qs if strata[q]['type'] == t]
        print(f"  {t:26s} n={len(tq):3d}  U6 {statistics.mean(m['U6'][q] for q in tq):.3f}  "
              f"UFR - U6 {statistics.mean(m['UFR'][q] - m['U6'][q] for q in tq):+.3f}")


if __name__ == '__main__':
    ap = argparse.ArgumentParser()
    ap.add_argument('cmd', choices=('evidence', 'export', 'judge', 'report'))
    ap.add_argument('run', nargs='?')
    ap.add_argument('--topf', type=int, default=192)
    ap.add_argument('--cursor', help='judge one item per call with this Cursor model (e.g. grok-4.7)')
    a = ap.parse_args()
    {'evidence': lambda: evidence(a.topf), 'export': lambda: export(a.run, a.topf),
     'judge': lambda: judge(a.run, cursor=a.cursor),
     'report': lambda: report(a.run)}[a.cmd]()
