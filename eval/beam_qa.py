"""BEAM-10M end to end: pm1 recall (local) -> answer (freellmapi) -> official BEAM judging (freellmapi judge).

  answer: for every probing question of a conversation, recall evidence from that conversation's pm1 store with
          the routed reranker, then ask the answer model using BEAM's own RAG answer prompt
          (`answer_generation_for_rag`). Output in the official answers format {category: [{question, llm_response}]}.
  judge : the official compute_metrics functions (vendored, MIT) with an OpenAI-compatible freellmapi model in
          place of gpt-4.1-mini. Event ordering uses the official *semantic* alignment (all-MiniLM-L6-v2, thr 0.65)
          instead of pairwise LLM alignment (~400 judge calls per question); the official report metric is kept
          (tau_norm for event ordering, llm_judge_score for the other nine categories).

  .venv/bin/python eval/beam_qa.py answer 1      # conversation id(s) or 'all'
  .venv/bin/python eval/beam_qa.py judge 1
  .venv/bin/python eval/beam_qa.py report

Offline (batch) path, when an external batch model does the answering / judging:
  .venv-gpu/bin/python eval/beam_qa.py export-answers all          -> eval/data/beam_answer_prompts.jsonl
  .venv/bin/python eval/beam_qa.py import-answers <answers.jsonl> <model>    ({"id","answer"} per line)
  .venv/bin/python eval/beam_qa.py export-judge                     -> eval/data/beam_judge_prompts.jsonl
  .venv/bin/python eval/beam_qa.py import-judge <judged.jsonl> <model>       ({"id","response"} per line)
"""
from __future__ import annotations

import ast
import json
import os
import statistics
import sys
import time
from pathlib import Path

import pyarrow.parquet as pq

HERE = Path(__file__).resolve().parent
sys.path.insert(0, str(HERE))
sys.path.insert(0, str(HERE.parent / 'src'))
from llm import chat  # noqa: E402

from pm1 import Recaller, Store  # noqa: E402
from pm1.embed import FastE5, HttpE5  # noqa: E402
from pm1.rerank import PROFILES, router_from_env  # noqa: E402

from beam_cfg import D, GLOB, NAME, STORES, conv_ids  # noqa: E402
TAG = os.environ.get('PM1_BEAM_TAG', '')               # e.g. 'v2': separate prompts/answers/results per run
OUT = HERE / 'reports' / (NAME + (f'_{TAG}' if TAG else ''))
SFX = f'_{TAG}' if TAG else ''
ANSWER_MODELS = os.environ.get('PM1_BEAM_ANSWER', 'gpt-oss-120b,gemini-3.6-flash,command-a').split(',')
JUDGE_MODELS = os.environ.get('PM1_BEAM_JUDGE', 'gpt-oss-120b,gemini-3.6-flash,command-a').split(',')
BUDGET = (None if os.environ.get('PM1_BEAM_CONTEXT') == 'adaptive'
          else int(os.environ.get('PM1_BEAM_CONTEXT', 12000)))           # evidence tokens per question
CATS = ['abstention', 'contradiction_resolution', 'event_ordering', 'information_extraction', 'instruction_following',
        'knowledge_update', 'multi_session_reasoning', 'preference_following', 'summarization', 'temporal_reasoning']


def questions(conv: str) -> dict:
    for f in sorted(D.glob(GLOB)):
        for r in pq.read_table(f, columns=['conversation_id', 'probing_questions']).to_pylist():
            if r['conversation_id'] == conv:
                return ast.literal_eval(r['probing_questions'])
    raise KeyError(conv)


def ask(prompt: str, models: list[str], max_tokens: int = 2048) -> tuple[str, str]:
    last = None
    for _round in range(6):
        for m in models:
            try:
                txt, used = chat([{'role': 'user', 'content': prompt}], model=m, max_tokens=max_tokens,
                                 timeout=180, retries=1)
                if txt.strip():
                    return txt.strip(), used
            except Exception as exc:
                last = exc
        time.sleep(60 * (_round + 1))            # free-tier quotas: wait and retry rather than give up
    raise RuntimeError(f'no model answered: {last}')


def answer(conv: str) -> Path:
    from beam_official.prompts import answer_generation_for_rag
    OUT.mkdir(parents=True, exist_ok=True)
    out = OUT / f'answers_conv{conv}.json'
    done = json.loads(out.read_text()) if out.exists() else {}
    store = Store(STORES / f'conv{conv}.sqlite')
    dense = FastE5(gpu=os.environ.get('PM1_GPU', '1') != '0') if os.environ.get('PM1_BEAM_DENSE', 'onnx') == 'onnx' else \
        HttpE5(os.environ.get('PM1_E5_QUERY_URL', 'http://127.0.0.1:18170/v1/embeddings'), name='e5-small-q8')
    rec = Recaller(store, dense=dense, router=router_from_env(), k_first=100,
                   profile=os.environ.get('PM1_RERANK_PROFILE', 'balanced'))
    qs = questions(conv)
    for cat in CATS:
        rows = done.setdefault(cat, [])
        for i, q in enumerate(qs[cat]):
            if i < len(rows):
                continue
            t0 = time.perf_counter()
            res = rec.recall('beam', conv, q['question'], budget_tokens=BUDGET, max_items=80)
            t_recall = (time.perf_counter() - t0) * 1e3
            prompt = answer_generation_for_rag.replace('<context>', res.context).replace('<question>', q['question'])
            resp, model = ask(prompt, ANSWER_MODELS)
            d = res.diagnostics
            rows.append({'question': q['question'], 'llm_response': resp, 'answer_model': model,
                         'recall_ms': round(t_recall, 1), 'context_tokens': d.get('tokens'),
                         'route': d.get('route'), 'rerank_model': d.get('rerank_model'),
                         'source_chat_ids': [int(it['source'].split(':')[1]) for it in res.items if it.get('source')]})
            out.write_text(json.dumps(done, indent=1, ensure_ascii=False))
            print(f'conv {conv} {cat}[{i}] recall {t_recall:.0f} ms, {d.get("tokens")} tok, '
                  f'{d.get("rerank_model")} -> {model}', flush=True)
    store.close()
    return out


class FreeLLM:
    """The tiny LangChain-like surface compute_metrics uses: llm.invoke(prompt_or_messages).content"""
    class _R:
        def __init__(self, content):
            self.content = content

    def __init__(self, models):
        self.models, self.used = models, {}

    def invoke(self, p):
        msgs = p if isinstance(p, list) else [{'role': 'user', 'content': p}]
        prompt = '\n\n'.join(m['content'] for m in msgs)
        txt, used = ask(prompt, self.models, max_tokens=1024)
        self.used[used] = self.used.get(used, 0) + 1
        return self._R(txt)


def judge(conv: str) -> Path:
    from beam_official import compute_metrics as cm
    cm.initialize_models = lambda: None
    llm = FreeLLM(JUDGE_MODELS)
    answers = json.loads((OUT / f'answers_conv{conv}.json').read_text())
    qs = questions(conv)
    out = OUT / f'evaluation-results_conv{conv}.json'
    done = json.loads(out.read_text()) if out.exists() else {}
    for cat in CATS:
        rows = done.setdefault(cat, [])
        for i, a in enumerate(answers.get(cat, [])):
            if i < len(rows):
                continue
            rubric = qs[cat][i]['rubric']
            if cat == 'event_ordering':
                score = cm.event_ordering_score(reference_list=rubric, system_list=a['llm_response'].split('\n'),
                                                align_type='semantic', llm=llm)
                score['llm_judge_score'] = None
                res = score
            else:
                fn = getattr(cm, f'evaluate_{cat}')
                res = fn(rubric=rubric, llm_response=a['llm_response'], probing_question=a['question'], model=llm)
            rows.append(res)
            out.write_text(json.dumps(done, indent=1, ensure_ascii=False))
            print(f'judged conv {conv} {cat}[{i}]: {res.get("tau_norm") if cat == "event_ordering" else res["llm_judge_score"]}',
                  flush=True)
    print('judge models used:', llm.used)
    return out


def report():
    per_cat = {c: [] for c in CATS}
    for f in sorted(OUT.glob('evaluation-results_conv*.json')):
        data = json.loads(f.read_text())
        for c in CATS:
            for r in data.get(c, []):
                per_cat[c].append(r['tau_norm'] if c == 'event_ordering' else r['llm_judge_score'])
    rep = {c: (round(statistics.mean(v), 3), len(v)) for c, v in per_cat.items() if v}
    overall = statistics.mean(m for m, _ in rep.values()) if rep else None
    print(json.dumps({'per_category': rep, 'overall_mean_of_categories': overall and round(overall, 3)}, indent=1))
    (OUT / 'summary.json').write_text(json.dumps({'per_category': rep, 'overall': overall}, indent=1))


# ---------- offline (batch) path: prompts out as JSONL, model outputs back in ----------
DATA = HERE / 'data'


def _recaller(conv):
    store = Store(STORES / f'conv{conv}.sqlite')
    dense = FastE5(gpu=os.environ.get('PM1_GPU', '1') != '0') if os.environ.get('PM1_BEAM_DENSE', 'onnx') == 'onnx' else \
        HttpE5(os.environ.get('PM1_E5_QUERY_URL', 'http://127.0.0.1:18170/v1/embeddings'), name='e5-small-q8')
    ledger = None
    if os.environ.get('PM1_LEDGER') == '1':           # P2: attach the consolidated state ledger
        from pm1.ledger import Ledger
        ledger = Ledger(store)
    tree_factory = None
    if os.environ.get('PM1_TREE_DIR'):                 # memory tree v2: teacher- or student-built trees, one per conv
        from pm1.tree import Tree

        def tree_factory(bank, scope, _d=Path(os.environ['PM1_TREE_DIR'])):
            p = _d / f'conv{scope}.sqlite'
            return Tree(Store(p), os.environ.get('PM1_TREE_BANK', 'beam'), str(scope)) if p.exists() else None
    return store, Recaller(store, dense=dense, router=router_from_env(), k_first=100, tree_factory=tree_factory,
                           profile=os.environ.get('PM1_RERANK_PROFILE', 'balanced'),
                           planning=os.environ.get('PM1_PLANNING', '1') != '0', ledger=ledger,
                           ledger_mode=os.environ.get('PM1_LEDGER_MODE', 'block'),
                           directive_min_sim=float(os.environ['PM1_DIR_MIN']) if os.environ.get('PM1_DIR_MIN') else None)


def export_answers(convs):
    """recall locally, write one answer prompt per question: {"id": "conv|category|index", "prompt": ...}"""
    from beam_official.prompts import answer_generation_for_rag
    DATA.mkdir(exist_ok=True)
    OUT.mkdir(parents=True, exist_ok=True)
    meta_f = OUT / 'recall_meta.json'
    meta = json.loads(meta_f.read_text()) if meta_f.exists() else {}
    out = DATA / f'beam_answer_prompts{SFX}.jsonl'
    have = {json.loads(l)['id'] for l in out.read_text().splitlines()} if out.exists() else set()
    with out.open('a', encoding='utf-8') as f:
        for conv in convs:
            store, rec = _recaller(conv)
            qs = questions(conv)
            for cat in CATS:
                for i, q in enumerate(qs[cat]):
                    qid = f'{conv}|{cat}|{i}'
                    if qid in have:
                        continue
                    t0 = time.perf_counter()
                    res = rec.recall('beam', conv, q['question'], budget_tokens=BUDGET, max_items=80)
                    d = res.diagnostics
                    meta[qid] = {'recall_ms': round((time.perf_counter() - t0) * 1e3, 1), 'tokens': d.get('tokens'),
                                 'route': d.get('route'), 'rerank_model': d.get('rerank_model'),
                                 'source_chat_ids': [int(it['source'].split(':')[1]) for it in res.items
                                                     if it.get('source')]}
                    prompt = answer_generation_for_rag.replace('<context>', res.context).replace('<question>', q['question'])
                    f.write(json.dumps({'id': qid, 'prompt': prompt}, ensure_ascii=False) + '\n')
                    f.flush()
            store.close()
            meta_f.write_text(json.dumps(meta, indent=1))
            print(f'conv {conv}: answer prompts written', flush=True)
    print(f'{out}: {sum(1 for _ in out.open())} prompts')


def _rows(path):
    raw = Path(path).read_text(encoding='utf-8').strip()
    return json.loads(raw) if raw.startswith('[') else [json.loads(l) for l in raw.splitlines() if l.strip()]


def import_answers(path, model='external'):
    OUT.mkdir(parents=True, exist_ok=True)
    by_conv = {}
    for r in _rows(path):
        conv, cat, i = r['id'].split('|')
        by_conv.setdefault(conv, []).append((cat, int(i), r.get('answer') or r.get('response') or ''))
    for conv, items in by_conv.items():
        qs = questions(conv)
        out = OUT / f'answers_conv{conv}.json'
        data = json.loads(out.read_text()) if out.exists() else {}
        for cat in CATS:
            data.setdefault(cat, [None] * len(qs[cat]))
        for cat, i, ans in items:
            data[cat][i] = {'question': qs[cat][i]['question'], 'llm_response': ans.strip(), 'answer_model': model}
        out.write_text(json.dumps(data, indent=1, ensure_ascii=False))
        print(f'conv {conv}: {len(items)} answers imported')


ALIGN_SYS = """
You are a binary classifier.
If the TWO snippets describe the SAME event/fact, reply **YES**
Otherwise reply **NO**. No extra words.
DO NOT provide any exaplanation.
"""


def _align_replay(reference, system, yes):
    """compute_metrics.align_with_llm with the LLM calls replaced by looked-up batch verdicts"""
    used, out = set(), []
    for si, s in enumerate(system):
        m = None
        for r in range(len(reference)):
            if r not in used and yes(si, r):
                m = r
                break
        if m is None:
            out.append(s)
        else:
            out.append(reference[m])
            used.add(m)
    return out


def export_judge():
    from beam_official.prompts import unified_llm_judge_base_prompt
    out = DATA / f'beam_judge_prompts{SFX}.jsonl'
    n = n_align = 0
    seen, alias = {}, {}
    with out.open('w', encoding='utf-8') as f:
        for af in sorted(OUT.glob('answers_conv*.json')):
            conv = af.stem.replace('answers_conv', '')
            qs, answers = questions(conv), json.loads(af.read_text())
            for cat in CATS:
                for i, a in enumerate(answers.get(cat, [])):
                    if not a:
                        continue
                    for r, item in enumerate(qs[cat][i]['rubric']):
                        prompt = unified_llm_judge_base_prompt.replace('<rubric_item>', item).replace(
                            '<llm_response>', a['llm_response'])
                        f.write(json.dumps({'id': f'{conv}|{cat}|{i}|{r}', 'prompt': prompt}, ensure_ascii=False) + '\n')
                        n += 1
                    if cat == 'event_ordering':   # official align_with_llm: pairwise YES/NO, replayed greedily on import
                        for si, sl in enumerate(a['llm_response'].split('\n')):
                            if not sl.strip():
                                continue
                            for r, ref in enumerate(qs[cat][i]['rubric']):
                                aid = f'align|{conv}|{i}|{si}|{r}'
                                ap = ALIGN_SYS.strip() + '\n\n' + f'First snippet: {ref} \n\nSecond snippet: {sl}'
                                if ap in seen:          # identical pair -> identical verdict; ask once
                                    alias[aid] = seen[ap]
                                    continue
                                seen[ap] = aid
                                f.write(json.dumps({'id': aid, 'prompt': ap}, ensure_ascii=False) + '\n')
                                n_align += 1
    (DATA / f'beam_judge_align_alias{SFX}.json').write_text(json.dumps(alias))
    print(f'{out}: {n} judge prompts + {n_align} event-ordering alignment prompts ({len(alias)} duplicates folded)')


def _order_score(reference_canon, system_canon):
    """tail of compute_metrics.event_ordering_score (after alignment), verbatim logic"""
    from scipy.stats import kendalltau
    tp = len(set(reference_canon) & set(system_canon))
    fp = len([x for x in system_canon if x not in reference_canon])
    fn = len([x for x in reference_canon if x not in system_canon])
    precision = tp / (tp + fp) if tp + fp else 0
    recall = tp / (tp + fn) if tp + fn else 0
    f1 = 2 * precision * recall / (precision + recall) if precision + recall else 0
    union = list(dict.fromkeys(reference_canon + system_canon))
    tie_rank = len(union) + 1

    def to_rank(seq):
        r = {item: i + 1 for i, item in enumerate(seq)}
        return [r.get(u, tie_rank) for u in union]
    tau_b, _ = kendalltau(to_rank(reference_canon), to_rank(system_canon), variant='b', method='auto')
    tau_b_norm = (tau_b + 1) / 2 if tau_b is not None else 0
    return dict(precision=precision, recall=recall, f1=f1, tau_norm=tau_b_norm, final_score=tau_b_norm * f1)


def import_judge(path, model='external'):
    from json_repair import repair_json
    from beam_official import compute_metrics as cm
    got = {}
    for r in _rows(path):
        got[r['id']] = r.get('response') or r.get('judgement') or r.get('output') or ''
    af_ = DATA / f'beam_judge_align_alias{SFX}.json'
    for dup, canon in (json.loads(af_.read_text()) if af_.exists() else {}).items():
        if canon in got:
            got[dup] = got[canon]
    for af in sorted(OUT.glob('answers_conv*.json')):
        conv = af.stem.replace('answers_conv', '')
        qs, answers = questions(conv), json.loads(af.read_text())
        results = {}
        for cat in CATS:
            rows = []
            for i, a in enumerate(answers.get(cat, [])):
                if not a:
                    continue
                rubric = qs[cat][i]['rubric']
                parsed = []
                for r in range(len(rubric)):
                    txt = got.get(f'{conv}|{cat}|{i}|{r}')
                    if txt is None:
                        parsed = None
                        break
                    try:
                        parsed.append(cm.parse_json_response(response=txt))
                    except Exception:
                        try:
                            parsed.append(json.loads(repair_json(txt)))
                        except ValueError:     # empty / unusable verdict: leave the question out (both arms
                            parsed = None      # of an A/B then drop it), never crash the whole import
                            print(f'unusable verdict {conv}|{cat}|{i}|{r}: {txt[:60]!r}', file=sys.stderr)
                            break
                if parsed is None:
                    rows.append(None)
                    continue
                res = {'llm_judge_score': sum(float(x.get('score', 0)) for x in parsed) / len(parsed),
                       'llm_judge_responses': parsed, 'judge_model': model, 'q': i}
                if cat == 'event_ordering':        # official report metric: tau_norm with align_type='llm'
                    sys_list = a['llm_response'].split('\n')
                    keys = [f'align|{conv}|{i}|{si}|{r}' for si, sl in enumerate(sys_list) if sl.strip()
                            for r in range(len(rubric))]
                    if keys and all(k in got for k in keys):
                        canon = _align_replay(rubric, sys_list,
                                              lambda si, r: 'yes' in got.get(f'align|{conv}|{i}|{si}|{r}', '').lower())
                        res |= _order_score(rubric, canon)
                        res['align_type'] = 'llm'
                    else:
                        res |= cm.event_ordering_score(reference_list=rubric, system_list=sys_list,
                                                       align_type='semantic', llm=None)
                        res['align_type'] = 'semantic'
                rows.append(res)
            results[cat] = [r for r in rows if r is not None]
        (OUT / f'evaluation-results_conv{conv}.json').write_text(json.dumps(results, indent=1, ensure_ascii=False))
        print(f'conv {conv}: judged {sum(len(v) for v in results.values())} questions')
    report()


if __name__ == '__main__':
    cmd, *convs = sys.argv[1:]
    ids = conv_ids() if convs == ['all'] else convs
    if cmd == 'report':
        report()
    elif cmd == 'export-answers':
        export_answers(ids)
    elif cmd == 'import-answers':
        import_answers(*convs)
    elif cmd == 'export-judge':
        export_judge()
    elif cmd == 'import-judge':
        import_judge(*convs)
    else:
        ids = conv_ids() if convs == ['all'] else convs
        for c in ids:
            {'answer': answer, 'judge': judge}[cmd](c)
