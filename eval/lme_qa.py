"""LongMemEval-S question answering through pm1 recall (gate G1/G3 metric), official prompts.

  lme_qa.py export  [--split dev|test|all] [--budget 2500] [--planning 1] [--tag T]
        -> data/lme_answer_prompts_<tag>.jsonl  {"id": question_id, "prompt"}   (+ reports/lme_qa_<tag>_recall.json)
  lme_qa.py judge-prompts <answers.jsonl> --tag T  -> data/lme_judge_prompts_<tag>.jsonl (official anscheck prompts)
  lme_qa.py report <judge_responses.jsonl> --tag T -> reports/lme_qa_<tag>.json (accuracy overall / by type)

Answer prompt = LongMemEval's own CoT template (run_generation.py), history = pm1's packed context.
Judge prompts = evaluate_qa.get_anscheck_prompt verbatim (answered yes/no by external labelers).
"""
import argparse, json, os, statistics, sys, time
from collections import Counter, defaultdict
from pathlib import Path

EV = Path(__file__).resolve().parent
sys.path.insert(0, str(EV)); sys.path.insert(0, str(EV.parent / 'src'))
DATA_DIR, REP = EV / 'data', EV / 'reports'

ANSWER_T = ('I will give you several history chats between you and a user. Please answer the question based on the '
            'relevant chat history. Answer the question step by step: first extract all the relevant information, and '
            'then reason over the information to get the answer.\n\n\nHistory Chats:\n\n{}\n\nCurrent Date: {}\n'
            'Question: {}\nAnswer (step by step):')

_BASE = ("I will give you a question, a correct answer, and a response from a model. Please answer yes if the response "
         "contains the correct answer. Otherwise, answer no. If the response is equivalent to the correct answer or "
         "contains all the intermediate steps to get the correct answer, you should also answer yes. If the response "
         "only contains a subset of the information required by the answer, answer no. ")


def anscheck(task, question, answer, response, abstention=False):
    if abstention:
        t = ("I will give you an unanswerable question, an explanation, and a response from a model. Please answer yes "
             "if the model correctly identifies the question as unanswerable. The model could say that the information "
             "is incomplete, or some other information is given but the asked information is not.\n\nQuestion: {}\n\n"
             "Explanation: {}\n\nModel Response: {}\n\nDoes the model correctly identify the question as unanswerable? "
             "Answer yes or no only.")
    elif task in ('single-session-user', 'single-session-assistant', 'multi-session'):
        t = _BASE + "\n\nQuestion: {}\n\nCorrect Answer: {}\n\nModel Response: {}\n\nIs the model response correct? Answer yes or no only."
    elif task == 'temporal-reasoning':
        t = (_BASE + "In addition, do not penalize off-by-one errors for the number of days. If the question asks for the "
             "number of days/weeks/months, etc., and the model makes off-by-one errors (e.g., predicting 19 days when "
             "the answer is 18), the model's response is still correct. \n\nQuestion: {}\n\nCorrect Answer: {}\n\n"
             "Model Response: {}\n\nIs the model response correct? Answer yes or no only.")
    elif task == 'knowledge-update':
        t = ("I will give you a question, a correct answer, and a response from a model. Please answer yes if the "
             "response contains the correct answer. Otherwise, answer no. If the response contains some previous "
             "information along with an updated answer, the response should be considered as correct as long as the "
             "updated answer is the required answer.\n\nQuestion: {}\n\nCorrect Answer: {}\n\nModel Response: {}\n\n"
             "Is the model response correct? Answer yes or no only.")
    elif task == 'single-session-preference':
        t = ("I will give you a question, a rubric for desired personalized response, and a response from a model. "
             "Please answer yes if the response satisfies the desired response. Otherwise, answer no. The model does not "
             "need to reflect all the points in the rubric. The response is correct as long as it recalls and utilizes "
             "the user's personal information correctly.\n\nQuestion: {}\n\nRubric: {}\n\nModel Response: {}\n\n"
             "Is the model response correct? Answer yes or no only.")
    else:
        raise ValueError(task)
    return t.format(question, answer, response)


def rows(p):
    return [json.loads(l) for l in Path(p).read_text(encoding='utf-8').split('\n') if l.strip()]


def questions(split):
    from lme import DATA, split_of
    qs = json.loads(DATA.read_text())
    return [q for q in qs if split == 'all' or split_of(q['question_id']) == split]


def export(a):
    from lme import VecCache, fill, ingest
    from pm1 import Recaller
    from pm1.embed import SplitE5
    from pm1.rerank import router_from_env
    dense, cache, router = SplitE5(), VecCache(), router_from_env()
    if router is None:      # the gate configuration reranks; a silent no-reranker export is not comparable
        raise SystemExit('lme_qa export: no reranker configured (set PM1_RERANK_QWEN / PM1_RERANK_GTE, '
                         'see scripts/beam_run.sh)')
    out = DATA_DIR / f'lme_answer_prompts_{a.tag}.jsonl'
    meta = {}
    with out.open('w', encoding='utf-8') as f:
        only = set(filter(None, a.only.split(',')))
        for n, q in enumerate(q for q in questions(a.split) if not only or q['question_id'] in only):
            store, smeta = ingest(q)
            fill(store, dense, 'dense', cache)
            rec = Recaller(store, dense=dense, router=router, planning=bool(a.planning),
                           directive_min_sim=a.dir_min, profile=a.profile)
            t0 = time.perf_counter()
            if not a.anchor:
                rec.anchor_k = 0
            res = rec.recall('lme', q['question_id'], q['question'], budget_tokens=a.budget or None, max_items=40,
                             now=q['question_date'])
            ms = (time.perf_counter() - t0) * 1e3
            sess = {smeta[it['span_id']]['session'] for it in res.items if it['span_id'] in smeta}
            ans = set(q['answer_session_ids'])
            meta[q['question_id']] = {'type': q['question_type'], 'ms': round(ms, 1),
                                      'tokens': res.diagnostics.get('tokens'), 'intent': res.diagnostics.get('intent'),
                                      'anchor': res.diagnostics.get('anchor'),
                                      'rerank': res.diagnostics.get('rerank_model') or res.diagnostics.get('fallback'),
                                      'sess_all': bool(ans) and ans <= sess, 'sess_any': bool(ans & sess)}
            prompt = ANSWER_T.format(res.context, q['question_date'], q['question'])
            leaked = [s for s in q['answer_session_ids'] if s in prompt]
            if leaked:      # gold labels must never reach the answerer (see lme.session_label)
                raise SystemExit(f"lme_qa export: {q['question_id']}: answer session id(s) {leaked} in the prompt")
            f.write(json.dumps({'id': q['question_id'], 'prompt': prompt}, ensure_ascii=False) + '\n')
            store.close()
            if n % 20 == 0:
                print(n, flush=True)
    (REP / f'lme_qa_{a.tag}_recall.json').write_text(json.dumps(meta, indent=1))
    ms = sorted(v['ms'] for v in meta.values())
    toks = [v['tokens'] or 0 for v in meta.values()]
    print(out, len(meta), 'p50 ms', round(statistics.median(ms)), 'p95', round(ms[int(.95 * len(ms))]),
          'median tokens', statistics.median(toks))
    rr = Counter(str(v['rerank']) for v in meta.values())
    print('rerank per question:', dict(rr.most_common()))
    skipped = sum(n for k, n in rr.items() if k.startswith('no reranker') or 'error' in k or 'fallback' in k)
    if a.require_rerank and skipped:
        raise SystemExit(f'lme_qa export: {skipped}/{len(meta)} questions were not reranked '
                         f'(profile {a.profile!r}); not a frozen-config export')


def judge_prompts(a):
    qs = {q['question_id']: q for q in questions('all')}
    out = DATA_DIR / f'lme_judge_prompts_{a.tag}.jsonl'
    with out.open('w', encoding='utf-8') as f:
        for r in rows(a.path):
            q = qs[r['id']]
            resp = r.get('response') or r.get('answer') or ''
            f.write(json.dumps({'id': r['id'], 'prompt': anscheck(q['question_type'], q['question'], q['answer'], resp,
                                                                   abstention=r['id'].endswith('_abs'))},
                               ensure_ascii=False) + '\n')
    print(out)


def report(a):
    qs = {q['question_id']: q for q in questions('all')}
    by = defaultdict(list)
    for r in rows(a.path):
        ok = 'yes' in str(r.get('response', '')).lower()[:10]
        t = 'abstention' if r['id'].endswith('_abs') else qs[r['id']]['question_type']
        by[t].append(ok)
        by['ALL'].append(ok)
    rep = {t: {'acc': round(sum(v) / len(v), 4), 'n': len(v)} for t, v in sorted(by.items())}
    (REP / f'lme_qa_{a.tag}.json').write_text(json.dumps(rep, indent=1))
    print(json.dumps(rep, indent=1))


if __name__ == '__main__':
    ap = argparse.ArgumentParser()
    ap.add_argument('cmd', choices=['export', 'judge-prompts', 'report'])
    ap.add_argument('path', nargs='?')
    ap.add_argument('--split', default='dev', choices=['dev', 'test', 'all'])
    ap.add_argument('--budget', type=int, default=0, help='0 = adaptive (2.5K / 8K)')
    ap.add_argument('--planning', type=int, default=1)
    ap.add_argument('--dir-min', type=float, default=None)
    ap.add_argument('--anchor', type=int, default=1, help='time-anchored recall from question_date (0 = off)')
    ap.add_argument('--only', default='', help='comma-separated question ids (debug)')
    ap.add_argument('--tag', default='dev2500')
    ap.add_argument('--profile', default=os.environ.get('PM1_RERANK_PROFILE', 'balanced'),
                    choices=['fast', 'balanced', 'best'],
                    help="rerank latency profile; 'best' = no latency budget, so load cannot skip the reranker")
    ap.add_argument('--require-rerank', type=int, default=0,
                    help='1 = fail the export if any question fell back to the unreranked order')
    a = ap.parse_args()
    {'export': export, 'judge-prompts': judge_prompts, 'report': report}[a.cmd](a)
