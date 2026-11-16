"""LongMemEval-S dev at the recall() level (what an agent actually receives), planning off vs on.

metrics per question (abstention questions skipped):
  sess_all : every answer session has >=1 item in the packed context
  sess_any : at least one answer session present
  ev_any   : at least one evidence turn present
usage: PM1_RERANK_QWEN=... PM1_RERANK_GTE=... .venv-gpu/bin/python eval/lme_recall_eval.py [--budget 4000] [--limit N]
"""
import argparse, json, statistics, sys, time
from collections import defaultdict
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent))
sys.path.insert(0, str(Path(__file__).resolve().parents[1] / 'src'))
from lme import DATA, VecCache, fill, ingest, split_of  # noqa: E402
from pm1 import Recaller  # noqa: E402
from pm1.embed import SplitE5  # noqa: E402
from pm1.rerank import router_from_env  # noqa: E402

ap = argparse.ArgumentParser()
ap.add_argument('--budget', type=int, default=4000)
ap.add_argument('--limit', type=int)
ap.add_argument('--out', default=str(Path(__file__).parent / 'reports' / 'lme_recall_planning_dev.json'))
a = ap.parse_args()
dense, cache, router = SplitE5(), VecCache(), router_from_env()
qs = [q for q in json.loads(DATA.read_text()) if split_of(q['question_id']) == 'dev'][:a.limit]
rows = {'off': [], 'on': []}
by_type = {'off': defaultdict(list), 'on': defaultdict(list)}
lat = {'off': [], 'on': []}
intents = defaultdict(int)
for q in qs:
    if q['question_id'].endswith('_abs'):
        continue
    store, meta = ingest(q)
    fill(store, dense, 'dense', cache)
    ans = set(q['answer_session_ids'])
    for name, planning in (('off', False), ('on', True)):
        rec = Recaller(store, dense=dense, router=router, planning=planning)
        t0 = time.perf_counter()
        res = rec.recall('lme', q['question_id'], q['question'], budget_tokens=a.budget, max_items=40)
        lat[name].append((time.perf_counter() - t0) * 1e3)
        if planning:
            intents[res.diagnostics.get('intent')] += 1
        sess = {meta[it['span_id']]['session'] for it in res.items if it['span_id'] in meta}
        ev = any(meta[it['span_id']]['evidence'] for it in res.items if it['span_id'] in meta)
        r = {'sess_all': ans <= sess, 'sess_any': bool(ans & sess), 'ev_any': ev}
        rows[name].append(r)
        by_type[name][q['question_type']].append(r)
    store.close()


def mean(rs):
    return {k: round(statistics.mean(float(r[k]) for r in rs), 3) for k in rs[0]} | {'n': len(rs)}


rep = {'budget_tokens': a.budget, 'intents_on': dict(intents),
       **{n: {'all': mean(rows[n]), 'p50_ms': round(statistics.median(lat[n])),
              'by_type': {t: mean(v) for t, v in sorted(by_type[n].items())}} for n in rows}}
Path(a.out).write_text(json.dumps(rep, indent=2) + '\n')
print(json.dumps({n: rep[n]['all'] | {'p50_ms': rep[n]['p50_ms']} for n in rows}, indent=1))
print(json.dumps({t: {n: rep[n]['by_type'][t]['sess_all'] for n in rows} for t in rep['on']['by_type']}, indent=1))
