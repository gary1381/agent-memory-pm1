"""End-to-end check of the language router on the frozen dev pools (60 q x top-16).

For each setting (en / zh / x) and quality profile (fast / balanced / best): which path and model the router
picked, T-MRR, and the rerank latency it actually paid. Uses the real servers:
  PM1_RERANK_QWEN, PM1_RERANK_GTE, PM1_RERANK_COLBERT  (see pm1.rerank.router_from_env)
"""
import json
import statistics
import sys
import time
from collections import Counter
from pathlib import Path

import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parent))
from pool_rerank_eval import judge  # noqa: E402

from pm1.lang import detect  # noqa: E402
from pm1.rerank import PROFILES, router_from_env  # noqa: E402

router = router_from_env()
print('rerankers:', [(r.name, r.multilingual) for r in router.rerankers], flush=True)
pool = [x for x in json.loads((Path(__file__).parent / 'data' / 'lme_dev_pool_zh.json').read_text()) if x['complete']]
# warm up every model once so latency estimates reflect this machine, not the priors
for r in router.rerankers:
    r.score('warm up', ['warm up document'] * 16)
report = {}
for setting in ('en', 'zh', 'x'):
    for profile in ('fast', 'balanced', 'best'):
        rows, models, routes, lat = [], Counter(), Counter(), []
        for x in pool:
            q = x['question_en'] if setting == 'en' else x['question']
            docs = [c['text_en'] if setting in ('en', 'x') else c['text'] for c in x['cands']]
            t0 = time.perf_counter()
            order, info = router.rerank(q, docs, detect(q), [detect(d) for d in docs], PROFILES[profile])
            lat.append((time.perf_counter() - t0) * 1e3)
            rows.append(judge(order if order is not None else list(range(len(docs))), x['cands']))
            models[info.get('rerank_model') or 'none(rrf)'] += 1
            routes[info['route']] += 1
        key = f'{setting}|{profile}'
        report[key] = {'T-MRR': round(statistics.mean(r['T-MRR'] for r in rows), 3),
                       'T@1': round(statistics.mean(float(r['T@1']) for r in rows), 3),
                       'routes': dict(routes), 'models': dict(models),
                       'p50_ms': round(float(np.percentile(lat, 50)), 1), 'p95_ms': round(float(np.percentile(lat, 95)), 1)}
        print(key, json.dumps(report[key], ensure_ascii=False), flush=True)
(Path(__file__).parent / 'reports' / 'route_eval.json').write_text(json.dumps(report, indent=2, ensure_ascii=False) + '\n')
