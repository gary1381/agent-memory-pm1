"""Latency of a llama.cpp rerank endpoint on realistic pools (LongMemEval dev, top-16 candidates), and agreement with
a reference endpoint (same model on the GPU).  usage: rerank_cpu_bench.py <url> [ref_url] [n_queries] [depth]"""
import json, statistics, sys, time
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / 'src'))
from pm1.rerank import llama_scorer  # noqa: E402

url = sys.argv[1]
ref = sys.argv[2] if len(sys.argv) > 2 and sys.argv[2] != '-' else None
n = int(sys.argv[3]) if len(sys.argv) > 3 else 10
depth = int(sys.argv[4]) if len(sys.argv) > 4 else 16
pool = json.loads((Path(__file__).parent / 'data' / 'lme_dev_pool_en.json').read_text())[:n]
f = llama_scorer(url, timeout=600)
g = llama_scorer(ref) if ref else None
lat, agree, chars = [], [], []
for x in pool:
    docs = [c['text'] for c in x['cands'][:depth]]
    chars.append(sum(len(d) for d in docs))
    t0 = time.perf_counter()
    s = f(x['question'], docs)
    lat.append((time.perf_counter() - t0) * 1e3)
    if g:
        r = g(x['question'], docs)
        top = lambda v: sorted(range(len(v)), key=lambda i: -v[i])[:3]  # noqa: E731
        agree.append(len(set(top(s)) & set(top(r))) / 3)
print(json.dumps({'url': url, 'n': n, 'depth': depth, 'avg_pool_chars': round(statistics.mean(chars)),
                  'p50_ms': round(statistics.median(lat)), 'max_ms': round(max(lat)),
                  'ms_per_doc': round(statistics.median(lat) / depth, 1),
                  'top3_agreement_vs_ref': round(statistics.mean(agree), 3) if agree else None}))
