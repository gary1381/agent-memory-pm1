"""Cross-encoder rerankers over the shipped RRF top-K on the LME dev split.

Each --variant is name=url[,ctx]: a llama.cpp `--rerank` server, optionally with context (ctx): the query gets
the date it is asked, each document its session date and speaker. Instructions live in the server's GGUF
(see scripts/rerank_template.py). Reports rerank-only and RRF(fused, reranked) orders.

  .venv/bin/python eval/rerank_eval.py --k 32 \
     --variant qwen=http://127.0.0.1:18161 --variant qwen_mem_ctx=http://127.0.0.1:18162,ctx
"""
from __future__ import annotations

import argparse
import json
import statistics
import sys
import time
import urllib.request
from pathlib import Path

import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parent))
from lme import DATA, VecCache, fill, ingest, span_texts, split_of  # noqa: E402

from pm1 import Recaller  # noqa: E402
from pm1.embed import SplitE5  # noqa: E402
from pm1.recall import rrf  # noqa: E402


def judge(order, meta, ans):
    sessions = []
    for sid in order:
        if meta[sid]['session'] not in sessions:
            sessions.append(meta[sid]['session'])
    ev = next((r for r, sid in enumerate(order) if meta[sid]['evidence']), None)
    out = {f'T@{k}': ev is not None and ev < k for k in (1, 3, 5, 10)}
    out['S@1'] = bool(sessions) and sessions[0] in ans
    out['S@5'] = any(s in ans for s in sessions[:5])
    out['T-MRR'] = 1 / (ev + 1) if ev is not None else 0.0
    return out


def rerank(url, query, docs):
    req = urllib.request.Request(url.rstrip('/') + '/v1/rerank',
                                 data=json.dumps({'query': query, 'documents': docs, 'top_n': len(docs)}).encode(),
                                 headers={'Content-Type': 'application/json'})
    with urllib.request.urlopen(req, timeout=300) as r:
        res = json.loads(r.read())['results']
    scores = [0.0] * len(docs)
    for x in res:
        scores[x['index']] = x['relevance_score']
    return scores


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument('--k', type=int, default=32)
    ap.add_argument('--variant', action='append', required=True)
    ap.add_argument('--limit', type=int)
    ap.add_argument('--out', default=str(Path(__file__).parent / 'reports' / 'rerank_dev.json'))
    a = ap.parse_args()
    variants = {}
    for v in a.variant:
        name, spec = v.split('=', 1)
        url, *flags = spec.split(',')
        variants[name] = (url, 'ctx' in flags)
    dense, cache = SplitE5(), VecCache()
    qs = [q for q in json.loads(DATA.read_text())
          if split_of(q['question_id']) == 'dev' and not q['question_id'].endswith('_abs')][:a.limit]
    res = {'rrf': []} | {f'{n}|{m}': [] for n in variants for m in ('rerank', 'fused')}
    lat = {n: [] for n in variants}
    for qi, q in enumerate(qs, 1):
        store, meta = ingest(q)
        fill(store, dense, 'dense', cache)
        scope, ans = q['question_id'], set(q['answer_session_ids'])
        shipped, _, _ = Recaller(store, dense=dense).rank('lme', scope, q['question'], 'rrf')
        res['rrf'].append(judge(shipped, meta, ans))
        head, tail = shipped[:a.k], shipped[a.k:]
        texts = span_texts(store, set(head))
        info = {sid: (role, when) for sid, role, when in store.con.execute(
            'SELECT s.id, e.role, e.occurred_at FROM spans s JOIN episodes e ON e.id=s.episode_id')}
        for name, (url, ctx) in variants.items():
            if ctx:
                query = f"Question asked on {q['question_date']}: {q['question']}"
                docs = [f'[{info[s][1]}, {info[s][0]}] {texts[s]}' for s in head]
            else:
                query, docs = q['question'], [texts[s] for s in head]
            t0 = time.perf_counter()
            scores = rerank(url, query, docs)
            lat[name].append((time.perf_counter() - t0) * 1e3)
            ranked = [head[j] for j in np.argsort(-np.asarray(scores), kind='stable')]
            res[f'{name}|rerank'].append(judge(ranked + tail, meta, ans))
            res[f'{name}|fused'].append(judge(rrf([head, ranked]) + tail, meta, ans))
        store.close()
        if qi % 10 == 0 or qi == len(qs):
            report = {'questions': qi, 'k': a.k,
                      'modes': {k: {m: round(statistics.mean(float(r[m]) for r in v), 3) for m in v[0]}
                                for k, v in res.items()},
                      'rerank_latency_ms': {n: {'p50': round(float(np.percentile(v, 50)), 1),
                                                'p95': round(float(np.percentile(v, 95)), 1)} for n, v in lat.items()}}
            Path(a.out).write_text(json.dumps(report, indent=2) + '\n')
            print(json.dumps({'n': qi, **{k: (v['T@1'], v['T@10'], v['T-MRR']) for k, v in report['modes'].items()}}),
                  flush=True)


if __name__ == '__main__':
    main()
