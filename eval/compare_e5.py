"""Compare a remote e5 endpoint against local int8 e5: throughput, vector agreement, ranking agreement.

Usage: .venv/bin/python eval/compare_e5.py --url http://127.0.0.1:17883/gateway/<svc>/v1/embeddings
"""
from __future__ import annotations

import argparse
import json
import random
import time
from pathlib import Path

import numpy as np

from pm1.embed import HttpE5, LocalE5
from pm1.text import split_spans

DATA = Path.home() / '.cache/pm1/data/longmemeval_s_cleaned.json'


def sample(n_pass: int, n_query: int, seed: int = 7):
    data = json.loads(DATA.read_text())
    rng = random.Random(seed)
    qs = rng.sample(data, n_query)
    passages = []
    for q in rng.sample(data, 20):
        for sess in q['haystack_sessions']:
            for turn in sess:
                c = turn['content']
                passages += [c[s:e] for s, e in split_spans(c)]
    rng.shuffle(passages)
    zh = ['我上周在杭州买了一台二手相机，花了三千块。', '下个月我要去东京出差，记得提醒我带转换插头。',
          '我女儿的生日是五月十二日，她喜欢画画。', '我们讨论过把数据库从 MySQL 迁移到 PostgreSQL 的方案。']
    return [q['question'] for q in qs] + ['我的相机是在哪里买的？', '我什么时候去东京？'], passages[:n_pass] + zh


def timed(fn, *a):
    t0 = time.perf_counter()
    r = fn(*a)
    return r, time.perf_counter() - t0


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument('--url', required=True)
    ap.add_argument('--n', type=int, default=400)
    ap.add_argument('--queries', type=int, default=30)
    a = ap.parse_args()
    queries, passages = sample(a.n, a.queries)
    chars = [len(p) for p in passages]
    print(f'{len(passages)} passages (chars p50={int(np.median(chars))} max={max(chars)}), {len(queries)} queries')

    local = LocalE5()
    lp, tl = timed(local.embed, passages, 'passage')
    lq, _ = timed(local.embed, queries, 'query')
    print(f'local int8 CPU : {len(passages)/tl:7.1f} docs/s')

    report = {}
    for batch in (8, 16, 32):
        remote = HttpE5(url=a.url, batch=batch, timeout=60)
        try:
            rp, tr = timed(remote.embed, passages, 'passage')
        except Exception as exc:
            print(f'remote batch={batch}: FAILED {type(exc).__name__}: {str(exc)[:200]}')
            continue
        print(f'remote batch={batch:2d}: {len(passages)/tr:7.1f} docs/s')
        report[batch] = len(passages) / tr
    remote = HttpE5(url=a.url, batch=16, timeout=60)
    rp = remote.embed(passages, 'passage')
    lat = []
    for q in queries:
        t0 = time.perf_counter(); rq1 = remote.embed([q], 'query'); lat.append((time.perf_counter() - t0) * 1e3)
    rq = remote.embed(queries, 'query')
    lat.sort()
    print(f'remote single-query latency: p50={lat[len(lat)//2]:.1f} ms p95={lat[int(len(lat)*.95)]:.1f} ms')

    cos = (lp * rp).sum(1)
    print(f'passage cosine local-vs-remote: mean={cos.mean():.4f} min={cos.min():.4f} p05={np.percentile(cos,5):.4f}')
    print(f'query   cosine local-vs-remote: mean={(lq*rq).sum(1).mean():.4f} min={(lq*rq).sum(1).min():.4f}')
    worst = np.argsort(cos)[:3]
    for i in worst:
        print(f'  low cos {cos[i]:.4f} len={len(passages[i])}: {passages[i][:80]!r}')

    sl, sr = lq @ lp.T, rq @ rp.T
    ov = [len(set(np.argsort(-sl[i])[:10]) & set(np.argsort(-sr[i])[:10])) / 10 for i in range(len(queries))]
    top1 = np.mean([np.argmax(sl[i]) == np.argmax(sr[i]) for i in range(len(queries))])
    print(f'ranking agreement: top10 overlap={np.mean(ov):.3f}  top1 same={top1:.3f}')
    # Cross-compat: remote query vs local passages (could the cache mix both?)
    sx = rq @ lp.T
    ovx = [len(set(np.argsort(-sl[i])[:10]) & set(np.argsort(-sx[i])[:10])) / 10 for i in range(len(queries))]
    print(f'mixed (remote query x local passages) top10 overlap vs local={np.mean(ovx):.3f}')


if __name__ == '__main__':
    main()
