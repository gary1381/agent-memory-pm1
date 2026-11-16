#!/usr/bin/env python3
"""LongMemEval-S (cleaned) at agentmemory's metric: strict recall-all@5 over distinct sessions, through pm1's Recaller.

Same questions as agentmemory's eval (470 answerable; abstention `_abs` skipped), same dataset (sha256 d6f21ea9...).
pm1 ranks turn-level spans; the ranked spans are collapsed to distinct sessions in first-appearance order and the first
k sessions are scored. Metrics: strict_all@k (every gold session), frac@k (share of gold sessions), any@k.
Per-question rows are saved so a paired bootstrap against agentmemory's scores.ndjson can be computed.

  PM1 env as in competitors/harness/pm1_smoke.py ENV; .venv/bin/python eval/lme_strict.py --modes lexical,rrf,routed
"""
from __future__ import annotations

import argparse
import json
import os
import statistics
import sys
import time
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent))
sys.path.insert(0, str(Path(__file__).resolve().parents[1] / 'src'))
os.environ.setdefault('PM1_RERANK_QWEN_VLLM', 'http://127.0.0.1:18166')
os.environ.setdefault('PM1_RERANK_PROFILE', 'wait')
os.environ.setdefault('PM1_QUALITY_FLOOR', 'wait')
from lme import DATA, VecCache, fill, ingest, split_of  # noqa: E402
from pm1 import Recaller  # noqa: E402
from pm1.rerank import router_from_env  # noqa: E402

KS = (1, 3, 5, 10)


def score(order, meta, gold):
    sessions = []
    for sid in order:
        s = meta[sid]['session']
        if s not in sessions:
            sessions.append(s)
    out = {}
    for k in KS:
        top = set(sessions[:k])
        out[f'strict@{k}'] = float(gold <= top)
        out[f'frac@{k}'] = len(gold & top) / len(gold)
        out[f'any@{k}'] = float(bool(gold & top))
    return out


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument('--modes', default='lexical,dense,rrf,routed')
    ap.add_argument('--dense', default='split', choices=['split', 'local'])
    ap.add_argument('--limit', type=int)
    ap.add_argument('--stride', type=int, default=1, help='keep every n-th question (systematic sample)')
    ap.add_argument('--out', required=True)
    a = ap.parse_args()
    from pm1.embed import LocalE5, SplitE5
    dense = SplitE5() if a.dense == 'split' else LocalE5()
    cache, router = VecCache(), router_from_env()
    modes = a.modes.split(',')
    qs = [q for q in json.loads(DATA.read_text()) if not q['question_id'].endswith('_abs')][::a.stride][:a.limit]
    out = Path(a.out)
    out.parent.mkdir(parents=True, exist_ok=True)
    done = {}
    if out.exists():                                   # resume
        done = {r['qid']: r for r in map(json.loads, out.read_text().splitlines())}
    t_start = time.time()
    with out.open('a') as f:
        for n, q in enumerate(qs, 1):
            if q['question_id'] in done:
                continue
            store, meta = ingest(q)
            idx_s = fill(store, dense, 'dense', cache)
            rec = Recaller(store, dense=dense, router=router, k_rerank=32, encode_missing=0)
            gold = set(q['answer_session_ids'])
            row = {'qid': q['question_id'], 'type': q['question_type'], 'split': split_of(q['question_id']),
                   'n_gold': len(gold), 'index_s': round(idx_s, 1)}
            for m in modes:
                t0 = time.perf_counter()
                order, _, diag = rec.rank('lme', q['question_id'], q['question'], m)
                row[m] = score(order, meta, gold) | {'ms': round((time.perf_counter() - t0) * 1e3),
                                                      'err': bool(diag.get('rerank_error') or diag.get('dense_error'))}
            f.write(json.dumps(row) + '\n')
            f.flush()
            store.close()
            if n % 10 == 0:
                import resource
                print(f'{n}/{len(qs)} {(time.time() - t_start) / 60:.1f} min load {os.getloadavg()[0]:.1f} '
                      f'peak_rss_mb {resource.getrusage(resource.RUSAGE_SELF).ru_maxrss // 1024}', flush=True)


if __name__ == '__main__':
    main()
