"""Reproducible CPU/SQLite audit on synthetic records; no models, private data or services.

Run from pm1: OMP_NUM_THREADS=2 .venv/bin/python eval/audit_bench.py --sizes 1000 10000 --output result.json
Use PYTHONPATH to point at another checkout for a before/after comparison.
"""
from __future__ import annotations

import argparse
import json
import math
import os
import platform
import resource
import statistics
import tempfile
import time
from pathlib import Path

# Explicit, standalone benchmark configuration; do not inherit a production service profile.
for key in list(os.environ):
    if key.startswith('PM1_'):
        del os.environ[key]
os.environ.update(PM1_CONFIG='/nonexistent/pm1-audit.toml', PM1_JUDGE='off', PM1_DENSE='none',
                  PM1_SESS_FULL='1', PM1_SESS_PACK='tokens', PM1_SESS_NEIGHBORS='0',
                  PM1_PROJECTS='/nonexistent/pm1-projects.json', PM1_PROJECT='audit')

from pm1 import Recaller, Store
from pm1.sessions import SessionSearch


def summary(values):
    ordered = sorted(values)
    return {'n': len(values), 'p50_ms': round(statistics.median(values), 3),
            'p95_ms': round(ordered[math.ceil(.95 * len(values)) - 1], 3),
            'max_ms': round(max(values), 3)}


def run(size, queries, storage):
    with tempfile.TemporaryDirectory(prefix='pm1-audit-') as directory:
        db = Path(directory) / 'audit.sqlite'
        s = Store(db if storage == 'file' else ':memory:')
        writes, ids = [], []
        for i in range(size):
            start = time.perf_counter()
            out = s.add_episode(bank='personal', scope='proj:/audit/project', session=f'codex:{i // 20}',
                                source=f'/synthetic/transcript{i // 20}.jsonl#{i % 20}',
                                occurred_at='2026-11-01T12:00:00Z',
                                text=f'ticket{i:07d}: queue worker retry policy for service{i % 31}. '
                                     'The retry limit is three and the timeout is thirty seconds. '
                                     f'Decision recorded by engineer{i % 19}.')
            writes.append((time.perf_counter() - start) * 1000)
            ids.append(out['episode_id'])
        ss = SessionSearch(s, Recaller(s))
        times, hits, over_budget = [], 0, 0
        # Deterministic spread across the corpus, with unique identifier queries.
        selected = [i * (size - 1) // max(1, queries - 1) for i in range(queries)]
        for i in selected:
            start = time.perf_counter()
            out = ss.search(f'ticket{i:07d}', k=1, budget_tokens=300)
            times.append((time.perf_counter() - start) * 1000)
            hits += bool(out['items'] and out['items'][0]['episode_id'] == ids[i])
            over_budget += out['tokens'] > 300
        broad = []
        for _ in range(queries):
            start = time.perf_counter()
            ss.search('queue worker retry policy', k=5, budget_tokens=1500)
            broad.append((time.perf_counter() - start) * 1000)
        s.con.execute('PRAGMA wal_checkpoint(TRUNCATE)')
        result = {'records': size, 'spans': s.stats('personal')['spans'], 'storage': storage,
                  'write': summary(writes), 'write_total_s': round(sum(writes) / 1000, 3),
                  'exact_search': summary(times), 'exact_hit_at_1': hits / queries,
                  'budget_violations': over_budget, 'common_terms_search': summary(broad),
                  'database_bytes': db.stat().st_size if storage == 'file' else None,
                  'process_peak_rss_kib': resource.getrusage(resource.RUSAGE_SELF).ru_maxrss}
        s.close()
        return result


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--sizes', type=int, nargs='+', default=[1000, 10000])
    parser.add_argument('--queries', type=int, default=30)
    parser.add_argument('--storage', choices=['file', 'memory'], default='file')
    parser.add_argument('--output', type=Path, required=True)
    args = parser.parse_args()
    if min(args.sizes) < 1 or args.queries < 1:
        parser.error('sizes and queries must be positive')
    report = {'python': platform.python_version(), 'platform': platform.platform(),
              'store_module': __import__('pm1.store', fromlist=['x']).__file__,
              'mode': 'SQLite lexical search; no dense model or reranker', 'results': []}
    for size in args.sizes:
        result = run(size, args.queries, args.storage)
        report['results'].append(result)
        print(json.dumps(result), flush=True)
        args.output.parent.mkdir(parents=True, exist_ok=True)
        args.output.write_text(json.dumps(report, indent=2) + '\n')


if __name__ == '__main__':
    main()
