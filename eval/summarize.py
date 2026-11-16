"""Render an LME retrieval report as a markdown table: python eval/summarize.py [report.json]"""
import json
import sys
from pathlib import Path

p = Path(sys.argv[1] if len(sys.argv) > 1 else Path(__file__).parent / 'reports' / 'lme_retrieval_dev.json')
d = json.loads(p.read_text())
keys = [k for k in ('S@5', 'S@10', 'T@5', 'T@10', 'T-MRR') if k in next(iter(d['modes'].values()))['metrics']]
print(f"split={d['split']} questions={d['questions']} dense={d.get('dense_model')} k_rerank={d['k_rerank']} "
      f"elapsed={d['elapsed_min']}min indexing_s={d['offline_indexing_s']} failures={d['encode_failures']} "
      f"dense_fallbacks={d.get('dense_fallbacks')}\n")
print('| mode | ' + ' | '.join(keys) + ' | p50 ms | p95 ms |')
print('|' + '---|' * (len(keys) + 3))
for m, v in d['modes'].items():
    print(f'| {m} | ' + ' | '.join(f"{v['metrics'].get(k, 0):.3f}" for k in keys)
          + f" | {v['online_latency_ms']['p50']} | {v['online_latency_ms']['p95']} |")
types = sorted({t for v in d['modes'].values() for t in v['by_type']})
print('\nS@5 by question type\n')
print('| type | ' + ' | '.join(d['modes']) + ' |')
print('|' + '---|' * (len(d['modes']) + 1))
for t in types:
    print(f'| {t} | ' + ' | '.join(f"{v['by_type'].get(t, {}).get('S@5', 0):.2f} (n={v['by_type'].get(t, {}).get('n', '?')})"
                                    for v in d['modes'].values()) + ' |')
