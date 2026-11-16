"""Side-by-side group coverage (+latency) for several recall_meta files.  usage: beam_cov_compare.py name=path ..."""
import json, statistics, sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent))
from beam_coverage import coverage  # noqa: E402

res = {}
for arg in sys.argv[1:]:
    name, f = arg.split('=', 1)
    m = json.loads(Path(f).read_text())
    per, _ = coverage(m, group=True)
    res[name] = {c: statistics.mean(v) for c, v in per.items() if v}
    ms = sorted(v['recall_ms'] for v in m.values())
    res[name]['_p50 ms'] = statistics.median(ms)
    res[name]['_p95 ms'] = ms[int(.95 * len(ms))]
cats = [c for c in next(iter(res.values())) if not c.startswith('_')]
print(f"{'':26s}" + ''.join(f'{n:>9s}' for n in res))
for c in cats:
    print(f'{c:26s}' + ''.join(f'{res[n][c]:9.3f}' for n in res))
print(f"{'MEAN (9 categories)':26s}" + ''.join(
    f"{statistics.mean(res[n][c] for c in cats):9.3f}" for n in res))
for c in ('_p50 ms', '_p95 ms'):
    print(f'{c[1:]:26s}' + ''.join(f'{res[n][c]:9.0f}' for n in res))
