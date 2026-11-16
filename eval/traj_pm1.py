#!/usr/bin/env python3
"""pm1 on the SWE-agent trajectory retrieval set (docs/EVALUATION.md E17). One store holds all 40 sessions; each turn is a record."""
import json, os, sys
from pathlib import Path
sys.path.insert(0, str(Path(__file__).resolve().parents[1] / 'src'))
os.environ.setdefault('PM1_RERANK_QWEN_VLLM', 'http://127.0.0.1:18166')
os.environ.setdefault('PM1_RERANK_PROFILE', 'wait')
os.environ.setdefault('PM1_QUALITY_FLOOR', 'wait')
from pm1 import Recaller, Store
from pm1.embed import LocalE5, SplitE5
from pm1.rerank import router_from_env

seed = sys.argv[1]
CPU = ('_cpu' if '--local' in sys.argv else '') + ('_gx' if os.environ.get('PM1_GREP_CHANNEL') == '1' else '')   # keep the production rows untouched
T = Path(os.environ.get('PM1_TRAJ_DIR', str(Path.home() / '.cache/pm1-eval/traj')))
hay = json.load(open(T / f'haystack_{seed}.json'))
VARIANTS = {v: json.load(open(T / f'questions_{seed}{v}.json')) for v in ('', 'p')}
dense = LocalE5() if '--local' in sys.argv else SplitE5()
store, meta = Store(':memory:'), {}
for s in hay:
    for i, t in enumerate(s['turns']):
        ep = store.add_episode(bank='tj', scope='tj', text=t['text'], role='assistant' if t['role'] == 'ai' else 'user',
                               session=s['sid'], seq=i, occurred_at='2026-01-01T00:00:00Z')
        for sp in ep['span_ids']:
            meta[sp] = s['sid']
spans = store.con.execute('SELECT s.id, s.start, s.end, e.text FROM spans s JOIN episodes e ON e.id=s.episode_id').fetchall()
print('spans', len(spans), flush=True)
for r, v in zip(spans, dense.embed([r[3][r[1]:r[2]] for r in spans], 'passage')):
    store.put_vector(r[0], dense.name, v)
rec = Recaller(store, dense=dense, router=router_from_env(), k_rerank=32, encode_missing=0)
out = Path(os.environ.get('PM1_EVAL_RUNS', str(Path.home() / '.cache/pm1-eval/runs'))) / 'traj'; out.mkdir(parents=True, exist_ok=True)
for var, qs in VARIANTS.items():
    rows = []
    for q in qs:
        row = {'id': q['id'], 'type': q['type']}
        for m in (('lexical', 'dense', 'rrf') if '--local' in sys.argv else ('lexical', 'dense', 'rrf', 'routed')):
            order, _, _ = rec.rank('tj', 'tj', q['question'], m)
            top = []
            for sp in order:
                if meta[sp] not in top:
                    top.append(meta[sp])
            row[m] = {f'hit@{k}': float(set(q['gold']) <= set(top[:k])) for k in (1, 3, 5)}
        rows.append(row)
    (out / f'pm1_{seed}{var}{CPU}.json').write_text(json.dumps(rows))
    for m in (('lexical', 'dense', 'rrf') if '--local' in sys.argv else ('lexical', 'dense', 'rrf', 'routed')):
        for t in ('file', 'issue', 'error', 'ALL'):
            rr = [r for r in rows if t in ('ALL', r['type'])]
            print(var or 'orig', m, t, len(rr), ' '.join(f"hit@{k}={sum(r[m][f'hit@{k}'] for r in rr) / len(rr):.3f}" for k in (1, 3, 5)), flush=True)
