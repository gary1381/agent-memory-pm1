"""Size up BEAM-10M: conversations, turns, characters (~tokens), probing questions per category."""
import ast
import json
from collections import Counter
from pathlib import Path

import pyarrow.parquet as pq

D = Path.home() / '.cache/pm1/data/beam10m'
rows = []
for f in sorted(D.glob('10M-*.parquet')):
    rows += pq.read_table(f, columns=['conversation_id', 'chat', 'probing_questions']).to_pylist()
print('conversations:', len(rows))
tot_chars = tot_turns = 0
cats = Counter()
for r in rows:
    turns = chars = 0
    for block in r['chat']:
        for plan, batches in block.items():
            for b in batches or []:
                for pair in b['turns'] or []:
                    for t in pair:
                        turns += 1
                        chars += len(t['content'] or '')
    pq_ = ast.literal_eval(r['probing_questions']) if r['probing_questions'] else {}
    n_q = sum(len(v) for v in pq_.values()) if isinstance(pq_, dict) else len(pq_)
    if isinstance(pq_, dict):
        cats.update({k: len(v) for k, v in pq_.items()})
    tot_chars += chars
    tot_turns += turns
    print(f"  {r['conversation_id']}: turns={turns} chars={chars/1e6:.1f}M (~{chars/4e6:.1f}M tokens) questions={n_q}")
print(f'total turns={tot_turns} chars={tot_chars/1e6:.0f}M (~{tot_chars/4e6:.0f}M tokens)')
print('question categories:', dict(cats))
first = ast.literal_eval(rows[0]['probing_questions'])
k = next(iter(first))
print('example question (', k, '):', json.dumps(first[k][0], ensure_ascii=False)[:600])
