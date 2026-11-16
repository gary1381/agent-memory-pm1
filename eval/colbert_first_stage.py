"""Would late interaction as the FIRST stage fix the misses? Exhaustive pplx MaxSim over every span.

Runs only on the questions listed (default: the T@10 misses from diag_dev.json) because pplx encodes
~5 docs/s on the shared GPU. Token vectors go to the shared vector cache, so reruns are free.
"""
from __future__ import annotations

import json
import sys
import time
from pathlib import Path

import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parent))
from lme import DATA, FAILURES, VecCache, fill, ingest  # noqa: E402

from pm1.embed import PplxLate, maxsim  # noqa: E402

diag = json.loads((Path(__file__).parent / 'reports' / 'diag_dev.json').read_text())
want = sys.argv[1].split(',') if len(sys.argv) > 1 else [m['qid'] for m in diag['misses']]
prior = {r['qid']: r for r in diag['rows']}
tok, cache = PplxLate(batch=16, timeout=120), VecCache()
out = []
for q in json.loads(DATA.read_text()):
    if q['question_id'] not in want:
        continue
    t0 = time.time()
    store, meta = ingest(q)
    fill(store, tok, 'tok', cache, batch=16)
    ids = list(meta)
    vecs = store.tokvecs(ids, tok.name)
    [qv] = tok.encode([q['question']], 'query')
    scores = {i: maxsim(qv, vecs[i]) for i in ids if i in vecs}
    order = sorted(scores, key=lambda i: -scores[i])
    ev = next((r for r, i in enumerate(order) if meta[i]['evidence']), None)
    row = {'qid': q['question_id'], 'type': q['question_type'], 'colbert_first_stage': ev,
           'unencoded_spans': len(ids) - len(scores),
           **{k: prior[q['question_id']][k] for k in ('bm25', 'e5', 'rrf_shipped')},
           'minutes': round((time.time() - t0) / 60, 1)}
    out.append(row)
    print(json.dumps(row), flush=True)
    store.close()
(Path(__file__).parent / 'reports' / 'colbert_first_stage_misses.json').write_text(
    json.dumps({'rows': out, 'encode_failures': FAILURES}, indent=2) + '\n')
