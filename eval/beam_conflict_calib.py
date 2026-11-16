"""Calibrate contradiction pairing on BEAM (no LLM): for every question record the top never-statements by query
similarity and their closest affirming user turn, then compare with the gold contradiction pairs.
usage: .venv-gpu/bin/python eval/beam_conflict_calib.py <out.json>"""
import json, sys
from pathlib import Path

import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parent))
from beam_cfg import conv_ids  # noqa: E402
from beam_coverage import gold_ids  # noqa: E402
from beam_qa import CATS, STORES, questions  # noqa: E402
from pm1 import Recaller, Store  # noqa: E402
from pm1.embed import FastE5  # noqa: E402

dense = FastE5(gpu=__import__('os').environ.get('PM1_GPU', '1') != '0')
out = {}
for conv in conv_ids():
    store = Store(STORES / f'conv{conv}.sqlite')
    rec = Recaller(store, dense=dense, n_conflicts=3, conflict_query_sim=0.0, conflict_pair_sim=0.0)
    meta = store.span_meta('beam', conv)
    seq = lambda sid: int(meta['seq'][meta['pos'][sid]])   # noqa: E731
    negs = store.negations('beam', conv)
    ids, mat, row, _ = rec._dense_rows('beam', conv)
    qs = questions(conv)
    for cat in CATS:
        for i, q in enumerate(qs[cat]):
            pairs = rec._conflicts('beam', conv, q['question'])
            qv = rec._qvec(q['question'])
            qsims = {p[0]: float(mat[row[p[0]]] @ qv) for p in pairs}
            out[f'{conv}|{cat}|{i}'] = {
                'pairs': [[seq(a), seq(b), round(s, 4), round(qsims[a], 4)] for a, b, s in pairs],
                'gold': sorted(gold_ids(q)) if cat == 'contradiction_resolution' else []}
    print(f'conv {conv}: {len(negs)} never-statements', flush=True)
    store.close()
Path(sys.argv[1]).write_text(json.dumps(out))
