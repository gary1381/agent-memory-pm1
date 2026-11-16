"""Load consolidation outputs (labeling schema) into each BEAM conversation store's ledger.

  PM1_BEAM_SET=1m beam_ledger_ingest.py <task-prefix> <labeler> [--source NAME]
      reads $PM1_LABEL_DIR/outbox/<labeler>/<prefix>_*.out.jsonl (ids conv<N>|session|day|k),
      replaces that source's ledger rows in conv<N>.sqlite (bank 'beam', scope '<N>'), resolves supersession.
"""
import os
import argparse, sys
from collections import defaultdict
from pathlib import Path

HERE = Path(__file__).resolve().parent
sys.path.insert(0, str(HERE)); sys.path.insert(0, str(HERE.parent / 'src'))
from beam_cfg import STORES  # noqa: E402
from pm1 import Store  # noqa: E402
from pm1.ledger import Ledger, load_labels  # noqa: E402

ap = argparse.ArgumentParser()
ap.add_argument('prefix'); ap.add_argument('labeler'); ap.add_argument('--source', default=None)
a = ap.parse_args()
src = a.source or f'{a.labeler}:{a.prefix}'
by = defaultdict(list)
for p in sorted((Path(os.environ.get('PM1_LABEL_DIR', str(Path.home() / 'pm1-labeling'))) / 'outbox' / a.labeler).glob(f'{a.prefix}_*.out.jsonl')):
    for r in load_labels(p):
        conv, rest = r['id'].split('|', 1)
        by[conv].append((rest, r['response']))
for conv, items in sorted(by.items(), key=lambda kv: int(kv[0][4:])):
    st = Store(STORES / f'{conv}.sqlite')
    led = Ledger(st)
    st.con.execute('BEGIN')         # one transaction per conversation (autocommit = one fsync per row: hours)
    led.clear('beam', conv[4:], source=src)
    n = sum(led.add_extraction('beam', conv[4:], rest, rest.split('|')[1], resp, source=src) for rest, resp in items)
    sup = led.resolve('beam', conv[4:])
    st.con.execute('COMMIT')
    print(conv, len(items), 'chunks', n, 'rows', sup, 'superseded', led.stats('beam', conv[4:]), flush=True)
    st.close()
