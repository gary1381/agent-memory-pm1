"""Recompute ledger supersession chains in every BEAM store of the current set (after a norm_key change)."""
import sys
from pathlib import Path
sys.path.insert(0, str(Path(__file__).resolve().parent))
sys.path.insert(0, str(Path(__file__).resolve().parents[1] / 'src'))
from beam_cfg import STORES  # noqa: E402
from pm1.ledger import Ledger  # noqa: E402
from pm1.store import Store  # noqa: E402

tot = 0
for f in sorted(STORES.glob('conv*.sqlite')):
    led = Ledger(Store(f))
    scope = f.stem[4:]
    led.con.execute('BEGIN')
    n = led.resolve('beam', scope)
    led.con.execute('COMMIT')
    heads = led.con.execute('SELECT count(*) FROM ledger_state WHERE bank=? AND scope=? AND superseded_by IS NULL',
                            ('beam', scope)).fetchone()[0]
    tot += n
    print(f.name, 'superseded', n, 'live', heads)
print('total superseded', tot)
