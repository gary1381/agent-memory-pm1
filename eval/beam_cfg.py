"""Which BEAM split the eval scripts work on: PM1_BEAM_SET=10m (default) or 1m (also 100k, 500k)."""
import os
from pathlib import Path

SET = os.environ.get('PM1_BEAM_SET', '10m').lower()
NAME = f'beam{SET}'                                    # beam10m / beam1m
D = Path.home() / '.cache/pm1/data' / NAME             # parquet files
STORES = Path.home() / '.cache/pm1' / NAME             # one SQLite per conversation
GLOB = f'{SET.upper()}-*.parquet'


def conv_ids() -> list[str]:
    import pyarrow.parquet as pq
    ids = []
    for f in sorted(D.glob(GLOB)):
        ids += pq.read_table(f, columns=['conversation_id']).column('conversation_id').to_pylist()
    return sorted(set(ids), key=lambda x: int(x) if str(x).isdigit() else x)
