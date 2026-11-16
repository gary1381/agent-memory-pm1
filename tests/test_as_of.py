"""As-of recall: a question asked part-way through a history sees only the records up to that time."""
from __future__ import annotations

import numpy as np

from pm1 import Recaller, Store


def _store():
    s = Store(index_models=('dense:word-dense',))
    ids = {}
    for i, (day, text) in enumerate([('2024-01-02', 'Maria: the pottery class costs 40 dollars'),
                                     ('2024-01-02', 'Maria: I signed up for it'),
                                     ('2024-03-05', 'Maria: correction, the pottery class costs 55 dollars now')]):
        ids[i] = s.add_episode(bank='b', scope='s', session='s1', seq=i, role='user', text=text,
                               occurred_at=f'{day}T10:0{i}:00Z')['episode_id']
    return s, ids


def test_until_hides_later_records(monkeypatch):
    s, ids = _store()
    rec = Recaller(s, dense=None, planning=False)              # keyword path; the vector path is checked below
    q = 'How much does the pottery class cost?'
    all_eps = {it['episode_id'] for it in rec.recall('b', 's', q, budget_tokens=500).items}
    assert ids[2] in all_eps
    early = rec.recall('b', 's', q, budget_tokens=500, until='2024-02-01T00:00:00')
    assert ids[2] not in {it['episode_id'] for it in early.items}
    assert ids[0] in {it['episode_id'] for it in early.items}
    assert '55 dollars' not in early.context
    assert s._until is None                          # restored after the call
    with s.as_of('2024-01-02T10:00:30'):             # same day: the exact time still counts
        assert {n['episode_id'] for n in s.neighbors('b', 's', 's1', 0, radius=2)} == {ids[0]}


def test_until_filters_vector_rows():
    s, ids = _store()
    sids = np.array(sorted(s.first_spans('b', 's', list(ids.values())).values()), dtype=np.int64)
    mat = np.eye(len(sids), dtype=np.float32)
    with s.as_of('2024-02-01'):
        kept, m = s._as_of_rows('b', 's', sids, mat)
    assert len(kept) == 2 and m.shape == (2, len(sids))
    assert len(s._as_of_rows('b', 's', sids, mat)[0]) == 3          # outside the block: everything
