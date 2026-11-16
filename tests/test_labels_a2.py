"""Label deployment test: A2 (label + value entries) after search_sessions' result. Off and shadow return
pm1's reply unchanged; on only appends records, within PM1_SESS_LABELS_X tokens, never one already shown."""
from __future__ import annotations

import json
import sqlite3

import numpy as np

from pm1 import Recaller, Store
from pm1.labels import LabelIndex
from pm1.sessions import SessionSearch, flat_items

TEXTS = [('user', 'what limit did we pick for the zebra queue?'), ('assistant', 'sixty per hour for the zebra queue'),
         ('user', 'unrelated chatter about lunch'), ('user', 'the backup host for the queue moved to gale-06'),
         ('assistant', 'noted'), ('tool', 'x' * 3000)]


def _store(tmp_path) -> tuple[Store, list[int]]:
    s = Store(str(tmp_path / 'm.sqlite'))
    ids = [s.add_episode(bank='personal', scope='proj:/w/a', session='codex:s1', seq=i, role=r, text=t,
                         occurred_at=f'2026-05-01T10:0{i}:00Z')['episode_id'] for i, (r, t) in enumerate(TEXTS)]
    return s, ids


def _index(tmp_path, ids: list[int], target: int) -> np.ndarray:
    """one entry per record, unit vectors; the query vector equals the target record's entry"""
    d = tmp_path / 'labels'
    d.mkdir()
    n = len(ids)
    mat = np.eye(n, 8, dtype=np.float32)
    at = np.array([np.datetime64(f'2026-05-01T10:0{i}:00').astype('datetime64[s]').astype(np.int64)
                   for i in range(n)])
    np.savez(d / 'entries.npz', lo=np.arange(n), hi=np.arange(1, n + 1), rec=np.array(ids), at=at,
             scope=np.zeros(n, np.int32), host=np.zeros(n, np.int32), mode=np.array([0] * (n - 1) + [1]),
             scope_names=np.array(['proj:/w/a']), host_names=np.array(['codex']))
    np.save(d / 'entries.c.npy', mat)
    return mat[ids.index(target)]


def _env(monkeypatch, tmp_path, mode):
    monkeypatch.setenv('PM1_SESS_FULL', '1')
    monkeypatch.setenv('PM1_SESS_NEIGHBORS', '0')
    monkeypatch.setenv('PM1_LABELS_DIR', str(tmp_path / 'labels'))
    monkeypatch.setenv('PM1_LABELS_SHADOW', str(tmp_path / 'shadow.sqlite'))
    if mode:
        monkeypatch.setenv('PM1_SESS_LABELS', mode)
    else:
        monkeypatch.delenv('PM1_SESS_LABELS', raising=False)
    import pm1.labels as L
    monkeypatch.setattr(L, 'DIR', tmp_path / 'labels')


def _same(out: dict) -> str:
    """the reply without timings and the freshness note (filled by a background thread, so it races in tests)"""
    out = json.loads(json.dumps(out))
    out.pop('latency_ms', None)
    for v in out.values():
        if isinstance(v, dict):
            v.pop('sources_up_to', None)
    return json.dumps(out, sort_keys=True)


def _search(s, **kw):
    out = SessionSearch(s, Recaller(s), 'personal').search('zebra queue limit', k=1, budget_ms=None, **kw)
    out.pop('latency_ms')
    return out


def test_shadow_reply_equals_off_and_logs_ids_only(monkeypatch, tmp_path):
    s, ids = _store(tmp_path)
    qv = _index(tmp_path, ids, ids[3])
    monkeypatch.setattr(LabelIndex, 'embed', lambda self, q: qv)
    _env(monkeypatch, tmp_path, None)
    off = _search(s)
    _env(monkeypatch, tmp_path, 'shadow')
    ss = SessionSearch(s, Recaller(s), 'personal')
    sh = ss.search('zebra queue limit', k=1, budget_ms=None)
    sh.pop('latency_ms')
    ss._lpool.shutdown(wait=True)
    assert _same(sh) == _same(off)
    row = sqlite3.connect(tmp_path / 'shadow.sqlite').execute('SELECT a2, ctl, error, lang FROM shadow').fetchone()
    assert row[2] is None and row[3] == 'en'
    assert ids[3] in [a for a, _ in json.loads(row[0])]
    assert 'gale' not in ''.join(str(x) for x in row)            # no record text in the log


def test_on_only_appends(monkeypatch, tmp_path):
    s, ids = _store(tmp_path)
    qv = _index(tmp_path, ids, ids[3])
    monkeypatch.setattr(LabelIndex, 'embed', lambda self, q: qv)
    _env(monkeypatch, tmp_path, None)
    off = flat_items(_search(s))
    _env(monkeypatch, tmp_path, 'on')
    res = _search(s)
    on = flat_items(res)
    assert on[:0] == [] and all(any(o == x for x in on) for o in off)    # every off record, unchanged
    added = [x for x in on if x['episode_id'] not in {o['episode_id'] for o in off}]
    assert added and added[0]['episode_id'] == ids[3]
    assert res['route']['labels']['tokens'] <= 512
    assert len({x['episode_id'] for x in on}) == len(on)                    # nothing shown twice


def test_on_falls_back_when_the_tower_fails(monkeypatch, tmp_path):
    s, ids = _store(tmp_path)
    _index(tmp_path, ids, ids[3])

    def boom(self, q):
        raise TimeoutError('down')
    monkeypatch.setattr(LabelIndex, 'embed', boom)
    _env(monkeypatch, tmp_path, None)
    off = _search(s)
    _env(monkeypatch, tmp_path, 'on')
    res = _search(s)
    assert res['route']['labels'] == {'skipped': 'TimeoutError'}
    res['route'].pop('labels')
    assert _same(res) == _same(off)


def test_long_record_shows_its_labeled_part(monkeypatch, tmp_path):
    s, ids = _store(tmp_path)
    qv = _index(tmp_path, ids, ids[5])           # the 3,000-char tool record, a doc-tail entry (mode 1)
    monkeypatch.setattr(LabelIndex, 'embed', lambda self, q: qv)
    _env(monkeypatch, tmp_path, 'on')
    out = SessionSearch(s, Recaller(s), 'personal').search('zebra queue limit', k=1, budget_ms=None,
                                                          roles=['user', 'assistant', 'tool'])
    tool = [x for x in flat_items(out) if x['episode_id'] == ids[5]]
    assert tool and tool[0]['partial'] and len(tool[0]['text']) <= 2000


def test_index_reloads_when_rebuilt(monkeypatch, tmp_path):
    s, ids = _store(tmp_path)
    _env(monkeypatch, tmp_path, 'on')
    ss = SessionSearch(s, Recaller(s), 'personal')
    assert ss._label_index() is None                    # no index yet: pm1's own result
    qv = _index(tmp_path, ids, ids[3])
    monkeypatch.setattr(LabelIndex, 'embed', lambda self, q: qv)
    first = ss._label_index()
    assert first is not None and ss._label_index() is first
    import os
    p = tmp_path / 'labels' / 'entries.npz'
    os.utime(p, ns=(p.stat().st_atime_ns, p.stat().st_mtime_ns + 10**9))
    assert ss._label_index() is not first
