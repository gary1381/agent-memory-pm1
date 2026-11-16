"""Step-bounded tools: max_turn refuses future turns; absent max_turn keeps legacy behaviour."""
import json
import sqlite3
from pathlib import Path

import pytest

from pm1.store import Store
from pm1.tree import Tree
from pm1.assertions import Ledger


RAW = {
    't1': ('user', 'My budget is $400.', '2024-12-05'),
    't2': ('assistant', 'Maybe $450?', '2024-12-05'),
    't3': ('user', 'OK, $450.', '2024-12-06'),
    't10': ('user', 'I adopted a dog named Rex.', '2024-12-07'),
    't11': ('user', 'Rex is a labrador.', '2024-12-08'),
}


def test_tree_refuse_future_evidence(tmp_path):
    t = Tree(Store(tmp_path / 't.sqlite'), 'b', 's', raw=RAW.get, max_turn=10)
    assert t.apply({'op': 'create_entity', 'name': 'User'})['ok']
    ok = t.apply({'op': 'add_version', 'entity': 'User', 'attribute': 'budget', 'value': '$400', 'evidence': ['t1']})
    assert ok['ok']
    bad = t.apply({'op': 'add_version', 'entity': 'User', 'attribute': 'dog', 'value': 'Rex', 'evidence': ['t11']})
    assert not bad['ok']
    assert 'after the current step' in bad['error']
    assert 't11' in bad['error'] and 't10' in bad['error']


def test_tree_raw_bound_via_evidence_message(tmp_path):
    """Digest turns after max_turn are rejected the same way."""
    t = Tree(Store(tmp_path / 't.sqlite'), 'b', 's', raw=RAW.get, max_turn=3)
    bad = t.apply({'op': 'add_digest', 'turns': ['t10'], 'source': 'user', 'text': 'Adopted Rex.'})
    assert not bad['ok'] and 'after the current step' in bad['error']


def test_tree_no_max_turn_unchanged(tmp_path):
    t = Tree(Store(tmp_path / 't.sqlite'), 'b', 's', raw=RAW.get)  # no max_turn
    assert t.apply({'op': 'create_entity', 'name': 'User'})['ok']
    assert t.apply({'op': 'add_version', 'entity': 'User', 'attribute': 'dog', 'value': 'Rex',
                    'evidence': ['t10']})['ok']


def test_ledger_refuse_future_turn():
    led = Ledger(sqlite3.connect(':memory:'), RAW.get, max_turn=10)
    e = led.apply('resolve_entity', {'surface_name': 'User', 'turn_id': 't1'})['entity_id']
    ok = led.apply('append_assertion', {'turn_id': 't1', 'entity_id': e, 'facet': 'other',
                                        'quote': 'My budget is $400.', 'modality': 'user_claim'})
    assert ok['ok']
    bad = led.apply('append_assertion', {'turn_id': 't11', 'entity_id': e, 'facet': 'other',
                                         'quote': 'Rex is a labrador.', 'modality': 'user_claim'})
    assert not bad['ok'] and 'after the current step' in bad['error']
    # resolve_entity turn_id also bounded
    bad2 = led.apply('resolve_entity', {'surface_name': 'Rex', 'turn_id': 't11'})
    assert not bad2['ok'] and 'after the current step' in bad2['error']


def test_ledger_no_max_turn_unchanged():
    led = Ledger(sqlite3.connect(':memory:'), RAW.get)
    e = led.apply('resolve_entity', {'surface_name': 'User', 'turn_id': 't11'})['entity_id']
    assert led.apply('append_assertion', {'turn_id': 't11', 'entity_id': e, 'facet': 'other',
                                          'quote': 'Rex is a labrador.', 'modality': 'user_claim'})['ok']


def _mem_env(tmp_path, max_turn=None):
    """Build a Mem-like workspace with CURRENT.json + raw sqlite (episodes)."""
    root = tmp_path / 'memfs'
    raws = tmp_path / 'raw'
    root.mkdir(); raws.mkdir()
    cur = {'conv': '1', 'step': 'test:r1:t1-t10'}
    if max_turn is not None:
        cur['max_turn'] = max_turn
    (root / 'CURRENT.json').write_text(json.dumps(cur))
    raw_db = sqlite3.connect(raws / 'conv1.sqlite')
    raw_db.execute('CREATE TABLE episodes (id INTEGER PRIMARY KEY, role TEXT, text TEXT, occurred_at TEXT)')
    for tid, (role, text, day) in RAW.items():
        raw_db.execute('INSERT INTO episodes VALUES (?,?,?,?)', (int(tid[1:]), role, text, day))
    raw_db.commit(); raw_db.close()
    return root, raws


def test_memfs_refuse_future_evidence(tmp_path, monkeypatch):
    import pm1.memfs_mcp as m
    root, raws = _mem_env(tmp_path, max_turn=10)
    monkeypatch.setattr(m, 'ROOT', root)
    monkeypatch.setattr(m, 'RAWS', raws)
    mem = m.Mem()
    # raw_turn past max
    out = mem.call('raw_turn', {'turn': 't11'})
    assert 'error' in out and 'after the current step' in out['error']
    # raw_turn within max
    out = mem.call('raw_turn', {'turn': 't1'})
    assert out.get('role') == 'user'
    # write citing future turn
    with pytest.raises(ValueError, match='after the current step'):
        mem.call('mem_write', {'path': '/dog', 'content': 'kind: fact\nRex', 'why': 'test', 'evidence': ['t11']})
    # write within max
    ok = mem.call('mem_write', {'path': '/budget', 'content': 'kind: fact\n$400', 'why': 'test', 'evidence': ['t1']})
    assert ok['ok']


def test_memfs_no_max_turn_unchanged(tmp_path, monkeypatch):
    import pm1.memfs_mcp as m
    root, raws = _mem_env(tmp_path, max_turn=None)
    monkeypatch.setattr(m, 'ROOT', root)
    monkeypatch.setattr(m, 'RAWS', raws)
    mem = m.Mem()
    assert mem.call('raw_turn', {'turn': 't11'})['role'] == 'user'
    assert mem.call('mem_write', {'path': '/dog', 'content': 'kind: fact\nRex', 'why': 'test',
                                 'evidence': ['t11']})['ok']


def test_tree_mcp_raw_turn_bound(tmp_path, monkeypatch):
    import pm1.tree_mcp as tm
    trees = tmp_path / 'trees'
    raws = tmp_path / 'raw'
    trees.mkdir(); raws.mkdir()
    (trees / 'CURRENT.json').write_text(json.dumps({'conv': '1', 'step': 's', 'max_turn': 10}))
    # empty tree store
    sqlite3.connect(trees / 'conv1.sqlite').close()
    raw_db = sqlite3.connect(raws / 'conv1.sqlite')
    raw_db.execute('CREATE TABLE episodes (id INTEGER PRIMARY KEY, role TEXT, text TEXT, occurred_at TEXT)')
    raw_db.execute("INSERT INTO episodes VALUES (11, 'user', 'later', '2024-12-08')")
    raw_db.execute("INSERT INTO episodes VALUES (1, 'user', 'early', '2024-12-05')")
    raw_db.commit(); raw_db.close()
    monkeypatch.setattr(tm, 'TREES', trees)
    monkeypatch.setattr(tm, 'RAWS', raws)
    out = tm.call('raw_turn', {'turn': 't11'})
    assert 'error' in out and 'after the current step' in out['error']
    out = tm.call('raw_turn', {'turn': 't1'})
    assert out['role'] == 'user'


def test_assert_mcp_raw_turn_bound(tmp_path, monkeypatch):
    import pm1.assert_mcp as am
    root = tmp_path / 'ledger'
    raws = tmp_path / 'raw'
    root.mkdir(); raws.mkdir()
    (root / 'CURRENT.json').write_text(json.dumps({'conv': '1', 'step': 's', 'max_turn': 10}))
    sqlite3.connect(root / 'conv1.sqlite').close()
    raw_db = sqlite3.connect(raws / 'conv1.sqlite')
    raw_db.execute('CREATE TABLE episodes (id INTEGER PRIMARY KEY, role TEXT, text TEXT, occurred_at TEXT)')
    raw_db.execute("INSERT INTO episodes VALUES (11, 'user', 'later', '2024-12-08')")
    raw_db.execute("INSERT INTO episodes VALUES (1, 'user', 'early', '2024-12-05')")
    raw_db.commit(); raw_db.close()
    monkeypatch.setattr(am, 'ROOT', root)
    monkeypatch.setattr(am, 'RAWS', raws)
    out = am.call('raw_turn', {'turn': 't11'})
    assert 'error' in out and 'after the current step' in out['error']
    out = am.call('raw_turn', {'turn': 't1'})
    assert out['role'] == 'user'
