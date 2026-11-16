"""memory_recall fills the token budget (no 12-item cap) and brings each top hit's session neighbours."""
from __future__ import annotations

from pm1 import Recaller, Store


def _store(n: int = 30) -> Store:
    s = Store()
    for i in range(n):
        role = 'user' if i % 2 == 0 else 'assistant'
        text = f'turn {i} about gardening tomatoes' if i == 10 else f'unrelated chatter number {i} weather'
        s.add_episode(bank='b', scope='s', session='sess1', seq=i, role=role, text=text,
                      occurred_at=f'2023-05-08T13:{i:02d}:00Z')
    for i in range(30):
        s.add_episode(bank='b', scope='s', session='sess2', seq=i, role='user', text=f'tomatoes note {i}',
                      occurred_at=f'2023-06-08T13:{i:02d}:00Z')
    return s


def test_fills_budget_past_twelve_items(monkeypatch):
    monkeypatch.setenv('PM1_NEIGHBORS', '0')
    res = Recaller(_store(), planning=False).recall('b', 's', 'tomatoes', budget_tokens=4000, budget_ms=None)
    assert len(res.items) > 12


def test_explicit_max_items_still_caps(monkeypatch):
    monkeypatch.setenv('PM1_NEIGHBORS', '0')
    res = Recaller(_store(), planning=False).recall('b', 's', 'tomatoes', budget_tokens=4000, max_items=5,
                                                    budget_ms=None)
    assert len(res.items) == 5


def test_top_hit_brings_its_neighbours(monkeypatch):
    monkeypatch.setenv('PM1_NEIGHBORS', '1,2')
    res = Recaller(_store(), planning=False).recall('b', 's', 'gardening', budget_tokens=4000, budget_ms=None)
    hit = res.items[0]
    assert hit['seq'] == 10 and hit['session'] == 'sess1'
    nb = sorted(it['seq'] for it in res.items if it.get('kind') == 'neighbor' and it['session'] == 'sess1')
    assert nb == [8, 9, 11, 12]
    assert '[2023-05-08T13:09:00Z · sess1 · assistant] unrelated chatter number 9' in res.context


def test_neighbours_off(monkeypatch):
    monkeypatch.setenv('PM1_NEIGHBORS', '0')
    res = Recaller(_store(), planning=False).recall('b', 's', 'gardening', budget_tokens=4000, budget_ms=None)
    assert not any(it.get('kind') == 'neighbor' for it in res.items)


def test_block_that_does_not_fit_keeps_the_hit(monkeypatch):
    monkeypatch.setenv('PM1_NEIGHBORS', '1,2')
    res = Recaller(_store(), planning=False).recall('b', 's', 'gardening', budget_tokens=30, budget_ms=None)
    assert [it['seq'] for it in res.items] == [10]
