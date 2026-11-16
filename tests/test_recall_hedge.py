"""Reranker hedges (docs/EVALUATION.md E9): the fused top N kept after the reranked top 8, and the next turn of later hits."""
from __future__ import annotations

from pm1 import Recaller, Store
from pm1.sessions import SessionSearch


def test_keep_fused_puts_the_fused_top_after_the_reranked_front(monkeypatch):
    monkeypatch.setenv('PM1_KEEP_FUSED', '2')
    r = Recaller(Store(), planning=False)
    reranked, fused = list(range(20, 0, -1)), list(range(1, 21))
    assert r._keep_fused(reranked, fused) == list(range(20, 12, -1)) + [1, 2] + list(range(12, 2, -1))
    assert r._keep_fused(reranked, reranked) == reranked             # already in front: nothing moves


def _store() -> Store:
    s = Store()
    for i, (role, text) in enumerate([('user', 'hello there'), ('user', 'had a call with Quin about the clinic'),
                                      ('tool', 'tool output'), ('user', 'she moved to 251 Marigold Road'),
                                      ('assistant', 'noted')]):
        s.add_episode(bank='b', scope='s', session='sess1', seq=i, role=role, text=text,
                      occurred_at=f'2023-05-08T13:0{i}:00Z')
    return s


def test_next_turn_skips_tool_records_and_only_looks_forward(monkeypatch):
    monkeypatch.setenv('PM1_NEIGHBORS', '0')
    monkeypatch.setenv('PM1_NEXT_TURN', '999')
    res = Recaller(_store(), planning=False).recall('b', 's', 'Quin clinic', budget_tokens=4000, budget_ms=None)
    assert res.items[0]['seq'] == 1
    assert [it['seq'] for it in res.items if it.get('kind') == 'neighbor' and it['rank'] == 0] == [3]


def test_next_turn_off_by_default(monkeypatch):
    monkeypatch.setenv('PM1_NEIGHBORS', '0')
    monkeypatch.delenv('PM1_NEXT_TURN', raising=False)
    res = Recaller(_store(), planning=False).recall('b', 's', 'Quin clinic', budget_tokens=4000, budget_ms=None)
    assert not any(it.get('kind') == 'neighbor' for it in res.items)


def test_next_turn_respects_as_of():
    s = _store()
    assert s.next_turn('b', 's', 'sess1', 1)[0]['seq'] == 3
    with s.as_of('2023-05-08T13:02:00Z'):
        assert s.next_turn('b', 's', 'sess1', 1) == []


def test_sessions_next_turn_for_hits_without_neighbours(monkeypatch):
    monkeypatch.setenv('PM1_SESS_FULL', '1')
    monkeypatch.setenv('PM1_SESS_NEIGHBORS', '0')
    monkeypatch.setenv('PM1_SESS_NEXT_TURN', '5')
    s = Store()
    ids = [s.add_episode(bank='personal', scope='proj:/w/a', session='codex:s', seq=i, role=r,
                         text=t, occurred_at=f'2026-05-01T10:0{i}:00Z')['episode_id']
           for i, (r, t) in enumerate([('user', 'hello'), ('user', 'what limit did we pick for the zebra queue?'),
                                       ('tool', 'grep output'), ('assistant', 'sixty per hour'), ('user', 'ok')])]
    out = SessionSearch(s, Recaller(s), 'personal').search('zebra queue limit', k=1, budget_ms=None)
    got = [i['episode_id'] for i in out['items']]
    assert got == [ids[1], ids[3]] and out['items'][1]['neighbor_of'] == ids[1]
    early = SessionSearch(s, Recaller(s), 'personal').search('zebra queue limit', k=1, until='2026-05-01T10:02:30Z',
                                                             budget_ms=None)
    assert ids[3] not in [i['episode_id'] for i in early['items']]         # as-of: no later turn


def test_sessions_next_turn_from_leftover_never_shortens_a_hit(monkeypatch):
    monkeypatch.setenv('PM1_SESS_FULL', '1')
    monkeypatch.setenv('PM1_SESS_NEIGHBORS', '0')
    monkeypatch.setenv('PM1_SESS_NEXT_TURN', '5')
    s = Store()
    def add(sess, day, i, role, text):
        return s.add_episode(bank='personal', scope='proj:/w/a', session=sess, seq=i, role=role, text=text,
                             occurred_at=f'2026-05-0{day}T10:0{i}:00Z')['episode_id']
    ask = add('codex:s', 1, 0, 'user', 'what limit did we pick for the zebra queue?')
    reply = add('codex:s', 1, 1, 'assistant', 'sixty per hour, ' + 'set by the upstream quota on each worker. ' * 12)
    other = add('codex:t', 2, 0, 'user', 'zebra queue limit review: ' + 'the queue drains in arrival order. ' * 30)

    def shown(budget, left):
        monkeypatch.setenv('PM1_SESS_NEXT_LEFTOVER', left)
        out = SessionSearch(s, Recaller(s), 'personal').search('zebra queue limit', k=2, budget_tokens=budget,
                                                               budget_ms=None)
        return [(i['episode_id'], i['partial']) for i in out['items']]
    assert shown(500, '0') == [(ask, False), (reply, False), (other, True)]     # the next turn cuts the later hit
    assert shown(500, '1') == [(ask, False), (other, False)]                    # leftover: the hit stays whole
    assert shown(4000, '1') == [(ask, False), (reply, False), (other, False)]   # room left: the next turn follows
