"""Possible conflicts: later user words that may change a rule are flagged for the user, never applied."""
from __future__ import annotations

from pm1 import Store
from pm1 import conflicts, rules
from pm1.brief import Brief


class FakeScorer:
    def score(self, query, docs):
        return [0.97 if ('quota' in d or '额度' in d) else 0.2 for d in docs]


def setup(tmp_path):
    s = Store(tmp_path / 'm.sqlite')
    sc = 'proj:/w/pm'

    def u(i, text, scope=sc, source=None):
        return s.add_episode(bank='personal', scope=scope, session='claude:x', role='user', seq=i,
                             occurred_at=f'2026-10-18T0{i}:00:00', text=text, source=source or f'u#{i}')['episode_id']
    e1 = u(1, '标注改用 Cursor 模型为主（composer-2.5 + grok-4.7）')
    u(2, 'my quota for Cursor models composer-2.5 and grok-4.7 is very little now; switch labeling to gpt-6-luna')
    u(3, '请读通知，标注改用 Cursor 模型为主（composer-2.5 + grok-4.7）')                     # restatement
    u(4, 'composer-2.5 and grok-4.7 quota is gone, switch labeling', scope='proj:/w/other')    # other project
    u(5, 'composer-2.5 grok-4.7 quota: switch labeling now', source='mcp:memory_capture:codex')  # agent-written
    return s, sc, e1


def test_find_flags_later_change_only(tmp_path):
    s, sc, e1 = setup(tmp_path)
    rid = rules.add(s.con, 'Labeling: use Cursor models (composer-2.5 + grok-4.7) first.', projects=['/w/pm'],
                    source_episode=e1)
    found = conflicts.find(s.con, FakeScorer(), rules.get(s.con, rid))
    assert [c['episode_id'] for c in found] == [e1 + 1] and found[0]['hint'].startswith('a condition')
    conflicts.record(s.con, rid, found)
    shown = rules.render_rules(rules.active(s.con, sc))
    assert '⚠ the user may have changed this later' in shown and 'pm1-rule conflicts' in shown
    assert 'may have changed a standing rule' in Brief.markdown(Brief(s).build(session='claude:x'))
    assert rules.get(s.con, rid)['status'] == 'active'                          # flagged, never changed
    rules.edit(s.con, rid, why='Codex quota runs out faster')                   # the user looked again: settled
    assert rules.open_conflicts(s.con) == []


def test_approve_holds_a_conflicting_proposal(tmp_path, monkeypatch, capsys):
    s, sc, e1 = setup(tmp_path)
    p = rules.propose(s.con, 'Labeling: use Cursor models (composer-2.5 + grok-4.7) first.', e1,
                      'Cursor 模型为主', projects=['/w/pm'])
    import pm1.rerank as rr
    fake = type('R', (), {'rerankers': [type('S', (), {'healthy': lambda self: True, 'quality': 1.0,
                                                         'score': FakeScorer().score})()]})()
    monkeypatch.setattr(rr, 'router_from_env', lambda: fake)
    db = s.con.execute('PRAGMA database_list').fetchone()[2]
    try:
        rules.main(['--db', db, 'approve', str(p['id'])])
    except SystemExit:
        pass
    assert 'NOT approved yet' in capsys.readouterr().out and rules.get(s.con, p['id'])['status'] == 'proposed'
    rules.main(['--db', db, 'approve', str(p['id']), '--anyway'])
    assert rules.get(s.con, p['id'])['status'] == 'active'
    cid = rules.open_conflicts(s.con)[0]['id']
    rules.dismiss(s.con, cid)
    assert rules.open_conflicts(s.con) == []


def test_judges_filter_what_reaches_the_user(tmp_path):
    s, sc, e1 = setup(tmp_path)
    rid = rules.add(s.con, 'Labeling: use Cursor models (composer-2.5 + grok-4.7) first.', projects=['/w/pm'],
                    source_episode=e1)
    r = rules.get(s.con, rid)
    pairs = [(r, {'episode_id': 101, 'score': 0.9, 'hint': 'changed later?', 'excerpt': 'quota low, switch to luna'}),
             (r, {'episode_id': 102, 'score': 0.8, 'hint': 'changed later?', 'excerpt': 'composer-2.5 is great'})]

    def fake_judge(kind, items, q, labels):
        assert kind == 'conflict' and set(labels) >= {'replaces', 'consistent'}
        return {'k0': {'composer': 'replaces', 'grok47': 'adds_condition'},
                'k1': {'composer': 'consistent', 'grok47': 'consistent'}}
    out = conflicts.judged(s.con, pairs, fake_judge)
    assert [st for _, _, st in out] == ['open', 'auto_dismissed']
    assert 'adds a condition' in out[0][1]['hint']
    assert [st for _, _, st in conflicts.judged(s.con, pairs, lambda *a: None)] == ['open', 'open']   # judges down
    assert s.con.execute('SELECT count(*) FROM judge_log').fetchone()[0] == 2


def test_judge_stats_follow_user_decisions(tmp_path):
    from pm1 import judge
    s, sc, e1 = setup(tmp_path)
    rid = rules.add(s.con, 'Labeling: use Cursor models (composer-2.5 + grok-4.7) first.', projects=['/w/pm'],
                    source_episode=e1)
    judge.log(s.con, 'conflict', f'rule:{rid}:ep:{e1 + 1}', 'x', {'grok47': 'replaces'}, 'open')
    conflicts.record(s.con, rid, [{'episode_id': e1 + 1, 'score': 0.9, 'hint': 'changed later?', 'excerpt': 'x'}])
    assert judge.stats(s.con)['user_decided'] == 0
    rules.edit(s.con, rid, why='quota')                        # the user acted on the flag: judges were right
    st = judge.stats(s.con)
    assert (st['user_decided'], st['agree'], st['until_finetune']) == (1, 1, judge.FINETUNE_AT - 1)
