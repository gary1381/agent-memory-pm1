"""Validity in time and condition, and freshness from observation (never deletion, never an automatic change)."""
from __future__ import annotations

import datetime as dt
import json

import pytest

from pm1 import Store
from pm1 import cc_hook, lesson_mine, rules
from pm1.brief import Brief


def test_until_review_when(tmp_path, monkeypatch):
    s = Store(tmp_path / 'm.sqlite')
    s.add_episode(bank='personal', scope='proj:/w/tw', session='claude:a', role='user', source='u#1', text='先不用27B了')
    rid = rules.add(s.con, 'Do not use the 27B model for now.', projects=['/w/tw'], until='2w',
                    cond='while the GPU host is busy with other jobs')
    r = rules.get(s.con, rid)
    assert r['until'] == (dt.date.today() + dt.timedelta(days=14)).isoformat()
    shown = rules.render_rules(rules.active(s.con, '/w/tw'))
    assert 'only when: while the GPU host is busy' in shown and f"valid until {r['until']}" in shown
    s.con.execute("UPDATE rules SET until='2020-01-01' WHERE id=?", (rid,))
    assert rules.active(s.con, '/w/tw') == [] and [x['id'] for x in rules.expired_rules(s.con, '/w/tw')] == [rid]
    assert 'Rules to re-check with the user' in Brief.markdown(Brief(s).build(session='claude:a'))
    monkeypatch.setattr(cc_hook, 'DB', str(tmp_path / 'm.sqlite'))
    ctx = cc_hook.session_start({'source': 'startup', 'cwd': '/w/tw'})['hookSpecificOutput']['additionalContext']
    assert 'passed their end date' in ctx and 'Do not use the 27B' not in ctx
    rules.main(['--db', str(tmp_path / 'm.sqlite'), 'renew', str(rid), '--until', '1w'])
    assert rules.active(s.con, '/w/tw')                                            # back in force
    rules.edit(s.con, rid, review='2020-01-02')
    assert 'review due' in rules.render_rules(rules.active(s.con, '/w/tw'))
    with pytest.raises(ValueError):
        rules.parse_when('next tuesday')


def test_freshness_from_observation(tmp_path):
    s = Store(tmp_path / 'm.sqlite')
    sc = 'proj:/w/sm'

    def tool(i, cmd, out, failed=False):
        return s.add_episode(bank='personal', scope=sc, session='claude:b', role='tool', seq=i, source=f't#{i}',
                             occurred_at=f'2026-10-1{i}T10:00:00', text=f"[Bash{' FAILED' if failed else ''}] "
                             + json.dumps({'command': cmd}) + '\n→ ' + out)['episode_id']
    src = tool(1, 'cd /w/sm/runtime && PYTHONPATH=src python3 -m pytest -q', '5 passed')
    rec = rules.add(s.con, "sm: run `python3 -m pytest` as `cd /w/sm/runtime && PYTHONPATH=src python3 -m pytest "
                    "-q` -- written the plain way it failed", kind='recipe', projects=['/w/sm'])
    s.con.execute('UPDATE rules SET source_episode=? WHERE id=?', (src, rec))
    les = rules.add(s.con, 'bge-small-en is English only', kind='lesson', triggers=['bge-small-en'])
    tool(2, 'python3 -m pytest -q', "ModuleNotFoundError: No module named 'sm'", failed=True)   # plain way: ignored
    tool(3, 'cd /w/sm/runtime && PYTHONPATH=src python3 -m pytest -q tests/x.py', '3 passed')
    tool(4, 'grep -r model cfg.yaml', 'model: BAAI/bge-small-en-v1.5')
    out = lesson_mine.freshness(s.con)
    assert out['recipes_confirmed'] == 1 and out['lessons_seen'] == 1 and out['recipes_flagged'] == 0
    assert rules.get(s.con, rec)['last_seen'].startswith('2026-10-13')
    assert rules.get(s.con, les)['last_seen'].startswith('2026-10-14')
    tool(5, 'cd /w/sm/runtime && PYTHONPATH=src python3 -m pytest -q',
         "ModuleNotFoundError: No module named 'sm_runtime'", failed=True)             # the recipe's way fails now
    out = lesson_mine.freshness(s.con)
    assert out['recipes_flagged'] == 1
    assert 'observed failing again' in rules.render_rules(rules.active(s.con, '/w/sm'))
    assert rules.get(s.con, rec)['status'] == 'active'                                # flagged, not changed
