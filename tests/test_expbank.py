"""Stored experience lessons: written once per card, served by code_history (pull) and pushed once per project and
session by the pre-tool-use hook, same project only, half the sessions (A/B) only logging."""
from __future__ import annotations

import hashlib
import json
import re

from pm1 import cc_hook, expbank
from pm1.codemap import CodeMap

from test_failsig import build


def fake_writer(text):
    ep = re.findall(r'\[ep (\d+)\]', text)
    d = {'kind': 'pitfall', 'lesson': 'Editing util.py can break the `runs` key lookup; rerun pytest after it.',
         'applies_to': ['util.py'], 'evidence': [int(ep[0])], 'supported': True}
    return json.dumps(d), 100, 20


def synced(tmp_path):
    db, s = build(tmp_path)
    st = expbank.sync(s.con, write=fake_writer)
    assert st['new'] == 1 and st['kept'] == 1
    assert expbank.sync(s.con, write=fake_writer)['new'] == 0           # each card is written once
    return db, s


def test_code_history_serves_the_lesson(tmp_path):
    db, s = synced(tmp_path)
    s.con.execute('DROP TABLE code_repos')               # build() left a stub without the graph columns
    cm = CodeMap(s, 'personal')
    cm.git_root = lambda p: '/repo'
    cm.current_repo = lambda: None
    out = cm.history('/repo/app/util.py')
    assert out['lessons'][0]['kind'] == 'pitfall' and 'util.py' in out['lessons'][0]['lesson']
    assert expbank.for_file(s.con, '/repo/app/other.py') == []


def test_private_code_never_served(tmp_path, monkeypatch):
    monkeypatch.setenv('PM1_PRIVATE_DIRS', '/private-code/')
    db, s = synced(tmp_path)
    assert expbank.for_file(s.con, '/home/u/private-code/x/util.py') == []


def _sid(parity):
    i = 0
    while int(hashlib.sha1(f'claude:s{i}'.encode()).hexdigest(), 16) % 2 != parity:
        i += 1
    return f's{i}'


def test_hook_pushes_same_project_once_and_holds_half(tmp_path, monkeypatch):
    db, s = synced(tmp_path)
    monkeypatch.setattr(cc_hook, 'DB', str(db))
    tr = tmp_path / 't.jsonl'
    tr.write_text(json.dumps({'type': 'user', 'message': {'role': 'user', 'content': 'fix the runs report in util'}}) + '\n')
    push = {'tool_name': 'Edit', 'session_id': _sid(0), 'transcript_path': str(tr),
            'tool_input': {'file_path': '/repo/app/new.py'}}
    out = cc_hook.pre_tool_use(push)
    assert 'pm1 experience' in out['hookSpecificOutput']['additionalContext']
    assert cc_hook.pre_tool_use({**push, 'tool_input': {'file_path': '/repo/app/b.py'}}) is None   # once per project
    hold = {**push, 'session_id': _sid(1)}
    assert cc_hook.pre_tool_use(hold) is None
    tools = [r[0] for r in s.con.execute("SELECT tool FROM mcp_calls WHERE tool LIKE 'hook_exp_%'")]
    assert sorted(tools) == ['hook_exp_hold', 'hook_exp_push']
    other = {**push, 'session_id': _sid(0) + 'x', 'tool_input': {'file_path': '/elsewhere/a.py'}}
    assert cc_hook.pre_tool_use(other) is None                                   # no lesson from another project


def test_correction_is_a_short_reply_that_opens_with_it():
    assert expbank._corrects('不对，改回原来的方式')
    assert expbank._corrects("that's wrong, use the old path")
    assert not expbank._corrects('这个方法对不对？')                     # a question
    assert not expbank._corrects('报告：' + '很长的内容' * 100 + '不对')    # a long report
