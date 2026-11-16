"""Failure signatures and the Claude Code push hooks: exact matches only, once per session, environment errors
left out of edit warnings."""
from __future__ import annotations

import json

from pm1 import Store
from pm1 import cc_hook, failsig
from pm1.codemap import CodeMap

TB = 'Traceback (most recent call last):\n  File "/repo/app/util.py", line 3\nKeyError: \'runs\'\n'


def tool(name, cmd, out, failed=False):
    return f"[{name}{' FAILED' if failed else ''}] " + json.dumps({'command': cmd}) + '\n→ ' + out


def build(tmp_path):
    db = tmp_path / 'm.sqlite'
    s = Store(db)
    sc, ss = 'proj:/repo', 'claude:old'
    s.add_episode(bank='personal', scope=sc, session=ss, role='user', text='fix the report', source='t#1')
    s.add_episode(bank='personal', scope=sc, session=ss, role='tool', source='t#2',
                  text=tool('Edit', '', 'ok').replace('{"command": ""}', '{"file_path": "/repo/app/util.py"}'))
    s.add_episode(bank='personal', scope=sc, session=ss, role='tool', source='t#3',
                  text=tool('Bash', 'cd /repo && python -m pytest -q tests', TB, failed=True))
    s.add_episode(bank='personal', scope=sc, session=ss, role='tool', source='t#4',
                  text=tool('Edit', '', 'ok').replace('{"command": ""}', '{"file_path": "/repo/app/util.py"}'))
    s.add_episode(bank='personal', scope=sc, session=ss, role='tool', source='t#5',
                  text=tool('Bash', 'cd /repo && python -m pytest -q tests', '3 passed'))
    s.con.execute('CREATE TABLE IF NOT EXISTS code_repos (root TEXT)')
    cm = CodeMap(s, 'personal')
    cm.git_root = lambda p: '/repo'
    cm.extract_touches()
    assert failsig.index_new(s.con) == 1
    return db, s


def test_signature_and_command_key():
    assert failsig.signature(TB) == ("KeyError: 'runs'", True)
    assert failsig.signature('E   AssertionError') == ('AssertionError', False)  # generic: same project only
    assert failsig.signature('Exit code 1') is None
    assert failsig.cmd_key('cd /a && PM1_DB=x .venv/bin/python -m pytest -q') == 'python -m pytest'
    assert failsig.cmd_key("python3 - <<'PY'\nprint(1)\nPY") == ''               # inline scripts never match


def test_past_fix_and_edit_warning(tmp_path):
    db, s = build(tmp_path)
    fx = failsig.past_fixes(s.con, 'Exit code 1\n' + TB, 'python -m pytest -q tests', session='claude:new',
                            scope='proj:/repo')
    assert len(fx) == 1 and fx[0]['edited'] == ['app/util.py'] and fx[0]['user_said'] is None
    assert failsig.past_fixes(s.con, TB, 'python -m pytest', session='claude:old') == []    # own session
    w = failsig.edit_warnings(s.con, '/repo/app/util.py', session='claude:new')
    assert len(w) == 1 and w[0]['edited_again_after'] and "KeyError" in w[0]['error']
    assert failsig.edit_warnings(s.con, '/repo/app/other.py', session='claude:new') == []


def test_hooks_push_once(tmp_path, monkeypatch):
    db, s = build(tmp_path)
    monkeypatch.setattr(cc_hook, 'DB', str(db))
    ev = {'tool_name': 'Bash', 'session_id': 'new', 'cwd': '/repo', 'error': 'Exit code 1\n' + TB,
          'tool_input': {'command': 'cd /repo && python -m pytest -q tests'}}
    out = cc_hook.post_tool_failure(ev)
    assert out['hookSpecificOutput']['hookEventName'] == 'PostToolUseFailure'
    assert 'app/util.py' in out['hookSpecificOutput']['additionalContext']
    assert cc_hook.post_tool_failure(ev) is None                                  # once per session
    pre = {'tool_name': 'Edit', 'session_id': 'new', 'tool_input': {'file_path': '/repo/app/util.py'}}
    out = cc_hook.pre_tool_use(pre)
    assert 'permissionDecision' not in out['hookSpecificOutput'] and 'KeyError' in out['hookSpecificOutput']['additionalContext']
    assert cc_hook.pre_tool_use(pre) is None


def test_environment_errors_not_edit_warnings(tmp_path):
    db, s = build(tmp_path)
    s.con.execute("UPDATE fail_sigs SET sig=\"ModuleNotFoundError: No module named 'x'\"")
    assert failsig.edit_warnings(s.con, '/repo/app/util.py', session='claude:new') == []
