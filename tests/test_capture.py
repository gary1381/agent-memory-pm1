"""Passive capture: Codex rollout and Claude Code transcript lines -> episodes, usage ledger, tool I/O;
incremental by byte offset; partial last line waits; injected environment text is skipped."""
import json

from pm1 import Store
from pm1.capture import Capture


def w(p, rows, mode='w', partial=None):
    with open(p, mode) as f:
        for r in rows:
            f.write(json.dumps(r) + '\n')
        if partial:
            f.write(partial)


def test_codex_incremental(tmp_path):
    p = str(tmp_path / 'rollout-x.jsonl')
    ts = '2026-10-17T10:00:00Z'
    w(p, [{'type': 'session_meta', 'timestamp': ts, 'payload': {'id': 'S1', 'cwd': '/repo'}},
          {'type': 'turn_context', 'timestamp': ts, 'payload': {'cwd': '/repo', 'model': 'gpt-6-luna'}},
          {'type': 'response_item', 'timestamp': ts, 'payload': {'type': 'message', 'role': 'user', 'content': [
              {'type': 'input_text', 'text': '<environment_context>cwd=/repo</environment_context>'}]}},
          {'type': 'response_item', 'timestamp': ts, 'payload': {'type': 'message', 'role': 'user', 'content': [
              {'type': 'input_text', 'text': 'fix the flaky ANN routing test'}]}},
          {'type': 'response_item', 'timestamp': ts, 'payload': {'type': 'custom_tool_call', 'call_id': 'c1',
                                                                 'name': 'exec', 'input': 'pytest -x tests/test_ann.py'}}],
      partial='{"type": "response_item", "timest')
    s = Store()
    cap = Capture(s)
    cap.ingest_file(p, 'codex')
    assert s.stats('personal', 'proj:/repo')['episodes'] == 1          # env text skipped, call pending
    with open(p) as f:
        body = f.read()
    with open(p, 'w') as f:                                           # complete the partial line
        f.write(body[:body.rindex('\n') + 1])
    w(p, [{'type': 'response_item', 'timestamp': ts, 'payload': {'type': 'custom_tool_call_output', 'call_id': 'c1',
                                                                 'output': [{'type': 'input_text',
                                                                             'text': 'E AssertionError ' + 'x' * 9000}]}},
          {'type': 'token_usage_record', 'timestamp': ts, 'payload': {'usage': {
              'input_tokens': 1000, 'cached_input_tokens': 900, 'output_tokens': 50, 'reasoning_output_tokens': 10}}},
          {'type': 'compacted', 'timestamp': ts, 'payload': {'message': 'summary of work so far'}}], mode='a')
    cap.ingest_file(p, 'codex')
    rows = [dict(r) for r in s.con.execute('SELECT role, text, source FROM episodes ORDER BY id')]
    assert [r['role'] for r in rows] == ['user', 'tool', 'event']
    assert rows[1]['text'].startswith('[exec] pytest -x') and 'omitted' in rows[1]['text']
    assert rows[1]['source'].endswith('#6')
    u = dict(s.con.execute('SELECT * FROM usage').fetchone())
    assert (u['input'], u['cached'], u['model'], u['cwd']) == (1000, 900, 'gpt-6-luna', '/repo')
    assert s.con.execute('SELECT out_chars FROM tool_io').fetchone()[0] > 9000
    assert cap.ingest_file(p, 'codex') == 0                           # nothing new


def test_claude_transcript(tmp_path):
    p = str(tmp_path / 'sess.jsonl')
    base = {'sessionId': 'C1', 'cwd': '/proj', 'timestamp': '2026-10-17T11:00:00Z'}
    w(p, [{**base, 'type': 'user', 'message': {'role': 'user', 'content': 'why does the import fail?'}},
          {**base, 'type': 'assistant', 'message': {'model': 'claude-x', 'usage': {
              'input_tokens': 5, 'cache_read_input_tokens': 900, 'cache_creation_input_tokens': 100,
              'output_tokens': 40}, 'content': [{'type': 'text', 'text': 'Checking the module path.'},
                                                {'type': 'tool_use', 'id': 't1', 'name': 'Bash',
                                                 'input': {'command': 'python -c "import pm1"'}}]}},
          {**base, 'type': 'user', 'message': {'role': 'user', 'content': [
              {'type': 'tool_result', 'tool_use_id': 't1', 'is_error': True, 'content': 'ModuleNotFoundError'}]}}])
    s = Store()
    cap = Capture(s)
    cap.ingest_file(p, 'claude')
    rows = [dict(r) for r in s.con.execute('SELECT role, text, session FROM episodes ORDER BY id')]
    assert [r['role'] for r in rows] == ['user', 'assistant', 'tool']
    assert rows[2]['text'].startswith('[Bash FAILED]') and rows[0]['session'] == 'claude:C1'
    u = dict(s.con.execute('SELECT input, cached, cache_write FROM usage').fetchone())
    assert u == {'input': 1005, 'cached': 900, 'cache_write': 100}
    rep = cap.report(days=36500)
    assert rep['by_day'][0]['calls'] == 1 and rep['tool_output'][0]['name'] == 'Bash'


def test_cursor_composer(tmp_path, monkeypatch):
    import sqlite3
    from pathlib import Path
    monkeypatch.setenv('HOME', str(tmp_path))
    g = tmp_path / '.config/Cursor/User/globalStorage'
    g.mkdir(parents=True)
    ws = tmp_path / '.config/Cursor/User/workspaceStorage/W1'
    ws.mkdir(parents=True)
    (ws / 'workspace.json').write_text(json.dumps({'folder': 'file:///home/x/my%20repo'}))
    db = sqlite3.connect(g / 'state.vscdb')
    db.execute('CREATE TABLE cursorDiskKV (key TEXT PRIMARY KEY, value BLOB)')
    db.execute('CREATE TABLE composerHeaders (composerId TEXT PRIMARY KEY, workspaceId TEXT)')
    db.execute("INSERT INTO composerHeaders VALUES ('C1','W1')")
    heads = [{'bubbleId': b} for b in ('b1', 'b2', 'b3', 'b4')]
    comp = {'lastUpdatedAt': 1000, 'fullConversationHeadersOnly': heads, 'contextTokensUsed': 29466,
            'modelConfig': {'modelName': 'composer-2.5'}, 'generatingBubbleIds': ['b4']}
    rows = {'composerData:C1': comp,
            'bubbleId:C1:b1': {'type': 1, 'text': 'find CLAUDE.md', 'createdAt': '2026-10-17T19:41:30Z'},
            'bubbleId:C1:b2': {'type': 2, 'text': 'Searching.', 'createdAt': '2026-10-17T19:41:31Z'},
            'bubbleId:C1:b3': {'type': 2, 'createdAt': '2026-10-17T19:41:32Z', 'toolFormerData': {
                'name': 'glob_file_search', 'rawArgs': '{"globPattern":"**/CLAUDE.md"}', 'result': {'files': ['a']},
                'status': 'completed'}},
            'bubbleId:C1:b4': {'type': 2, 'text': 'half-writt', 'createdAt': '2026-10-17T19:41:33Z'}}
    db.executemany('INSERT INTO cursorDiskKV VALUES (?,?)', [(k, json.dumps(v)) for k, v in rows.items()])
    db.commit()
    s = Store()
    cap = Capture(s)
    cap.CURSOR_DB = g / 'state.vscdb'
    assert cap.ingest_cursor() == 3                         # b4 is still generating
    eps = [dict(r) for r in s.con.execute('SELECT role, scope, text FROM episodes ORDER BY id')]
    assert [e['role'] for e in eps] == ['user', 'assistant', 'tool']
    assert eps[0]['scope'] == 'proj:/home/x/my repo' and eps[2]['text'].startswith('[glob_file_search]')
    assert s.con.execute("SELECT input, model FROM usage WHERE host='cursor'").fetchone()[:] == (29466, 'composer-2.5')
    comp.update(lastUpdatedAt=2000, generatingBubbleIds=[])
    rows['bubbleId:C1:b4']['text'] = 'half-written answer, now complete'
    db.execute("UPDATE cursorDiskKV SET value=? WHERE key='composerData:C1'", (json.dumps(comp),))
    db.execute("UPDATE cursorDiskKV SET value=? WHERE key='bubbleId:C1:b4'", (json.dumps(rows['bubbleId:C1:b4']),))
    db.commit()
    assert cap.ingest_cursor() == 1
    assert s.con.execute("SELECT text FROM episodes ORDER BY id DESC").fetchone()[0].endswith('now complete')
    assert cap.ingest_cursor() == 0


def test_config_timeline(tmp_path, monkeypatch):
    monkeypatch.setenv('HOME', str(tmp_path))
    (tmp_path / '.cursor').mkdir()
    (tmp_path / '.cursor/mcp.json').write_text(json.dumps({'mcpServers': {'graphify': {'env': {'TOKEN': 'secret'}}}}))
    (tmp_path / '.claude/skills/graphify').mkdir(parents=True)
    cap = Capture(Store())
    assert sorted(cap.snapshot_config()) == ['claude.skills', 'cursor.mcp']
    assert cap.snapshot_config() == []                                 # unchanged -> no new row
    (tmp_path / '.cursor/mcp.json').write_text(json.dumps({'mcpServers': {'pm1': {}}}))
    assert cap.snapshot_config() == ['cursor.mcp']
    last = dict(cap.con.execute("SELECT added, removed, items FROM config_snapshots ORDER BY id DESC").fetchone())
    assert (json.loads(last['added']), json.loads(last['removed'])) == (['pm1'], ['graphify'])
    assert 'secret' not in ''.join(r[0] for r in cap.con.execute('SELECT items FROM config_snapshots'))


def test_cloud_conversation_import_idempotent():
    """A Cowork session's own export: stable keys, so re-importing a larger export only adds what is new;
    usage already imported by import-usage (same response) is not counted twice."""
    s = Store()
    cap = Capture(s)
    ts = '2026-10-17T23:00:00Z'
    cap._usage({'host': 'cowork', 'session': 'S9', 'cwd': 'cowork:kb', 'model': 'm'}, ts, (100, 90, 5, 7, 0),
               'cowork:S9#12')
    ev = [{'k': 'turn:2', 'ts': None, 'kind': 'turn', 'role': 'user', 'text': 'review the knowledge base project'},
          {'k': 'l:u1:0', 'ts': ts, 'kind': 'user', 'text': 'save my cowork conversations'},
          {'k': 't:c1', 'ts': ts, 'kind': 'tool', 'name': 'Bash', 'input': '{"command": "ls"}', 'output': 'a.md'},
          {'k': 'm:msg1', 'ts': ts, 'kind': 'usage', 'model': 'm', 'u': [100, 90, 5, 7]},
          {'k': 'c:x', 'ts': ts, 'kind': 'event', 'text': '[context compacted]'}]
    r = cap.import_conversation(ev, 'cowork', 'S9', 'kb')
    assert (r['episodes'], r['tools'], r['usage']) == (3, 1, 0)
    ev.append({'k': 'l:a1:0', 'ts': ts, 'kind': 'assistant', 'text': 'here is a skill that does it'})
    r = cap.import_conversation(ev, 'cowork', 'S9', 'kb')
    assert (r['episodes'], r['tools'], r['usage'], r['session_episodes']) == (1, 0, 0, 5)
    assert s.stats('personal', 'proj:cowork:kb')['episodes'] == 5


def test_cloud_dedupe_keeps_local_detail():
    s = Store()
    cap = Capture(s)
    ts = '2026-10-18T10:00:00Z'
    ev = [{'k': 'turn:5', 'ts': None, 'kind': 'turn', 'role': 'user', 'text': 'please save my cowork conversations'},
          {'k': 'turn:6', 'ts': None, 'kind': 'turn', 'role': 'assistant',
           'text': 'I will add an import command first.\n\nThen I write the skill and test it.\n[tools: Bash, Write]'},
          {'k': 'turn:2', 'ts': None, 'kind': 'turn', 'role': 'user', 'text': 'an older question nobody kept locally'},
          {'k': 'l:u1:0', 'ts': ts, 'kind': 'user', 'text': 'please save my cowork conversations'},
          {'k': 'l:a1:0', 'ts': ts, 'kind': 'assistant', 'text': 'I will add an import command first.'},
          {'k': 'l:a2:0', 'ts': ts, 'kind': 'assistant', 'text': 'Then I write the skill and test it.'},
          {'k': 'm:1', 'ts': ts, 'kind': 'usage', 'model': 'm', 'u': [10, 5, 0, 2]}]
    cap.import_conversation(ev, 'cowork', 'S1', 'kb')
    cap._usage({'host': 'cowork', 'session': 'S1', 'cwd': 'cowork:kb', 'model': 'm'}, ts, (10, 5, 0, 2, 0),
               'cowork:S1#99')                                   # same response under a line key
    r = cap.dedupe_cloud('cowork')
    assert (r['usage'], r['episodes_same_text'], r['turns_covered']) == (1, 1, 1)
    left = {x[0] for x in s.con.execute("SELECT source FROM episodes WHERE session='cowork:S1'")}
    assert left == {'cowork:S1#turn:2', 'cowork:S1#l:u1:0', 'cowork:S1#l:a1:0', 'cowork:S1#l:a2:0'}
    r = cap.import_conversation(ev, 'cowork', 'S1', 'kb')        # re-import does not bring dropped turns back
    assert r['episodes'] == 0 and r['session_episodes'] == 4


def test_own_outputs_not_recaptured():
    """pm1's (and precision-memory's) tool outputs are copies of stored records: counted, not stored again."""
    from pm1.capture import SELF_TOOL
    assert all(SELF_TOOL.search(n) for n in ('mcp__pm1__search_sessions', 'mcp-pm1-search_docs',
                                             'mcp__remote-devices__pm1__session_brief',
                                             'mcp__precision-memory__memory_recall'))
    assert not any(SELF_TOOL.search(n) for n in ('Bash', 'mcp__graphify__query_graph', 'mcp__pm1x__a'))
    s = Store()
    cap = Capture(s)
    ts = '2026-10-18T23:00:00Z'
    ev = [{'k': 't:1', 'ts': ts, 'kind': 'tool', 'name': 'mcp__pm1__search_sessions', 'input': '{"query": "x"}',
           'output': '{"items": [...]}'},
          {'k': 't:2', 'ts': ts, 'kind': 'tool', 'name': 'Bash', 'input': '{"command": "ls"}', 'output': 'a.md'}]
    cap.import_conversation(ev, 'cowork', 'S1', 'kb')
    assert s.stats('personal', 'proj:cowork:kb')['episodes'] == 1
    assert s.con.execute("SELECT count(*) FROM tool_io").fetchone()[0] == 2


def test_reread_after_midfile_stop_adds_nothing(tmp_path):
    """The byte offset is saved per file, so lines written before a crash are read again on restart."""
    p = str(tmp_path / 'rollout-y.jsonl')
    ts = '2026-10-17T10:00:00Z'
    w(p, [{'type': 'session_meta', 'timestamp': ts, 'payload': {'id': 'S2', 'cwd': '/repo'}}] +
         [{'type': 'response_item', 'timestamp': ts, 'payload': {'type': 'message', 'role': 'user', 'content': [
             {'type': 'input_text', 'text': f'question {i}'}]}} for i in range(3)])
    s = Store()
    cap = Capture(s)
    cap.ingest_file(p, 'codex')
    s.con.execute('DELETE FROM cap_files')                            # offset lost, as after a crash
    cap.ingest_file(p, 'codex')
    assert s.con.execute('SELECT COUNT(*) FROM episodes').fetchone()[0] == 3
