"""Code history: graphify graph import, file touches parsed from captured tool calls, history and session search."""
import json
import os
import subprocess

from pm1 import Store
from pm1.codemap import CodeMap
from pm1.sessions import SessionSearch


def _repo(tmp_path):
    root = tmp_path / 'proj'
    (root / 'src').mkdir(parents=True)
    subprocess.run(['git', 'init', '-q', str(root)], check=True)
    (root / 'src' / 'limiter.py').write_text('import os\n\n\nclass Limiter:\n    def acquire(self):\n        return 1\n')
    (root / 'graphify-out').mkdir()
    g = {'nodes': [
        {'id': 'limiter', 'label': 'Limiter', 'norm_label': 'limiter', 'file_type': 'code', '_origin': 'ast',
         'source_file': 'src/limiter.py', 'source_location': 'L4', '_callable_class': True},
        {'id': 'acquire', 'label': 'acquire()', 'norm_label': 'acquire()', 'file_type': 'code', '_origin': 'ast',
         'source_file': 'src/limiter.py', 'source_location': 'L5'},
        {'id': 'main', 'label': 'main()', 'norm_label': 'main()', 'file_type': 'code', '_origin': 'ast',
         'source_file': 'src/app.py', 'source_location': 'L1'},
        {'id': 'why', 'label': 'rationale', 'file_type': 'rationale', '_origin': 'llm', 'source_file': 'src/app.py'}],
        'links': [{'source': 'main', 'target': 'limiter', 'relation': 'calls'},
                  {'source': 'limiter', 'target': 'why', 'relation': 'rationale_for'}],
        'built_at_commit': 'abc123'}
    (root / 'graphify-out' / 'graph.json').write_text(json.dumps(g))
    return root


def test_code_history_end_to_end(tmp_path):
    root = _repo(tmp_path)
    s = Store()
    cm = CodeMap(s, home=str(tmp_path))
    r = cm.import_graphs([str(root / 'graphify-out' / 'graph.json')])
    assert (r['units'], r['edges']) == (3, 1)                     # rationale node and its edge left out
    assert cm.import_graphs([str(root / 'graphify-out' / 'graph.json')])['skipped'] == 1
    scope, sess, ts = f'proj:{root}', 'codex:S1', '2026-10-18T10:00:0{}Z'
    add = lambda role, text, i: s.add_episode(bank='personal', scope=scope, session=sess, role=role, text=text,  # noqa
                                              occurred_at=ts.format(i), source=f'f#{i}')
    add('user', 'make the limiter release slots on timeout', 0)
    add('tool', '[exec] text(await tools.exec_command({cmd:"sed -n 1,40p src/limiter.py"}))\n→ class Limiter', 1)
    add('tool', '[apply_patch] *** Begin Patch\n*** Update File: src/limiter.py\n@@\n-x\n+y\n*** End Patch\n→ ok', 2)
    add('tool', '[exec] text(await tools.exec_command({cmd:"cd ' + str(root) + ' && .venv/bin/python -m pytest -q"}))'
        '\n→ F\nE   AssertionError: slot leak\n1 failed, 3 passed', 3)
    add('tool', '[exec] text(await tools.exec_command({cmd:"cat src/limiter.py"}))\n→ E AssertionError in a docstring', 4)
    add('tool', '[exec] text(await tools.exec_command({cmd:"cd ' + str(root) + ' && git commit -am fix"}))\n→ '
        '[main 1a2b3c4] limiter: release slots on timeout\n 1 file changed', 5)
    t = cm.extract_touches()
    assert t['commits'] == 1 and t['touches'] >= 3
    assert cm.extract_touches()['episodes'] == 0                  # incremental
    h = cm.history('Limiter')
    assert h['target']['symbol'] == 'Limiter' and h['target']['on_disk'] == 'current'
    assert h['callers'] == ['main() (src/app.py:1, calls)']
    sess0 = h['sessions'][0]
    assert sess0['edits'] == 1 and sess0['request']['text'].startswith('make the limiter')
    assert len(sess0['failures_after']) == 1 and 'slot leak' in sess0['failures_after'][0]['error']
    assert sess0['commits'][0]['sha'] == '1a2b3c4'
    assert cm.history('src/limiter.py')['totals']['edits'] == 1
    assert cm.history(str(root / 'src' / 'limiter.py'))['target']['file'] == 'src/limiter.py'

    ss = SessionSearch(s)
    res = ss.search('release slots timeout')
    assert res['items'][0]['episode_id'] and res['items'][0]['host'] == 'codex'
    assert ss.search('release slots', host='cursor')['items'] == []
    ctx = ss.open_record(res['items'][0]['episode_id'], before=1, after=1)
    assert any(t['is_target'] for t in ctx['turns'])
    ss.log('test', 'search_sessions', {'query': 'x'}, res, 12.0)
    assert s.con.execute('SELECT count(*) FROM mcp_calls').fetchone()[0] == 1


def test_code_history_prefers_the_current_repo(tmp_path, monkeypatch):
    s = Store()
    cm = CodeMap(s, home=str(tmp_path))
    roots = []
    for name in ('a', 'b'):
        root = tmp_path / name
        (root / 'src').mkdir(parents=True)
        subprocess.run(['git', 'init', '-q', str(root)], check=True)
        (root / 'src' / 'db.py').write_text('x = 1\n')
        roots.append(root)
        for i in range(1 if name == 'a' else 3):           # b has more touches: it wins when nothing is preferred
            s.add_episode(bank='personal', scope=f'proj:{root}', session=f'codex:{name}{i}', role='tool',
                          text=f'[apply_patch] *** Begin Patch\n*** Update File: src/db.py\n@@\n-x\n+y\n*** End Patch\n→ ok',
                          occurred_at=f'2026-10-18T10:00:0{i}Z', source=f'{name}#{i}')
    cm.extract_touches()
    monkeypatch.setenv('PM1_REPO', str(tmp_path / 'elsewhere'))
    assert cm.history('db.py')['target']['repo'] == str(roots[1])
    monkeypatch.setenv('PM1_REPO', str(roots[0] / 'src'))
    h = cm.history('db.py')
    assert h['target']['repo'] == str(roots[0]) and h['searched'].startswith('the current repo')
    assert h['also_in_other_repos'][0] == f'{roots[1]}/src/db.py'
    assert cm.history('db.py', repo='/b')['target']['repo'] == str(roots[1])


def test_code_history_finds_edits_made_from_a_parent_folder(tmp_path):
    """Real misses (production log, 10-18..11-15): sessions started in ~/work, whose .git is an empty sandbox
    placeholder, stored repo=~/work, file=<repo>/...; .mjs files were skipped; Codex code-mode patches kept the
    escaped newline in the path. A repo that became a git repo after the edits must still find them."""
    work = tmp_path / 'work'
    (work / '.git').mkdir(parents=True)                         # placeholder: no HEAD
    (work / '.git' / 'config').write_text('')
    app = work / 'app'
    (app / 'lib').mkdir(parents=True)
    s = Store()
    cm = CodeMap(s, home=str(tmp_path))
    add = lambda text, i, scope=work: s.add_episode(bank='personal', scope=f'proj:{scope}', session='codex:P',  # noqa
                                                    role='tool', text=text, occurred_at=f'2026-11-11T10:00:0{i}Z',
                                                    source=f'p#{i}')
    add('[exec] text(await tools.apply_patch("*** Begin Patch\\n*** Add File: app/lib/providers.mjs\\n+export {}\\n'
        '*** End Patch"))\n→ ok', 0)
    add('[exec] text(await tools.exec_command({cmd:"cd app && sed -i s/a/b/ lib/agent.py"}))\n→ ok', 1)
    cm.extract_touches()
    assert s.con.execute("SELECT repo FROM code_touches WHERE file LIKE '%providers.mjs'").fetchone()[0] == str(work)
    h = cm.history('lib/providers.mjs', repo='app')
    assert h['totals']['edits'] == 1
    subprocess.run(['git', 'init', '-q', str(app)], check=True)  # app becomes a repo; old touches stay keyed to work
    cm._git_root.clear()
    add('[apply_patch] *** Begin Patch\n*** Update File: lib/agent.py\n@@\n-x\n+y\n*** End Patch\n→ ok', 2, scope=app)
    cm.extract_touches()
    h = cm.history('lib/agent.py', repo='app')
    assert h['target']['repo'] == str(app) and h['totals']['edits'] == 2
    assert cm.history(str(app / 'lib' / 'agent.py'))['totals']['edits'] == 2


def test_touches_are_reparsed_when_the_parser_changes(tmp_path):
    root = _repo(tmp_path)
    s = Store()
    cm = CodeMap(s, home=str(tmp_path))
    s.add_episode(bank='personal', scope=f'proj:{root}', session='codex:V', role='tool', occurred_at='2026-11-11T10:00:00Z',
                  text='[apply_patch] *** Begin Patch\n*** Update File: src/limiter.py\n@@\n-x\n+y\n*** End Patch\n→ ok',
                  source='v#0')
    assert cm.extract_touches()['touches'] == 1
    s.con.execute("UPDATE code_state SET value='1' WHERE key='touch_version'")
    assert cm.extract_touches()['touches'] == 1                 # old rows dropped, every tool call parsed again
    assert s.con.execute('SELECT count(*) FROM code_touches').fetchone()[0] == 1
