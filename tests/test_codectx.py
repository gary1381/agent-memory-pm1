"""codectx: code context and repo outline from a graphify graph.json plus the source tree (no database)."""
import io
import json

from pm1.codectx import Graph, serve

A = '''"""store module"""


class Store:
    """keeps rows"""

    def add(self, row: dict,
            *, check: bool = True) -> int:
        """add one row"""
        return helper(row)


def helper(row):
    return len(row)


def runner():
    s = Store()
    s.add({})
    return helper({})
'''
T = '''from pkg.a import Store


def test_add():
    assert Store().add({'x': 1}) == 1
'''


def node(i, label, f, line, **kw):
    return {'id': i, 'label': label, 'file_type': 'code', 'source_file': f, 'source_location': f'L{line}', **kw}


def edge(s, t, rel, f, line):
    return {'source': s, 'target': t, 'relation': rel, 'source_file': f, 'source_location': f'L{line}'}


def repo(tmp_path):
    (tmp_path / 'pkg').mkdir()
    (tmp_path / 'tests').mkdir()
    (tmp_path / 'pkg' / 'a.py').write_text(A)
    (tmp_path / 'tests' / 'test_a.py').write_text(T)
    nodes = [node('fa', 'a.py', 'pkg/a.py', 1), node('store', 'Store', 'pkg/a.py', 4, _callable_class=True),
             node('add', '.add()', 'pkg/a.py', 7), node('helper', 'helper()', 'pkg/a.py', 13),
             node('runner', 'runner()', 'pkg/a.py', 17), node('ft', 'test_a.py', 'tests/test_a.py', 1),
             node('tadd', 'test_add()', 'tests/test_a.py', 4)]
    links = [edge('fa', 'store', 'contains', 'pkg/a.py', 4), edge('store', 'add', 'method', 'pkg/a.py', 7),
             edge('fa', 'helper', 'contains', 'pkg/a.py', 13), edge('fa', 'runner', 'contains', 'pkg/a.py', 17),
             edge('add', 'helper', 'calls', 'pkg/a.py', 10), edge('runner', 'helper', 'calls', 'pkg/a.py', 20),
             edge('runner', 'store', 'calls', 'pkg/a.py', 18), edge('ft', 'store', 'imports', 'tests/test_a.py', 1)]
    (tmp_path / 'graphify-out').mkdir()
    (tmp_path / 'graphify-out' / 'graph.json').write_text(json.dumps({'nodes': nodes, 'links': links}))
    return Graph(str(tmp_path))


def test_function_context(tmp_path):
    out = repo(tmp_path).code_context('helper')
    assert out['defined_at'] == 'pkg/a.py:13' and out['signature'] == 'def helper(row):'
    assert [c.split(' (')[0] for c in out['called_by']] == ['.add()', 'runner()']
    assert 'pkg/a.py:20' in out['called_by'][1]


def test_method_context_has_signature_doc_and_name_matched_call_sites(tmp_path):
    out = repo(tmp_path).code_context('Store.add')
    assert out['kind'] == 'method' and out['name'] == 'Store.add'
    assert out['signature'] == 'def add(self, row: dict, *, check: bool = True) -> int:'
    assert out['doc'] == 'add one row'
    assert out['called_by'] == [] and out['call_sites_by_name'] == ['pkg/a.py:19: s.add({})']
    assert out['tests_mentioning_it'] == ['tests/test_a.py:5']
    assert 'helper() (pkg/a.py:13)' in out['calls']


def test_class_and_file_context(tmp_path):
    g = repo(tmp_path)
    c = g.code_context('Store')
    assert c['kind'] == 'class' and c['methods'][0].startswith('L7 def add(') and c['tests_mentioning_it'] == ['tests/test_a.py:1']
    f = g.code_context('pkg/a.py')
    assert f['kind'] == 'file' and f['outline'][0] == 'L4 class Store:' and f['outline'][1].startswith('    L7 def add(')
    assert f['tests'] == ['tests/test_a.py'] and g.code_context('a.py')['target'] == 'pkg/a.py'


def test_unknown_target_suggests_names(tmp_path):
    out = repo(tmp_path).code_context('help')
    assert out['found'] is False and out['similar_names'] == ['helper']


def test_repo_outline_lists_top_level_symbols(tmp_path):
    g = repo(tmp_path)
    assert g.repo_outline()['tree'][0] == 'pkg/a.py (20 lines): Store, helper, runner'
    assert g.repo_outline('tests')['tree'] == ['tests/test_a.py (5 lines): test_add']


def test_mcp_tools_are_read_only(tmp_path):
    g = repo(tmp_path)
    msgs = [{'jsonrpc': '2.0', 'id': 1, 'method': 'tools/list'},
            {'jsonrpc': '2.0', 'id': 2, 'method': 'tools/call', 'params': {'name': 'code_context',
                                                                          'arguments': {'target': 'runner'}}}]
    out = io.StringIO()
    serve(g, stdin=io.StringIO(''.join(json.dumps(m) + '\n' for m in msgs)), stdout=out)
    r = [json.loads(x) for x in out.getvalue().splitlines()]
    assert all(t['annotations']['readOnlyHint'] for t in r[0]['result']['tools'])
    assert json.loads(r[1]['result']['content'][0]['text'])['calls'][0].startswith('Store (pkg/a.py:4)')
