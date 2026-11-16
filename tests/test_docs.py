"""Document layer: parse with locators, evidence IDs, versions (supersede / revert / delete), search and cite,
MCP tools. Lexical-only recall (no models needed)."""
from __future__ import annotations

import json
import os

import pytest

from pm1 import Recaller, Store
from pm1.docparse import merge_small, parse
from pm1.docs import DocIndex
from pm1.mcp import Service, handle


@pytest.fixture
def env(tmp_path):
    root = tmp_path / 'notes'
    (root / 'sub').mkdir(parents=True)
    (root / '.git').mkdir()
    (root / '.git' / 'x.md').write_text('# hidden\nshould not be indexed zanzibar\n')
    (root / 'plan.md').write_text('# Storage\n\nWe chose SQLite with WAL for the memory store.\n\n'
                                  '# Search\n\nRetrieval uses BM25 plus dense vectors fused with RRF.\n')
    (root / 'sub' / 'page.html').write_text('<html><body><script>var x=1</script><h1>Budget</h1>'
                                            '<p>The hosting budget is 40 dollars per month.</p></body></html>')
    (root / 'ignore.bin').write_bytes(b'\x00\x01')
    s = Store(str(tmp_path / 'm.sqlite'))
    d = DocIndex(s)
    d.add_root('b', 'docs', str(root))
    return s, d, Recaller(s, planning=False), root


def test_parse_locators_md():
    blocks = parse(b'# A\n\none\n\ntwo\n\n# B\n\nthree\n', '.md')
    assert {b['locator']['section'] for b in blocks} == {'A', 'B'}
    assert next(b for b in blocks if 'three' in b['body'])['locator']['line_start'] >= 7
    merged = merge_small(blocks)
    assert {b['locator'].get('section') for b in merged} == {'A', 'B'}
    assert all('line_start' in b['locator'] for b in merged)


def test_scan_search_cite(env):
    s, d, rec, root = env
    res = d.scan()
    assert res['new'] == 2 and res['errors'] == 0
    out = d.search(rec, 'b', 'docs', 'hosting budget per month', k=3)
    assert out['items'] and 'page.html' in out['items'][0]['path']
    assert out['items'][0]['evidence_id'].startswith('ev_')
    assert 'zanzibar' not in json.dumps(d.search(rec, 'b', 'docs', 'zanzibar'))   # dot-dirs skipped
    c = d.cite(out['items'][0]['evidence_id'])
    assert c['current'] and c['file_matches'] and '40 dollars' in c['text']
    assert c['locator']['section'] == 'Budget'
    assert d.scan()['unchanged'] == 2


def test_change_supersedes_and_revert(env):
    s, d, rec, root = env
    d.scan()
    old = d.search(rec, 'b', 'docs', 'SQLite WAL memory store')['items'][0]['evidence_id']
    p = root / 'plan.md'
    original = p.read_text()
    p.write_text(original.replace('SQLite with WAL', 'PostgreSQL with pgvector'))
    os.utime(p, (1e9, 1e9))
    assert d.scan()['changed'] == 1
    c = d.cite(old)
    assert not c['current'] and c['version_status'] == 'superseded' and 'superseded_by' in c
    assert c['file_matches'] is False
    hits = d.search(rec, 'b', 'docs', 'SQLite WAL memory store')['items']
    assert all(h['evidence_id'] != old for h in hits)
    assert any('PostgreSQL' in h['text'] for h in d.search(rec, 'b', 'docs', 'PostgreSQL pgvector')['items'])
    p.write_text(original)                                   # content comes back: same evidence id is current again
    os.utime(p, (2e9, 2e9))
    d.scan()
    assert d.cite(old)['current']
    assert d.search(rec, 'b', 'docs', 'SQLite WAL memory store')['items'][0]['evidence_id'] == old


def test_delete_and_disable(env):
    s, d, rec, root = env
    d.scan()
    ev = d.search(rec, 'b', 'docs', 'hosting budget')['items'][0]['evidence_id']
    (root / 'sub' / 'page.html').unlink()
    assert d.scan()['deleted'] == 1
    assert not d.search(rec, 'b', 'docs', 'hosting budget')['items']
    c = d.cite(ev)
    assert not c['current'] and c['file_matches'] is False and '40 dollars' in c['text']
    d.disable_root(d.roots()[0]['id'])
    assert s.stats('b', 'docs')['spans'] == 0


def test_unparsable_keeps_previous(env, tmp_path):
    s, d, rec, root = env
    (root / 'bad.pdf').write_bytes(b'not a pdf')
    res = d.scan()
    assert res['errors'] == 1
    assert d.status()['errors'] and 'bad.pdf' in d.status()['errors'][0]['path']


def test_office_formats(tmp_path):
    docx = pytest.importorskip('docx')
    openpyxl = pytest.importorskip('openpyxl')
    root = tmp_path / 'r'
    root.mkdir()
    doc = docx.Document()
    doc.add_heading('Contract', 1)
    doc.add_paragraph('The renewal date is March 3.')
    doc.save(root / 'c.docx')
    wb = openpyxl.Workbook()
    wb.active['A1'] = 'rent'
    wb.active['B1'] = 1250
    wb.save(root / 'b.xlsx')
    s = Store()
    d = DocIndex(s)
    d.add_root('b', 'docs', str(root))
    assert d.scan()['new'] == 2
    rec = Recaller(s, planning=False)
    hit = d.search(rec, 'b', 'docs', 'renewal date')['items'][0]
    assert hit['locator']['section'] == 'Contract'
    assert 'Sheet!' in d.search(rec, 'b', 'docs', 'rent')['items'][0]['text']


def test_mcp_doc_tools(env):
    s, d, rec, root = env
    d.scan()
    svc = Service(s, rec, bank='b', default_scope='p', docs=d)
    names = [t['name'] for t in handle(svc, {'jsonrpc': '2.0', 'id': 1, 'method': 'tools/list'})['result']['tools']]
    assert names[-3:] == ['search_docs', 'cite', 'docs_status'] and 'forget' not in ' '.join(names)

    def call(name, args):
        r = handle(svc, {'jsonrpc': '2.0', 'id': 2, 'method': 'tools/call',
                         'params': {'name': name, 'arguments': args}})['result']
        return r['isError'], (r['content'][0]['text'] if r['isError'] else json.loads(r['content'][0]['text']))
    err, out = call('search_docs', {'query': 'RRF fused retrieval'})
    assert not err and 'EXCERPTS' in out['notice'] and out['items'][0]['where']
    err, c = call('cite', {'evidence_id': out['items'][0]['evidence_id']})
    assert not err and c['current']
    assert call('cite', {'evidence_id': 'ev_nope'})[0]
    assert call('docs_status', {})[1]['files_active'] == 2
    # conversation memory and documents stay apart
    call('memory_capture', {'text': 'I prefer tea over coffee', 'scope': 'p'})
    assert not call('search_docs', {'query': 'tea coffee'})[1]['items']
