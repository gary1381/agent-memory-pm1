"""search_sessions / open_record contracts from the 10-27 audit: long records page, a failed or slow query-embedding
service falls back to keywords, and vector candidates are checked against the database (reused span ids)."""
from __future__ import annotations

import time

import numpy as np

from pm1 import Recaller, Store
from pm1.sessions import SessionSearch


def add(s, text, scope='proj:/audit/alpha', seq=None, role='user'):
    return s.add_episode(bank='personal', scope=scope, session='codex:audit', seq=seq, role=role, text=text,
                         occurred_at='2026-10-01T00:00:00Z')


class WordDense:
    name = 'audit-word-dense'

    def __init__(self):
        self.calls = 0

    def embed(self, texts, kind):
        self.calls += 1
        return np.array([[1., 0.] if 'europe' in t.lower() else [0., 1.] for t in texts], dtype=np.float32)


class BrokenDense(WordDense):
    def embed(self, texts, kind):
        self.calls += 1
        raise ConnectionError('embedding service down')


class SlowDense(WordDense):
    def embed(self, texts, kind):
        self.calls += 1
        time.sleep(0.6)
        return super().embed(texts, kind)


def test_long_record_pages_to_the_end_and_find_locates_the_middle():
    s = Store()
    text = 'Introduction. ' * 300 + 'MIDDLE_NEEDLE_73921 was the cause.' + ' Ending.' * 300
    eid = add(s, text)['episode_id']
    ss = SessionSearch(s, Recaller(s))
    got, off, pages = '', 0, 0
    while off is not None:
        t = ss.open_record(eid, before=0, after=0, offset=off, max_chars=1000)['turns'][0]
        assert t['total_chars'] == len(text) and t['offset'] == off and t['partial']
        got += t['text']
        off, pages = t['next_offset'], pages + 1
    assert got == text and pages >= 7                       # the pages join up to the whole record, nothing skipped
    t = ss.open_record(eid, before=0, after=0, find='middle_needle_73921')['turns'][0]
    assert t['matches'][0]['offset'] == text.index('MIDDLE_NEEDLE') and 'MIDDLE_NEEDLE_73921' in t['text']
    short = next(t for t in ss.open_record(add(s, 'a short turn')['episode_id'])['turns'] if t['is_target'])
    assert short['text'] == 'a short turn' and short['next_offset'] is None and not short['partial']


def test_search_hint_points_at_the_page_of_the_shown_part(monkeypatch):
    monkeypatch.setenv('PM1_SESS_FULL', '1')
    s = Store()
    text = 'Opening remarks about layout. ' * 60 + 'The zebra queue limit is sixty per hour. ' + 'Tail notes. ' * 200
    eid = add(s, text)['episode_id']
    it = SessionSearch(s, Recaller(s), 'personal').search('zebra queue limit', k=1, budget_tokens=400,
                                                          budget_ms=None)['items'][0]
    off = int(it['text'].split(f'open_record({eid}, offset=')[1].split(')')[0])
    page = SessionSearch(s, Recaller(s)).open_record(eid, offset=off)['turns'][0]['text']
    assert 'zebra queue limit' in page


def test_embedding_outage_falls_back_to_keywords_and_says_so(monkeypatch):
    monkeypatch.setenv('PM1_SESS_FULL', '1')
    s = Store()
    eid = add(s, 'The zebra queue has a limit of sixty per hour.')['episode_id']
    dense = BrokenDense()
    ss = SessionSearch(s, Recaller(s, dense=dense))
    out = ss.search('zebra queue limit', budget_ms=None)
    assert [i['episode_id'] for i in out['items']] == [eid]
    assert out['route']['degraded'][0].startswith('dense: ConnectionError')
    n = dense.calls
    again = ss.search('zebra queue limit', budget_ms=None)            # within the pause: not asked again
    assert dense.calls == n and 'skipped' in again['route']['degraded'][0] and again['items']


def test_slow_embedding_does_not_hold_the_search(monkeypatch):
    monkeypatch.setenv('PM1_SESS_FULL', '1')
    monkeypatch.setenv('PM1_DENSE_WAIT_S', '0.1')
    s = Store()
    add(s, 'The zebra queue has a limit of sixty per hour.')
    out = SessionSearch(s, Recaller(s, dense=SlowDense())).search('zebra queue limit', budget_ms=None)
    assert out['items'] and out['route']['degraded'][0].startswith('dense: TimeoutError')
    assert out['latency_ms']['total'] < 500 and set(out['latency_ms']) >= {'lexical', 'dense', 'rerank', 'pack'}


def test_reused_span_id_does_not_leak_another_project(monkeypatch):
    monkeypatch.setenv('PM1_SESS_FULL', '1')
    s = Store()
    a, b = add(s, 'weather alpha'), add(s, 'European alpha office')
    dense = WordDense()
    for rec, text in ((a, 'weather alpha'), (b, 'European alpha office')):
        s.put_vector(rec['span_ids'][0], dense.name, dense.embed([text], 'passage')[0])
    ss = SessionSearch(s, Recaller(s, dense=dense))
    ss.search('Europe', project='alpha', budget_ms=None)              # builds the in-memory vector index
    s.forget_episode('personal', 'proj:/audit/alpha', b['episode_id'])
    c = add(s, 'pottery beta private', scope='proj:/audit/beta')    # SQLite hands it the deleted span's id
    assert c['span_ids'][0] == b['span_ids'][0]
    s.put_vector(c['span_ids'][0], dense.name, dense.embed(['pottery beta private'], 'passage')[0])
    out = ss.search('Europe', project='alpha', k=5, budget_ms=None)
    assert {i['project'] for i in out['items']} == {'/audit/alpha'}
    assert 'pottery beta private' not in [i['text'] for i in out['items']]


def test_call_log_waits_briefly_then_writes_in_the_background(tmp_path, monkeypatch):
    import sqlite3
    from pm1.mcp import Service
    monkeypatch.setenv('PM1_SESS_FULL', '1')
    db = str(tmp_path / 'm.sqlite')
    s = Store(db)
    add(s, 'zebra queue policy')
    ss = SessionSearch(s, Recaller(s))
    svc = Service(s, Recaller(s), bank='personal', default_scope='audit', sessions=ss)
    assert 'call_id' in svc.call('search_sessions', {'query': 'zebra queue', 'k': 1})
    writer = sqlite3.connect(db, isolation_level=None)
    writer.execute('BEGIN IMMEDIATE')                     # another process holds the write lock
    t = time.perf_counter()
    out = svc.call('search_sessions', {'query': 'zebra queue', 'k': 1})
    waited = time.perf_counter() - t
    assert out['items'] and 'call_id' not in out and waited < 1.0
    writer.execute('ROLLBACK')
    writer.close()
    for _ in range(50):                                   # the background writer lands it once the lock is free
        if s.con.execute('SELECT count(*) FROM mcp_calls').fetchone()[0] == 2:
            break
        time.sleep(0.1)
    assert s.con.execute('SELECT count(*) FROM mcp_calls').fetchone()[0] == 2


def test_document_path_filter_comes_before_ranking(tmp_path):
    from pm1.docs import DocIndex
    root = tmp_path / 'docs'
    (root / 'wanted').mkdir(parents=True)
    for i in range(60):
        (root / f'distractor{i:02}.txt').write_text('zebra queue policy\n')
    (root / 'wanted' / 'target.txt').write_text('zebra queue policy ' + 'additional description ' * 50)
    s = Store()
    docs = DocIndex(s)
    docs.add_root('personal', 'docs', str(root))
    docs.scan()
    out = docs.search(Recaller(s), 'personal', 'docs', 'zebra queue policy', path_prefix='wanted/', k=1)
    assert [i['path'].endswith('wanted/target.txt') for i in out['items']] == [True]
    absolute = docs.search(Recaller(s), 'personal', 'docs', 'zebra queue policy', path_prefix=str(root / 'wanted'), k=1)
    assert len(absolute['items']) == 1
    none = docs.search(Recaller(s), 'personal', 'docs', 'zebra queue policy', path_prefix='nowhere/', k=1)
    assert none['items'] == [] and 'no indexed file under nowhere/' in none['note']


def test_search_says_how_far_each_source_is_captured(monkeypatch):
    monkeypatch.setenv('PM1_SESS_FULL', '1')
    s = Store()
    add(s, 'zebra queue policy')
    s.add_episode(bank='personal', scope='proj:/audit/alpha', session='cursor:c1', seq=0, role='user',
                  text='an older cursor turn', occurred_at='2026-10-23T20:04:54Z')
    out = SessionSearch(s, Recaller(s)).search('zebra queue', budget_ms=None)
    assert out['searched']['sources_up_to'] == {'codex': '2026-10-01T00:00', 'cursor': '2026-10-23T20:04'}


def test_chinese_part_keeps_to_the_budget_and_to_the_matching_sentence(monkeypatch):
    monkeypatch.setenv('PM1_SESS_PACK', 'tokens')
    from pm1.recall import est_tokens
    monkeypatch.setenv('PM1_SESS_FULL', '1')
    s = Store()
    add(s, '开头说明一些背景。' * 60 + '斑马队列的上限是每小时六十个。' + '后面是别的讨论内容。' * 120)
    out = SessionSearch(s, Recaller(s)).search('斑马队列 上限', budget_tokens=300, k=1, budget_ms=None)
    it = out['items'][0]
    assert '斑马队列的上限是每小时六十个' in it['text']           # the evidence, not the opening lines
    assert out['tokens'] <= 300 and est_tokens(it['text']) + 50 <= 300   # audit 10-27: 760 for a 300 budget


def test_a_record_that_does_not_fit_is_named_not_stubbed(monkeypatch):
    monkeypatch.setenv('PM1_SESS_PACK', 'tokens')
    monkeypatch.setenv('PM1_SESS_FULL', '1')
    s = Store()
    first = add(s, 'zebra queue limit decided: sixty per hour. ' + 'details of the decision follow. ' * 25)
    second = add(s, 'zebra queue limit was discussed again later. ' + 'more notes from that day. ' * 60)
    out = SessionSearch(s, Recaller(s)).search('zebra queue limit', budget_tokens=330, k=5, budget_ms=None)
    shown = [i['episode_id'] for i in out['items']]
    assert first['episode_id'] in shown and out['tokens'] <= 330
    assert [x['episode_id'] for x in out.get('not_shown', [])] == [second['episode_id']]
    assert 'raise budget_tokens' in out['not_shown_note']
