"""search_sessions with PM1_SESS_FULL=1: e5 candidates from every project join the BM25 pool (same filters)."""
from __future__ import annotations

import os

import numpy as np

from pm1 import Recaller, Store
from pm1.sessions import SessionSearch
from pm1.worker import run_once

HOME = os.path.expanduser("~")


class WordDense:
    name = 'word-dense'
    WORDS = ('europe', 'england', 'turtle', 'weather', 'pottery')

    def embed(self, texts, kind):
        out = []
        for t in texts:
            t = t.lower()
            v = np.array([float(w in t) for w in self.WORDS] + [0.05], dtype=np.float32)
            out.append(v / np.linalg.norm(v))
        return np.vstack(out)


def _store():
    s = Store(index_models=('dense:word-dense',))
    ids = {}
    for p, scope in enumerate(('proj:/w/alpha', 'proj:cowork:beta')):
        for i in range(10):
            text = 'the European office plan moved to friday' if (p, i) == (1, 3) else \
                f'chatting about the weather again, day {i}'
            ids[(p, i)] = s.add_episode(bank='personal', scope=scope, session=f'codex:s{p}', seq=i, role='user',
                                        text=text, occurred_at=f'2026-0{5 + p}-01T10:{i:02d}:00Z')['episode_id']
    run_once(s, dense=WordDense())
    return s, ids


def test_dense_finds_what_bm25_cannot(monkeypatch):
    s, ids = _store()
    q = 'Europe trip countries'          # no word in common with the record (European != Europe)
    monkeypatch.setenv('PM1_SESS_FULL', '0')
    lex = SessionSearch(s, Recaller(s, dense=WordDense()), 'personal')
    try:
        got = [it['episode_id'] for it in lex.search(q, budget_ms=None)['items']]
    except ValueError:
        got = []
    assert ids[(1, 3)] not in got
    monkeypatch.setenv('PM1_SESS_FULL', '1')
    full = SessionSearch(s, Recaller(s, dense=WordDense()), 'personal')
    hits = [i for i in full.search(q, budget_ms=None)['items'] if 'neighbor_of' not in i]
    assert hits[0]['episode_id'] == ids[(1, 3)]


def test_filters_apply_to_dense_candidates(monkeypatch):
    s, ids = _store()
    monkeypatch.setenv('PM1_SESS_FULL', '1')
    ss = SessionSearch(s, Recaller(s, dense=WordDense()), 'personal')
    q = 'Europe office plan'
    assert ids[(1, 3)] not in [it['episode_id'] for it in ss.search(q, project='alpha', strict=True,
                                                                    budget_ms=None)['items']]
    assert ids[(1, 3)] not in [it['episode_id'] for it in ss.search(q, until='2026-05-30', budget_ms=None)['items']]
    assert ids[(1, 3)] in [it['episode_id'] for it in ss.search(q, project='beta', budget_ms=None)['items']]
    # a record added later is appended to the index without a reload
    new = s.add_episode(bank='personal', scope='proj:/w/alpha', session='codex:s0', seq=50, role='user',
                        text='pottery class moved to europe trip', occurred_at='2026-07-01T10:00:00Z')['episode_id']
    run_once(s, dense=WordDense())
    assert new in [it['episode_id'] for it in ss.search('pottery', budget_ms=None)['items']]


def _proj_store(tmp_path, monkeypatch, isolated=()):
    import json
    cfg = tmp_path / 'projects.json'
    cfg.write_text(json.dumps({'agent-lab': ['aglab'], 'isolated': list(isolated)}))
    monkeypatch.setenv('PM1_PROJECTS', str(cfg))
    monkeypatch.setenv('PM1_SESS_FULL', '1')
    s = Store(index_models=('dense:word-dense',))
    ids = {}
    for scope in (f'proj:{HOME}/1.work/agent_lab', 'proj:cowork:aglab', f'proj:{HOME}/1.work/orbit_gpu',
                  'proj:cowork:client-x'):
        for i in range(3):
            ids[(scope, i)] = s.add_episode(bank='personal', scope=scope, session=f'codex:{scope}', seq=i, role='user',
                                            text=f'the weather pottery plan number {i} at {scope[-9:]}',
                                            occurred_at=f'2026-06-0{i + 1}T10:00:00Z')['episode_id']
    run_once(s, dense=WordDense())
    return s, ids


def test_logical_project_filter_takes_every_scope(tmp_path, monkeypatch):
    s, ids = _proj_store(tmp_path, monkeypatch)
    ss = SessionSearch(s, Recaller(s, dense=WordDense()), 'personal')
    out = ss.search('pottery plan', project='aglab', strict=True, k=20, budget_ms=None)
    assert {i['project_name'] for i in out['items']} == {'agent-lab'}
    assert len(out['items']) == 6                                  # the folder and the Cowork copy
    assert out['searched']['projects'] == "only 'agent-lab'"


def test_current_project_moves_up_and_isolated_stays_hidden(tmp_path, monkeypatch):
    s, ids = _proj_store(tmp_path, monkeypatch, isolated=['client-x'])
    monkeypatch.setenv('PM1_PROJECT', f'{HOME}/1.work/orbit_gpu/orbit')
    monkeypatch.setenv('PM1_SESS_HOME_BOOST', '100')
    ss = SessionSearch(s, Recaller(s, dense=WordDense()), 'personal')
    out = ss.search('pottery plan', k=20, budget_ms=None)
    names = [i['project_name'] for i in out['items']]
    assert names[:3] == ['orbit'] * 3 and 'client-x' not in names
    assert out['searched']['current_project'] == 'orbit'
    assert any(i['project_name'] == 'client-x' for i in ss.search('pottery plan', project='client-x',
                                                                     budget_ms=None)['items'])


def test_coverage_points_to_other_projects(tmp_path, monkeypatch):
    s, ids = _proj_store(tmp_path, monkeypatch)
    ss = SessionSearch(s, Recaller(s, dense=WordDense()), 'personal')
    out = ss.search('pottery plan', k=2, budget_ms=None)
    shown = {i['project_name'] for i in out['items']}
    assert any("project='" in h for h in out['next']) and all(f"project='{p}'" not in h for p in shown for h in out['next'])
    assert out['next'][-1].startswith('open_record')


def test_tool_descriptions_follow_the_switch(monkeypatch):
    from pm1.mcp import Service
    s = Store()
    for full in ('0', '1'):
        monkeypatch.setenv('PM1_SESS_FULL', full)
        svc = Service(s, Recaller(s), bank='personal', default_scope='p', sessions=SessionSearch(s, Recaller(s)))
        t = {x['name']: x for x in svc.tools()}
        assert ('ranks the current project first' in t['search_sessions']['description']) == (full == '1')
        assert ('use search_sessions for those' in t['search_notes']['description']) == (full == '1')


def test_shared_index_file_plus_new_rows(tmp_path, monkeypatch):
    from pm1 import sessindex
    monkeypatch.setenv('PM1_SESS_FULL', '1')
    db = str(tmp_path / 'm.sqlite')
    s = Store(db, index_models=('dense:word-dense',))
    a = s.add_episode(bank='personal', scope='proj:/w/alpha', session='codex:s', seq=0, role='user',
                      text='the European office plan', occurred_at='2026-05-01T10:00:00Z')['episode_id']
    s.add_episode(bank='personal', scope='proj:/w/alpha', session='codex:s', seq=1, role='user',
                  text='chatting about the weather', occurred_at='2026-05-01T10:01:00Z')
    run_once(s, dense=WordDense())
    r = sessindex.build(db, 'word-dense')
    assert r['rows'] == 2 and sessindex.load(db, 'word-dense')['mat'].shape == (2, 6)
    b = s.add_episode(bank='personal', scope='proj:/w/beta', session='codex:t', seq=0, role='user',
                      text='pottery class plan', occurred_at='2026-05-02T10:00:00Z')['episode_id']
    run_once(s, dense=WordDense())
    ss = SessionSearch(s, Recaller(s, dense=WordDense()), 'personal')
    assert ss.search('Europe trip', budget_ms=None)['items'][0]['episode_id'] == a       # from the file
    assert ss.search('pottery trip', budget_ms=None)['items'][0]['episode_id'] == b      # added after the build
    g = ss._global_dense()
    assert g['base'] is not None and g['n_rows'] == 1


def test_common_words_leave_the_bm25_query(monkeypatch):
    monkeypatch.setenv('PM1_SESS_DF_MAX', '0.3')
    s = Store()
    for i in range(10):
        s.add_episode(bank='personal', scope='proj:/w/a', session='codex:s', seq=i, role='user',
                      text=f'model run output {i}' + (' zebra limiter' if i == 4 else ''),
                      occurred_at=f'2026-05-01T10:0{i}:00Z')
    ss = SessionSearch(s, Recaller(s), 'personal')
    toks = ss._lexical_tokens(['model', 'run', 'output', 'zebra', 'limiter', 'absent'])
    assert set(toks) == {'zebra', 'limiter', 'absent'}            # the three common words are left out
    assert len(ss._lexical_tokens(['model', 'run', 'output', 'zebra'])) == 3     # at least the rarest 3 stay
    assert ss.search('model run output zebra limiter', budget_ms=None)['items'][0]['text'].endswith('zebra limiter')


def test_v2_brings_the_turns_around_a_hit(monkeypatch):
    monkeypatch.setenv('PM1_SESS_FULL', '1')
    s = Store()
    ids = [s.add_episode(bank='personal', scope='proj:/w/a', session='codex:s', seq=i, role=('user', 'assistant')[i % 2],
                         text=t, occurred_at=f'2026-05-01T10:0{i}:00Z')['episode_id']
           for i, t in enumerate(['hello', 'what limit did we pick for the zebra queue?', 'sixty per hour', 'ok thanks'])]
    out = SessionSearch(s, Recaller(s), 'personal').search('zebra queue limit', k=1, budget_ms=None)
    got = [i['episode_id'] for i in out['items']]
    assert got == ids[0:3] and out['items'][2]['neighbor_of'] == ids[1]
    early = SessionSearch(s, Recaller(s), 'personal').search('zebra queue limit', k=1, until='2026-05-01T10:01:30Z',
                                                             budget_ms=None)
    assert ids[2] not in [i['episode_id'] for i in early['items']]         # as-of: no later turn


def test_own_folder_ranks_above_the_rest_of_the_project(tmp_path, monkeypatch):
    s, ids = _proj_store(tmp_path, monkeypatch)
    monkeypatch.setenv('PM1_PROJECT', 'proj:cowork:aglab')
    monkeypatch.setenv('PM1_SESS_FOLDER_BOOST', '100')
    monkeypatch.setenv('PM1_SESS_HOME_BOOST', '50')
    ss = SessionSearch(s, Recaller(s, dense=WordDense()), 'personal')
    assert ss.home == 'agent-lab' and ss.home_scope == 'proj:cowork:aglab'
    folders = [i['project'] for i in ss.search('pottery plan', k=20, budget_ms=None)['items']]   # hits + their turns
    assert folders[:3] == ['cowork:aglab'] * 3 and folders[3:6] == [f'{HOME}/1.work/agent_lab'] * 3
    assert ss.search('pottery plan', project='*', k=20, budget_ms=None)['searched']['projects'].startswith('all')


def test_a_named_project_is_a_filter(tmp_path, monkeypatch):
    # since 2026-10-27: a search aimed at one project must not bring in other folders (the 10-26 soft preference did)
    s, ids = _proj_store(tmp_path, monkeypatch)
    hit = s.add_episode(bank='personal', scope=f'proj:{HOME}/1.work/neuro_engine', session='codex:neuro', seq=0,
                        role='assistant', text='the turtle europe adapter result: 4B loads fine',
                        occurred_at='2026-06-09T10:00:00Z')['episode_id']
    run_once(s, dense=WordDense())
    ss = SessionSearch(s, Recaller(s, dense=WordDense()), 'personal')
    for kw in ({}, {'strict': False}, {'strict': True}):          # strict is ignored: old callers still work
        out = ss.search('turtle europe adapter', project='orbit', k=8, budget_ms=None, **kw)
        assert hit not in [i['episode_id'] for i in out['items']]
        assert {i['project_name'] for i in out['items']} <= {'orbit'}
        assert out['searched']['projects'] == "only 'orbit'"
        assert any("project='*'" in n for n in out['next'])
    assert hit in [i['episode_id'] for i in ss.search('turtle europe adapter', k=8, budget_ms=None)['items']]


def test_long_record_says_it_is_a_part_and_adds_its_next_part(monkeypatch):
    monkeypatch.setenv('PM1_SESS_FULL', '1')
    s = Store()
    long = ('Intro about the panel layout and colours. ' * 30 + 'Zebra queue limit is sixty per hour. ' +
            'Middle notes on unrelated build steps. ' * 30 + 'The zebra cause was the stopped translation service. ' +
            'Closing words about tests. ' * 30)
    e = s.add_episode(bank='personal', scope='proj:/w/a', session='codex:s', seq=0, role='assistant', text=long,
                      occurred_at='2026-05-01T10:00:00Z')['episode_id']
    it = SessionSearch(s, Recaller(s), 'personal').search('zebra queue limit', k=1, budget_ms=None)['items'][0]
    assert it['episode_id'] == e and it['partial']
    assert f'open_record({e}, offset=' in it['text'] and f'{len(long):,}-char record' in it['text']
    assert it['text'].count('\n…\n') == 1                               # the shown part plus one more part


def test_lexical_query_starts_from_the_full_text_index(monkeypatch):
    # 10-27: with strict (scope IN ...) the planner walked every span of the project and tested MATCH on each: 52 s
    s, _ = _store()                    # a project with a subfolder (two scopes) + since: the plan that flipped
    s.add_episode(bank='personal', scope='proj:/w/alpha/sub', session='codex:s9', seq=0, role='user',
                  text='weather plan for the sub folder', occurred_at='2026-05-02T10:00:00Z')
    monkeypatch.setenv('PM1_SESS_FULL', '1')
    ss = SessionSearch(s, Recaller(s, dense=WordDense()), 'personal')
    seen = []
    s.con.set_trace_callback(lambda q: seen.append(q) if 'bm25(span_fts)' in q else None)
    ss.search('weather plan', project='alpha', strict=True, since='2026-01-01', k=5, budget_ms=None)
    s.con.set_trace_callback(None)
    assert seen and any(q.count("'proj:/w/alpha") >= 2 for q in seen)
    for q in seen:
        plan = [r[3] for r in s.con.execute('EXPLAIN QUERY PLAN ' + q)]
        assert plan[0].startswith('SCAN span_fts'), plan


class OrderRouter:
    """Reranks by whether the text holds 'european', so a chosen record outranks everything."""
    def rerank(self, query, docs, query_lang, cand_langs, budget):
        return sorted(range(len(docs)), key=lambda i: 'european' not in docs[i].lower()), {'rerank_model': 'order'}


def _outside(monkeypatch, n):
    s, ids = _store()
    monkeypatch.setenv('PM1_SESS_FULL', '1')
    monkeypatch.setenv('PM1_SESS_OUTSIDE', str(n))
    return ids, SessionSearch(s, Recaller(s, dense=WordDense(), router=OrderRouter()), 'personal')


def test_outside_off_keeps_the_project_filter(monkeypatch):
    ids, ss = _outside(monkeypatch, 0)
    out = ss.search('European plan weather', project='alpha', budget_ms=None)
    assert ids[(1, 3)] not in [i['episode_id'] for i in out['items']]
    assert 'outside_note' not in out


def test_outside_record_is_marked_and_named(monkeypatch):
    ids, ss = _outside(monkeypatch, 2)
    out = ss.search('European plan weather', project='alpha', budget_ms=None)
    hit = [i for i in out['items'] if i['episode_id'] == ids[(1, 3)]]
    assert hit and hit[0]['outside_project'] is True
    assert sum(bool(i.get('outside_project')) for i in out['items']) <= 2
    assert "'alpha'" in out['outside_note']
    assert not any(i.get('neighbor_of') == ids[(1, 3)] for i in out['items'])


def test_outside_does_nothing_without_a_project(monkeypatch):
    ids, ss = _outside(monkeypatch, 2)
    out = ss.search('European plan weather', budget_ms=None)
    assert not any(i.get('outside_project') for i in out['items'])
    assert 'outside_note' not in out


def test_outside_needs_to_beat_the_project(monkeypatch):
    ids, ss = _outside(monkeypatch, 2)
    ss.recaller.router = type('R', (), {'rerank': lambda self, q, d, ql, cl, b: (
        sorted(range(len(d)), key=lambda i: 'european' in d[i].lower()), {})})()
    out = ss.search('European plan weather', project='alpha', budget_ms=None)
    assert not any(i.get('outside_project') for i in out['items'])


class GramDense:
    """distinct texts get distinct vectors (hashed letter trigrams), so no two scores tie"""
    name = 'word-dense'

    def embed(self, texts, kind):
        import zlib
        out = []
        for t in texts:
            v = np.zeros(64, dtype=np.float32)
            for i in range(len(t) - 2):
                v[zlib.crc32(t[i:i + 3].lower().encode()) % 64] += 1
            out.append(v / (np.linalg.norm(v) or 1))
        return np.vstack(out)


def test_lanes_share_one_scan(monkeypatch):
    s = Store(index_models=('dense:word-dense',))
    for p, scope in enumerate(('proj:/w/alpha', 'proj:/w/beta')):
        for i in range(20):
            s.add_episode(bank='personal', scope=scope, session=f'codex:s{p}', seq=i, role='user',
                          text=f'notes {i * 7 + p} on the {("weather", "pottery", "office plan")[i % 3]} day {i}',
                          occurred_at=f'2026-0{5 + p}-01T10:{i:02d}:00Z')
    run_once(s, dense=GramDense())
    monkeypatch.setenv('PM1_SESS_FULL', '1')
    monkeypatch.setenv('PM1_SESS_POOL', '3')          # a lane's top 9 of 40 rows: argpartition, not every row
    monkeypatch.setenv('PM1_SESS_HOME_POOL', '2')
    got = {}
    for share, topk in (('0', 'sort'), ('1', 'sort'), ('1', 'part')):
        monkeypatch.setenv('PM1_SESS_SCAN_SHARE', share)
        monkeypatch.setenv('PM1_SESS_TOPK', topk)
        ss = SessionSearch(s, Recaller(s, dense=GramDense()), 'personal')
        ss.home_scope = 'proj:/w/alpha'                # the current-folder lane runs as well
        scans, real = [], ss._sims
        ss._sims = lambda g, q: scans.append(1) or real(g, q)
        got[share, topk] = [[r['eid'] for r in ss._dense_cands(q, None, [], None, None, None, ['user'], 3)]
                            + [(i['episode_id'], i.get('neighbor_of')) for i in ss.search(q, budget_ms=None)['items']]
                            for q in ('office plan notes', 'weather day 4', 'pottery 15')]
        assert len(scans) == (3 if share == '1' else 9)   # per query: one scan, or one per lane
    assert got['0', 'sort'] == got['1', 'sort'] == got['1', 'part']


class TurtleDense(WordDense):
    """A second model that knows a turtle is a reptile: a record no other lane ranks."""
    name = 'turtle-dense'

    def embed(self, texts, kind):
        v = np.array([[float('turtle' in t.lower() or 'reptile' in t.lower()), 0.05] for t in texts], np.float32)
        return v / np.linalg.norm(v, axis=1, keepdims=True)


def test_second_dense_lane_off_by_default_and_fused_when_set(monkeypatch):
    s = Store(index_models=('dense:word-dense', 'dense:turtle-dense'))
    for i in range(10):
        s.add_episode(bank='personal', scope='proj:/w/alpha', session='codex:s0', seq=i, role='user',
                      text=f'chatting about the weather again, day {i}', occurred_at=f'2026-05-01T10:{i:02d}:00Z')
    eid = s.add_episode(bank='personal', scope='proj:/w/alpha', session='codex:s9', seq=0, role='user',
                        text='a small reptile with a shell lives at the pond', occurred_at='2026-07-01T10:00:00Z')['episode_id']
    run_once(s, dense=WordDense())
    for sid, text in s.con.execute('SELECT s.id, substr(e.text, s.start + 1, s.end - s.start) FROM spans s '
                                   'JOIN episodes e ON e.id = s.episode_id').fetchall():
        s.con.execute('INSERT OR REPLACE INTO vectors(span_id, model, vec) VALUES (?, ?, ?)',
                      (sid, 'turtle-dense', TurtleDense().embed([text], 'passage')[0].astype(np.float16).tobytes()))
    s.con.commit()
    monkeypatch.setenv('PM1_SESS_FULL', '1')
    ss = SessionSearch(s, Recaller(s, dense=WordDense()), 'personal')
    assert ss.dense2 is None
    q = 'turtle weather notes'
    ss.search(q, budget_ms=None)
    ss2 = SessionSearch(s, Recaller(s, dense=WordDense()), 'personal')
    ss2.dense2 = TurtleDense()
    ss2.search(q, budget_ms=None)
    roles = ['user', 'assistant', 'event', 'tool']
    assert ss2._dense_cands(q, None, [], None, None, None, roles, 10, True)[0]['eid'] == eid
    assert ss2.last_order != ss.last_order       # its ranking is one more RRF list in the fusion
