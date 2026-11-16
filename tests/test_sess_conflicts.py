"""search_sessions with PM1_SESS_CONFLICTS=1: the user's own never-statements closest to the query go first in
possible_conflicts, from the same budget; off by default."""
from __future__ import annotations

import numpy as np

from pm1 import Recaller, Store
from pm1.sessions import SessionSearch
from pm1.worker import run_once


class WordDense:
    name = 'word-dense'
    WORDS = ('hospital', 'interview', 'turtle', 'weather')

    def embed(self, texts, kind):
        out = []
        for t in texts:
            t = t.lower()
            v = np.array([float(w in t) for w in self.WORDS] + [0.05], dtype=np.float32)
            out.append(v / np.linalg.norm(v))
        return np.vstack(out)


def _store():
    s = Store(index_models=('dense:word-dense',))
    texts = ['I have never attended an interview at the hospital, honestly.',                      # 0 never, close
             'My hospital interview went well on March 22.',                                         # 1 the other side
             'I have never owned a turtle.',                                                         # 2 never, far
             'chatting about the weather again']
    for i, t in enumerate(texts):
        s.add_episode(bank='personal', scope='proj:/w/alpha', session='codex:s1', seq=i, role='user', text=t,
                      occurred_at=f'2026-05-01T10:{i:02d}:00Z')
    s.add_episode(bank='personal', scope='proj:/w/alpha', session='codex:s2', seq=0, role='user',
                  text='I have never had a hospital interview', occurred_at='2026-05-02T10:00:00Z',
                  source='mcp:memory_capture:codex')                                                # agent-written
    s.add_episode(bank='personal', scope='proj:/v/beta', session='codex:s3', seq=0, role='user',
                  text='I have never done a hospital interview in beta', occurred_at='2026-05-03T10:00:00Z')
    run_once(s, dense=WordDense())
    return s


def test_off_by_default(monkeypatch):
    monkeypatch.setenv('PM1_SESS_FULL', '1')
    s = _store()
    r = SessionSearch(s, Recaller(s, dense=WordDense()), 'personal').search('hospital interview', budget_ms=None)
    assert 'possible_conflicts' not in r and 'conflicts' not in r['route']


def test_never_statements_first_from_the_same_budget(monkeypatch):
    monkeypatch.setenv('PM1_SESS_FULL', '1')
    s = _store()
    off = SessionSearch(s, Recaller(s, dense=WordDense()), 'personal').search('hospital interview', budget_ms=None,
                                                                               budget_tokens=600)
    monkeypatch.setenv('PM1_SESS_CONFLICTS', '1')
    ss = SessionSearch(s, Recaller(s, dense=WordDense()), 'personal')
    r = ss.search('hospital interview', budget_ms=None, budget_tokens=600)
    keys = list(r)
    assert keys.index('possible_conflicts') < keys.index('sessions' if 'sessions' in r else 'items')
    st = r['possible_conflicts']['statements']
    texts = [x['text'] for x in st]
    assert 'never attended an interview at the hospital' in texts[0]
    assert not any('turtle' in t for t in texts)                    # below the similarity floor
    assert not any('mcp' in t or t == 'I have never had a hospital interview' for t in texts)   # agent notes
    assert len(st) <= 2 and r['route']['conflicts']['n'] == len(st)
    assert r['tokens'] <= 600 and r['tokens'] >= off['tokens'] - 1000
    # a project filter applies to them as to the records
    r2 = ss.search('hospital interview', project='/w/alpha', budget_ms=None)
    assert all(x['project'] != 'beta' and 'beta' not in x['text'] for x in r2['possible_conflicts']['statements'])
    # nothing close enough: no field at all
    assert 'possible_conflicts' not in ss.search('weather', budget_ms=None)


def test_new_records_are_picked_up(monkeypatch):
    monkeypatch.setenv('PM1_SESS_FULL', '1')
    monkeypatch.setenv('PM1_SESS_CONFLICTS', '1')
    s = _store()
    ss = SessionSearch(s, Recaller(s, dense=WordDense()), 'personal')
    assert 'possible_conflicts' not in ss.search('turtle weather', budget_ms=None)
    s.add_episode(bank='personal', scope='proj:/w/alpha', session='codex:s1', seq=9, role='user',
                  text='I have never discussed turtle weather', occurred_at='2026-05-04T10:00:00Z')
    run_once(s, dense=WordDense())
    r = ss.search('turtle weather', budget_ms=None)
    assert 'turtle weather' in r['possible_conflicts']['statements'][0]['text']


def test_only_never_sentences_count():
    N = SessionSearch.NEVER
    assert N.search('I have never attended it') and N.search('我从来没有用过 Cursor') and N.search('我从未去过')
    assert not N.search('我没有看到这个错误') and not N.search('我还没跑完') and not N.search('they never came')


def test_not_yet_forms_pass_the_prefilter(monkeypatch):
    monkeypatch.setenv('PM1_SESS_FULL', '1')
    monkeypatch.setenv('PM1_SESS_CONFLICTS', '1')
    s = Store(index_models=('dense:word-dense',))
    s.add_episode(bank='personal', scope='proj:/w/alpha', session='codex:s1', seq=0, role='user',
                  text="I haven't ever been to a hospital interview", occurred_at='2026-05-01T10:00:00Z')
    run_once(s, dense=WordDense())
    r = SessionSearch(s, Recaller(s, dense=WordDense()), 'personal').search('hospital interview', budget_ms=None)
    assert r['possible_conflicts']['statements']


def test_first_look_runs_in_the_background_on_a_file_store(tmp_path, monkeypatch):
    import time
    monkeypatch.setenv('PM1_SESS_FULL', '1')
    monkeypatch.setenv('PM1_SESS_CONFLICTS', '1')
    s = Store(tmp_path / 'm.sqlite', index_models=('dense:word-dense',))
    s.add_episode(bank='personal', scope='proj:/w/alpha', session='codex:s1', seq=0, role='user',
                  text='I have never attended a hospital interview', occurred_at='2026-05-01T10:00:00Z')
    run_once(s, dense=WordDense())
    ss = SessionSearch(s, Recaller(s, dense=WordDense()), 'personal')
    ss.search('hospital interview', budget_ms=None)               # starts the look; may or may not be done
    for _ in range(100):
        if ss._neg['ready'] and ss._neg['job'] is None:
            break
        time.sleep(0.05)
    assert ss.search('hospital interview', budget_ms=None)['possible_conflicts']['statements']
