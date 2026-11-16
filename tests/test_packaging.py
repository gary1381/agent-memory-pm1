"""Defaults an outside user gets with nothing configured (T026): one database, no GPU service needed, no cloud labeler
when the judges are off."""
from __future__ import annotations

import os

from pm1 import lesson_mine, meter, paths, rules


def test_one_default_database_for_every_entry_point(monkeypatch, tmp_path):
    monkeypatch.delenv('PM1_DB', raising=False)
    monkeypatch.setenv('XDG_STATE_HOME', str(tmp_path))
    assert paths.default_db() == str(tmp_path / 'pm1' / 'memory.sqlite')
    monkeypatch.setenv('PM1_DB', '~/x.sqlite')
    assert paths.default_db() == os.path.expanduser('~/x.sqlite')


def test_meter_audit_sends_nothing_when_judges_are_off(monkeypatch, tmp_path):
    import sqlite3
    monkeypatch.setenv('PM1_JUDGE', 'off')
    out = meter.audit(sqlite3.connect(tmp_path / 'm.sqlite'), '2026-10-19', 5, False, False)
    assert out['ran'] is False and 'skipped' in out


def test_suggest_without_reranker_or_judges_is_not_an_error(monkeypatch, tmp_path, capsys):
    monkeypatch.setenv('PM1_JUDGE', 'off')
    for k in ('PM1_RERANK_QWEN_VLLM', 'PM1_RERANK_QWEN', 'PM1_RERANK_GTE'):
        monkeypatch.delenv(k, raising=False)
    lesson_mine.main(['--db', str(tmp_path / 'm.sqlite')])
    assert 'no suggestions' in capsys.readouterr().out


def test_miner_ignores_context_summaries_and_agent_messages(tmp_path):
    from pm1 import Store
    s = Store(tmp_path / 'm.sqlite')
    rules.ensure(s.con)
    for src, text in (('a#1', 'This session is being continued. The fix failed because `bge-small-en` is English only.'),
                      ('a#2', '<task-notification>It failed because `bge-small-en` is English only.</task-notification>'),
                      ('a#3', 'Another Claude session sent a message: broken because `bge-small-en` is English only.'),
                      ('a#4', 'It failed because `bge-small-en` is English only and the vectors were useless.')):
        s.add_episode(bank='personal', scope='proj:/w/a', session='claude:x', role='user', source=src,
                      occurred_at='2026-10-19', text=text)
    out = lesson_mine.candidates(s.con, '2026-10-01')
    assert [c['sentence'][:9] for c in out] == ['It failed']


def test_config_file_sets_defaults_and_the_environment_wins(tmp_path, monkeypatch):
    cfg = tmp_path / 'c.toml'
    cfg.write_text('[pm1]\ndb = "/x/y.sqlite"\ndense = "none"\njudge = "off"\nthreads = 3\n')
    monkeypatch.setenv('PM1_CONFIG', str(cfg))
    monkeypatch.delenv('PM1_DB', raising=False)
    monkeypatch.delenv('PM1_DENSE', raising=False)
    monkeypatch.setenv('PM1_JUDGE', 'cursor')              # set in the environment: the file must not override it
    monkeypatch.delenv('PM1_THREADS', raising=False)
    done = paths.apply_config()
    assert sorted(done) == ['PM1_DB', 'PM1_DENSE', 'PM1_THREADS']
    assert os.environ['PM1_DB'] == '/x/y.sqlite' and os.environ['PM1_JUDGE'] == 'cursor' and os.environ['PM1_THREADS'] == '3'
    for k in done:
        monkeypatch.delenv(k, raising=False)


def test_missing_or_broken_config_is_ignored(tmp_path, monkeypatch):
    monkeypatch.setenv('PM1_CONFIG', str(tmp_path / 'none.toml'))
    assert paths.apply_config() == []
    (tmp_path / 'bad.toml').write_text('[pm1\nnope')
    monkeypatch.setenv('PM1_CONFIG', str(tmp_path / 'bad.toml'))
    assert paths.apply_config() == []


def _user_first_case(tmp_path, monkeypatch, env, query):
    from pm1 import Recaller, Store
    if env is not None:
        monkeypatch.setenv('PM1_USER_FIRST', env)
    s = Store(tmp_path / 'u.sqlite')
    # the assistant's turn mentions the topic more often, so plain ranking puts it first
    s.add_episode(bank='personal', scope='p', session='x', role='assistant', source='a#1',
                  text='The staging database password rotation policy: rotate the staging database password monthly, '
                       'document the staging database password owner, and review the staging database password vault.')
    s.add_episode(bank='personal', scope='p', session='x', role='user', source='a#2',
                  text='We keep the staging database password in the team vault.')
    out = Recaller(s, planning=False).recall('personal', 'p', query, max_items=1, mode='bm25')
    return [it['role'] for it in out.items]


def test_user_words_come_first_when_switched_on(tmp_path, monkeypatch):
    assert _user_first_case(tmp_path, monkeypatch, '1', 'staging database password') == ['user']


def test_user_first_is_off_by_default(tmp_path, monkeypatch):
    monkeypatch.delenv('PM1_USER_FIRST', raising=False)
    assert _user_first_case(tmp_path, monkeypatch, None, 'staging database password') == ['assistant']


def test_a_question_about_what_you_said_keeps_the_plain_order(tmp_path, monkeypatch):
    assert _user_first_case(tmp_path, monkeypatch, '1', 'what you recommended about the staging database password') == ['assistant']


def test_long_passages_are_windowed_not_refused():
    """A span over the 512-token context used to make the server refuse the whole batch (then one request per item and
    a head-only cut). Now it is split into windows that fit, the vectors averaged, and the batch goes through at once."""
    import numpy as np

    from pm1.embed import HttpE5

    calls = []

    class Fake(HttpE5):
        def _call(self, inputs):
            calls.append(list(inputs))
            assert all(HttpE5.est_tokens(x) <= 500 for x in inputs), 'a window is over the context'
            return [[float(len(x) % 7), 1.0, 0.0] for x in inputs]
    e = Fake(batch=8)
    short, long_en, long_zh = 'a short passage', 'word ' * 1200, '这是一个很长的中文段落。' * 300
    v = e.embed([short, long_en, long_zh], 'doc')
    assert v.shape == (3, 3) and np.allclose(np.linalg.norm(v, axis=1), 1.0)
    assert sum(len(c) for c in calls) > 3                       # the long ones became several windows
    assert HttpE5.windows(short) == [short]
    assert ''.join(HttpE5.windows(long_zh)) == long_zh and ''.join(HttpE5.windows(long_en)) == long_en   # nothing lost


def test_a_refused_batch_is_halved_and_the_long_input_split_by_the_real_count():
    import io
    import json as _json
    import urllib.error

    import numpy as np

    from pm1.embed import HttpE5

    calls = []

    class Fake(HttpE5):
        def _call(self, inputs):
            calls.append(len(inputs))
            tokens = [len(x) for x in inputs]                       # pretend: one token per character, context 100
            if max(tokens) > 100:
                code = 400 if len(inputs) == 1 else 500
                body = _json.dumps({'error': {'type': 'exceed_context_size_error', 'n_prompt_tokens': max(tokens)}}).encode()
                raise urllib.error.HTTPError('u', code, 'x', {}, io.BytesIO(body))
            return [[float(len(x)), 1.0] for x in inputs]
    f = Fake(batch=64)
    f.max_tokens = 100
    f.window_tokens = 10 ** 6                                       # no windows: the safety net alone has to cope
    texts = ['x' * 20] * 31 + ['y' * 260] + ['x' * 20] * 32
    v = f.embed(texts, 'doc')
    assert v.shape == (64, 2) and np.allclose(np.linalg.norm(v, axis=1), 1.0)
    assert len(calls) < 30                                           # not one request per item


def test_remind_me_questions_about_the_assistant_keep_the_plain_order(tmp_path, monkeypatch):
    """LongMemEval single-session-assistant questions are phrased 'our previous conversation ... can you remind me ...' without
    'you said': they must not be reordered (0.889 -> 0.578 when they were)."""
    for q in ("I'm going back to our previous conversation about staging. Can you remind me of the database password policy",
              'Could you suggest a good policy for the staging database password?'):
        assert _user_first_case(tmp_path, monkeypatch, '1', q) == ['assistant'], q


def test_search_sessions_says_when_a_topic_has_no_prior_work(tmp_path):
    from pm1 import Store
    from pm1.sessions import SessionSearch
    s = Store(tmp_path / 's.sqlite')
    s.add_episode(bank='personal', scope='proj:/w/tide', session='codex:a', role='user', source='a#1',
                  text='orbit safety test results for the planner are in the report')
    ss = SessionSearch(s, None)
    out = ss.search('google-ai-mode-scraper oxylabs safety test', k=3)
    assert out['not_found_terms'] == ['google-ai-mode-scraper', 'oxylabs'] and 'No record anywhere' in out['note']
    assert out['items']                                                   # the nearest records are still returned
    assert 'not_found_terms' not in ss.search('orbit safety test', k=3)
