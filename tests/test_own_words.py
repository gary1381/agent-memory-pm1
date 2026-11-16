"""Pasted or quoted text in a user turn is not the user's words (pm1-rule proposals #25-#30, 2026-10-25: five of six
came from a skill body, <pasted_content> blocks, or the quoted half of "quoted::my reply")."""
from __future__ import annotations

import json

import pytest

from pm1 import Store, lesson_mine, rules
from pm1.capture import Capture
from pm1.userwords import is_user_words, own_words

SC = 'proj:/repo'


def test_own_words_drops_pasted_blocks_and_quoted_halves():
    t = ('Please read this:\n\n<pasted_content id="b694">\nIt fails because GGUF does not support multi-LoRA.\n'
         '</pasted_content id="b694">\nwhat do you think?')
    assert 'GGUF' not in own_words(t) and own_words(t).split() == ['Please', 'read', 'this:', 'what', 'do', 'you', 'think?']
    assert own_words('原因是 llama.cpp 不支持 SWA 缓存。：：思考模式可以用') == '思考模式可以用'
    assert own_words('the guard fails because Chinese has no word boundaries.:: Please provide more details') \
        == 'Please provide more details'
    assert own_words('而不是追 LongMemEval 总分::同意') == '同意'
    assert own_words('use std::vector here') == 'use std::vector here'        # code, not a quote separator


def test_a_skill_body_is_not_the_users_words():
    assert not is_user_words('# Schedule Cloud Agents\n\nYou are helping the user ' + 'x' * 4000)
    assert is_user_words('# a short heading the user typed')


def test_miner_ignores_pasted_and_quoted_text(tmp_path):
    s = Store(tmp_path / 'm.sqlite')
    rules.ensure(s.con)
    for i, text in enumerate([
            'read:\n<pasted_content id="1">\nPrompt caching fails because `llama-server` drops the SWA slots.\n'
            '</pasted_content id="1">',
            'Prompt caching fails because `llama-server` drops the SWA slots on reload.::好的，记下来']):
        s.add_episode(bank='personal', scope=SC, session='claude:a', role='user', source=f't#{i}',
                      occurred_at='2026-10-24', text=text)
    assert lesson_mine.candidates(s.con, '2026-10-01') == []
    s.add_episode(bank='personal', scope=SC, session='claude:a', role='user', source='t#9', occurred_at='2026-10-24',
                  text='ok:: It fails because `bge-small-en` is English only, so Chinese queries miss.')
    assert [c['sentence'][:9] for c in lesson_mine.candidates(s.con, '2026-10-01')] == ['It fails ']


def test_quote_from_pasted_or_quoted_text_is_refused(tmp_path):
    s = Store(tmp_path / 'm.sqlite')
    e = s.add_episode(bank='personal', scope=SC, session='claude:a', role='user', source='t#1',
                      text='the guard treats 张三 as inside 张三丰 because Chinese has no word boundaries.:: '
                           'Please provide more details')
    with pytest.raises(ValueError, match='pasted or quoted'):
        rules.verify_quote(s.con, e['episode_id'], 'Chinese has no word boundaries', 'lesson')
    e = s.add_episode(bank='personal', scope=SC, session='claude:a', role='user', source='t#2',
                      text='Please read this:\n<pasted_content id="2">\nalways run the tests twice\n'
                           '</pasted_content id="2">')
    with pytest.raises(ValueError, match='pasted or quoted'):
        rules.verify_quote(s.con, e['episode_id'], 'always run the tests twice')


def test_quoted_text_with_a_plain_approval_is_the_users(tmp_path):
    """A real rule came from 'quoted proposal::同意'."""
    s = Store(tmp_path / 'm.sqlite')
    e = s.add_episode(bank='personal', scope=SC, session='claude:a', role='user', source='t#1',
                      text='按同模型同预算比较，而不是追 LongMemEval 总分::同意')
    assert rules.verify_quote(s.con, e['episode_id'], '而不是追 LongMemEval 总分')['how'] == \
        'user approved the text they quoted'
    e = s.add_episode(bank='personal', scope=SC, session='claude:a', role='user', source='t#2',
                      text='按同模型同预算比较，而不是追 LongMemEval 总分::同意吗？')
    with pytest.raises(ValueError):
        rules.verify_quote(s.con, e['episode_id'], '而不是追 LongMemEval 总分')


def test_capture_skips_harness_expanded_user_text(tmp_path):
    p = tmp_path / 'sess.jsonl'
    base = {'sessionId': 'C1', 'cwd': '/proj', 'timestamp': '2026-10-21T12:37:53Z'}
    rows = [{**base, 'type': 'user', 'message': {'role': 'user', 'content': 'schedule a nightly run'}},
            {**base, 'type': 'user', 'isMeta': True, 'message': {'role': 'user', 'content': [
                {'type': 'text', 'text': '# Schedule Cloud Agents\n\nYou are helping the user schedule agents.'}]}}]
    p.write_text(''.join(json.dumps(r) + '\n' for r in rows))
    s = Store()
    Capture(s).ingest_file(str(p), 'claude')
    assert [r[0] for r in s.con.execute('SELECT text FROM episodes')] == ['schedule a nightly run']


def test_no_recipe_from_another_utilitys_error(tmp_path):
    """#27: `pytest ...; tail queue.log` failed on tail; the later PYTHONPATH=... pytest run is not its fix."""
    from pm1 import failsig
    repo = tmp_path / 'proj'
    (repo / '.git').mkdir(parents=True)
    s = Store(tmp_path / 'm.sqlite')
    rules.ensure(s.con)
    sc = f'proj:{repo}'
    for i, (cmd, out, bad) in enumerate([
            ('python -m pytest -q tests; tail -n 5 queue.log',
             "tail: cannot open 'queue.log' for reading: No such file or directory", True),
            ('PYTHONPATH=/tmp/deps python -m pytest -q tests', '5 passed', False)], 1):
        s.add_episode(bank='personal', scope=sc, session='codex:r', role='tool', source=f'r#{i}', seq=i,
                      occurred_at='2026-10-21T10:00:00', text=f"[Bash{' FAILED' if bad else ''}] "
                      + json.dumps({'command': cmd}) + '\n→ ' + out)
    failsig.index_new(s.con)
    assert lesson_mine.recipes(s.con, since='2026-10-01') == []
