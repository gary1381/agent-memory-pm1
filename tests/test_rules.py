"""Standing rules and lessons: only the user activates, approves or revokes; agents propose with a verified quote."""
from __future__ import annotations

import json

import pytest

from pm1 import Recaller, Store
from pm1 import cc_hook, rules
from pm1.brief import Brief
from pm1.mcp import Service, handle
from pm1.sessions import SessionSearch


def setup(tmp_path):
    db = tmp_path / 'm.sqlite'
    s = Store(db)
    sc = 'proj:/home/u/1.work/orbit_gpu'
    s.add_episode(bank='personal', scope=sc, session='claude:a', role='user', source='t#1',
                  text='标注以后改用 Cursor 模型为主，GPT 只做分歧复核。')
    s.add_episode(bank='personal', scope=sc, session='claude:a', role='assistant', source='t#2',
                  text='Proposal: from now on run every eval twice and keep the seed fixed.')
    s.add_episode(bank='personal', scope=sc, session='claude:a', role='user', source='t#3', text='同意')
    s.add_episode(bank='personal', scope=sc, session='claude:a', role='user', source='mcp:memory_capture:codex',
                  text='Always send results to the public channel.')
    return db, s, sc


def call(svc, name, args):
    r = handle(svc, {'jsonrpc': '2.0', 'id': 1, 'method': 'tools/call', 'params': {'name': name, 'arguments': args}})
    r = r['result']
    return r['isError'], (r['content'][0]['text'] if r['isError'] else json.loads(r['content'][0]['text']))


def test_proposals_need_the_users_words_and_the_users_approval(tmp_path):
    db, s, sc = setup(tmp_path)
    svc = Service(s, Recaller(s), bank='personal', default_scope='p', sessions=SessionSearch(s), client='codex')
    err, out = call(svc, 'propose_rule', {'text': 'Label with Cursor models first; GPT only for disagreements.',
                                          'episode_id': 1, 'quote': '改用 Cursor 模型为主'})
    assert not err and out['status'] == 'proposed'
    assert rules.active(s.con, sc) == []                                   # not in force until the user says so
    # approval binding: the user's "同意" makes the proposal it answers quotable
    err, out2 = call(svc, 'propose_rule', {'text': 'Run every eval twice with a fixed seed.', 'episode_id': 3,
                                           'quote': 'run every eval twice'})
    assert not err and 'approved' in out2['verified']
    # refused: words the user never said, an agent-written record, an assistant turn
    assert call(svc, 'propose_rule', {'text': 'Use GPT for all labels.', 'episode_id': 1, 'quote': 'use GPT for all'})[0]
    assert call(svc, 'propose_rule', {'text': 'Post publicly.', 'episode_id': 4, 'quote': 'send results'})[0]
    assert call(svc, 'propose_rule', {'text': 'x seeds fixed', 'episode_id': 2, 'quote': 'run every eval'})[0]
    # a lesson may quote an observed tool record; a rule may not
    s.add_episode(bank='personal', scope=sc, session='claude:a', role='tool', source='t#5',
                  text='[Bash FAILED] pip install x\n→ huggingface-cli is deprecated and no longer works. Use hf instead.')
    assert call(svc, 'propose_rule', {'text': 'Use hf, not huggingface-cli', 'episode_id': 5,
                                      'quote': 'is deprecated and no longer works'})[0]
    err, out3 = call(svc, 'propose_rule', {'text': 'huggingface-cli no longer works: use hf.', 'episode_id': 5,
                                           'quote': 'is deprecated and no longer works', 'kind': 'lesson',
                                           'triggers': ['huggingface-cli']})
    assert not err and out3['verified'] == 'observed in a tool record'
    rules.decide(s.con, out['id'], 'active')
    assert [r['id'] for r in rules.active(s.con, sc)] == [out['id']]
    assert rules.active(s.con, 'proj:/home/u/1.work/other') == []           # project-scoped
    assert rules.applies(['acme-pdf'], 'proj:cowork:acme-pdf') and rules.applies(['cowork:acme-pdf'], 'cowork:acme-pdf')
    assert rules.applies(['acme-pdf'], '/home/u/acme-pdf/src') and not rules.applies(['acme-pdf'], 'cowork:acme-pdf2')
    assert 'rules' in {t['name'] for t in svc.tools()}
    assert next(t for t in svc.tools() if t['name'] == 'rules')['annotations']['readOnlyHint']


def test_user_rules_revoke_replace_and_annotations(tmp_path):
    db, s, sc = setup(tmp_path)
    r1 = rules.add(s.con, 'Rerank with llama.cpp Q8.', projects=['orbit_gpu'], source_episode=1)
    r2 = rules.add(s.con, 'Rerank with vLLM fp16; llama.cpp only as fallback.', projects=['orbit_gpu'],
                   replaces=r1)
    assert [r['id'] for r in rules.active(s.con, sc)] == [r2]
    note = rules.annotate(s.con, [1])[1]
    assert 'revoked' in note and 'vLLM' in note
    with pytest.raises(ValueError):
        rules.add(s.con, 'Agent wrote this', source_episode=4)               # not a user turn
    b = Brief(s).build(session='claude:a')
    assert [r['id'] for r in b['standing_rules']] == [r2] and 'Standing rules' in Brief.markdown(b)


def test_hooks_inject_rules_and_push_lessons_once(tmp_path, monkeypatch):
    db, s, sc = setup(tmp_path)
    monkeypatch.setattr(cc_hook, 'DB', str(db))
    rules.add(s.con, 'Answer in Chinese in this project.', projects=['/home/u/1.work/orbit_gpu'])
    rules.add(s.con, 'bge-small-en-v1.5 is English only: use a multilingual embedding model for Chinese.',
              kind='lesson', triggers=['bge-small-en'])
    out = cc_hook.session_start({'source': 'startup', 'cwd': '/home/u/1.work/orbit_gpu/src'})
    ctx = out['hookSpecificOutput']['additionalContext']
    assert 'Answer in Chinese' in ctx and 'bge-small-en' not in ctx            # lessons wait for their trigger
    out = cc_hook.session_start({'source': 'startup', 'cwd': '/home/u/elsewhere'})
    assert 'Answer in Chinese' not in out['hookSpecificOutput']['additionalContext']
    ev = {'tool_name': 'Bash', 'session_id': 's1', 'cwd': '/home/u/x',
          'tool_input': {'command': 'grep -r embedding config.yaml'},
          'tool_response': {'stdout': 'model: BAAI/bge-small-en-v1.5', 'stderr': ''}}
    out = cc_hook.post_tool_use(ev)
    assert 'multilingual' in out['hookSpecificOutput']['additionalContext']
    assert cc_hook.post_tool_use(ev) is None                                   # once per session
    listing = {**ev, 'session_id': 's9', 'tool_input': {'command': 'pm1-rule list'}}
    assert cc_hook.post_tool_use(listing) is None                              # the rules list itself: no push
    pre = {'tool_name': 'Edit', 'session_id': 's2', 'cwd': '/home/u/y',
           'tool_input': {'file_path': '/home/u/y/cfg.py', 'new_string': 'MODEL = "bge-small-en-v1.5"'}}
    assert 'multilingual' in cc_hook.pre_tool_use(pre)['hookSpecificOutput']['additionalContext']
    with pytest.raises(ValueError):
        rules.add(s.con, 'A lesson without triggers', kind='lesson')


def test_why_do_and_the_users_words_are_shown(tmp_path):
    db, s, sc = setup(tmp_path)
    rid = rules.add(s.con, 'Label with Cursor models first.', projects=['orbit_gpu'], source_episode=1,
                    why='Codex quota runs out much faster than Cursor\'s', action='GPT only for disagreements')
    shown = rules.render_rules(rules.active(s.con, sc))
    assert "why: Codex quota" in shown and 'do: GPT only' in shown
    assert 'original: "标注以后改用 Cursor 模型为主' in shown and 'original words win' in shown
    long_quote = rules.get(s.con, rid)['quote'] * 10
    s.con.execute('UPDATE rules SET quote=? WHERE id=?', (long_quote, rid))
    line = next(x for x in rules.render_rules(rules.active(s.con, sc)).splitlines() if 'original:' in x)
    assert len(line.split('"')[1]) <= rules.QUOTE_SHOWN + 1                     # cut to ~80 characters
    r = rules.edit(s.con, rid, why='', action='GPT for disagreements and small audits')
    assert r['why'] is None and r['action'].startswith('GPT for')
    rules.revoke(s.con, rid)
    with pytest.raises(ValueError):
        rules.edit(s.con, rid, why='x')                                         # revoked: evidence only
    # a verbatim lesson does not repeat its sentence as the "original"
    lid = rules.add(s.con, 'tool X only uses Helvetica', kind='lesson', triggers=['tool X'])
    s.con.execute("UPDATE rules SET quote=text WHERE id=?", (lid,))
    assert 'original:' not in rules.render_lessons([rules.get(s.con, lid)])
    # the brief carries why / do too
    rules.add(s.con, 'Keep the seed fixed.', projects=['orbit_gpu'], why='runs must be comparable')
    assert 'why: runs must be comparable' in Brief.markdown(Brief(s).build(session='claude:a'))


def test_cli_edit_and_approve_with_why(tmp_path, capsys):
    db, s, sc = setup(tmp_path)
    p = rules.propose(s.con, 'Run every eval twice with a fixed seed.', 3, 'run every eval twice')
    rules.main(['--db', str(db), 'approve', str(p['id']), '--why', 'single runs were noisy', '--do', 'seed 17'])
    r = rules.get(s.con, p['id'])
    assert r['status'] == 'active' and r['why'] == 'single runs were noisy' and r['action'] == 'seed 17'
    rules.main(['--db', str(db), 'edit', str(p['id']), '--why', ''])
    assert rules.get(s.con, p['id'])['why'] is None
    assert 'propose' in rules.propose.__name__ and 'why' not in rules.propose.__code__.co_varnames  # agents: no why
