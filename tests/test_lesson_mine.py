"""Lesson miner: candidates from stated causes, ranked, rare triggers only, proposals never active."""
from __future__ import annotations

from pm1 import Store
from pm1 import lesson_mine, rules


class FakeScorer:
    def score(self, query, docs):
        return [0.9 if 'Helvetica' in d else 0.3 if 'llama.cpp' in d else 0.01 for d in docs]


def test_suggest_proposes_ranked_lessons_with_rare_triggers():
    s = Store()
    rules.ensure(s.con)
    rules.add(s.con, 'llama.cpp batch 500', kind='lesson', triggers=['HttpE5'])
    s.con.execute("UPDATE rules SET quote='returns 500, not 400, for a batch'")
    sc = 'proj:/home/u/pdf'
    s.add_episode(bank='personal', scope=sc, session='claude:a', role='user', source='t#1', occurred_at='2026-10-18',
                  text='Callouts and `draw-text` return a 502 error for 你好世界 because they only use Helvetica. Thanks!')
    s.add_episode(bank='personal', scope=sc, session='claude:a', role='user', source='t#2', occurred_at='2026-10-18',
                  text='[123:456:0927/180927.84:ERROR:gcm/registration.cc:291] response: DEPRECATED_ENDPOINT because x.y')
    s.add_episode(bank='personal', scope=sc, session='claude:b', role='tool', source='t#3', occurred_at='2026-10-18',
                  text='[Bash] git commit -m x\n→ [pm1 a12084d] embed: retry item by item (llama.cpp returns 500, '
                       'not 400, for a batch holding one over-long input)')
    for i in range(12):        # 'llama.cpp' is everywhere: not a usable trigger
        s.add_episode(bank='personal', scope=sc, session=f'claude:n{i}', role='tool', source=f'n#{i}',
                      occurred_at='2026-10-18', text='[Bash] ./llama.cpp/build/bin/llama-server -m x.gguf\n→ ok')
    out = lesson_mine.suggest(s.con, FakeScorer(), since='2026-10-01', df_since='2026-10-01', max_new=5)
    assert [o['episode_id'] for o in out][:1] == [1] and out[0]['triggers'] == ['draw-text']
    assert all('DEPRECATED_ENDPOINT' not in o['sentence'] for o in out)          # pasted log line dropped
    assert all(t != 'llama.cpp' for o in out for t in o['triggers'])
    assert all('a12084d' not in o['sentence'] for o in out)                     # already a lesson
    assert all(r['proposed_by'] != 'lesson-miner' for r in rules.active(s.con, None, ('lesson',)))  # proposals only
    assert {p['proposed_by'] for p in rules.pending(s.con)} == {'lesson-miner'}
    again = lesson_mine.suggest(s.con, FakeScorer(), since='2026-10-01', df_since='2026-10-01', max_new=5)
    assert again == []                                                           # no duplicates on the next run


def test_run_recipes_only_for_environment_fixes(tmp_path):
    import json
    from pm1 import failsig
    repo = tmp_path / 'proj'
    (repo / '.git').mkdir(parents=True)
    s = Store(tmp_path / 'm.sqlite')
    rules.ensure(s.con)
    sc = f'proj:{repo}'

    def tool(i, cmd, out, failed=False):
        s.add_episode(bank='personal', scope=sc, session='claude:r', role='tool', source=f'r#{i}', seq=i,
                      occurred_at='2026-10-18T10:00:00', text=f"[Bash{' FAILED' if failed else ''}] "
                      + json.dumps({'command': cmd}) + '\n→ ' + out)
    tool(1, 'python3 -m pytest -q tests', "Traceback (most recent call last):\nModuleNotFoundError: No module named 'proj'",
         failed=True)
    tool(2, f'cd {repo}/runtime && PYTHONPATH=src python3 -m pytest -q tests', '5 passed')
    tool(3, 'python3 scripts/run.py', "Traceback (most recent call last):\nModuleNotFoundError: No module named 'x'",
         failed=True)
    tool(4, 'python3 scripts/run.py --other-flag', 'ok')                         # no environment change
    failsig.index_new(s.con)
    found = lesson_mine.recipes(s.con, since='2026-10-01')
    assert [r['cmd'] for r in found] == ['python3 -m pytest'] and found[0]['project'] == str(repo)
    out = lesson_mine.propose_recipes(s.con, since='2026-10-01')
    assert out[0]['id'] and rules.pending(s.con)[0]['kind'] == 'recipe'
    rules.decide(s.con, out[0]['id'], 'active')
    shown = rules.render_rules(rules.active(s.con, str(repo / 'runtime')))
    assert 'known-good ways to run' in shown and 'PYTHONPATH=src' in shown
    assert lesson_mine.propose_recipes(s.con, since='2026-10-01') == []          # not proposed twice


def test_judged_lessons_need_both_judges():
    s = Store()
    rules.ensure(s.con)
    sc = 'proj:/home/u/pdf'
    for i, t in enumerate(['Callouts and `draw-text` return a 502 error for 你好世界 because they only use Helvetica.',
                           'The `fido2` wheel is missing because `requirements-runtime.lock` was never regenerated.']):
        s.add_episode(bank='personal', scope=sc, session='claude:j', role='user', source=f'j#{i}',
                      occurred_at='2026-10-18', text=t)

    def fake_judge(kind, items, q, labels):
        return {iid: ({'composer': 'lesson', 'grok47': 'lesson'} if 'Helvetica' in text
                      else {'composer': 'lesson', 'grok47': 'not_lesson'}) for iid, text in items}
    out = lesson_mine.suggest(s.con, None, since='2026-10-01', df_since='2026-10-01', judge_fn=fake_judge)
    assert [o['sentence'][:9] for o in out] == ['Callouts '] and out[0]['judged']
    assert s.con.execute("SELECT count(*) FROM judge_log WHERE outcome='dropped'").fetchone()[0] == 1
