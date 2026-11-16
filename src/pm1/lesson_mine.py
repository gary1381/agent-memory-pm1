"""Find candidate lessons automatically; the user decides (they are only ever PROPOSED, see pm1.rules).

Failures with a later fix are already pushed by the failure hook (pm1.failsig); what is left are explained causes
written down in words: "X returns 500, not 400, for ...", "... only uses Helvetica, so CJK text fails",
"Y is deprecated", "Z cannot reach ... because ...". Pipeline, no generative model:
  1. sentences from the user's turns (including reports the user relayed) and from commit lines in tool output
     that state a cause or a limitation (CAUSE) about something specific (TECH);
  2. drop pasted logs, near-duplicates and anything a rule already came from;
  3. rank with the local reranker against a description of "a reusable technical lesson" (2026-10-19, 156 unique
     candidates: top 5 all real lessons, top 12 ~9 real; unranked ~1 in 4); keep the best few above a floor;
  4. triggers = the sentence's technical tokens that are rare across recent sessions (a common one such as
     'llama.cpp' would push the lesson into unrelated sessions); a candidate without one is not proposed.
Proposals verify their quote like any other (pm1.rules.propose) and wait for `pm1-rule approve`.
"""
from __future__ import annotations

import re
import sqlite3

from . import rules
from .userwords import is_user_words, own_words
from .paths import default_db

CAUSE = re.compile(r'(只能用于|不能用于|不支持|已弃用|已废弃|会导致|导致了|原因是|是因为|根因|deprecated|no longer works|'
                   r'caused by|root cause|because\b|returns? \d{3},? not \d{3}|refuse[sd]?\b|silently|only listens|'
                   r'only works|only uses?|does not support|doesn\'t support|cannot (?:reach|read|be)|breaks when|'
                   r'fails when|times? out when|hardcodes?)', re.I)
TECH = re.compile(r'`[^`]{3,60}`|\b[\w.-]*[._/-][\w.-]*[A-Za-z][\w.-]*\b|\b[A-Z][a-z]+[A-Z]\w+\b')
PASTED_LOG = re.compile(r'^\[\d+:\d+:|^\s*\[\d+\]\s|^\s*(?:print|def|class|import|from|return)\b|^\s*\|.*\|\s*$|'
                        r'^\s*\[\[?\d+\]?\]\(|https?://\S+\)?\s*$')
COMMIT_LINE = re.compile(r'\[[\w./+-]+ [0-9a-f]{7,40}\] [^\n]{10,400}')
QUERY = ('A reusable technical lesson: a specific tool, library, model, service or config behaves in a surprising '
         'way (a limitation, a misleading error, a default that breaks things), with its cause, so another project '
         'can avoid it.')
GENERIC = {'e.g', 'i.e', 'etc', 'config', 'error', 'test', 'tests', 'docs', 'api', 'http', 'https', 'json', 'yaml'}


def _sentences(text: str):
    for s in re.split(r'(?<=[.!?。！？;；])\s+|\n+', text or ''):
        s = s.strip(' -*•\t')
        if 25 <= len(s) <= 400 and not PASTED_LOG.search(s):
            yield s


def candidates(con: sqlite3.Connection, since: str, bank: str = 'personal') -> list[dict]:
    rows = con.execute(
        "SELECT id, role, scope, session, text FROM episodes WHERE bank=? AND scope LIKE 'proj:%' AND occurred_at >= ? "
        "AND (role='user' OR (role='tool' AND text LIKE '%commit%')) AND COALESCE(source, '') NOT LIKE 'mcp:%' "
        "ORDER BY id", (bank, since)).fetchall()
    used = {r[0] for r in con.execute('SELECT source_episode FROM rules WHERE source_episode IS NOT NULL')}
    norm = lambda x: re.sub(r'\W+', '', (x or '').lower())    # noqa: E731
    quotes = [q for q in (norm(r[0]) for r in con.execute('SELECT quote FROM rules')) if len(q) >= 12]
    out, seen = [], set(q[:120] for q in quotes)
    for eid, role, scope, sess, text in rows:
        if eid in used:
            continue
        if role == 'user' and not is_user_words(text):
            continue                  # a context summary, task notification or another agent's message: not the user's words
        body = own_words(text) if role == 'user' else '\n'.join(m.group(0) for m in COMMIT_LINE.finditer(
            text.partition('\n→ ')[2]))
        for s in _sentences(body):
            if not (CAUSE.search(s) and TECH.search(s)):
                continue
            full = norm(s)
            key = full[:120]
            if key in seen or any(q in full for q in quotes):     # already a rule / lesson (or a proposal)
                continue
            seen.add(key)
            out.append({'episode_id': eid, 'role': role, 'scope': scope, 'session': sess, 'sentence': s})
    return out


def triggers_for(con: sqlite3.Connection, sentence: str, *, days_since: str, max_share: float = 0.1,
                 k: int = 3) -> list[str]:
    """Technical tokens of the sentence that appear in few recent sessions (rare enough not to be noise)."""
    n_sess = con.execute("SELECT count(DISTINCT session) FROM episodes WHERE role='tool' AND occurred_at >= ?",
                         (days_since,)).fetchone()[0] or 1
    toks = []
    for m in TECH.finditer(sentence):
        raw = m.group(0)
        t = raw.strip('`')
        t = re.split(r'[(\s"\'=]', t)[0] if raw.startswith('`') else t   # `f("x")` -> f: the name, not the call
        t = t.strip('.,;:()')
        if len(t) < 4 or t.lower() in GENERIC or not re.search(r'[A-Za-z]{3,}', t) or t in toks:
            continue
        if re.fullmatch(r'[a-z]+(?:-[a-z]+)+', t) and not raw.startswith('`'):
            continue                          # 'over-long', 'reranker-specific': ordinary words, not names
        toks.append(t)
    picked = []
    for t in sorted(toks, key=lambda x: (not sentence.count(f'`{x}`'), -len(x))):     # backticked first, longer first
        df = con.execute("SELECT count(DISTINCT session) FROM episodes WHERE role='tool' AND occurred_at >= ? "
                         'AND instr(lower(text), ?) > 0', (days_since, t.lower())).fetchone()[0]
        if df / n_sess <= max_share:
            picked.append(t)
        if len(picked) >= k:
            break
    return picked


def suggest(con: sqlite3.Connection, scorer, *, since: str, df_since: str, max_new: int = 5,
            min_score: float = 0.05, dry_run: bool = False, judge_fn=None, max_judged: int = 60) -> list[dict]:
    """judge_fn (pm1.judge.run): when it answers, a candidate is proposed only if every judge says 'lesson'
    (composer-2.5 + grok-4.7: P 0.92 R 0.80 vs reranker P 0.73 R 0.27 on the 2026-10-19 gold set); otherwise
    the reranker ranking decides, as before."""
    cands = candidates(con, since)
    if not cands:
        return []
    scores = []
    if scorer is not None:
        for i in range(0, len(cands), 32):
            scores += scorer.score(QUERY, [c['sentence'] for c in cands[i:i + 32]])
    else:
        scores = [0.0] * len(cands)
    ranked = sorted(zip(cands, scores), key=lambda t: -t[1])
    verdicts = None
    if judge_fn is not None:
        from . import judge
        pool = ranked[:max_judged]
        verdicts = judge_fn('lesson', [(f'c{i}', c['sentence']) for i, (c, _) in enumerate(pool)], judge.LESSON_Q,
                            judge.LESSON_LABELS)
        if verdicts is not None:
            kept = []
            for i, (c, sc) in enumerate(pool):
                v = verdicts.get(f'c{i}', {})
                if len(v) < len(judge.LABELERS):         # a judge missing for this one: today's reranker decides it
                    ok, outcome = sc >= min_score, 'fallback'
                    if not dry_run:
                        judge.log(con, 'lesson', str(c['episode_id']), c['sentence'], v, outcome)
                    if ok:
                        kept.append((c, sc))
                    continue
                ok = all(x == 'lesson' for x in v.values())
                if not dry_run:
                    judge.log(con, 'lesson', str(c['episode_id']), c['sentence'], v, 'kept' if ok else 'dropped')
                if ok:
                    kept.append((c, max(sc, 1.0)))       # judged: past the reranker floor
            ranked = kept
    out = []
    for c, sc in ranked:
        if sc < min_score or len(out) >= max_new:
            break
        trig = triggers_for(con, c['sentence'], days_since=df_since)
        if not trig:
            continue
        item = {**c, 'score': round(float(sc), 3), 'triggers': trig,
                'judged': verdicts is not None}
        if not dry_run:
            try:
                r = rules.propose(con, c['sentence'], c['episode_id'], c['sentence'], kind='lesson', projects=[],
                                  triggers=trig, client='lesson-miner')
            except ValueError as exc:            # quote check failed (should not happen: the quote is the sentence)
                item['refused'] = str(exc)
                out.append(item)
                continue
            if r.get('duplicate'):
                continue
            item['id'] = r['id']
        out.append(item)
    return out


ENV_SIG = re.compile(r"No module named|command not found|No such file or directory|cannot open shared object|"
                     r"Connection refused|not recognized|cannot find module", re.I)


CD_TO = re.compile(r'(?:^|&&|;)\s*cd\s+([^\s;&|]+)')
ENV_FIX = re.compile(r'(?:^|[\s;&(])(?:[A-Z_][A-Z0-9_]*=\S|source\s|\.\s+\S+/activate|conda activate|'
                     r'\S*\.venv/bin/|\S*/envs/[\w.-]+/bin/|cd\s)')
SANDBOX = re.compile(r'/root/\.claude|-home-claude|/sessions/')
SHELL_ONLY = {'export', 'cd', 'echo', 'source', 'set', 'true', 'test', '[', 'env'}
# "tail: cannot open 'queue.log'": an error from another part of a compound command, not from the command itself
UTIL_ERR = re.compile(r'^\s*(tail|head|cat|ls|grep|rg|sed|awk|find|cp|mv|rm|mkdir|wc|sort|stat|du|xargs): ')


def _umbrella(d: str) -> bool:
    """A repo that holds several other repos (~/1.work): a recipe scoped to it would reach all of them."""
    import os

    try:
        subs = [x for x in os.listdir(d) if os.path.isdir(os.path.join(d, x, '.git'))]
    except OSError:
        return False
    return len(subs) >= 3


def _project_of(scope: str, command: str = '') -> str | None:
    """A recipe's project: the git repo the command ran in (its own `cd` first, else the session folder), or the
    Cowork project name. None for folders that hold many projects: a recipe there would leak into all of them."""
    import os
    w = scope[5:] if scope.startswith('proj:') else scope
    m = CD_TO.search(command or '')
    if m and m.group(1).startswith(('/', '~')):
        w = os.path.expanduser(m.group(1))
    elif m and not w.startswith('cowork:'):
        w = os.path.normpath(os.path.join(w, m.group(1)))
    if w.startswith('cowork:'):
        return w[7:] or None
    d = w
    while d and d != '/':
        if os.path.exists(os.path.join(d, '.git')):
            return None if _umbrella(d) else d
        d = os.path.dirname(d)
    return None


def _env_difference(failed: str, passed: str) -> bool:
    """The passing command changes the environment the failing one lacked (PYTHONPATH=, PATH=, cd, a venv
    interpreter, source / conda activate): that is a recipe. Other differences are just different commands."""
    got = {m.group(0).strip(' ;&(') for m in ENV_FIX.finditer(passed)}
    had = {m.group(0).strip(' ;&(') for m in ENV_FIX.finditer(failed)}
    return bool(got - had)


def recipes(con: sqlite3.Connection, *, since: str, window: int = 40) -> list[dict]:
    """Environment failures in a session followed by the same command, written differently, passing."""
    from .failsig import cmd_key, parse_record
    try:
        rows = con.execute('SELECT episode_id, scope, session, seq, ts, cmd, sig FROM fail_sigs WHERE ts >= ? '
                           "AND cmd != '' ORDER BY ts", (since,)).fetchall()
    except sqlite3.OperationalError:
        return []
    have = [(r[0], r[1]) for r in con.execute("SELECT projects, text FROM rules WHERE kind='recipe' "
                                              "AND status IN ('proposed', 'active', 'rejected')")]
    out, seen = [], set()
    for eid, scope, sess, seq, ts, ck, sig in rows:
        if not ENV_SIG.search(sig) or ck.split()[0] in SHELL_ONLY:
            continue
        u = UTIL_ERR.match(sig)
        if u and u.group(1) != ck.split()[0]:
            continue
        first = parse_record(con.execute('SELECT text FROM episodes WHERE id=?', (eid,)).fetchone()[0])
        if first is None or SANDBOX.search(first[1]):
            continue
        for lid, text in con.execute("SELECT id, text FROM episodes WHERE scope=? AND session=? AND role='tool' "
                                     'AND seq > ? AND seq <= ? ORDER BY seq', (scope, sess, seq, seq + window)):
            p = parse_record(text)
            if not p or cmd_key(p[1]) != ck:
                continue
            if p[3]:
                continue                       # failed again: keep looking
            if not _env_difference(first[1], p[1]) or SANDBOX.search(p[1]):
                break                          # passed without an environment change: not a recipe
            proj = _project_of(scope, p[1])
            if proj is None or (proj, ck) in seen:
                break
            if any(proj in __import__('json').loads(pj) and f'`{ck}`' in t for pj, t in have):
                break
            seen.add((proj, ck))
            out.append({'project': proj, 'cmd': ck, 'error': sig, 'failed': first[1][:200],
                        'passed': p[1][:300], 'episode_id': lid, 'record': text, 'when': ts})
            break
    return out


def propose_recipes(con: sqlite3.Connection, *, since: str, max_new: int = 3, dry_run: bool = False) -> list[dict]:
    import json as _json
    out = []
    for r in recipes(con, since=since)[-max_new:]:            # the most recent ones
        name = r['project'].rstrip('/').rsplit('/', 1)[-1]
        text = (f"{name}: run `{r['cmd']}` as `{r['passed'][:200]}` -- written the plain way it failed with "
                f"\"{r['error'][:120]}\"")
        raw = r['record']                                      # quote = the passing command as stored
        quote = next((q for q in (r['passed'][:60], _json.dumps(r['passed'])[1:61]) if q.lower() in raw.lower()), None)
        item = {**r, 'text': text}
        item.pop('record')
        if quote is None:
            item['refused'] = 'passing command not found verbatim in its record'
        elif not dry_run:
            try:
                res = rules.propose(con, text, r['episode_id'], quote, kind='recipe', projects=[r['project']],
                                    client='lesson-miner')
                item['id'] = res['id']
            except ValueError as exc:
                item['refused'] = str(exc)
        out.append(item)
    return out


RECIPE_TEXT = re.compile(r"run `(?P<cmd>[^`]+)` as `(?P<passed>[^`]+)`")


def freshness(con: sqlite3.Connection) -> dict:
    """Evidence, not time: update last_seen of lessons (a trigger appeared again) and recipes (the recipe's way of
    running passed again); a recipe that failed again on the environment despite its fix is flagged for the user
    (rule_conflicts, hint 'observed failing again'). Incremental over records captured since the last run."""
    from . import conflicts
    from .failsig import ENV_ERROR, cmd_key, parse_record
    conflicts.ensure(con)
    con.execute('CREATE TABLE IF NOT EXISTS code_state (key TEXT PRIMARY KEY, value TEXT)')
    row = con.execute("SELECT value FROM code_state WHERE key='freshness_last_episode'").fetchone()
    last = int(row[0]) if row else 0
    top = con.execute('SELECT COALESCE(max(id), 0) FROM episodes').fetchone()[0]
    out = {'lessons_seen': 0, 'recipes_confirmed': 0, 'recipes_flagged': 0}
    entries = rules._rows(con.execute("SELECT * FROM rules WHERE status='active' AND kind IN ('lesson', 'recipe')"))
    for r in entries:
        lo = max(last, r.get('source_episode') or 0)
        if r['kind'] == 'lesson':
            seen = None
            for t in r['triggers']:
                x = con.execute("SELECT max(occurred_at) FROM episodes WHERE id > ? AND role='tool' "
                                "AND text NOT LIKE '%pm1-rule%' AND instr(lower(text), ?) > 0", (lo, t.lower())).fetchone()[0]
                seen = max(seen or '', x or '') or None
            if seen and seen > (r.get('last_seen') or ''):
                con.execute('UPDATE rules SET last_seen=? WHERE id=?', (seen, r['id']))
                con.commit()
                out['lessons_seen'] += 1
            continue
        m = RECIPE_TEXT.search(r['text'])
        if not m:
            continue
        markers = {x.group(0).strip(' ;&(') for x in ENV_FIX.finditer(m.group('passed'))}
        first = m.group('cmd').split()[0]
        best, best_id, bad = None, 0, []
        for eid, ts, text in con.execute("SELECT id, occurred_at, text FROM episodes WHERE id > ? AND role='tool' "
                                         'AND instr(text, ?) > 0 ORDER BY id', (lo, first)):
            p = parse_record(text)
            if not p or cmd_key(p[1]) != m.group('cmd'):
                continue
            used = {x.group(0).strip(' ;&(') for x in ENV_FIX.finditer(p[1])}
            if not markers <= used:
                continue                       # not run the recipe's way
            if not p[3]:
                best, best_id = max(best or '', ts), eid
            elif ENV_SIG.search(p[2][-4000:]) or ENV_ERROR.search(p[2][-4000:]):
                bad.append({'episode_id': eid, 'score': 1.0, 'hint': 'observed failing again the recipe\'s way',
                            'excerpt': (p[1][:200] + ' → ' + p[2][-300:]).replace('\n', ' ')})
        if best and best > (r.get('last_seen') or ''):
            con.execute('UPDATE rules SET last_seen=? WHERE id=?', (best, r['id']))
            con.commit()
            out['recipes_confirmed'] += 1
        if bad and bad[-1]['episode_id'] > best_id:    # the latest run the recipe's way failed: may be stale
            out['recipes_flagged'] += conflicts.record(con, r['id'], bad[-1:])
    con.execute("INSERT OR REPLACE INTO code_state VALUES ('freshness_last_episode', ?)", (str(top),))
    con.commit()
    return out


def main(argv=None) -> None:
    import argparse
    import datetime as dt
    import os
    from .rerank import router_from_env
    ap = argparse.ArgumentParser(prog='pm1-rule suggest', description='Propose lessons found in recent records '
                                 '(they stay inactive until you approve them).')
    ap.add_argument('--db', default=default_db())
    ap.add_argument('--days', type=int, default=2, help='look at records from the last N days (default 2)')
    ap.add_argument('--max', type=int, default=5, help='at most this many new proposals (default 5)')
    ap.add_argument('--min-score', type=float, default=0.05)
    ap.add_argument('--max-recipes', type=int, default=3, help='at most this many new run recipes (default 3)')
    ap.add_argument('--dry-run', action='store_true', help='show what would be proposed, change nothing')
    ap.add_argument('--no-judge', action='store_true', help='rank with the reranker only (no Cursor judges)')
    a = ap.parse_args(argv)
    from . import judge
    router = router_from_env()
    scorer = max((r for r in router.rerankers if r.healthy()), key=lambda r: r.quality) if router else None
    judge_fn = judge.run if (judge.enabled() and not a.no_judge) else None
    if scorer is None and judge_fn is None:
        print('pm1-rule suggest: no reranker (PM1_RERANK_QWEN_VLLM / PM1_RERANK_QWEN) and no judges (PM1_JUDGE): nothing to rank '
              'with, no suggestions today')
        return
    con = sqlite3.connect(a.db, timeout=30)
    rules.ensure(con)
    today = dt.datetime.now(dt.timezone.utc)
    since = (today - dt.timedelta(days=a.days)).date().isoformat()
    df_since = (today - dt.timedelta(days=30)).date().isoformat()
    res = suggest(con, scorer, since=since, df_since=df_since, max_new=a.max, min_score=a.min_score,
                  dry_run=a.dry_run, judge_fn=judge_fn)
    for r in propose_recipes(con, since=since, max_new=a.max_recipes, dry_run=a.dry_run):
        tag = 'would propose' if a.dry_run else (f"#{r['id']} proposed" if 'id' in r else f"refused: {r['refused']}")
        print(f"{tag} · recipe · {r['project']} · ep {r['episode_id']}\n    {r['text'][:300]}")
    for r in res:
        tag = 'would propose' if a.dry_run else (f"#{r['id']} proposed" if 'id' in r else f"refused: {r['refused']}")
        how = 'judged lesson by both models' if r.get('judged') else f"score {r['score']}"
        print(f"{tag} · {how} · triggers {r['triggers']} · ep {r['episode_id']}\n    {r['sentence'][:300]}")
    if not a.dry_run:
        from . import conflicts
        fr = freshness(con)
        if fr['recipes_flagged']:
            print(f"{fr['recipes_flagged']} recipe(s) observed failing again: pm1-rule conflicts")
        sc = conflicts.scan(con, scorer, judge_fn=judge_fn) if scorer is not None else {'new_conflicts': 0}
        if sc['new_conflicts']:
            print(f"{sc['new_conflicts']} possible conflict(s) with standing rules: pm1-rule conflicts")
    if not res:
        print('(no new lessons)')
    elif not a.dry_run:
        print('Review: pm1-rule pending  ·  approve: pm1-rule approve <id> [--trigger …]  ·  reject: pm1-rule reject <id>')
