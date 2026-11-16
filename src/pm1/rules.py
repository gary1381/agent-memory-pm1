"""Standing rules and cross-project lessons: what the user decided, carried to every session it applies to.

Authority comes from the source, never from a model's confidence:
  - the user adds a rule in a terminal (``pm1-rule add``): active at once;
  - an agent can only PROPOSE one (MCP ``propose_rule``), quoting the user's own words verbatim -- the quote is
    checked against a captured user turn (or the assistant proposal the user's next turn approved); a proposal
    stays inactive until the user approves it in a terminal (``pm1-rule approve``). Agents cannot activate,
    approve, revoke or edit anything.
  - rules are revoked or replaced by the user only (``pm1-rule revoke`` / ``add --replaces``); nothing is
    inferred, so there is no automatic supersession to get wrong.

Three kinds:
  rule    shown when a session starts in a matching project (Claude Code SessionStart hook; MCP ``rules``)
  recipe  how to run something in a project, observed (a command that passed after the obvious one failed on the
          environment); shown at session start like a rule; its evidence is the tool record, approval the user's
  lesson  a pitfall worth remembering across projects, with trigger strings: pushed once per session when a trigger
          appears in a command, its output or an edit ("bge-small-en" -> "English only; use a multilingual model").

Standard library only: the Claude Code hooks import this on every tool call.
"""
from __future__ import annotations

import datetime as _dt
import json
import os
import re
import sqlite3

from .paths import default_db
from .redact import redact
from .userwords import is_user_words, own_words

SCHEMA = """
CREATE TABLE IF NOT EXISTS rules (id INTEGER PRIMARY KEY, kind TEXT NOT NULL, text TEXT NOT NULL,
  projects TEXT NOT NULL DEFAULT '[]', triggers TEXT NOT NULL DEFAULT '[]', quote TEXT, source_episode INTEGER,
  said_at TEXT, status TEXT NOT NULL, proposed_by TEXT, created_at TEXT NOT NULL, decided_at TEXT, note TEXT,
  replaced_by INTEGER, why TEXT, action TEXT, until TEXT, review TEXT, cond TEXT, last_seen TEXT, via TEXT);
CREATE INDEX IF NOT EXISTS ix_rules_status ON rules(status);
"""
QUOTE_SHOWN = 80          # characters of the user's own words shown next to each paraphrase
KINDS = ('rule', 'lesson', 'recipe')
AGENT_SOURCE = 'mcp:memory_capture:'      # records an agent wrote (pm1.store.AGENT_SOURCE); never a user's words
APPROVAL = re.compile(r'^\s*(?:好的|好|可以|同意|行|执行|按(?:照)?(?:你的|这个)?(?:方案|建议)|就这样|没问题|'
                      r'ok(?:ay)?|yes|go ahead|agreed?|approved?|do it|sounds good|lgtm)\b', re.I)
MAX_RULES_CHARS = 2500
# after the approving word only these may follow ("ok, go ahead", "好的，按你的方案做", "yes thanks"): anything else is a
# comment or a hesitation, never an approval (T030 M1: "ok, never mind", "好的，算了")
APPROVAL_TAIL = re.compile(r"^[\s,.!，。！、~]*(?:(?:please|thanks?|thank you|go ahead|do it|sounds good|that works|proceed|"
                           r"谢谢|吧|啊|的|了|就|这样|做|办|执行|开始|继续|按照?(?:你的|这个)?(?:方案|建议)|可以)[\s,.!，。！、~]*)*$", re.I)
# commands that change what is in force: only a person at a terminal may run them (cli(), not main(): tests and
# the daily timer call main). A speed bump, not a wall: anything that can write the SQLite file can bypass it.
HUMAN_ONLY = ('add', 'approve', 'reject', 'revoke', 'edit', 'renew', 'dismiss')
AGENT_ENV = ('CLAUDECODE', 'CLAUDE_CODE_ENTRYPOINT', 'CODEX_SANDBOX', 'CODEX_THREAD_ID', 'CURSOR_AGENT',
             'GROKBOT_TASK')
# approval = a short reply that is only an approval: no contrast, question or refusal anywhere in it
NOT_APPROVAL = re.compile(r"\b(but|however|except|unless|wait|no|not|don't|dont|instead|why|how)\b|不|但|别|等一下|等等|"
                          r"先别|[?？]", re.I)


def now() -> str:
    return _dt.datetime.now(_dt.timezone.utc).isoformat(timespec='seconds')


def ensure(con: sqlite3.Connection) -> None:
    con.executescript(SCHEMA)
    have = {r[1] for r in con.execute('PRAGMA table_info(rules)')}
    for col in ('why', 'action', 'until', 'review', 'cond', 'last_seen', 'via'):   # older databases
        if col not in have:
            con.execute(f'ALTER TABLE rules ADD COLUMN {col} TEXT')


def _local_date(ts: str) -> str:
    """The local calendar date of a stored UTC timestamp ('2026-10-19T23:30:00+00:00' is 10-19 6 pm in Texas)."""
    try:
        d = _dt.datetime.fromisoformat(ts.replace('Z', '+00:00'))
        return (d.astimezone() if d.tzinfo else d).date().isoformat()
    except ValueError:
        return ts[:10]


def today() -> str:
    return _dt.date.today().isoformat()


def parse_when(v: str | None) -> str | None:
    """'2026-11-01' | '2w' | '10d' | '3m' -> ISO date; '' -> None (clears)."""
    v = (v or '').strip().lower()
    if not v:
        return None
    m = re.fullmatch(r'(\d+)\s*([dwm])', v)
    if m:
        n, unit = int(m.group(1)), m.group(2)
        return (_dt.date.today() + _dt.timedelta(days=n * {'d': 1, 'w': 7, 'm': 30}[unit])).isoformat()
    try:
        return _dt.date.fromisoformat(v).isoformat()
    except ValueError:
        raise ValueError(f'not a date or a span like 2w / 10d / 3m: {v!r}')


def expired(r: dict, on: str | None = None) -> bool:
    return bool(r.get('until')) and r['until'] < (on or today())


def due_review(r: dict, on: str | None = None) -> bool:
    return bool(r.get('review')) and r['review'] <= (on or today())


def _norm(s: str) -> str:
    return re.sub(r'\s+', ' ', s or '').strip().lower()


def _rows(cur) -> list[dict]:
    """Rows of a 'SELECT * FROM rules ...' cursor as dicts, by column name."""
    names = [c[0] for c in cur.description]
    return [_row(dict(zip(names, r))) for r in cur.fetchall()]


def _row(d: dict) -> dict:
    d = dict(d)
    if 'projects' not in d:          # rows of other tables (rule_conflicts)
        return d
    d['projects'], d['triggers'] = json.loads(d['projects']), json.loads(d['triggers'])
    return d


def _check(kind: str, text: str, triggers) -> None:
    if kind not in KINDS:
        raise ValueError(f'kind must be one of {KINDS}')
    if len((text or '').strip()) < 6:
        raise ValueError('rule text too short')
    if kind == 'lesson' and not [t for t in triggers or [] if len(t.strip()) >= 3]:
        raise ValueError('a lesson needs at least one trigger string (3+ characters) to know when to speak up')


# ---------------------------------------------------------------- writes (user: add / approve / revoke)
def add(con: sqlite3.Connection, text: str, *, kind: str = 'rule', projects=(), triggers=(),
        source_episode: int | None = None, replaces: int | None = None, why: str | None = None,
        action: str | None = None, until: str | None = None, review: str | None = None,
        cond: str | None = None) -> int:
    """The user's own rule: active at once. ``replaces`` revokes an earlier one in the same step. ``why`` (the
    reason / root cause) and ``action`` (what to do instead) are the user's words; nothing fills them in."""
    _check(kind, text, triggers)
    ensure(con)
    quote = said = None
    if source_episode is not None:
        e = con.execute('SELECT role, text, occurred_at, source FROM episodes WHERE id=?', (source_episode,)).fetchone()
        if e is None or e[0] != 'user' or not is_user_words(e[1], e[3]):
            raise ValueError(f'episode {source_episode} is not a captured user turn')
        quote, said = e[1][:600], e[2]
    cur = con.execute('INSERT INTO rules(kind, text, projects, triggers, quote, source_episode, said_at, status, '
                      "proposed_by, created_at, decided_at, why, action) VALUES (?,?,?,?,?,?,?,'active','user',?,?,?,?)",
                      (kind, text.strip(), json.dumps(list(projects)), json.dumps([t.strip() for t in triggers]),
                       quote, source_episode, said or now(), now(), now(), _clean(why), _clean(action)))
    rid = cur.lastrowid
    if until or review or cond:
        con.execute('UPDATE rules SET until=?, review=?, cond=? WHERE id=?',
                    (parse_when(until), parse_when(review), _clean(cond), rid))
    if replaces is not None:
        revoke(con, replaces, f'replaced by #{rid}', replaced_by=rid)
    con.commit()
    return rid


def decide(con: sqlite3.Connection, rid: int, status: str, note: str = '', projects=None, triggers=None) -> dict:
    """User only: approve or reject a proposal; ``projects`` / ``triggers`` (when given) replace the proposed ones."""
    ensure(con)
    if status not in ('active', 'rejected'):
        raise ValueError('status must be active or rejected')
    if projects is not None:
        con.execute("UPDATE rules SET projects=? WHERE id=? AND status='proposed'", (json.dumps(list(projects)), rid))
    if triggers:
        con.execute("UPDATE rules SET triggers=? WHERE id=? AND status='proposed'",
                    (json.dumps([t.strip() for t in triggers if len(t.strip()) >= 3]), rid))
    n = con.execute("UPDATE rules SET status=?, decided_at=?, note=? WHERE id=? AND status='proposed'",
                    (status, now(), note or None, rid)).rowcount
    con.commit()
    if not n:
        cur = get(con, rid)
        scope = ', '.join(cur['projects']) or 'all projects'
        raise ValueError(f"#{rid} is already {cur['status']} ({scope}); nothing to do" if cur['status'] != 'proposed'
                         else f'#{rid} is not a pending proposal')
    return get(con, rid)


def revoke(con: sqlite3.Connection, rid: int, note: str = '', replaced_by: int | None = None) -> dict:
    ensure(con)
    n = con.execute("UPDATE rules SET status='revoked', decided_at=?, note=?, replaced_by=? WHERE id=? "
                    "AND status='active'", (now(), note or None, replaced_by, rid)).rowcount
    if n:
        _resolve_conflicts(con, rid)
    con.commit()
    if not n:
        raise ValueError(f'#{rid} is not an active rule')
    return get(con, rid)


def _clean(v: str | None) -> str | None:
    v = (v or '').strip()
    return v or None


def edit(con: sqlite3.Connection, rid: int, *, text: str | None = None, why: str | None = None,
         action: str | None = None, triggers=None, projects=None, until: str | None = None,
         review: str | None = None, cond: str | None = None, settle: bool = True) -> dict:
    """User only (the pm1-rule CLI): change what a proposed or active entry says. An empty string clears why /
    action. The quote and its source never change: they are the evidence."""
    ensure(con)
    cur = get(con, rid)
    if cur['status'] not in ('proposed', 'active'):
        raise ValueError(f"#{rid} is {cur['status']}; only proposed or active entries can be edited")
    sets, args = [], []
    if text is not None:
        _check(cur['kind'], text, cur['triggers'])
        sets.append('text=?'); args.append(text.strip())
    for col, v in (('why', why), ('action', action), ('cond', cond)):
        if v is not None:
            sets.append(f'{col}=?'); args.append(_clean(v))
    for col, v in (('until', until), ('review', review)):
        if v is not None:
            sets.append(f'{col}=?'); args.append(parse_when(v))
    if triggers:
        sets.append('triggers=?'); args.append(json.dumps([t.strip() for t in triggers if len(t.strip()) >= 3]))
    if projects is not None:
        sets.append('projects=?'); args.append(json.dumps(list(projects)))
    if not sets:
        raise ValueError('nothing to change (--text / --why / --do / --when / --until / --review / --trigger / '
                         '--project / --global)')
    con.execute(f"UPDATE rules SET {', '.join(sets)} WHERE id=?", (*args, rid))
    if settle:
        _resolve_conflicts(con, rid)      # the user edited the rule after seeing the flags: they are settled
    con.commit()
    return get(con, rid)


def get(con: sqlite3.Connection, rid: int) -> dict:
    rows = _rows(con.execute('SELECT * FROM rules WHERE id=?', (rid,)))
    if not rows:
        raise KeyError(rid)
    return rows[0]


# ---------------------------------------------------------------- agent proposals (never active on their own)
def verify_quote(con: sqlite3.Connection, episode_id: int, quote: str, kind: str = 'rule') -> dict:
    """The quote must be the user's own words: in a captured user turn, or in the assistant proposal immediately
    before a user turn that approved it ("同意 / ok / go ahead ..."). A lesson may also quote an observation: a
    captured tool record (error text, command output, commit message) -- what happened, not what an agent said.
    Returns where it was found."""
    q = _norm(quote)
    if len(q) < 6:
        raise ValueError('quote too short (6+ characters of the user\'s own words)')
    e = con.execute('SELECT id, scope, session, seq, role, occurred_at, text, source FROM episodes WHERE id=?',
                    (episode_id,)).fetchone()
    if e is None:
        raise ValueError(f'episode {episode_id} not found')
    if (e[7] or '').startswith(AGENT_SOURCE):
        raise ValueError(f'episode {episode_id} was written by an agent')
    if e[4] == 'user' and not is_user_words(e[6], e[7]):
        raise ValueError(f'episode {episode_id} is a summary, notification or another agent\'s message, not the user\'s words')
    if kind in ('lesson', 'recipe') and e[4] == 'tool':
        if q in _norm(e[6]):
            return {'how': 'observed in a tool record', 'episode_id': e[0], 'said_at': e[5], 'scope': e[1]}
        raise ValueError('quote not found verbatim in that tool record')
    if e[4] != 'user':
        raise ValueError(f'episode {episode_id} is not a captured user turn' +
                         (' (a lesson or recipe may also quote a tool record)' if kind != 'rule' else ''))
    if q in _norm(own_words(e[6])):
        return {'how': 'user said it', 'episode_id': e[0], 'said_at': e[5], 'scope': e[1]}
    if q in _norm(e[6]):          # in what the user pasted or quoted: theirs only with a plain approval ("X::同意")
        mine = own_words(e[6])
        am = APPROVAL.search(mine)
        if am and len(mine) < 200 and not NOT_APPROVAL.search(mine) and APPROVAL_TAIL.match(mine[am.end():]):
            return {'how': 'user approved the text they quoted', 'episode_id': e[0], 'said_at': e[5], 'scope': e[1]}
        raise ValueError('quote is in text the user pasted or quoted, not in their own words')
    am = APPROVAL.search(e[6] or '')
    if am and len(e[6]) < 200 and not NOT_APPROVAL.search(e[6]) and APPROVAL_TAIL.match(e[6][am.end():]):
        # the proposal is what the assistant itself said just before: never a record an agent wrote into the session
        prev = con.execute("SELECT id, text FROM episodes WHERE scope=? AND session=? AND seq < ? AND role IN "
                           "('assistant', 'user') AND COALESCE(source, '') NOT LIKE ? ORDER BY seq DESC LIMIT 1",
                           (e[1], e[2], e[3], AGENT_SOURCE + '%')).fetchone()
        if prev is not None and con.execute('SELECT role FROM episodes WHERE id=?', (prev[0],)).fetchone()[0] \
                == 'assistant' and q in _norm(prev[1]):
            return {'how': 'user approved the assistant proposal that says it', 'episode_id': e[0],
                    'proposal_episode_id': prev[0], 'said_at': e[5], 'scope': e[1]}
    raise ValueError('quote not found verbatim in that user turn (or in the proposal it approved: a plain "ok" '
                     'reply, no "but" / question / refusal)')


def propose(con: sqlite3.Connection, text: str, episode_id: int, quote: str, *, kind: str = 'rule',
            projects=None, triggers=(), client: str = '') -> dict:
    text, quote = redact(text), redact(quote)
    triggers = [redact(t) for t in triggers]
    _check(kind, text, triggers)
    ensure(con)
    where = verify_quote(con, int(episode_id), quote, kind)
    if projects is None:          # default: the project the user said it in (a Cowork label by its name, so the
        sc = where['scope']       # same project's terminal sessions match too)
        projects = [sc[12:]] if sc.startswith('proj:cowork:') else [sc[5:]] if sc.startswith('proj:') else []
    dup = con.execute("SELECT id, status FROM rules WHERE source_episode=? AND status IN ('proposed', 'active', 'rejected') "
                      'AND kind=?', (where['episode_id'], kind)).fetchone()
    if dup:
        return {'id': dup[0], 'status': dup[1], 'duplicate': True}
    cur = con.execute('INSERT INTO rules(kind, text, projects, triggers, quote, source_episode, said_at, status, '
                      "proposed_by, created_at, via) VALUES (?,?,?,?,?,?,?,'proposed',?,?,?)",
                      (kind, text.strip(), json.dumps(list(projects)), json.dumps([t.strip() for t in triggers]),
                       quote.strip()[:600], where['episode_id'], where['said_at'], client or 'agent', now(),
                       where['how']))
    con.commit()
    return {'id': cur.lastrowid, 'status': 'proposed', 'verified': where['how'],
            'note': f'Inactive: only the user can activate it, in their own terminal (proposal #{cur.lastrowid}). '
                    'Do not run pm1-rule yourself.'}


# ---------------------------------------------------------------- reads
def applies(projects: list[str], where: str | None) -> bool:
    """[] = every project. '/abs/path' matches that folder and below; a bare name matches one whole path component
    or Cowork label exactly ('agent_lab' matches /home/u/work/agent_lab/x and cowork:agent_lab;
    'orbit' does not match orbit_gpu)."""
    if not projects:
        return True
    if not where:
        return False
    w = where[5:] if where.startswith('proj:') else where
    parts = set(p for p in re.split(r'[/:]', w) if p)
    for p in projects:
        if p.startswith('/'):
            if w.rstrip('/') == p.rstrip('/') or w.startswith(p.rstrip('/') + '/'):
                return True
        elif ':' in p:                     # a Cowork label as written: 'cowork:acme-pdf'
            if w == p:
                return True
        elif p in parts:
            return True
    return False


def active(con: sqlite3.Connection, where: str | None, kinds=('rule', 'recipe')) -> list[dict]:
    ensure(con)
    rows = _rows(con.execute(
        f"SELECT * FROM rules WHERE status='active' AND kind IN ({','.join('?' * len(kinds))}) ORDER BY said_at",
        kinds))
    rows = [r for r in rows if applies(r['projects'], where) and not expired(r)]
    attach_conflicts(con, rows)
    return rows


def expired_rules(con: sqlite3.Connection, where: str | None = None) -> list[dict]:
    """Active entries past their 'until' date: no longer shown to agents; the user is asked whether they still hold."""
    ensure(con)
    rows = _rows(con.execute("SELECT * FROM rules WHERE status='active' AND until IS NOT NULL AND until < ?",
                             (today(),)))
    return [r for r in rows if where is None or applies(r['projects'], where)]


def _has(con: sqlite3.Connection, table: str) -> bool:
    return con.execute("SELECT 1 FROM sqlite_master WHERE type='table' AND name=?", (table,)).fetchone() is not None


def attach_conflicts(con: sqlite3.Connection, rows: list[dict]) -> None:
    """Open possible conflicts (pm1.conflicts) on each rule, newest first."""
    if not rows or not _has(con, 'rule_conflicts'):
        return
    by = {r['id']: r for r in rows}
    for d in _rows(con.execute(f"SELECT * FROM rule_conflicts WHERE status='open' AND rule_id IN "
                               f"({','.join('?' * len(by))}) ORDER BY episode_id DESC", list(by))):
        by[d['rule_id']].setdefault('conflicts', []).append(d)


def open_conflicts(con: sqlite3.Connection, where: str | None = None) -> list[dict]:
    if not _has(con, 'rule_conflicts'):
        return []
    rows = _rows(con.execute("SELECT c.*, r.text AS rule_text, r.projects AS rule_projects FROM rule_conflicts c "
                             "JOIN rules r ON r.id = c.rule_id WHERE c.status='open' AND r.status='active' ORDER BY c.id"))
    return [c for c in rows if where is None or applies(json.loads(c['rule_projects']), where)]


def _resolve_conflicts(con: sqlite3.Connection, rid: int, status: str = 'resolved') -> None:
    if _has(con, 'rule_conflicts'):
        con.execute("UPDATE rule_conflicts SET status=?, decided_at=? WHERE rule_id=? AND status='open'",
                    (status, now(), rid))


def dismiss(con: sqlite3.Connection, cid: int) -> None:
    """User only: a flagged message is not a conflict."""
    if not _has(con, 'rule_conflicts') or not con.execute(
            "UPDATE rule_conflicts SET status='dismissed', decided_at=? WHERE id=? AND status='open'",
            (now(), cid)).rowcount:
        raise ValueError(f'no open conflict #{cid}')
    con.commit()


def pending(con: sqlite3.Connection) -> list[dict]:
    ensure(con)
    return _rows(con.execute("SELECT * FROM rules WHERE status='proposed' ORDER BY id"))


def lessons_hit(con: sqlite3.Connection, text: str, where: str | None) -> list[dict]:
    """Active lessons whose trigger string appears in ``text`` (case-insensitive)."""
    t = (text or '').lower()
    return [r for r in active(con, where, ('lesson',)) if any(x.lower() in t for x in r['triggers'])]


def annotate(con: sqlite3.Connection, episode_ids) -> dict[int, str]:
    """Search results that are the source of a rule: say whether it still stands."""
    ensure(con)
    ids = [int(i) for i in episode_ids]
    if not ids:
        return {}
    out = {}
    for d in _rows(con.execute(f"SELECT * FROM rules WHERE source_episode IN ({','.join('?' * len(ids))}) "
                               "AND status IN ('active', 'revoked')", ids)):
        if d['status'] == 'active':
            out[d['source_episode']] = f"standing {d['kind']} #{d['id']} (still in force): {d['text']}"
        else:
            new = get(con, d['replaced_by'])['text'] if d['replaced_by'] else None
            out[d['source_episode']] = (f"{d['kind']} #{d['id']} from this record was revoked by the user on "
                                        f"{(d['decided_at'] or '')[:10]}" + (f'; now: {new}' if new else '') +
                                        (f" ({d['note']})" if d['note'] and not new else ''))
    return out


def _detail(r: dict, quote: bool = True) -> list[str]:
    """The indented lines under an entry: why / what to do (the user's words) and the user's own words it came from
    (cut to QUOTE_SHOWN characters)."""
    out = []
    if r.get('cond'):
        out.append(f"  only when: {r['cond']}")
    wd = ' · '.join(x for x in (f"why: {r['why']}" if r.get('why') else '',
                                f"do: {r['action']}" if r.get('action') else '') if x)
    if wd:
        out.append(f'  {wd}')
    tl = ' · '.join(x for x in (f"valid until {r['until']}" if r.get('until') else '',
                                (f"review due ({r['review']}): check with the user that it still holds"
                                 if due_review(r) else f"review on {r['review']}") if r.get('review') else '',
                                f"last confirmed by observation {r['last_seen'][:10]}" if r.get('last_seen') else '')
                    if x)
    if tl:
        out.append(f'  {tl}')
    q = re.sub(r'\s+', ' ', r.get('quote') or '').strip()
    if quote and q:
        who = 'original (the assistant\'s proposal, which you approved)' if (r.get('via') or '').startswith(
            'user approved') else 'original'
        out.append(f'  {who}: "{q[:QUOTE_SHOWN]}{"…" if len(q) > QUOTE_SHOWN else ""}"'
                   + (f" [ep {r['source_episode']}]" if r.get('source_episode') else ''))
    elif r.get('source_episode'):
        out.append(f"  [ep {r['source_episode']}]")
    for c in (r.get('conflicts') or [])[:2]:
        ex = c['excerpt'] if len(c['excerpt']) <= 160 else '…' + c['excerpt'][len(c['excerpt']) // 2 - 80:][:160] + '…'
        who = 'observed' if c['hint'].startswith('observed') else 'the user may have changed this later'
        out.append(f"  ⚠ {who} ({c['hint']}): \"{ex}\" [ep {c['episode_id']}] -- "
                   'ask the user before relying on this')
    return out


PARAPHRASE_NOTE = ' Each line is a paraphrase; where it and the original words differ, the original words win.'


def render_rules(rows: list[dict], n_pending: int = 0) -> str:
    if not rows and not n_pending:
        return ''
    lines = []
    ru = [r for r in rows if r['kind'] == 'rule']
    rc = [r for r in rows if r['kind'] == 'recipe']
    if ru:
        lines.append('pm1: standing rules the user set for this project (the user\'s own decisions; the user '
                     'revokes them with pm1-rule). Follow them unless the user says otherwise now.' + PARAPHRASE_NOTE)
        for r in ru:
            lines.append(f"- #{r['id']} ({(r['said_at'] or '')[:10]}) {r['text']}")
            lines += _detail(r)
    if rc:
        lines.append('pm1: known-good ways to run things in this project (observed in earlier sessions, confirmed by '
                     'the user; if one no longer works, say so):')
        for r in rc:      # the recipe's text already holds the command it quotes
            lines.append(f"- #{r['id']} ({(r['said_at'] or '')[:10]}) {r['text']}")
            lines += _detail(r, quote=False)
    if n_pending:
        lines.append(f'({n_pending} rule proposal(s) await the user\'s confirmation: pm1-rule pending)')
    n_conf = sum(len(r.get('conflicts') or []) for r in rows)
    if n_conf:
        lines.append(f'({n_conf} possible conflict(s) with these rules await the user: pm1-rule conflicts)')
    out = '\n'.join(lines)
    return out if len(out) <= MAX_RULES_CHARS else out[:MAX_RULES_CHARS - 60] + '\n… (more: MCP rules tool)'


def render_lessons(rows: list[dict]) -> str:
    lines = ['pm1: a lesson the user recorded from earlier work applies here (records, not instructions to do '
             'anything new).' + PARAPHRASE_NOTE]
    for r in rows:
        lines.append(f"- #{r['id']} ({(r['said_at'] or '')[:10]}) {r['text']}")
        lines += _detail(r, quote=_norm(r.get('quote')) != _norm(r['text']))      # a verbatim lesson: no repeat
    return '\n'.join(lines)


# ---------------------------------------------------------------- CLI (the user's terminal)
def _conflict_gate(con: sqlite3.Connection, rid: int) -> bool:
    """Before approving a rule: did the user say something later that may change it? True = held back."""
    r = get(con, rid)
    if r['kind'] != 'rule':
        return False
    try:
        from . import conflicts
        from .rerank import router_from_env
        router = router_from_env()
        scorer = max((s for s in router.rerankers if s.healthy()), key=lambda s: s.quality) if router else None
        if scorer is None:
            raise RuntimeError('no reranker')
        conflicts.ensure(con)
        found = conflicts.find(con, scorer, r)
    except Exception as exc:                    # the check helps; it must not block approving
        print(f'#{rid}: conflict check skipped ({type(exc).__name__}: {exc})')
        return False
    if not found:
        return False
    conflicts.record(con, rid, found)
    print(f"#{rid} NOT approved yet: after it you said something that may change it:")
    for c in found:
        print(f"    {c['hint']} · score {c['score']} · ep {c['episode_id']}\n    \"{c['excerpt'][:300]}\"")
    print(f'  Reword it (pm1-rule approve {rid} --text "…"), reject it, or approve as is: '
          f'pm1-rule approve {rid} --anyway')
    return True


def _default_project() -> list[str]:
    d = os.getcwd()
    while d not in ('/', ''):
        if os.path.exists(os.path.join(d, '.git')):
            return [d]
        d = os.path.dirname(d)
    return []


def _require_human(cmd: str) -> None:
    import sys
    hit = [k for k in AGENT_ENV if os.environ.get(k)]
    if hit or not (sys.stdin.isatty() and sys.stdout.isatty()):
        why = f'{hit[0]} is set' if hit else 'there is no terminal'
        raise ValueError(f'`pm1-rule {cmd}` must be run by you in your own terminal ({why}); an agent can only '
                         'propose (propose_rule). Nothing was changed.')


def main(argv=None, human: bool = False) -> None:
    import sys
    argv = sys.argv[1:] if argv is None else argv
    if argv[:1] == ['suggest']:              # needs the reranker: imported only here, never by the hooks
        from .lesson_mine import main as suggest_main
        return suggest_main(argv[1:])
    import argparse
    ap = argparse.ArgumentParser(prog='pm1-rule', description='Standing rules and lessons (only you can activate, '
                                 'approve or revoke them; agents can only propose).')
    ap.add_argument('--db', default=default_db())
    sub = ap.add_subparsers(dest='cmd', required=True)
    a = sub.add_parser('add', help='add a rule (active at once)')
    a.add_argument('text')
    a.add_argument('--lesson', action='store_true', help='a cross-project lesson (needs --trigger)')
    a.add_argument('--trigger', action='append', default=[], help='string that makes a lesson speak up (repeatable)')
    g = a.add_mutually_exclusive_group()
    g.add_argument('--project', action='append', help='folder path or name (repeatable); default: this git repo')
    g.add_argument('--global', dest='glob', action='store_true', help='every project')
    a.add_argument('--from-episode', type=int, help='your own turn it comes from (pm1 episode id)')
    a.add_argument('--replaces', type=int, help='revoke this rule id in the same step')
    a.add_argument('--when', dest='cond', help='only when this holds (a condition, in your words)')
    a.add_argument('--until', help='valid until (2026-11-01, or 2w / 10d / 3m from today)')
    a.add_argument('--review', help='ask again on (date or 2w / 10d / 3m)')
    a.add_argument('--why', help='the reason (a rule) or root cause (a lesson), in your words')
    a.add_argument('--do', dest='act', help='what to do instead / how to avoid it')
    sub.add_parser('list', help='active rules and lessons').add_argument('--all', action='store_true')
    sub.add_parser('pending', help='proposals waiting for you')
    sub.add_parser('suggest', help='find candidate lessons in recent records and propose them (see suggest --help)')
    sub.add_parser('judge-stats', help='how often the Cursor judges agreed with your decisions')
    sub.add_parser('conflicts', help='later messages of yours that may change a standing rule').add_argument(
        '--all', action='store_true', help='also the ones the judges set aside (consistent / unrelated / other scope)')
    d = sub.add_parser('dismiss', help='a flagged message is not a conflict')
    d.add_argument('cid', type=int, nargs='+')
    e = sub.add_parser('edit', help='change the text, reason or advice of a proposed or active entry')
    e.add_argument('id', type=int)
    e.add_argument('--text')
    e.add_argument('--why', help='reason / root cause ("" clears it)')
    e.add_argument('--do', dest='act', help='what to do instead ("" clears it)')
    e.add_argument('--when', dest='cond', help='only when this holds ("" clears it)')
    e.add_argument('--until', help='valid until: date or 2w / 10d / 3m ("" clears it)')
    e.add_argument('--review', help='ask again on: date or 2w / 10d / 3m ("" clears it)')
    rn = sub.add_parser('renew', help='still valid: move until / review forward (default: review in 4 weeks)')
    rn.add_argument('id', type=int, nargs='+')
    rn.add_argument('--until')
    rn.add_argument('--review')
    e.add_argument('--trigger', action='append', help='replace the triggers (repeatable)')
    g3 = e.add_mutually_exclusive_group()
    g3.add_argument('--project', action='append', help='replace the scope (repeatable)')
    g3.add_argument('--global', dest='glob', action='store_true', help='every project')
    for name in ('approve', 'reject', 'revoke', 'show'):
        p = sub.add_parser(name)
        p.add_argument('id', type=int, nargs='+' if name != 'show' else None)
        if name != 'show':
            p.add_argument('--note', default='')
        if name == 'approve':
            g2 = p.add_mutually_exclusive_group()
            g2.add_argument('--project', action='append', help='set the scope (folder path or name; repeatable)')
            g2.add_argument('--global', dest='glob', action='store_true', help='every project')
            p.add_argument('--trigger', action='append', help='lesson: replace the proposed triggers (repeatable)')
            p.add_argument('--text', help='reword the rule or lesson as you want it shown')
            p.add_argument('--anyway', action='store_true', help='approve even if later messages may conflict')
            p.add_argument('--why', help='the reason / root cause, in your words')
            p.add_argument('--do', dest='act', help='what to do instead / how to avoid it')
            p.add_argument('--when', dest='cond', help='only when this holds')
            p.add_argument('--until', help='valid until (date or 2w / 10d / 3m)')
            p.add_argument('--review', help='ask again on (date or 2w / 10d / 3m)')
    x = ap.parse_args(argv)
    if human and x.cmd in HUMAN_ONLY:
        _require_human(x.cmd)
    con = sqlite3.connect(x.db, timeout=30)
    ensure(con)
    if x.cmd == 'add':
        projects = [] if x.glob else (x.project or _default_project())
        rid = add(con, x.text, kind='lesson' if x.lesson else 'rule', projects=projects, triggers=x.trigger,
                  source_episode=x.from_episode, replaces=x.replaces, why=x.why, action=x.act, until=x.until,
                  review=x.review, cond=x.cond)
        print(f"#{rid} active · {'every project' if not projects else ', '.join(projects)}")
        return
    if x.cmd in ('list', 'pending'):
        rows = pending(con) if x.cmd == 'pending' else _rows(con.execute(
            'SELECT * FROM rules ' + ('' if getattr(x, 'all', False) else "WHERE status='active' ") + 'ORDER BY id'))
        for r in rows:
            scope = ', '.join(r['projects']) or 'all projects'
            trig = f" · triggers {r['triggers']}" if r['triggers'] else ''
            st = 'expired' if r['status'] == 'active' and expired(r) else r['status']
            if r['kind'] == 'lesson' and r['status'] == 'active':
                ref = _local_date(r.get('last_seen') or r.get('decided_at') or r['created_at'])
                days = (_dt.date.today() - _dt.date.fromisoformat(ref)).days
                st += f" · {'triggered' if r.get('last_seen') else 'not triggered since approval'}"
                st += f' {days} d ago' if r.get('last_seen') else (f', {days} d' if days else '')
            print(f"#{r['id']} [{st}] {r['kind']} · {scope}{trig}\n    {r['text']}")
            for line in _detail({**r, 'quote': None, 'source_episode': None, 'why': None, 'action': None}):
                print('  ' + line)
            if r.get('why'):
                print(f"    why: {r['why']}")
            if r.get('action'):
                print(f"    do:  {r['action']}")
            if r['quote']:
                print(f"    quote [ep {r['source_episode']}, {(r['said_at'] or '')[:10]}]"
                      f"{' (assistant proposal you approved)' if (r.get('via') or '').startswith('user approved') else ''}: "
                      f"{r['quote'][:200]}")
        if not rows:
            print('(none)')
        return
    if x.cmd == 'judge-stats':
        from .judge import stats
        st = stats(con)
        print(f"verdicts {st['verdicts']} · decided by you {st['user_decided']} · agreed {st['agree']}"
              + (f" ({st['agreement']:.0%})" if st['agreement'] is not None else '')
              + (f" · {st['until_finetune']} more of your decisions before training a local decider is worth it"
                 if st['until_finetune'] else ' · enough to train and compare a local decider'))
        return
    if x.cmd == 'conflicts':
        cs = open_conflicts(con)
        if x.all and _has(con, 'rule_conflicts'):
            cs += _rows(con.execute("SELECT c.*, r.text AS rule_text FROM rule_conflicts c JOIN rules r ON "
                                    "r.id = c.rule_id WHERE c.status='auto_dismissed' ORDER BY c.id"))
        for c in cs:
            print(f"conflict #{c['id']}{' (set aside by the judges)' if c['status'] == 'auto_dismissed' else ''} · "
                  f"rule #{c['rule_id']}: {c['rule_text'][:100]}\n"
                  f"    {c['hint']} · score {c['score']} · ep {c['episode_id']}\n    \"{c['excerpt'][:400]}\"")
        if any(c['hint'].startswith('a condition') for c in cs):
            print('A condition or a temporary change? Put it on the rule: pm1-rule edit <rule> --when "…" '
                  'or --until <date|2w>')
        if cs:
            print('Settle: pm1-rule revoke <rule> / edit <rule> --text|--why … / add … --replaces <rule> / '
                  'dismiss <conflict #>')
        else:
            print('(none)')
        return
    if x.cmd == 'renew':
        for rid in x.id:
            r = edit(con, rid, until=x.until if x.until is not None else None,
                     review=x.review if x.review is not None else ('4w' if x.until is None else None),
                     settle=False)             # 'still valid' says nothing about the flags: they stay open
            print(f"#{rid} {r['status']} · until {r['until'] or '-'} · review {r['review'] or '-'}")
        return
    if x.cmd == 'dismiss':
        for cid in x.cid:
            dismiss(con, cid)
            print(f'conflict #{cid} dismissed')
        return
    if x.cmd == 'show':
        print(json.dumps(get(con, x.id), ensure_ascii=False, indent=1))
        return
    if x.cmd == 'edit':
        r = edit(con, x.id, text=x.text, why=x.why, action=x.act, triggers=x.trigger,
                 projects=[] if x.glob else x.project, until=x.until, review=x.review, cond=x.cond)
        print(f"#{r['id']} {r['status']} · {r['text']}")
        for line in _detail(r):
            print('  ' + line)
        return
    errors = []
    for rid in x.id:
        try:
            if x.cmd in ('approve', 'reject'):
                scope = ([] if x.glob else x.project) if x.cmd == 'approve' else None
                if x.cmd == 'approve' and any(v is not None for v in (x.text, x.why, x.act, x.cond, x.until,
                                                                       x.review)):
                    edit(con, rid, text=x.text, why=x.why, action=x.act, cond=x.cond, until=x.until,
                         review=x.review)
                if x.cmd == 'approve' and not x.anyway and _conflict_gate(con, rid):
                    errors.append(rid)
                    continue
                r = decide(con, rid, 'active' if x.cmd == 'approve' else 'rejected', x.note, projects=scope,
                           triggers=getattr(x, 'trigger', None))
            else:
                r = revoke(con, rid, x.note)
            print(f"#{r['id']} {r['status']}" + (f" · triggers {r['triggers']}" if r['triggers'] else ''))
        except (ValueError, KeyError) as exc:
            errors.append(exc)
            print(f'pm1-rule: {exc}' if isinstance(exc, ValueError) else f'pm1-rule: no rule #{rid}')
    if errors:
        raise SystemExit(1)


def cli() -> None:
    import sys
    try:
        main(human=True)
    except (ValueError, KeyError) as exc:      # user mistakes get one line, not a traceback
        msg = f'no rule #{exc.args[0]}' if isinstance(exc, KeyError) else str(exc)
        print(f'pm1-rule: {msg}  (pm1-rule list --all shows every rule and its status)', file=sys.stderr)
        sys.exit(1)


if __name__ == '__main__':
    cli()
