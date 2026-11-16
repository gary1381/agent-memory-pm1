"""Possible conflicts: something the user said AFTER a rule that may change, reverse or put a condition on it.

Only ever a flag for the user to settle (revoke, edit, --replaces, or dismiss); never an automatic change.
  1. candidates: the user's later turns (not agent-written) that share at least two of the rule's keywords and
     contain a change word ("改用 / 切到 / 暂停 / 额度 / instead / switch / quota ..."); turns outside the rule's
     projects are a different scope, not a conflict; a turn that repeats the rule's own words is a restatement;
  2. ranked by the local reranker on a window around the keyword hits (the user's words often follow a pasted
     reply: scoring the whole turn buried "my quota for Cursor models ... is very little now" at 0.31, its window
     scored 0.96), kept above a floor;
  3. a hint, not a verdict: condition / temporary words ("额度, 暂时, 等到, 恢复, until, quota, for now") -> "a condition
     or a temporary change?", otherwise "changed later?".
Runs when a rule is approved (pm1-rule approve) and every morning for active rules (pm1-rule suggest).
"""
from __future__ import annotations

import re
import sqlite3

from . import rules
from .userwords import is_user_words

CHANGE = re.compile(r'改用|改成|改为|换成|切到|切换|不要|不再|别再|暂停|先不|停掉|恢复|额度|quota|instead|switch|'
                    r'stop|pause|no longer|from now on|以后|今后|现在|暂时|等到|until|anymore', re.I)
CONDITION = re.compile(r'额度|quota|暂时|temporar|等到|until|恢复|for now|这次|this time|如果.{0,12}就|只有当|only when', re.I)
STOP = set('the and for use with only from that this your their when what which into first very little now please '
           'work project then than also just more most some such them they will would should could about after '
           'before other these those used uses using 我们 你 我 的 了 是 在 和 用 先 为主 以后 现在 这个 一个 可以 如果 '
           '不要 改用 不用'.split())
SCHEMA = """
CREATE TABLE IF NOT EXISTS rule_conflicts (id INTEGER PRIMARY KEY, rule_id INTEGER NOT NULL, episode_id INTEGER NOT NULL,
  score REAL, hint TEXT, excerpt TEXT, status TEXT NOT NULL DEFAULT 'open', found_at TEXT NOT NULL, decided_at TEXT,
  UNIQUE(rule_id, episode_id));
"""
QUERY = 'A later message from the user that changes, reverses or adds a condition to this standing rule: '
WINDOW = 300


def ensure(con: sqlite3.Connection) -> None:
    rules.ensure(con)
    con.executescript(SCHEMA)


def keywords(rule: dict) -> set[str]:
    from .text import tokenize
    src = ' '.join(x for x in (rule['text'], rule.get('quote') or '', rule.get('why') or '') if x)
    ks = {t.lower().rstrip('.,;:') for t in re.findall(r'[A-Za-z0-9][\w.+-]{2,}', src)
          if re.search(r'[A-Za-z]', t) and t.lower().rstrip('.,;:') not in STOP and len(t) >= 3}
    ks |= {t for t in tokenize(rule.get('quote') or '') if re.search(r'[一-鿿]', t) and len(t) >= 2
           and t not in STOP}
    return {k for k in ks if k}


def _windows(text: str, hits: list[str]) -> list[str]:
    low, out = text.lower(), []
    for k in hits:
        for m in re.finditer(re.escape(k), low):
            out.append(text[max(0, m.start() - WINDOW):m.end() + WINDOW])
            if len(out) >= 6:
                return out
    return out


def find(con: sqlite3.Connection, scorer, rule: dict, *, after_id: int | None = None, floor: float = 0.5,
         k: int = 3) -> list[dict]:
    """Possible conflicts for one rule among the user's turns after its source (or after ``after_id``)."""
    if rule['kind'] != 'rule':
        return []
    ks = keywords(rule)
    if len(ks) < 2:
        return []
    src_id = rule.get('source_episode') or 0
    said = rule.get('said_at') or ''
    lo = max(src_id, after_id or 0)
    rows = con.execute("SELECT id, scope, text FROM episodes WHERE role='user' AND COALESCE(source, '') NOT LIKE 'mcp:%' "
                       'AND id > ? AND (occurred_at > ? OR occurred_at = ?) AND length(text) < 20000 ORDER BY id',
                       (lo, said, said) if said else (lo, '', '')).fetchall()
    quote = rules._norm(rule.get('quote') or '')
    cands = []
    for eid, scope, text in rows:
        if rule['projects'] and not rules.applies(rule['projects'], scope):
            continue                                   # another project: a different scope, not a conflict
        if not is_user_words(text):
            continue                                   # a summary / notification / another agent's message
        low = text.lower()
        hits = [x for x in ks if x in low]
        if len(hits) < 2 or not CHANGE.search(text):
            continue
        if quote and len(quote) >= 6 and quote in rules._norm(text):
            continue                                   # the rule's own words again: a restatement
        wins = [w for w in _windows(text, hits) if CHANGE.search(w)]
        if wins:
            cands.append((eid, wins))
    if not cands:
        return []
    flat = [(eid, w) for eid, ws in cands for w in ws]
    scores = []
    for i in range(0, len(flat), 32):
        scores += scorer.score(QUERY + rule['text'], [w for _, w in flat[i:i + 32]])
    best: dict[int, tuple[float, str]] = {}
    for (eid, w), sc in zip(flat, scores):
        if sc > best.get(eid, (-1.0, ''))[0]:
            best[eid] = (float(sc), w)
    out = [{'episode_id': eid, 'score': round(sc, 3), 'excerpt': re.sub(r'\s+', ' ', w).strip(),
            'hint': 'a condition or a temporary change?' if CONDITION.search(w) else 'changed later?'}
           for eid, (sc, w) in best.items() if sc >= floor]
    return sorted(out, key=lambda c: -c['score'])[:k]


def record(con: sqlite3.Connection, rule_id: int, found: list[dict], status: str = 'open') -> int:
    ensure(con)
    n = 0
    for c in found:
        n += con.execute("INSERT OR IGNORE INTO rule_conflicts(rule_id, episode_id, score, hint, excerpt, found_at, "
                         'status) VALUES (?,?,?,?,?,?,?)', (rule_id, c['episode_id'], c['score'], c['hint'],
                                                            c['excerpt'][:700], rules.now(), status)).rowcount
    con.commit()
    return n


def judged(con: sqlite3.Connection, pairs: list[tuple[dict, dict]], judge_fn) -> list[tuple[dict, dict, str]]:
    """(rule, candidate) -> (rule, candidate, status). With the judges: 'open' only when one of them says the
    message replaces the rule or adds a condition (grok-4.7 P/R 1.00 on 23 pairs, 2026-10-19), with its label as
    the hint; else 'auto_dismissed' (kept, listed by pm1-rule conflicts --all). Without them: all 'open'."""
    if not pairs or judge_fn is None:
        return [(r, c, 'open') for r, c in pairs]
    from . import judge
    items = [(f'k{i}', f"Standing rule: {r['text']}\n\nLater message from the user: {c['excerpt']}")
             for i, (r, c) in enumerate(pairs)]
    v = judge_fn('conflict', items, judge.CONFLICT_Q, judge.CONFLICT_LABELS)
    if v is None:
        return [(r, c, 'open') for r, c in pairs]
    out = []
    for i, (r, c) in enumerate(pairs):
        labs = v.get(f'k{i}', {})
        need = [x for x in labs.values() if x in judge.NEEDS_USER]
        lab = labs.get('grok47') or (need[0] if need else next(iter(labs.values()), None))
        # set aside only when every judge answered; with one missing the user still gets the flag
        status = 'open' if need or len(labs) < len(judge.LABELERS) else 'auto_dismissed'
        if lab:
            c = {**c, 'hint': {'replaces': 'changed later? (judged: replaces)',
                               'adds_condition': 'a condition or a temporary change? (judged: adds a condition)'}
                 .get(lab, f'judged: {lab}')}
        judge.log(con, 'conflict', f"rule:{r['id']}:ep:{c['episode_id']}", c['excerpt'], labs, status)
        out.append((r, c, status))
    return out


def scan(con: sqlite3.Connection, scorer, judge_fn=None) -> dict:
    """Daily: every active rule against the user's turns captured since the last scan."""
    ensure(con)
    row = con.execute("SELECT value FROM code_state WHERE key='conflict_scan_last_episode'").fetchone() \
        if con.execute("SELECT 1 FROM sqlite_master WHERE name='code_state'").fetchone() else None
    last = int(row[0]) if row else 0
    top = con.execute('SELECT COALESCE(max(id), 0) FROM episodes').fetchone()[0]
    pairs = [(r, c) for r in rules._rows(con.execute("SELECT * FROM rules WHERE status='active' AND kind='rule'"))
             for c in find(con, scorer, r, after_id=last)]
    found = 0
    for r, c, status in judged(con, pairs, judge_fn):
        n = record(con, r['id'], [c], status=status)      # set-aside ones are kept too (pm1-rule conflicts --all)
        found += n if status == 'open' else 0
    con.execute('CREATE TABLE IF NOT EXISTS code_state (key TEXT PRIMARY KEY, value TEXT)')
    con.execute("INSERT OR REPLACE INTO code_state VALUES ('conflict_scan_last_episode', ?)", (str(top),))
    con.commit()
    return {'new_conflicts': found, 'scanned_after_episode': last}
