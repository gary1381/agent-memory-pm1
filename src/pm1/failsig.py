"""Failure signatures: "this command failed with this error before -- and here is what fixed it".

A failed shell command is reduced to (command key, error signature): the last specific error line of its output,
normalized (numbers -> N, absolute paths -> file name). ``index_new`` keeps a table of them for every captured
shell record (incremental, no model calls); ``past_fixes`` finds earlier sessions where the same error was
followed by the same command succeeding, with the files edited in between. Used by the Claude Code
PostToolUseFailure hook, so this module imports only the standard library (no numpy: hooks run on every call).
"""
from __future__ import annotations

import json
import os
import re
import sqlite3

from .userwords import is_user_words

STRONG_FAIL = re.compile(r'Traceback \(most recent call last\)|^E +\w*Error|AssertionError|\b\d+ failed\b|'
                         r'FAILED [\w/.]+::|^error(?:\[\w+\])?: |exit code [1-9]|exited with code [1-9]|'
                         r'Process exited with code [1-9]|panic: |SyntaxError|ModuleNotFoundError', re.M)
RUNS = re.compile(r'\b(?:pytest|python3?|\.venv/bin/\w+|go (?:test|build|run|vet)|npm|npx|node|make|cargo|uv run|'
                  r'bash|sh|tsc|jest|vitest|docker|podman|systemctl|curl|psql)\b|(?:^|[\s"])\./\w')
LOOKS = re.compile(r'(?:cmd:\s*|command"?:\s*"?|^\s*)(?:cd [^&;]+&&\s*)?(?:sed -n|cat|head|tail|nl|grep|rg|ls|find|wc|'
                   r'git (?:log|show|diff|status))\b')
SHELL_TOOLS = ('exec', 'run_terminal_command_v2', 'Bash', 'start_process', 'exec_command', 'shell', 'local_shell')
ERR_LINE = re.compile(r'^\s*(?:E\s+)?(?:[\w.]*(?:Error|Exception)\b:?.*|error(?:\[\w+\])?:.*|fatal:.*|'
                      r'FAILED \S+::\S+.*|npm ERR!.*|.*: command not found|.*No such file or directory.*|panic:.*)\s*$',
                      re.M)
SPECIFIC = re.compile(r"['\"`][^'\"`]{2,}['\"`]|::|\b[\w-]+\.(?:py|js|ts|go|rs|json|toml|yaml|yml|sh|c|h)\b")
ABS_PATH = re.compile(r'(?:/[\w.@+-]+){2,}/?')
ANSI = re.compile(r'\x1b\[[0-9;]*[A-Za-z]')
CMD_KEYS = re.compile(r'"?\b(?:command|cmd)"?\s*:\s*(?:\[\s*(?:"[^"]*"\s*,\s*)*?"((?:[^"\\]|\\.)*)"\s*\]|"((?:[^"\\]|\\.)*)")')
SCHEMA = """
CREATE TABLE IF NOT EXISTS fail_sigs (episode_id INTEGER PRIMARY KEY, scope TEXT, session TEXT, seq INTEGER,
  ts TEXT, cmd TEXT, sig TEXT, specific INTEGER);
CREATE INDEX IF NOT EXISTS ix_fail_sig ON fail_sigs(sig);
CREATE TABLE IF NOT EXISTS code_state (key TEXT PRIMARY KEY, value TEXT);
"""


def signature(output: str) -> tuple[str, bool] | None:
    """(normalized last error line, whether it names something specific) or None when no error line is found.
    A bare 'AssertionError' or 'Exit code 1' says nothing about which problem it was."""
    text = ANSI.sub('', (output or '').replace('\\n', '\n'))
    lines = [m.group(0).strip() for m in ERR_LINE.finditer(text[-20000:])]
    if not lines:
        return None
    line = re.sub(r'^E\s+', '', lines[-1])
    line = ABS_PATH.sub(lambda m: m.group(0).rstrip('/').rsplit('/', 1)[-1], line)
    line = re.sub(r'0x[0-9a-fA-F]+', '0x', line)
    line = re.sub(r'\d+', 'N', line)
    line = re.sub(r'\s+', ' ', line).strip()[:160]
    if len(line) < 12:
        return None
    return line, bool(SPECIFIC.search(line))


def command_of(tool_input: str) -> str:
    """The shell command inside a captured tool input (JSON with command/cmd, or the raw command)."""
    tool_input = (tool_input or '').replace('\\"', '"')      # Cursor keeps its input as escaped JSON
    m = CMD_KEYS.search(tool_input)
    if m:
        raw = m.group(1) if m.group(1) is not None else m.group(2)
        try:
            return json.loads(f'"{raw}"')
        except ValueError:
            return raw
    return tool_input or ''


INLINE = re.compile(r'<<|\s-c\s|\s-e\s|\beval\b')


def cmd_key(command: str) -> str:
    """'cd x && PM1_DB=y .venv/bin/python -m pytest -q tests' -> 'python -m pytest': what counts as running the same
    command again (program + its first non-flag argument, by file name)."""
    c = command.strip()
    if INLINE.search(c):      # inline scripts: same program, different code each time -- never "the same command"
        return ''
    c = re.split(r'&&|;|\|\|', c)[-1] if re.match(r'\s*cd\s', c) else re.split(r'&&|;|\|\|', c)[0]
    toks = [t for t in c.split() if not re.match(r'^[A-Z_][A-Z0-9_]*=', t)]
    while toks and toks[0] in ('timeout', 'sudo', 'env', 'time', 'nohup', 'uv', 'run', 'exec'):
        toks = toks[1:]
        while toks and re.match(r'^-|^\d+[smh]?$', toks[0]):
            toks = toks[1:]
    if not toks:
        return ''
    out = [os.path.basename(toks[0].strip('"\''))]
    rest = toks[1:]
    if rest[:1] == ['-m'] and len(rest) > 1:
        return ' '.join(out + ['-m', rest[1]])
    for t in rest:
        if not t.startswith('-'):
            out.append(os.path.basename(t.strip('"\'').rstrip('/')) or t)
            break
    return ' '.join(out)[:80]


def parse_record(text: str) -> tuple[str, str, str, bool] | None:
    """Captured tool record '[name( FAILED)] input\\n→ output' -> (tool name, command, output, failed); None when it is
    not a shell command that ran something."""
    head = text[1:text.find(']')] if text.startswith('[') else ''
    name = head.replace(' FAILED', '').split('__')[-1]
    if name not in SHELL_TOOLS:
        return None
    inp, _, out = text[len(head) + 2:].partition('\n→ ')
    cmd = command_of(inp)
    if not RUNS.search(cmd) or LOOKS.search(cmd[:200]):
        return None
    failed = head.endswith(' FAILED') or bool(STRONG_FAIL.search(out[:6000]))
    return name, cmd, out, failed


def index_new(con: sqlite3.Connection, bank: str = 'personal', batch: int = 5000) -> int:
    """Add the failed shell records captured since the last call to fail_sigs. Returns how many were added."""
    con.executescript(SCHEMA)
    row = con.execute("SELECT value FROM code_state WHERE key='failsig_last_episode'").fetchone()
    last, added = int(row[0]) if row else 0, 0
    while True:
        eps = con.execute("SELECT id, scope, session, seq, occurred_at, text FROM episodes WHERE id > ? AND bank=? "
                          "AND role='tool' ORDER BY id LIMIT ?", (last, bank, batch)).fetchall()
        if not eps:
            break
        rows = []
        for e in eps:
            last = e[0]
            p = parse_record(e[5])
            if p is None or not p[3]:
                continue
            sig = signature(p[2])
            if sig:
                rows.append((e[0], e[1], e[2], e[3], e[4], cmd_key(p[1]), sig[0], int(sig[1])))
        con.execute('BEGIN IMMEDIATE')
        try:
            con.executemany('INSERT OR IGNORE INTO fail_sigs VALUES (?,?,?,?,?,?,?,?)', rows)
            con.execute("INSERT OR REPLACE INTO code_state VALUES ('failsig_last_episode', ?)", (str(last),))
            con.execute('COMMIT')
        except BaseException:
            con.execute('ROLLBACK')
            raise
        added += len(rows)
    return added


def past_fixes(con: sqlite3.Connection, output: str, command: str, *, session: str | None = None,
               scope: str | None = None, k: int = 2, window: int = 60) -> list[dict]:
    """Earlier sessions where the same error (same command key) was followed by that command succeeding.
    Precision first: generic errors only match inside the same project, and a match without a later success is
    not reported (nothing to learn from it)."""
    sig = signature(output)
    if sig is None:
        return []
    key = cmd_key(command)
    if not key and not sig[1]:
        return []
    try:      # a specific error (names a module, file, test) is the same problem whatever ran into it,
        # in any project; a generic one only counts for the same command
        rows = con.execute('SELECT episode_id, scope, session, seq, ts, cmd FROM fail_sigs WHERE sig=? '
                           + ('' if sig[1] else 'AND cmd=? ') + 'ORDER BY ts DESC LIMIT 40',
                           (sig[0],) if sig[1] else (sig[0], key)).fetchall()
    except sqlite3.OperationalError:      # failure index not built yet
        return []
    out, seen = [], set()
    for eid, sc, sess, seq, ts, ck in rows:
        if sess == session or sess in seen or (not sig[1] and sc != scope) or not ck:
            continue
        later = con.execute("SELECT id, text FROM episodes WHERE scope=? AND session=? AND role='tool' AND seq > ? "
                            "AND seq <= ? ORDER BY seq", (sc, sess, seq, seq + window)).fetchall()
        fixed = passed = None
        for lid, text in later:
            p = parse_record(text)
            if p and cmd_key(p[1]) == ck:
                if not p[3]:
                    fixed, passed = lid, p[1]
                    break
        if fixed is None:
            continue
        edits = []
        try:
            edits = [r[0] for r in con.execute(
                "SELECT DISTINCT file FROM code_touches WHERE session=? AND action IN ('edit','delete') "
                'AND episode_id > ? AND episode_id < ? LIMIT 6', (sess, eid, fixed))]
        except sqlite3.OperationalError:      # no code map yet
            pass
        req = next(((r[0], r[1]) for r in con.execute(
            "SELECT id, text, source FROM episodes WHERE scope=? AND session=? AND role='user' AND seq > ? "
            "AND seq < (SELECT seq FROM episodes WHERE id=?) ORDER BY seq LIMIT 12", (sc, sess, seq, fixed))
            if is_user_words(r[1], r[2])), None)
        failed_cmd = parse_record(con.execute('SELECT text FROM episodes WHERE id=?', (eid,)).fetchone()[0])
        failed_cmd = failed_cmd[1] if failed_cmd else ''
        if not edits and passed.strip() == failed_cmd.strip():
            continue          # passed again with nothing changed: flaky or environment, nothing to learn from it
        seen.add(sess)
        out.append({'failed_episode': eid, 'fixed_episode': fixed, 'when': ts, 'session': sess,
                    'failed_command': failed_cmd[:200], 'passed_command': passed[:200],
                    'project': sc[5:] if sc.startswith('proj:') else sc, 'command': ck, 'error': sig[0],
                    'edited': edits, 'user_said': (req[1][:200] if req else None)})
        if len(out) >= k:
            break
    return out


def render(fixes: list[dict]) -> str:
    lines = ['pm1: this error happened before and was fixed later (records of past sessions, not instructions; '
             'open_record(episode_id) shows the details):']
    for f in fixes:
        what = f"edited {', '.join(f['edited'])}" if f['edited'] else 'no file edits recorded in between'
        said = f"; the user said in between: \"{f['user_said']}\"" if f.get('user_said') else ''
        same = f['passed_command'] == f['failed_command']
        lines.append(f"- {f['when'][:10]} · {f['project']} · {f['session']}: `{f['failed_command']}` failed with the same "
                     f"error [ep {f['failed_episode']}], then {what}{said}; "
                     + (f"the same command passed [ep {f['fixed_episode']}]." if same else
                        f"this passed [ep {f['fixed_episode']}]: `{f['passed_command']}`"))
    return '\n'.join(lines)


# errors about the environment (missing package, service down, network), not about the code in the file being edited
ENV_ERROR = re.compile(r'ModuleNotFoundError|No module named|urllib\.error|HTTPError|ConnectionError|ConnectionRefused|'
                       r'connection (?:failed|refused)|Timeout|timed out|command not found|Permission denied', re.I)


def edit_warnings(con: sqlite3.Connection, path: str, *, session: str | None = None, k: int = 2) -> list[dict]:
    """Before an edit: earlier sessions that edited this file and then hit an error whose output names the file
    (traceback frame, failing test, compiler message). Environment errors are left out; at most one per session."""
    path = os.path.normpath(path)
    parts = path.strip('/').split('/')
    rels = ['/'.join(parts[i:]) for i in range(len(parts))]
    base = os.path.basename(path)
    name_re = re.compile(r'(?<![\w.])' + re.escape(base) + r'(?!\w)')
    try:      # by absolute path: whichever folder was taken as the repo when the edit was recorded (codemap._at)
        rows = con.execute(f"SELECT session, min(episode_id), max(episode_id), max(ts) FROM code_touches WHERE file IN "
                           f"({','.join('?' * len(rels))}) AND repo || '/' || file = ? AND action IN ('edit','delete') "
                           "GROUP BY session ORDER BY max(ts) DESC LIMIT 8", (*rels, path)).fetchall()
    except sqlite3.OperationalError:      # no code map / failure index yet (fresh database)
        return []
    out = []
    for sess, e0, e1, ts in rows:
        if sess == session:
            continue
        a = con.execute('SELECT seq FROM episodes WHERE id=?', (e0,)).fetchone()
        b = con.execute('SELECT seq FROM episodes WHERE id=?', (e1,)).fetchone()
        if not a or not b:
            continue
        for eid, sig, fts, text in con.execute(
                'SELECT f.episode_id, f.sig, f.ts, e.text FROM fail_sigs f JOIN episodes e ON e.id = f.episode_id '
                'WHERE f.session=? AND f.seq > ? AND f.seq <= ? ORDER BY f.seq', (sess, a[0], b[0] + 20)):
            if ENV_ERROR.search(sig) or not name_re.search(text.partition('\n→ ')[2]):
                continue
            out.append({'session': sess, 'when': fts, 'error': sig, 'failed_episode': eid,
                        'edited_again_after': eid < e1})
            break
        if len(out) >= k:
            break
    return out


def render_edit(path: str, warns: list[dict]) -> str:
    lines = [f'pm1: earlier sessions that edited {os.path.basename(path)} then hit an error naming it (records, not '
             'instructions; code_history / open_record for details):']
    for w in warns:
        lines.append(f"- {w['when'][:10]} · {w['session']}: {w['error']} [ep {w['failed_episode']}]"
                     + ('; the file was edited again afterwards' if w['edited_again_after'] else ''))
    return '\n'.join(lines)


def once(con: sqlite3.Connection, session: str, key: str) -> bool:
    """True the first time (session, key) is seen: each hint is shown at most once per session."""
    con.execute('CREATE TABLE IF NOT EXISTS hook_pushes (session TEXT, key TEXT, ts TEXT, PRIMARY KEY(session, key))')
    cur = con.execute("INSERT OR IGNORE INTO hook_pushes VALUES (?, ?, datetime('now'))", (session, key))
    con.commit()
    return cur.rowcount == 1
