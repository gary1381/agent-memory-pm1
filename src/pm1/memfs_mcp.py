"""Free-form memory workspace for structure discovery (deliverable 1, step A2) — stdio MCP server, stdlib only.

The teacher can build ANY structure: folders (tree), [[path]] links (graph), front-matter records (typed facts),
summaries per topic/period, etc. Everything lives in one sqlite per conversation; nothing is ever deleted (moves
keep history). Every write is logged with its reason and evidence -> the recorded standard.

Active conversation: $PM1_MEMFS_DIR/CURRENT.json {"conv": "6", "step": "...", "max_turn": N}; store: $PM1_MEMFS_DIR/conv<conv>.sqlite;
raw turns (read-only): $PM1_RAW_DIR/conv<conv>.sqlite.
"""
import json, math, os, re, sqlite3, sys
from collections import Counter
from pathlib import Path

ROOT = Path(os.environ.get('PM1_MEMFS_DIR', '/work/memfs'))
RAWS = Path(os.environ.get('PM1_RAW_DIR', '/raw'))
MAX_NOTE = 6000
SCHEMA = """
CREATE TABLE IF NOT EXISTS note (path TEXT PRIMARY KEY, content TEXT NOT NULL, created TEXT, updated TEXT,
  evidence TEXT DEFAULT '[]');
CREATE TABLE IF NOT EXISTS note_history (id INTEGER PRIMARY KEY, path TEXT, content TEXT, step TEXT);
CREATE TABLE IF NOT EXISTS memlog (id INTEGER PRIMARY KEY, step TEXT, tool TEXT, args TEXT, why TEXT, ok INTEGER,
  result TEXT);
"""
S, A = {'type': 'string'}, {'type': 'array', 'items': {'type': 'string'}}


def _p(props, req=()):
    return {'type': 'object', 'properties': props, 'required': list(req), 'additionalProperties': True}


TOOLS = {
    'mem_list': ('List notes and folders under a path (default "/"), with sizes.', _p({'path': S})),
    'mem_read': ('Read a note.', _p({'path': S}, ['path'])),
    'mem_search': ('Keyword (BM25) search over all notes; returns paths + snippets.', _p({'query': S, 'k': {'type': 'integer'}}, ['query'])),
    'mem_backlinks': ('Notes that link to this path with [[path]].', _p({'path': S}, ['path'])),
    'mem_write': ('Create or replace a note (<=6000 chars). Use [[/path]] for links. evidence = turn ids it is based on.',
                  _p({'path': S, 'content': S, 'why': S, 'evidence': A}, ['path', 'content', 'why'])),
    'mem_append': ('Append text to a note (creates it if missing).',
                   _p({'path': S, 'text': S, 'why': S, 'evidence': A}, ['path', 'text', 'why'])),
    'mem_move': ('Move/rename a note (history kept; links are not rewritten automatically).',
                 _p({'src': S, 'dst': S, 'why': S}, ['src', 'dst', 'why'])),
    'raw_turn': ('Read one raw conversation turn by id (e.g. "t123").', _p({'turn': S}, ['turn'])),
}


def norm(p):
    p = '/' + '/'.join(x for x in re.split(r'/+', (p or '/').strip()) if x not in ('', '.', '..'))
    return p


class Mem:
    def __init__(self):
        cur = json.loads((ROOT / 'CURRENT.json').read_text())
        self.conv, self.step = str(cur['conv']), cur.get('step')
        self.max_turn = int(cur['max_turn']) if cur.get('max_turn') is not None else None
        self.con = sqlite3.connect(ROOT / f'conv{self.conv}.sqlite')
        self.con.executescript(SCHEMA)
        self.raw = sqlite3.connect(f'file:{RAWS / f"conv{self.conv}.sqlite"}?mode=ro&immutable=1', uri=True)   # read-only mount

    def _bound_err(self, tid):
        if self.max_turn is None:
            return None
        m = re.fullmatch(r't?(\d+)', str(tid))
        if m and int(m.group(1)) > self.max_turn:
            return (f'turn t{m.group(1)} is after the current step; '
                    f'only turns ≤ t{self.max_turn} are available')
        return None

    def turn(self, tid):
        err = self._bound_err(tid)
        if err:
            raise ValueError(err)
        m = re.fullmatch(r't?(\d+)', str(tid))
        r = self.raw.execute('SELECT role, text, substr(occurred_at,1,10) FROM episodes WHERE id=?',
                             (int(m.group(1)),)).fetchone() if m else None
        return r

    def _ev(self, ev):
        ev = [str(e) for e in (ev or [])]
        for e in ev:
            err = self._bound_err(e)
            if err:
                raise ValueError(err)
        bad = [e for e in ev if self.turn(e) is None]
        if bad:
            raise ValueError(f'evidence turn(s) not found: {bad}')
        return ev

    def _day(self, ev):
        days = [self.turn(e)[2] for e in ev] if ev else []
        return max(days) if days else None

    def call(self, name, a):
        if name == 'mem_list':
            base = norm(a.get('path', '/')).rstrip('/') + '/'
            rows = self.con.execute('SELECT path, length(content) FROM note WHERE path LIKE ? ORDER BY path',
                                    (base.replace('//', '/') + '%' if base != '//' else '/%',)).fetchall()
            out, dirs = [], Counter()
            for p, n in rows:
                rest = p[len(base):] if base != '//' else p[1:]
                if '/' in rest:
                    dirs[rest.split('/')[0]] += 1
                else:
                    out.append({'note': p, 'chars': n})
            return {'path': base, 'folders': [{'folder': base + d, 'notes': c} for d, c in sorted(dirs.items())],
                    'notes': out[:200]}
        if name == 'mem_read':
            r = self.con.execute('SELECT content, updated, evidence FROM note WHERE path=?', (norm(a['path']),)).fetchone()
            return {'path': norm(a['path']), 'content': r[0], 'updated': r[1], 'evidence': json.loads(r[2])} if r \
                else {'error': 'no such note'}
        if name == 'mem_search':
            return {'results': self.search(a['query'], int(a.get('k', 8)))}
        if name == 'mem_backlinks':
            p = norm(a['path'])
            return {'backlinks': [x for (x,) in self.con.execute('SELECT path FROM note WHERE content LIKE ?',
                                                                  (f'%[[{p}]]%',))]}
        if name == 'raw_turn':
            err = self._bound_err(a['turn'])
            if err:
                return {'error': err}
            m = re.fullmatch(r't?(\d+)', str(a['turn']))
            r = self.raw.execute('SELECT role, text, substr(occurred_at,1,10) FROM episodes WHERE id=?',
                                 (int(m.group(1)),)).fetchone() if m else None
            return {'turn': a['turn'], 'role': r[0], 'day': r[2], 'text': r[1][:4000]} if r else {'error': 'not found'}
        # writes
        if not (a.get('why') or '').strip():
            raise ValueError('every write needs a "why"')
        if name in ('mem_write', 'mem_append'):
            p = norm(a['path'])
            ev = self._ev(a.get('evidence'))
            old = self.con.execute('SELECT content, evidence, created FROM note WHERE path=?', (p,)).fetchone()
            text = a['content'] if name == 'mem_write' else ((old[0] + '\n') if old else '') + a['text']
            if len(text) > MAX_NOTE:
                raise ValueError(f'note would be {len(text)} chars > {MAX_NOTE}; split it into linked notes')
            if old:
                self.con.execute('INSERT INTO note_history(path, content, step) VALUES (?,?,?)', (p, old[0], self.step))
            evs = sorted(set((json.loads(old[1]) if old else []) + ev))
            day = self._day(ev)
            self.con.execute('INSERT OR REPLACE INTO note(path, content, created, updated, evidence) VALUES (?,?,?,?,?)',
                             (p, text, old[2] if old else day, day or (old[2] if old else None), json.dumps(evs)))
            return {'ok': True, 'path': p, 'chars': len(text), 'created': not old}
        if name == 'mem_move':
            s, d = norm(a['src']), norm(a['dst'])
            r = self.con.execute('SELECT content FROM note WHERE path=?', (s,)).fetchone()
            if not r:
                raise ValueError('no such note')
            if self.con.execute('SELECT 1 FROM note WHERE path=?', (d,)).fetchone():
                raise ValueError('destination exists')
            self.con.execute('INSERT INTO note_history(path, content, step) VALUES (?,?,?)', (s, r[0], self.step))
            self.con.execute('UPDATE note SET path=? WHERE path=?', (d, s))
            return {'ok': True, 'moved': [s, d]}
        raise ValueError(f'unknown tool {name}')

    def search(self, q, k):
        rows = self.con.execute('SELECT path, content FROM note').fetchall()
        tok = lambda s: re.findall(r'[a-z0-9]+(?:[.,][0-9]+)*', s.lower())
        docs = [tok(p + ' ' + c) for p, c in rows]
        df = Counter(w for d in docs for w in set(d))
        n, avg = len(docs), (sum(map(len, docs)) / max(1, len(docs)))
        qs, sc = tok(q), []
        for (p, c), d in zip(rows, docs):
            tf, s = Counter(d), 0.0
            for w in qs:
                if w in tf:
                    s += math.log(1 + (n - df[w] + 0.5) / (df[w] + 0.5)) * tf[w] * 2.2 / (tf[w] + 1.2 * (0.25 + 0.75 * len(d) / avg))
            if s > 0:
                sc.append((s, p, c))
        sc.sort(key=lambda x: -x[0])
        return [{'path': p, 'snippet': c[:300]} for _s, p, c in sc[:k]]

    def log(self, name, a, ok, res):
        self.con.execute('INSERT INTO memlog(step, tool, args, why, ok, result) VALUES (?,?,?,?,?,?)',
                         (self.step, name, json.dumps(a, ensure_ascii=False)[:20000], a.get('why'), int(ok),
                          json.dumps(res, ensure_ascii=False)[:2000]))
        self.con.commit()


def main():
    for line in sys.stdin:
        if not line.strip():
            continue
        msg = json.loads(line)
        mid, method = msg.get('id'), msg.get('method')
        if mid is None:
            continue
        if method == 'initialize':
            result = {'protocolVersion': msg.get('params', {}).get('protocolVersion', '2024-11-05'),
                      'capabilities': {'tools': {}}, 'serverInfo': {'name': 'pm1memfs', 'version': '0.1'}}
        elif method == 'tools/list':
            result = {'tools': [{'name': n, 'description': d, 'inputSchema': s} for n, (d, s) in TOOLS.items()]}
        elif method == 'tools/call':
            p = msg.get('params', {})
            name, a = p.get('name'), p.get('arguments') or {}
            m = None
            try:
                m = Mem()
                out, ok = m.call(name, a), True
            except Exception as e:  # noqa: BLE001
                out, ok = {'error': f'{type(e).__name__}: {e}'}, False
            if m is not None and name not in ('mem_list', 'mem_read', 'mem_search', 'mem_backlinks', 'raw_turn'):
                m.log(name, a, ok, out)
            result = {'content': [{'type': 'text', 'text': json.dumps(out, ensure_ascii=False)}], 'isError': not ok}
        elif method == 'ping':
            result = {}
        else:
            sys.stdout.write(json.dumps({'jsonrpc': '2.0', 'id': mid, 'error': {'code': -32601, 'message': method}}) + '\n')
            sys.stdout.flush()
            continue
        sys.stdout.write(json.dumps({'jsonrpc': '2.0', 'id': mid, 'result': result}) + '\n')
        sys.stdout.flush()


if __name__ == '__main__':
    main()
