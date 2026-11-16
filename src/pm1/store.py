"""SQLite store: immutable raw episodes (layer E) plus rebuildable derived indexes.

Every query is filtered by (bank, scope) in SQL. Derived rows (spans, FTS,
vectors, token vectors, jobs) are always deletable/rebuildable from episodes.
"""
from __future__ import annotations

import contextlib
import datetime as dt
import hashlib
import json
import os
import sqlite3
import threading
from pathlib import Path

import numpy as np

from .lang import detect
from .redact import redact
from .text import split_spans, tokenize

SCHEMA = """
PRAGMA journal_mode=WAL;
CREATE TABLE IF NOT EXISTS episodes (
  id INTEGER PRIMARY KEY, bank TEXT NOT NULL, scope TEXT NOT NULL, session TEXT NOT NULL,
  seq INTEGER NOT NULL, role TEXT NOT NULL, occurred_at TEXT NOT NULL, text TEXT NOT NULL,
  sha TEXT NOT NULL, source TEXT, created_at TEXT NOT NULL,
  UNIQUE(bank, scope, session, seq, sha));
CREATE INDEX IF NOT EXISTS ix_ep_scope ON episodes(bank, scope, session, seq);
CREATE TABLE IF NOT EXISTS spans (
  id INTEGER PRIMARY KEY, episode_id INTEGER NOT NULL REFERENCES episodes(id) ON DELETE CASCADE,
  bank TEXT NOT NULL, scope TEXT NOT NULL, start INTEGER NOT NULL, end INTEGER NOT NULL,
  salience REAL NOT NULL DEFAULT 1.0, lang TEXT);
CREATE INDEX IF NOT EXISTS ix_span_scope ON spans(bank, scope);
CREATE VIRTUAL TABLE IF NOT EXISTS span_fts USING fts5(toks, tokenize='porter unicode61 remove_diacritics 0');
CREATE TABLE IF NOT EXISTS vectors (
  span_id INTEGER NOT NULL REFERENCES spans(id) ON DELETE CASCADE, model TEXT NOT NULL,
  vec BLOB NOT NULL, PRIMARY KEY(span_id, model));
CREATE TABLE IF NOT EXISTS tokvecs (
  span_id INTEGER NOT NULL REFERENCES spans(id) ON DELETE CASCADE, model TEXT NOT NULL,
  n INTEGER NOT NULL, dim INTEGER NOT NULL, vec BLOB NOT NULL, PRIMARY KEY(span_id, model));
CREATE TABLE IF NOT EXISTS jobs (
  span_id INTEGER NOT NULL REFERENCES spans(id) ON DELETE CASCADE, kind TEXT NOT NULL,
  model TEXT NOT NULL, state TEXT NOT NULL DEFAULT 'pending', attempts INTEGER NOT NULL DEFAULT 0,
  error TEXT, PRIMARY KEY(span_id, kind, model));
CREATE INDEX IF NOT EXISTS ix_jobs ON jobs(state, kind, model);
CREATE INDEX IF NOT EXISTS ix_jobs_order ON jobs(state, kind, model, span_id);
"""


# source prefix of records written through memory_capture (an agent's claim, not a captured user turn)
AGENT_SOURCE = 'mcp:memory_capture:'


def now() -> str:
    return dt.datetime.now(dt.timezone.utc).isoformat(timespec='seconds')


class Store:
    def __init__(self, path: str | Path = ':memory:', *, index_models: tuple[str, ...] = ()):
        """``index_models``: embedding job kinds queued for every new span, e.g.
        ('dense:e5-small', 'tok:pplx-0.6b'). Jobs are processed by the worker."""
        # ~20 processes share one file (capture, every MCP server, nightly backfill): a writer waits up to
        # PM1_BUSY_S for the lock instead of failing after sqlite3's default 5 s (capture exited 3 times on 10-27)
        self.con = sqlite3.connect(str(path), check_same_thread=False, isolation_level=None,
                                   timeout=float(os.environ.get('PM1_BUSY_S', '30')))
        self.con.row_factory = sqlite3.Row
        self.con.execute('PRAGMA foreign_keys=ON')
        self.con.executescript(SCHEMA)
        if 'lang' not in {r[1] for r in self.con.execute('PRAGMA table_info(spans)')}:
            self.con.execute('ALTER TABLE spans ADD COLUMN lang TEXT')   # pre-routing databases
        self.lock = threading.RLock()
        self.index_models = index_models
        self._dense_cache: dict[tuple, tuple] = {}
        self._heads: dict[tuple, str] = {}
        self._writes = 0                    # this connection's own commits (data_version does not count them)
        self._seen: tuple | None = None     # (data_version, _writes) the fingerprint below was taken at
        self._fp: tuple | None = None

    def close(self) -> None:
        self.con.close()

    @property
    def _version(self) -> tuple:
        """Key for the derived-index caches below. Changes when spans or vectors change in this process or in any
        other one (worker, capture service, docs watch): a counter of this process's own writes kept the MCP
        server on the vectors it saw at its first query until restart (2026-10-18). The fingerprint query
        (~3 ms on 137K spans) only runs after some connection committed."""
        with self.lock:
            dv = self.con.execute('PRAGMA data_version').fetchone()[0]
            if self._seen != (dv, self._writes):
                self._fp = tuple(self.con.execute(
                    'SELECT (SELECT max(id) FROM spans), (SELECT count(*) FROM spans), '
                    '(SELECT max(rowid) FROM vectors), (SELECT count(*) FROM vectors), '
                    # ids are reused after the newest record is forgotten: its content tells the two apart
                    '(SELECT sha FROM episodes ORDER BY id DESC LIMIT 1)').fetchone())
                self._seen = (dv, self._writes)
            return self._fp

    # ---------- write path (synchronous, no model calls) ----------
    def add_episode(self, *, bank: str, scope: str, text: str, role: str = 'user',
                    session: str = 'default', seq: int | None = None,
                    occurred_at: str | None = None, source: str | None = None) -> dict:
        if not bank or not scope:
            raise ValueError('bank and scope are required')
        if not text or not text.strip():
            raise ValueError('empty text')
        text = redact(text)              # secrets never reach the store (pm1.redact)
        sha = hashlib.sha256(text.encode()).hexdigest()
        with self.lock:
            self.con.execute('BEGIN IMMEDIATE')
            try:
                if seq is None:
                    seq = self.con.execute('SELECT COALESCE(MAX(seq), -1) + 1 FROM episodes '
                                           'WHERE bank=? AND scope=? AND session=?',
                                           (bank, scope, session)).fetchone()[0]
                dup = self.con.execute('SELECT id FROM episodes WHERE bank=? AND scope=? AND session=? '
                                       'AND seq=? AND sha=?', (bank, scope, session, seq, sha)).fetchone()
                if not dup and source:   # a host line read again (capture stopped mid-file) gets a new seq
                    dup = self.con.execute('SELECT id FROM episodes WHERE bank=? AND scope=? AND session=? '
                                           'AND source=? AND sha=?',
                                           (bank, scope, session, source, sha)).fetchone()
                if dup:
                    self.con.execute('COMMIT')
                    return {'episode_id': dup['id'], 'duplicate': True, 'span_ids': self.span_ids(dup['id'])}
                cur = self.con.execute(
                    'INSERT INTO episodes(bank,scope,session,seq,role,occurred_at,text,sha,source,created_at) '
                    'VALUES (?,?,?,?,?,?,?,?,?,?)',
                    (bank, scope, session, seq, role, occurred_at or now(), text, sha, source, now()))
                eid = cur.lastrowid
                span_ids = []
                for start, end in split_spans(text):
                    sid = self.con.execute('INSERT INTO spans(episode_id,bank,scope,start,end,lang) '
                                           'VALUES (?,?,?,?,?,?)',
                                           (eid, bank, scope, start, end, detect(text[start:end]))).lastrowid
                    self.con.execute('INSERT INTO span_fts(rowid, toks) VALUES (?, ?)',
                                     (sid, ' '.join(tokenize(text[start:end]))))
                    for model in self.index_models:
                        kind, name = model.split(':', 1)
                        self.con.execute('INSERT INTO jobs(span_id,kind,model) VALUES (?,?,?)', (sid, kind, name))
                    span_ids.append(sid)
                self.con.execute('COMMIT')
            except BaseException:
                self.con.execute('ROLLBACK')
                raise
            self._writes += 1
        return {'episode_id': eid, 'duplicate': False, 'span_ids': span_ids}

    def rewrite_episode(self, episode_id: int, text: str) -> dict:
        """Replace an episode's text (only for redaction: records are otherwise immutable). Its spans, FTS rows and
        vectors are rebuilt; the embedding jobs the old spans had are queued again for the new ones."""
        sha = hashlib.sha256(text.encode()).hexdigest()
        with self.lock:
            e = self.con.execute('SELECT bank, scope FROM episodes WHERE id=?', (episode_id,)).fetchone()
            if e is None:
                raise KeyError(episode_id)
            self.con.execute('BEGIN IMMEDIATE')
            try:
                old = [r[0] for r in self.con.execute('SELECT id FROM spans WHERE episode_id=?', (episode_id,))]
                q = ','.join('?' * len(old))
                models = [tuple(r) for r in self.con.execute(
                    f'SELECT DISTINCT kind, model FROM jobs WHERE span_id IN ({q}) UNION '
                    f"SELECT DISTINCT 'dense', model FROM vectors WHERE span_id IN ({q})", old + old)] if old else []
                models = list(dict.fromkeys(models + [tuple(m.split(':', 1)) for m in self.index_models]))
                if old:
                    self.con.execute(f'DELETE FROM span_fts WHERE rowid IN ({q})', old)
                    self.con.execute(f'DELETE FROM spans WHERE id IN ({q})', old)     # cascades vectors, jobs
                self.con.execute('UPDATE episodes SET text=?, sha=? WHERE id=?', (text, sha, episode_id))
                span_ids = []
                for start, end in split_spans(text):
                    sid = self.con.execute('INSERT INTO spans(episode_id,bank,scope,start,end,lang) VALUES (?,?,?,?,?,?)',
                                           (episode_id, e['bank'], e['scope'], start, end,
                                            detect(text[start:end]))).lastrowid
                    self.con.execute('INSERT INTO span_fts(rowid, toks) VALUES (?, ?)',
                                     (sid, ' '.join(tokenize(text[start:end]))))
                    for kind, name in models:
                        self.con.execute('INSERT OR IGNORE INTO jobs(span_id,kind,model) VALUES (?,?,?)', (sid, kind, name))
                    span_ids.append(sid)
                self.con.execute('COMMIT')
            except BaseException:
                self.con.execute('ROLLBACK')
                raise
            self._writes += 1
        return {'episode_id': episode_id, 'span_ids': span_ids}

    def span_ids(self, episode_id: int) -> list[int]:
        return [r[0] for r in self.con.execute('SELECT id FROM spans WHERE episode_id=? ORDER BY start', (episode_id,))]

    # ---------- read helpers ----------
    def spans(self, bank: str, scope: str, ids: list[int]) -> dict[int, dict]:
        if not ids:
            return {}
        q = ','.join('?' * len(ids))
        rows = self.con.execute(
            f'SELECT s.id, s.start, s.end, s.salience, s.lang, s.bank, s.scope, e.id AS episode_id, e.session, e.seq, e.role, '
            f'e.occurred_at, e.text, e.source FROM spans s JOIN episodes e ON e.id = s.episode_id '
            f'WHERE s.bank=? AND s.scope=? AND s.id IN ({q})', (bank, scope, *ids)).fetchall()
        out = {}
        for r in rows:
            text = r['text'][r['start']:r['end']]
            out[r['id']] = {**dict(r), 'text': text, 'lang': r['lang'] or detect(text)}
        return out

    def span_header(self, r: dict) -> str:
        """'project · start of the session · date · role' for a span row: context for the vector and the reranker,
        never shown. The session has no stored title, so its first record stands in for one."""
        key = (r['scope'], r['session'])
        if key not in self._heads:
            first = self.con.execute('SELECT text FROM episodes WHERE bank=? AND scope=? AND session=? ORDER BY seq, id '
                                     'LIMIT 1', (r['bank'], r['scope'], r['session'])).fetchone()
            self._heads[key] = ' '.join((first[0] if first else '').split())[:80]
        return ' · '.join((r['scope'].rsplit(':', 1)[-1], self._heads[key], str(r['occurred_at'])[:10], r['role']))

    def widen(self, r: dict, chars: int) -> str:
        """The span grown evenly inside its record to at most `chars`; cuts are marked with '…'."""
        size = r['end'] - r['start']
        if size >= chars:
            return r['text']
        text = self.con.execute('SELECT text FROM episodes WHERE id=?', (r['episode_id'],)).fetchone()[0]
        a = max(0, r['start'] - (chars - size) // 2)
        b = min(len(text), a + chars)
        a = max(0, b - chars)
        return ('… ' if a else '') + text[a:b] + (' …' if b < len(text) else '')

    _until: str | None = None

    @staticmethod
    def _iso(t) -> str:
        return str(t).replace('/', '-').replace(' ', 'T')[:19]

    @contextlib.contextmanager
    def as_of(self, until):
        """Inside this block, searches see only records that happened at or before `until` (a question asked at a
        known time must not see later records; 'as of' recall). None = everything."""
        old, self._until = self._until, (self._iso(until) if until else None)
        try:
            yield self
        finally:
            self._until = old

    def _visible(self, occurred_at) -> bool:
        return self._until is None or self._iso(occurred_at) <= self._until

    def first_spans(self, bank: str, scope: str, episode_ids: list[int]) -> dict[int, int]:
        """episode id -> its first span id (the fact index points at episodes; recall ranks spans)"""
        if not episode_ids:
            return {}
        q = ','.join('?' * len(episode_ids))
        return {r[0]: r[1] for r in self.con.execute(
            f'SELECT s.episode_id, min(s.id), e.occurred_at FROM spans s JOIN episodes e ON e.id = s.episode_id '
            f'WHERE s.bank=? AND s.scope=? AND s.episode_id IN ({q}) GROUP BY s.episode_id', (bank, scope, *episode_ids))
            if self._visible(r[2])}

    def neighbors(self, bank: str, scope: str, session: str, seq: int, radius: int = 1) -> list[dict]:
        return [dict(r) for r in self.con.execute(
            'SELECT id AS episode_id, seq, role, occurred_at, source, text FROM episodes '
            'WHERE bank=? AND scope=? AND session=? AND seq BETWEEN ? AND ? ORDER BY seq',
            (bank, scope, session, seq - radius, seq + radius)) if self._visible(r['occurred_at'])]

    def next_turn(self, bank: str, scope: str, session: str, seq: int) -> list[dict]:
        """The first user or assistant turn after seq in its session (tool records skipped), [] if none is visible."""
        r = self.con.execute(
            'SELECT id AS episode_id, seq, role, occurred_at, source, text FROM episodes WHERE bank=? AND scope=? '
            "AND session=? AND seq > ? AND role IN ('user', 'assistant') ORDER BY seq LIMIT 1",
            (bank, scope, session, seq)).fetchone()
        return [dict(r)] if r is not None and self._visible(r['occurred_at']) else []

    def bm25(self, bank: str, scope: str, query: str, limit: int = 50,
             window: tuple[str, str] | None = None, ids: list[int] | None = None) -> list[tuple[int, float]]:
        """ids: rank only these spans (a filter applied before the limit, e.g. one folder of a document root)"""
        toks = sorted(set(tokenize(query)))
        if not toks:
            return []
        only, extra = ('', ()) if ids is None else (' AND s.id IN (SELECT value FROM json_each(?))',
                                                    (json.dumps(list(ids)),))
        match = ' OR '.join('"' + t.replace('"', '""') + '"' for t in toks)
        if self._until:                       # as-of: only records up to that day (exact time checked below)
            window = (window[0] if window else '0000-00-00', min(window[1], self._until[:10]) if window else self._until[:10])
        if window:   # only spans whose episode happened inside [start, end] (ISO dates, inclusive)
            rows = self.con.execute(
                'SELECT f.rowid AS id, bm25(span_fts) AS score FROM span_fts f JOIN spans s ON s.id = f.rowid '
                'JOIN episodes e ON e.id = s.episode_id WHERE span_fts MATCH ? AND s.bank=? AND s.scope=? '
                "AND replace(substr(e.occurred_at,1,10),'/','-') BETWEEN ? AND ?" + only + ' ORDER BY score LIMIT ?',
                (match, bank, scope, window[0], window[1], *extra, limit)).fetchall()
        else:
            rows = self.con.execute(
                'SELECT f.rowid AS id, bm25(span_fts) AS score FROM span_fts f JOIN spans s ON s.id = f.rowid '
                'WHERE span_fts MATCH ? AND s.bank=? AND s.scope=?' + only + ' ORDER BY score LIMIT ?',
                (match, bank, scope, *extra, limit)).fetchall()
        if self._until and rows:
            ok = {r[0] for r in self.con.execute(
                f"SELECT s.id, e.occurred_at FROM spans s JOIN episodes e ON e.id = s.episode_id WHERE s.id IN "
                f"({','.join('?' * len(rows))})", [r['id'] for r in rows]) if self._visible(r[1])}
            rows = [r for r in rows if r['id'] in ok]
        return [(r['id'], -r['score']) for r in rows]  # FTS5 bm25 is lower-is-better

    def span_meta(self, bank: str, scope: str) -> dict:
        """Per-span position in time: thread = (session, day). Cached until the store changes.
        Returns {'ids': int64[n], 'day': [str], 'thread': [str], 'seq': int64[n], 'user': bool[n],
                 'pos': {span_id: row}, 'opener': {thread: span_id}}; opener = first user span of the thread."""
        key = ('meta', bank, scope)
        v = self._version
        hit = self._dense_cache.get(key)
        if hit and hit[0] == v:
            return hit[1]
        rows = self.con.execute(
            "SELECT s.id, e.session, replace(substr(e.occurred_at,1,10),'/','-') AS day, e.seq, e.role, s.start FROM spans s "
            'JOIN episodes e ON e.id = s.episode_id WHERE s.bank=? AND s.scope=? ORDER BY e.seq, s.start',
            (bank, scope)).fetchall()
        ids = np.array([r[0] for r in rows], dtype=np.int64)
        day = [r[2] for r in rows]
        thread = [f'{r[1]}|{r[2]}' for r in rows]
        user = np.array([r[4] == 'user' for r in rows], dtype=bool)
        opener: dict[str, int] = {}
        lead: dict[str, list[int]] = {}           # first user spans of each thread (where it states its topic)
        for r, th, u in zip(rows, thread, user):
            if u:
                opener.setdefault(th, int(r[0]))
                ls = lead.setdefault(th, [])
                if len(ls) < 6:
                    ls.append(int(r[0]))
        meta = {'ids': ids, 'day': day, 'thread': thread, 'seq': np.array([r[3] for r in rows], dtype=np.int64),
                'user': user, 'pos': {int(i): j for j, i in enumerate(ids)}, 'opener': opener, 'lead': lead}
        self._dense_cache[key] = (v, meta)
        return meta

    _DIRECTIVE = __import__('re').compile(
        r'^\s*(?:please\s+)?(?:always|never|from now on|going forward|make sure (?:to|you)|remember to|'
        r"don't ever|do not ever)\b|\b(?:always|never)\b[^.?!\n]{0,160}\bwhen I\b|"
        r'^\s*(?:请|以后|今后)?(?:总是|一律|务必|永远不要|不要再|每次)', __import__('re').I)

    _NEGATION = __import__('re').compile(
        r"\bI(?:'ve| have)? never\b|\bI(?: have|'ve) not (?:ever|yet)\b|\bI haven't (?:ever|yet)\b|"
        r"\bI(?: did not|'d never| had never|'d not) ever\b|\bnever (?:once )?(?:have|did) I\b|"
        r'我(?:从来)?(?:没有|从没|从未|还没)', __import__('re').I)

    def negations(self, bank: str, scope: str) -> list[int]:
        """Span ids of the user's own "I have never ..." statements (candidates for contradictions). Cached."""
        key = ('neg', bank, scope)
        v = self._version
        hit = self._dense_cache.get(key)
        if hit and hit[0] == v:
            return hit[1]
        out = [int(r[0]) for r in self.con.execute(
            "SELECT s.id, substr(e.text, s.start + 1, s.end - s.start) AS t FROM spans s "
            "JOIN episodes e ON e.id = s.episode_id WHERE s.bank=? AND s.scope=? AND e.role='user'", (bank, scope))
            if self._NEGATION.search(r[1])]
        self._dense_cache[key] = (v, out)
        return out

    def span_texts(self, sids: list[int]) -> dict[int, str]:
        """Text of the given spans."""
        if not sids:
            return {}
        q = ','.join('?' * len(sids))
        return {int(r[0]): r[1] for r in self.con.execute(
            f"SELECT s.id, substr(e.text, s.start + 1, s.end - s.start) FROM spans s JOIN episodes e ON e.id = s.episode_id "
            f"WHERE s.id IN ({q})", [int(x) for x in sids])}

    def directives(self, bank: str, scope: str) -> list[int]:
        """Span ids of standing user instructions ("Always include X when I ask about Y"). Cached. Only records the
        user said (captured turns, ingested conversations), never ones an agent stored through memory_capture."""
        key = ('dir', bank, scope)
        v = self._version
        hit = self._dense_cache.get(key)
        if hit and hit[0] == v:
            return hit[1]
        out = []
        for r in self.con.execute(
                "SELECT s.id, substr(e.text, s.start + 1, s.end - s.start) AS t FROM spans s "
                "JOIN episodes e ON e.id = s.episode_id WHERE s.bank=? AND s.scope=? AND e.role='user' "
                "AND s.end - s.start <= 600 AND (e.source IS NULL OR e.source NOT LIKE ?)",
                (bank, scope, AGENT_SOURCE + '%')):
            if self._DIRECTIVE.search(r[1]):
                out.append(int(r[0]))
        self._dense_cache[key] = (v, out)
        return out

    # ---------- derived vectors ----------
    def put_vector(self, span_id: int, model: str, vec: np.ndarray) -> None:
        with self.lock:
            self.con.execute('INSERT OR REPLACE INTO vectors(span_id,model,vec) VALUES (?,?,?)',
                             (span_id, model, np.asarray(vec, dtype=np.float16).tobytes()))
            self._writes += 1

    def put_vectors(self, model: str, pairs) -> None:
        """many (span_id, vec) in one transaction: one lock and one sync instead of one per vector"""
        rows = [(sid, model, np.asarray(v, dtype=np.float16).tobytes()) for sid, v in pairs]
        with self.lock:
            self.con.execute('BEGIN IMMEDIATE')
            try:
                self.con.executemany('INSERT OR REPLACE INTO vectors(span_id,model,vec) VALUES (?,?,?)', rows)
                self.con.execute('COMMIT')
            except BaseException:
                self.con.execute('ROLLBACK')
                raise
            self._writes += 1

    def put_tokvecs(self, span_id: int, model: str, mat: np.ndarray) -> None:
        m = np.asarray(mat, dtype=np.float16)
        with self.lock:
            self.con.execute('INSERT OR REPLACE INTO tokvecs(span_id,model,n,dim,vec) VALUES (?,?,?,?,?)',
                             (span_id, model, m.shape[0], m.shape[1], m.tobytes()))

    def tokvecs(self, ids: list[int], model: str) -> dict[int, np.ndarray]:
        if not ids:
            return {}
        q = ','.join('?' * len(ids))
        return {r['span_id']: np.frombuffer(r['vec'], dtype=np.float16).reshape(r['n'], r['dim']).astype(np.float32)
                for r in self.con.execute(f'SELECT span_id,n,dim,vec FROM tokvecs WHERE model=? AND span_id IN ({q})',
                                          (model, *ids))}

    def dense_matrix(self, bank: str, scope: str, model: str) -> tuple[np.ndarray, np.ndarray]:
        """(span_ids, matrix) for brute-force cosine search; cached until the store changes."""
        key = (bank, scope, model)
        v = self._version
        hit = self._dense_cache.get(key)
        if hit and hit[0] == v:
            return self._as_of_rows(bank, scope, hit[1], hit[2])
        rows = self.con.execute('SELECT v.span_id, v.vec FROM vectors v JOIN spans s ON s.id = v.span_id '
                                'WHERE s.bank=? AND s.scope=? AND v.model=?', (bank, scope, model)).fetchall()
        ids = np.array([r[0] for r in rows], dtype=np.int64)
        mat = (np.vstack([np.frombuffer(r[1], dtype=np.float16) for r in rows]).astype(np.float32)
               if rows else np.zeros((0, 1), dtype=np.float32))
        self._dense_cache[key] = (v, ids, mat)
        return self._as_of_rows(bank, scope, ids, mat)

    def _as_of_rows(self, bank: str, scope: str, ids: np.ndarray, mat: np.ndarray):
        if not self._until or not len(ids):
            return ids, mat
        key = ('when', bank, scope)
        hit = self._dense_cache.get(key)
        if not (hit and hit[0] == self._version):
            when = dict(self.con.execute('SELECT s.id, e.occurred_at FROM spans s JOIN episodes e ON e.id = s.episode_id '
                                         'WHERE s.bank=? AND s.scope=?', (bank, scope)).fetchall())
            hit = (self._version, {k: self._iso(t) for k, t in when.items()})
            self._dense_cache[key] = hit
        keep = np.array([hit[1].get(int(i), '') <= self._until for i in ids], dtype=bool)
        return ids[keep], mat[keep]

    # ---------- background jobs ----------
    def claim_jobs(self, kind: str, model: str, limit: int = 32, newest_first: bool = False) -> list[dict]:
        order = 'DESC' if newest_first else 'ASC'
        with self.lock:
            # one write transaction: two workers (nightly backfill + pm1-worker) must not claim the same jobs
            self.con.execute('BEGIN IMMEDIATE')
            try:
                rows = self._claim(kind, model, limit, order)
                self.con.execute('COMMIT')
            except BaseException:
                self.con.execute('ROLLBACK')
                raise
        return [{'span_id': r['span_id'], 'text': r['text'][r['start']:r['end']]} for r in rows]

    def _claim(self, kind: str, model: str, limit: int, order: str) -> list:
        rows = self.con.execute(
            # pick the ids on the index first: joining all pending rows to their episode text before the
            # ORDER BY took 12 s per batch with 110K pending jobs (2026-10-18)
            "SELECT j.span_id, e.text, s.start, s.end FROM (SELECT span_id FROM jobs WHERE kind=? AND model=? "
            f"AND state='pending' AND attempts < 3 ORDER BY span_id {order} LIMIT ?) j "
            "JOIN spans s ON s.id = j.span_id JOIN episodes e ON e.id = s.episode_id "
            f"ORDER BY j.span_id {order}", (kind, model, limit)).fetchall()
        ids = [r['span_id'] for r in rows]
        if ids:
            self.con.execute(f"UPDATE jobs SET state='running', attempts=attempts+1 WHERE kind=? AND model=? "
                             f"AND span_id IN ({','.join('?' * len(ids))})", (kind, model, *ids))
        return rows

    def finish_jobs(self, kind: str, model: str, span_ids: list[int], error: str | None = None) -> None:
        if not span_ids:
            return
        state = 'pending' if error else 'done'
        with self.lock:
            self.con.execute(f"UPDATE jobs SET state=?, error=? WHERE kind=? AND model=? "
                             f"AND span_id IN ({','.join('?' * len(span_ids))})",
                             (state, error, kind, model, *span_ids))

    # ---------- forgetting ----------
    def forget_episode(self, bank: str, scope: str, episode_id: int) -> dict:
        """Delete an episode and every derived row; verifies nothing is left."""
        with self.lock:
            sids = [r[0] for r in self.con.execute('SELECT id FROM spans WHERE episode_id=? AND bank=? AND scope=?',
                                                   (episode_id, bank, scope))]
            self.con.execute('BEGIN IMMEDIATE')
            try:
                if sids:
                    q = ','.join('?' * len(sids))
                    self.con.execute(f'DELETE FROM span_fts WHERE rowid IN ({q})', sids)
                n = self.con.execute('DELETE FROM episodes WHERE id=? AND bank=? AND scope=?',
                                     (episode_id, bank, scope)).rowcount
                self.con.execute('COMMIT')
            except BaseException:
                self.con.execute('ROLLBACK')
                raise
            self._writes += 1
            left = 0
            if sids:
                q = ','.join('?' * len(sids))
                for table, col in (('spans', 'id'), ('vectors', 'span_id'), ('tokvecs', 'span_id'),
                                   ('jobs', 'span_id'), ('span_fts', 'rowid')):
                    left += self.con.execute(f'SELECT count(*) FROM {table} WHERE {col} IN ({q})', sids).fetchone()[0]
        if left:
            raise RuntimeError(f'forget left {left} derived rows behind')
        return {'deleted_episodes': n, 'deleted_spans': len(sids), 'verified': True}

    def stats(self, bank: str, scope: str | None = None) -> dict:
        where, args = ('bank=?', [bank]) if scope is None else ('bank=? AND scope=?', [bank, scope])
        one = lambda sql: self.con.execute(sql, args).fetchone()[0]  # noqa: E731
        return {'episodes': one(f'SELECT count(*) FROM episodes WHERE {where}'),
                'spans': one(f'SELECT count(*) FROM spans WHERE {where}'),
                'jobs_pending': one(f"SELECT count(*) FROM jobs j JOIN spans s ON s.id=j.span_id "
                                    f"WHERE j.state IN ('pending','running') AND s.{where.replace(' AND ', ' AND s.')}")}
