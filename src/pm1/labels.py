"""A2 in search_sessions (label deployment test, off by default): records found through lab2 label + value entries, added after
pm1's packed result.

PM1_SESS_LABELS=off (default) | shadow | on. The entries come from scripts/label_pass.py (labels of every record by the
frozen lab2 labeler, embedded with the c tower as 'passage: ' + fact text) and live under PM1_LABELS_DIR:
  entries.npz        per entry: first/last offsets into rec, the record ids (rec), date (latest record, seconds),
                     scope / host codes (from its first record), mode (0 chat chunk, 1 doc tail), vocab
  entries.c.npy      float32 vectors, one row per entry, memory-mapped read-only (shared by every pm1-mcp process)
Per call: the query is embedded by the c tower server (PM1_SESS_LABELS_EMB, 'query: ' + text), entries are masked by
the call's filters, the top PM1_SESS_LABELS_TOP (192) give their records in rank order, and records not already shown
are added while the total stays within PM1_SESS_LABELS_X (512) tokens beyond pm1's result (addonly.take_more: a record
that does not fit is skipped, later ones may still fit). Long records show the part their entry was labeled from:
chat chunk -> the first 600 chars, doc tail -> chars 600-2,600, cut to the room left (at least MIN_PART tokens).
"""
from __future__ import annotations

import os
import re
import time
from pathlib import Path

from .recall import est_tokens

DIR = Path(os.environ.get('PM1_LABELS_DIR', Path.home() / '.local/state/pm1/labels'))
CHAT_PART, DOC_PART = (0, 600), (600, 2600)     # what lab2 read: turns cut to 600 chars; doc tail 600 .. 2,600


def mode() -> str:
    m = os.environ.get('PM1_SESS_LABELS', 'off').strip().lower()
    return m if m in ('shadow', 'on') else 'off'


class LabelIndex:
    def __init__(self, directory: Path | None = None, url: str | None = None) -> None:
        import numpy as np
        self.dir = Path(directory or os.environ.get('PM1_LABELS_DIR', DIR))
        self.url = (url or os.environ.get('PM1_SESS_LABELS_EMB', 'http://127.0.0.1:18197')).rstrip('/')
        self.x = int(os.environ.get('PM1_SESS_LABELS_X', '512'))
        self.top = int(os.environ.get('PM1_SESS_LABELS_TOP', '192'))
        self.wait_s = float(os.environ.get('PM1_SESS_LABELS_MS', '400')) / 1e3
        self.min_part = int(os.environ.get('PM1_SESS_MIN_PART', '100'))
        z = np.load(self.dir / 'entries.npz', allow_pickle=False)
        self.lo, self.hi, self.rec = z['lo'], z['hi'], z['rec']
        self.at, self.scope, self.host, self.mode = z['at'], z['scope'], z['host'], z['mode']
        self.scopes = {s: i for i, s in enumerate(z['scope_names'].tolist())}
        self.hosts = {s: i for i, s in enumerate(z['host_names'].tolist())}
        self.mat = np.load(self.dir / 'entries.c.npy', mmap_mode='r')
        assert len(self.mat) == len(self.at), 'entries.npz and entries.c.npy differ in length'
        self.down_until = 0.0

    # ---- query side ----
    def embed(self, query: str):
        import numpy as np
        import requests
        if time.time() < self.down_until:
            raise TimeoutError('c tower skipped: it failed less than a minute ago')
        try:
            r = requests.post(self.url + '/embedding', json={'content': 'query: ' + query}, timeout=self.wait_s)
            r.raise_for_status()
        except Exception:
            self.down_until = time.time() + 60
            raise
        j = r.json()
        v = j[0]['embedding'] if isinstance(j, list) else j['embedding']
        v = np.asarray(v[0] if v and isinstance(v[0], list) else v, np.float32)
        return v / (np.linalg.norm(v) or 1.0)

    def ranked(self, qv, *, scopes_in=None, scopes_out=(), host=None, since=None, until=None) -> list[tuple[int, int]]:
        """(record id, entry mode) in entry rank order, records of one entry in id order, each record once."""
        import numpy as np
        from .sessindex import secs
        mask = np.ones(len(self.at), bool)
        if scopes_in is not None:
            mask &= np.isin(self.scope, [self.scopes[s] for s in scopes_in if s in self.scopes])
        if scopes_out:
            mask &= ~np.isin(self.scope, [self.scopes[s] for s in scopes_out if s in self.scopes])
        if host:
            mask &= np.isin(self.host, [self.hosts[h] for h in (host,) if h in self.hosts])
        hi_t = secs([until + ('T23:59:59' if len(until) == 10 else '')])[0] if until else int(time.time())
        mask &= self.at <= hi_t
        if since:
            mask &= self.at >= secs([since])[0]
        idx = np.flatnonzero(mask)
        if not len(idx):
            return []
        s = np.asarray(self.mat[idx] @ qv) if len(idx) < len(self.at) // 4 else np.asarray(self.mat @ qv)[idx]
        n = min(self.top, len(idx))
        part = np.argpartition(-s, n - 1)[:n] if n < len(idx) else np.arange(len(idx))
        top = idx[part[np.argsort(-s[part], kind='stable')]]
        out, seen = [], set()
        for e in top:
            for r in sorted(self.rec[self.lo[e]:self.hi[e]].tolist()):
                if r not in seen:
                    seen.add(r)
                    out.append((r, int(self.mode[e])))
        return out

    # ---- packing ----
    def take(self, con, ranked: list[tuple[int, int]], have: set, seen_text: set, roles: list[str], used: int,
             item_cost: int, hit_of) -> tuple[list[dict], int]:
        """Records in rank order while the added total stays within X tokens beyond `used` (addonly.take_more)."""
        out, budget = [], used + self.x
        roles = set(roles)
        for rid, m in ranked:
            if rid in have:
                continue
            r = con.execute('SELECT id, scope, session, role, occurred_at, text, source FROM episodes WHERE id=?',
                            (rid,)).fetchone()
            if r is None or r['role'] not in roles:
                continue
            key = re.sub(r'\s+', ' ', r['text']).strip()
            if key in seen_text:
                continue
            room = budget - used - item_cost
            text = r['text']
            if est_tokens(text) > room or len(text) > 1500:
                a, b = DOC_PART if m == 1 else CHAT_PART
                text = text[a:b] if len(text) > a else text[:b]
                while text and est_tokens(text) > room:
                    text = text[:int(len(text) * 0.85)]
                if est_tokens(text) < min(self.min_part, est_tokens(r['text'])):
                    continue
            cost = est_tokens(text) + item_cost
            if used + cost > budget:
                continue
            used += cost
            have.add(rid)
            seen_text.add(key)
            out.append(hit_of(r, text))
        return out, used


SHADOW_COLS = ('ts', 'qsha', 'lang', 'filters', 'head', 'head_tokens', 'a2', 'a2_tokens', 'ctl', 'ctl_tokens',
               'n_ranked', 'embed_ms', 'scan_ms', 'pack_ms', 'error')


def shadow_log(row: dict, path: Path | None = None) -> None:
    """One shadow row per search_sessions call: ids, token counts and timings only, never query or record text
    (the query is joined back to mcp_calls by its sha1 and time)."""
    import sqlite3
    path = Path(path or os.environ.get('PM1_LABELS_SHADOW')
                or Path(os.environ.get('PM1_LABELS_DIR', DIR)) / 'labels_shadow.sqlite')
    con = sqlite3.connect(path, timeout=10)
    try:
        con.execute(f"CREATE TABLE IF NOT EXISTS shadow (id INTEGER PRIMARY KEY, {', '.join(SHADOW_COLS)})")
        con.execute(f"INSERT INTO shadow ({', '.join(SHADOW_COLS)}) VALUES ({', '.join('?' * len(SHADOW_COLS))})",
                    [row.get(c) for c in SHADOW_COLS])
        con.commit()
    finally:
        con.close()


def stamp(directory: Path | None = None) -> int | None:
    """Change marker of the entry index (the nightly rebuild replaces entries.npz last): processes reload on change."""
    try:
        return os.stat(Path(directory or os.environ.get('PM1_LABELS_DIR', DIR)) / 'entries.npz').st_mtime_ns
    except OSError:
        return None
