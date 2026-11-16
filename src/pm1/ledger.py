"""State ledger (layer L, P2): what the background consolidation agent extracted from the raw episodes.

Derived and rebuildable, like every layer except episodes. Each row keeps its provenance (chunk id = session|day|k
and the day it was said). Rows: dated events, state facts with supersession, standing instructions, never-statements.

Supersession: state rows with the same normalized (subject, attribute) key form a chain ordered by day; every row but
the newest gets superseded_by = the next newer row, so recall can say "now X (since D2), previously Y (since D1)".
"""
from __future__ import annotations

import json
import re
from collections import defaultdict

import numpy as np

SCHEMA = """
CREATE TABLE IF NOT EXISTS ledger_event (
  id INTEGER PRIMARY KEY, bank TEXT NOT NULL, scope TEXT NOT NULL, chunk TEXT NOT NULL, session TEXT, day TEXT,
  date TEXT, when_text TEXT, what TEXT NOT NULL, kind TEXT, source TEXT);
CREATE INDEX IF NOT EXISTS ix_lev ON ledger_event(bank, scope);
CREATE TABLE IF NOT EXISTS ledger_state (
  id INTEGER PRIMARY KEY, bank TEXT NOT NULL, scope TEXT NOT NULL, chunk TEXT NOT NULL, session TEXT, day TEXT,
  subject TEXT, attribute TEXT, value TEXT, previous TEXT, evidence TEXT, key TEXT NOT NULL,
  superseded_by INTEGER, source TEXT);
CREATE INDEX IF NOT EXISTS ix_lst ON ledger_state(bank, scope, key);
CREATE TABLE IF NOT EXISTS ledger_note (
  id INTEGER PRIMARY KEY, bank TEXT NOT NULL, scope TEXT NOT NULL, chunk TEXT NOT NULL, session TEXT, day TEXT,
  kind TEXT NOT NULL, text TEXT NOT NULL, evidence TEXT, source TEXT);
CREATE INDEX IF NOT EXISTS ix_lno ON ledger_note(bank, scope, kind);
CREATE TABLE IF NOT EXISTS ledger_mention (
  id INTEGER PRIMARY KEY, bank TEXT NOT NULL, scope TEXT NOT NULL, chunk TEXT NOT NULL, day TEXT,
  name TEXT NOT NULL, type TEXT, aliases TEXT, event_id INTEGER, source TEXT);
CREATE INDEX IF NOT EXISTS ix_lme ON ledger_mention(bank, scope);
"""

_STOP = frozenset('the a an of for to in on my our user users current currently its their his her'.split())


def _stem(w: str) -> str:
    """'gifts' -> 'gift' so 'holiday gifts budget' and 'holiday gift budget' chain together."""
    return w[:-1] if len(w) > 3 and w.endswith('s') and not w.endswith('ss') else w


def norm_key(subject: str, attribute: str) -> str:
    def toks(s):
        return [_stem(w) for w in re.findall(r'[a-z0-9]+', (s or '').lower()) if w not in _STOP]
    return ' '.join(sorted(set(toks(subject)))) + ' | ' + ' '.join(sorted(set(toks(attribute))))


class Ledger:
    def __init__(self, store):
        self.store = store
        self.con = store.con
        self.con.executescript(SCHEMA)
        self._vec: dict[tuple, tuple] = {}

    # ---------- write ----------
    def clear(self, bank: str, scope: str, source: str | None = None) -> None:
        for t in ('ledger_event', 'ledger_state', 'ledger_note', 'ledger_mention'):
            q = f'DELETE FROM {t} WHERE bank=? AND scope=?' + (' AND source=?' if source else '')
            self.con.execute(q, (bank, scope, source) if source else (bank, scope))
        self._vec.clear()

    def add_extraction(self, bank: str, scope: str, chunk: str, day: str, resp: dict, source: str = '') -> int:
        """One consolidation output (the labeling schema) for one chunk. Returns rows written."""
        session = chunk.split('|')[0] if '|' in chunk else ''
        n = 0
        for e in resp.get('events') or []:
            if isinstance(e, dict) and e.get('what'):
                cur = self.con.execute('INSERT INTO ledger_event(bank,scope,chunk,session,day,date,when_text,what,kind,'
                                       'source) VALUES (?,?,?,?,?,?,?,?,?,?)',
                                       (bank, scope, chunk, session, day, e.get('date') or None, e.get('when_text') or '',
                                        str(e['what']), e.get('kind'), source))
                n += 1
                for name in e.get('entities') or []:          # graph schema: event -> entity links
                    if isinstance(name, str) and name.strip():
                        self.con.execute('INSERT INTO ledger_mention(bank,scope,chunk,day,name,event_id,source) '
                                         'VALUES (?,?,?,?,?,?,?)', (bank, scope, chunk, day, name.strip(), cur.lastrowid, source))
        for x in resp.get('entities') or []:                  # graph schema: the chunk's canonical entities
            if isinstance(x, dict) and str(x.get('name') or '').strip():
                al = [a for a in (x.get('aliases') or []) if isinstance(a, str) and a.strip()]
                self.con.execute('INSERT INTO ledger_mention(bank,scope,chunk,day,name,type,aliases,source) '
                                 'VALUES (?,?,?,?,?,?,?,?)', (bank, scope, chunk, day, str(x['name']).strip(),
                                                              x.get('type'), json.dumps(al), source))
        for s in resp.get('state') or []:
            if isinstance(s, dict) and s.get('value') not in (None, ''):
                self.con.execute('INSERT INTO ledger_state(bank,scope,chunk,session,day,subject,attribute,value,previous,'
                                 'evidence,key,source) VALUES (?,?,?,?,?,?,?,?,?,?,?,?)',
                                 (bank, scope, chunk, session, day, str(s.get('subject') or ''),
                                  str(s.get('attribute') or ''), str(s.get('value')), s.get('previous'),
                                  s.get('evidence'), norm_key(s.get('subject'), s.get('attribute')), source))
                n += 1
        for kind, field in (('instruction', 'instructions'), ('negation', 'negations')):
            for x in resp.get(field) or []:
                text = x.get('text') if isinstance(x, dict) else x
                if text:
                    self.con.execute('INSERT INTO ledger_note(bank,scope,chunk,session,day,kind,text,evidence,source) '
                                     'VALUES (?,?,?,?,?,?,?,?,?)',
                                     (bank, scope, chunk, session, day, kind, str(text),
                                      x.get('evidence') if isinstance(x, dict) else None, source))
                    n += 1
        self._vec.clear()
        return n

    def _subject_roots(self, bank, scope) -> dict:
        """normalised entity name -> canonical root, merged through the extractor's aliases (graph schema)."""
        from .graph import nkey
        parent: dict[str, str] = {}

        def find(k):
            while parent.get(k, k) != k:
                k = parent[k]
            return k
        for name, aliases in self.con.execute('SELECT name, aliases FROM ledger_mention WHERE bank=? AND scope=? '
                                              'AND aliases IS NOT NULL', (bank, scope)):
            k = nkey(name)
            parent.setdefault(k, k)
            for a in json.loads(aliases or '[]'):
                ak = nkey(a)
                if ak and ak != k:
                    parent.setdefault(ak, ak)
                    ra, rb = find(k), find(ak)
                    if ra != rb:
                        parent[rb] = ra
        return {k: find(k) for k in parent}

    def resolve(self, bank: str, scope: str) -> int:
        """Recompute supersession chains per (entity, attribute): subjects are merged through the extractor's
        aliases when the graph schema provided them, else by the normalized key. Returns rows superseded."""
        from .graph import nkey
        roots = self._subject_roots(bank, scope)
        rows = self.con.execute('SELECT id, key, day, subject, attribute FROM ledger_state WHERE bank=? AND scope=?',
                                (bank, scope)).fetchall()
        self.con.execute('UPDATE ledger_state SET superseded_by=NULL WHERE bank=? AND scope=?', (bank, scope))
        chains = defaultdict(list)
        for i, k, d, subj, attr in rows:
            sk = nkey(subj)
            ck = (roots[sk] + ' | ' + norm_key('', attr).split(' | ')[1]) if sk in roots else norm_key(subj, attr)
            chains[ck].append((d or '', i))
        n = 0
        for ids in chains.values():
            ids = [i for _d, i in sorted(ids)]
            for a, b in zip(ids, ids[1:]):
                self.con.execute('UPDATE ledger_state SET superseded_by=? WHERE id=?', (b, a))
                n += 1
        return n

    # ---------- read ----------
    def _index(self, bank, scope, dense):
        key = (bank, scope)
        if key not in self._vec:
            st = self.con.execute('SELECT id, day, subject, attribute, value, superseded_by, key FROM ledger_state '
                                  'WHERE bank=? AND scope=?', (bank, scope)).fetchall()
            ev = self.con.execute('SELECT id, day, date, when_text, what, kind FROM ledger_event WHERE bank=? AND scope=?',
                                  (bank, scope)).fetchall()
            st_txt = [f'{r[2]} {r[3]}: {r[4]}' for r in st]
            ev_txt = [r[4] for r in ev]
            sv = dense.embed(st_txt, 'passage') if st_txt else np.zeros((0, 384), np.float32)
            evv = dense.embed(ev_txt, 'passage') if ev_txt else np.zeros((0, 384), np.float32)
            self._vec[key] = (st, sv, ev, evv)
        return self._vec[key]

    def state_lines(self, bank, scope, qv, dense, k=6, min_sim=0.80) -> list[str]:
        """Current value of the k state facts closest to the query, each with its history chain."""
        st, sv, _, _ = self._index(bank, scope, dense)
        if not st:
            return []
        sims = sv @ qv
        by_id = {r[0]: r for r in st}
        chains = defaultdict(list)
        for r in st:
            chains[r[6]].append(r)
        out, keys = [], []
        for j in np.argsort(-sims):
            if sims[j] < min_sim or len(keys) >= k:
                break
            kkey = st[j][6]
            if kkey in keys:
                continue
            keys.append(kkey)
            chain = sorted(chains[kkey], key=lambda r: (r[1] or '', r[0]))
            cur = chain[-1]
            line = f'- {cur[2]} — {cur[3]}: {cur[4]} (as of {cur[1]})'
            if len(chain) > 1:
                line += '; earlier: ' + '; '.join(f'{r[4]} ({r[1]})' for r in reversed(chain[:-1][-3:]))
            out.append(line)
        return out

    def event_lines(self, bank, scope, qv, dense, k=12, min_sim=0.80) -> list[str]:
        """The k events closest to the query, in date order (resolved date, else the day it was mentioned)."""
        _, _, ev, evv = self._index(bank, scope, dense)
        if not ev:
            return []
        sims = evv @ qv
        top = [j for j in np.argsort(-sims)[:k] if sims[j] >= min_sim]
        rows = sorted((ev[j] for j in top), key=lambda r: (r[2] or r[1] or '', r[0]))
        return [f"- {r[2] or r[1]}{' (' + r[3] + ')' if r[3] else ''}: {r[4]}" for r in rows]

    def evolution_lines(self, bank, scope, query, qv, dense, k=3, min_sim=0.83, max_steps=4) -> list[str]:
        """Query-matched value histories ('match' mode): resolved supersession chains whose value really changed,
        closest to the query (embedding) AND sharing a content word with it; oldest -> newest. [] when none match."""
        st, sv, _, _ = self._index(bank, scope, dense)
        if not st:
            return []
        sims = sv @ qv
        row = {r[0]: (j, r) for j, r in enumerate(st)}
        nxt = {r[0]: r[5] for r in st if r[5] is not None}
        heads = [r[0] for r in st if r[0] not in set(nxt.values())]
        qw = {_stem(w) for w in re.findall(r'[a-z0-9]+', query.lower()) if w not in _STOP and len(w) > 2}
        cands = []
        for h in heads:
            ids, i = [], h
            while i is not None and i in row and len(ids) < 200:
                ids.append(i); i = nxt.get(i)
            steps = []
            for i in ids:
                v = (row[i][1][4] or '').strip()
                if not steps or v.lower() != steps[-1][0].lower():
                    steps.append((v, row[i][1][1]))
            if len(steps) < 2:
                continue
            r0 = row[ids[-1]][1]
            words = {_stem(w) for w in re.findall(r'[a-z0-9]+', f'{r0[2]} {r0[3]}'.lower()) if w not in _STOP}
            if not qw & words:
                continue
            s = max(sims[row[i][0]] for i in ids)
            if s >= min_sim:
                cands.append((s, r0, steps))
        cands.sort(key=lambda c: -c[0])
        out = []
        for _s, r0, steps in cands[:k]:
            sh = steps[-max_steps:]
            out.append(f'- {r0[2]} — {r0[3]}: ' + ' → '.join(f'{v[:140]} ({d})' for v, d in sh)
                       + (f'  [{len(steps) - len(sh)} earlier values omitted]' if len(steps) > len(sh) else ''))
        return out

    def update_notes(self, bank, scope) -> dict:
        """(session, day) -> notes for state values said there that were later replaced, e.g.
        'memory note: Dr. Kaya — role was later updated to "cardiology coordinator" (2024-03-01)'. Used to annotate
        raw records inline (knowledge update / contradiction) instead of adding a separate ledger block."""
        key = ('notes', bank, scope)
        if key not in self._vec:
            rows = self.con.execute(
                'SELECT a.session, a.day, a.subject, a.attribute, a.value, b.value, b.day FROM ledger_state a '
                'JOIN ledger_state b ON b.id = a.superseded_by WHERE a.bank=? AND a.scope=?', (bank, scope)).fetchall()
            out = defaultdict(list)
            for sess, day, subj, attr, old, new, nday in rows:
                if (old or '').strip().lower() == (new or '').strip().lower() or nday == day:
                    continue                   # same value restated, or a same-day correction
                out[(sess, (day or '')[:10])].append(f'{subj} — {attr}: "{old}" was later updated to "{new}" ({nday})')
            self._vec[key] = dict(out)
        return self._vec[key]

    def stats(self, bank, scope) -> dict:
        q = lambda t, w='': self.con.execute(f'SELECT count(*) FROM {t} WHERE bank=? AND scope=? {w}', (bank, scope)).fetchone()[0]
        return {'events': q('ledger_event'), 'state': q('ledger_state'),
                'superseded': q('ledger_state', 'AND superseded_by IS NOT NULL'), 'notes': q('ledger_note')}


def load_labels(path) -> list[dict]:
    out = []
    for l in open(path, encoding='utf-8'):
        if l.strip():
            r = json.loads(l)
            v = r.get('response')
            if isinstance(v, str):
                try:
                    v = json.loads(v)
                except ValueError:
                    v = {}
            out.append({'id': r['id'], 'response': v or {}})
    return out
