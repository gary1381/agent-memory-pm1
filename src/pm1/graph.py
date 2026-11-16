"""Memory graph (P2, 2026-10-15): the ledger reorganised around entities, for navigation by the answering model.

Built from the ledger tables (derived and rebuildable):
  entity  : canonical name + type, merged across chunks by normalised name and aliases (union-find)
  attribute chains : ledger_state rows whose subject resolves to the entity, grouped by key, oldest -> newest
  events  : ledger_event rows linked to the entity (event entity links, or subject/name match)
  pointers: every row keeps its chunk id (session|day|k) -> the raw records it came from

The graph is navigation and annotation only; raw records stay authoritative. Tools (see mcp.py / eval harness):
  map(query)            a few relevant nodes, one or two lines each, with "N earlier values / M events" hints
  expand(node)          attribute chains + events of one node, each with its chunk pointer
  timeline(node|window) events in date order
"""
from __future__ import annotations

import json
import re
from collections import defaultdict

import numpy as np

_STOP = frozenset('the a an of for to in on my our user users dr mr mrs ms'.split())


def nkey(name: str) -> str:
    return ' '.join(w for w in re.findall(r'[a-z0-9]+', (name or '').lower()) if w not in _STOP)


class Graph:
    def __init__(self, store, bank: str, scope: str):
        self.con, self.bank, self.scope = store.con, bank, scope
        self.nodes: list[dict] = []
        self._vec = None
        self.build()

    # ---------- build ----------
    def build(self):
        q = lambda sql: self.con.execute(sql, (self.bank, self.scope)).fetchall()
        ments = q('SELECT name, type, aliases, event_id, day FROM ledger_mention WHERE bank=? AND scope=?')
        parent: dict[str, str] = {}

        def find(k):
            while parent.get(k, k) != k:
                parent[k] = parent.get(parent[k], parent[k])
                k = parent[k]
            return k

        def union(a, b):
            ra, rb = find(a), find(b)
            if ra != rb:
                parent[rb] = ra

        names: dict[str, dict] = {}
        for name, typ, aliases, _ev, day in ments:
            k = nkey(name)
            if not k:
                continue
            parent.setdefault(k, k)
            d = names.setdefault(k, {'names': defaultdict(int), 'types': defaultdict(int), 'days': []})
            d['names'][name] += 1
            if typ:
                d['types'][typ] += 1
            if day:
                d['days'].append(day)
            for a in json.loads(aliases) if aliases else []:
                ak = nkey(a)
                if ak and len(ak) > 2:
                    parent.setdefault(ak, ak)
                    names.setdefault(ak, {'names': defaultdict(int), 'types': defaultdict(int), 'days': []})
                    union(k, ak)
        groups: dict[str, list[str]] = defaultdict(list)
        for k in names:
            groups[find(k)].append(k)
        key2node: dict[str, int] = {}
        self.nodes = []
        for root, ks in groups.items():
            nm = defaultdict(int); ty = defaultdict(int); days = []
            for k in ks:
                for n, c in names[k]['names'].items():
                    nm[n] += c
                for t, c in names[k]['types'].items():
                    ty[t] += c
                days += names[k]['days']
            if not nm:
                continue
            node = {'id': len(self.nodes), 'name': max(nm, key=lambda n: (nm[n], len(n))), 'keys': set(ks),
                    'type': max(ty, key=ty.get) if ty else 'other', 'mentions': sum(nm.values()),
                    'first': min(days) if days else None, 'last': max(days) if days else None,
                    'aliases': sorted(set(nm) - {max(nm, key=lambda n: (nm[n], len(n)))}),
                    'events': set(), 'states': []}
            for k in ks:
                key2node[k] = node['id']
            self.nodes.append(node)
        user = {'id': len(self.nodes), 'name': 'User', 'keys': {'user'}, 'type': 'person', 'mentions': 0,
                'first': None, 'last': None, 'aliases': [], 'events': set(), 'states': []}
        self.nodes.append(user)
        key2node['user'] = key2node[''] = user['id']
        for name, _t, _a, ev, _d in ments:
            if ev is not None and nkey(name) in key2node:
                self.nodes[key2node[nkey(name)]]['events'].add(ev)
        # state subjects -> nodes (exact normalised name, else the longest node key contained in the subject)
        longkeys = sorted((k for k in key2node if k), key=len, reverse=True)
        for sid, subj in q('SELECT id, subject FROM ledger_state WHERE bank=? AND scope=?'):
            k = nkey(subj)
            nid = key2node.get(k)
            if nid is None:
                nid = next((key2node[lk] for lk in longkeys if len(lk) > 3 and re.search(rf'\b{re.escape(lk)}\b', k)),
                           user['id'])
            self.nodes[nid]['states'].append(sid)
        self._vec = None

    # ---------- read ----------
    def _cards(self):
        out = []
        for n in self.nodes:
            st = self._states(n, latest_only=True)[:3]
            out.append(f"{n['name']} ({n['type']}): " + '; '.join(f"{a}: {v}" for a, v, _d, _h in st))
        return out

    def _states(self, n, latest_only=False):
        if not n['states']:
            return []
        rows = self.con.execute(f"SELECT id, key, attribute, value, day, superseded_by, chunk FROM ledger_state "
                                f"WHERE id IN ({','.join('?' * len(n['states']))})", n['states']).fetchall()
        chains = defaultdict(list)          # per node and attribute (subject spellings differ across chunks)
        for r in rows:
            chains[nkey(r[2]) or r[1]].append(r)
        out = []
        for key, ch in chains.items():
            ch.sort(key=lambda r: (r[4] or '', r[0]))
            cur = ch[-1]
            hist = [(r[3], r[4], r[6]) for r in ch[:-1]]
            out.append((cur[2], cur[3], cur[4], hist if not latest_only else len(hist)))
        out.sort(key=lambda x: x[2] or '', reverse=True)
        return out

    def map(self, query: str, qv=None, dense=None, k: int = 5, min_sim: float = 0.82) -> list[str]:
        """Nodes relevant to the query: named in it, or close to it in embedding space."""
        ql = nkey(query)
        hit = [n['id'] for n in self.nodes if n['name'] != 'User' and any(
            len(key) > 3 and re.search(rf'\b{re.escape(key)}\b', ql) for key in n['keys'])]
        if dense is not None and qv is not None:
            if self._vec is None:
                self._vec = dense.embed(self._cards(), 'passage')
            sims = self._vec @ qv
            for j in np.argsort(-sims):
                if len(hit) >= k or sims[j] < min_sim:
                    break
                if j not in hit and self.nodes[j]['name'] != 'User':
                    hit.append(int(j))
        lines = []
        for nid in hit[:k]:
            n = self.nodes[nid]
            st = self._states(n, latest_only=True)
            cur = '; '.join(f"{a}: {v} (as of {d})" for a, v, d, _h in st[:3])
            earlier = sum(h for *_x, h in st)
            lines.append(f"[node {nid}] {n['name']} ({n['type']}"
                         + (f", also: {', '.join(n['aliases'][:3])}" if n['aliases'] else '') + ')'
                         + (f" — {cur}" if cur else '')
                         + f" | {len(st)} attributes, {earlier} earlier values, {len(n['events'])} events")
        return lines

    def expand(self, nid: int, max_events: int = 25) -> str:
        if not 0 <= nid < len(self.nodes):
            return f'(no node {nid})'
        n = self.nodes[nid]
        out = [f"[node {nid}] {n['name']} ({n['type']}); first mentioned {n['first']}, last {n['last']}"]
        for a, v, d, hist in self._states(n):
            line = f"- {a}: {v} (since {d})"
            if hist:
                line += '; earlier: ' + '; '.join(f"{hv} ({hd})" for hv, hd, _c in reversed(hist))
            out.append(line)
        out += self._event_lines(sorted(n['events']), max_events)
        return '\n'.join(out)

    def timeline(self, nid: int | None = None, start: str | None = None, end: str | None = None,
                 max_events: int = 40) -> str:
        if nid is not None and 0 <= nid < len(self.nodes):
            ids = sorted(self.nodes[nid]['events'])
        else:
            ids = [r[0] for r in self.con.execute('SELECT id FROM ledger_event WHERE bank=? AND scope=?',
                                                  (self.bank, self.scope))]
        return '\n'.join(self._event_lines(ids, max_events, start, end)) or '(no events)'

    def _event_lines(self, ids, max_events, start=None, end=None):
        if not ids:
            return []
        rows = self.con.execute(f"SELECT id, COALESCE(date, day), when_text, what, chunk FROM ledger_event "
                                f"WHERE id IN ({','.join('?' * len(ids))})", ids).fetchall()
        rows = [r for r in rows if (not start or (r[1] or '') >= start) and (not end or (r[1] or '') <= end)]
        rows.sort(key=lambda r: (r[1] or '', r[0]))
        out = [f"- {r[1]}{' (' + r[2] + ')' if r[2] else ''}: {r[3]} [src {r[4]}]" for r in rows[-max_events:]]
        if len(rows) > max_events:
            out.insert(0, f"(showing the latest {max_events} of {len(rows)} events)")
        return out
