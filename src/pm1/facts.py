"""Fact index (layer F): short self-contained facts extracted from a session, each pointing back to the turns it came
from. The facts are a search key, never shown instead of the records: recall matches the question against the facts
and pulls the facts' source turns (the user's own words) into its candidate list.

Why (LoCoMo competitor run, docs/EVALUATION.md E2-E3): the evidence pm1 still missed usually
needs a small inference to connect it to the question ("which European countries" <- "a trip to England"). A fact written
once per session at write time ("Maria visited England (a European country)") makes that link; dictionaries (WordNet,
ConceptNet) and graph / session-vector expansion did not (offline evidence check, competitors/harness/pm1_*_eval.py).

Derived and rebuildable, like every layer except episodes.
"""
from __future__ import annotations

import json
import re
from collections import defaultdict

import numpy as np

# entity graph over facts (pm1-v4): an entity is a capitalised name written in a fact; facts sharing an entity are linked
NAME = re.compile(r"\b([A-Z][a-z]+(?:[ '-][A-Z][a-z]+)*)\b")
NOT_NAMES = frozenset('The A An In On At As By For Of And But May June July August January February March April '
                      'September October November December Monday Tuesday Wednesday Thursday Friday Saturday Sunday '
                      'What When Who How Which Why Where Did Does Do Is Was Would Has Have'.split())


def entities(text: str) -> set:
    return {m for m in NAME.findall(text or '') if m not in NOT_NAMES}

SCHEMA = """
CREATE TABLE IF NOT EXISTS facts (
  id INTEGER PRIMARY KEY, bank TEXT NOT NULL, scope TEXT NOT NULL, session TEXT NOT NULL, text TEXT NOT NULL,
  episodes TEXT NOT NULL, model TEXT, source TEXT, created_at TEXT DEFAULT (strftime('%Y-%m-%dT%H:%M:%fZ','now')));
CREATE INDEX IF NOT EXISTS ix_facts_scope ON facts(bank, scope, session);
CREATE TABLE IF NOT EXISTS fact_vectors (fact_id INTEGER NOT NULL, model TEXT NOT NULL, vec BLOB NOT NULL,
  PRIMARY KEY (fact_id, model));
CREATE TABLE IF NOT EXISTS fact_sessions (bank TEXT NOT NULL, scope TEXT NOT NULL, session TEXT NOT NULL,
  extractor TEXT, n_facts INTEGER, last_seq INTEGER, done_at TEXT, error TEXT, PRIMARY KEY (bank, scope, session));
"""


class FactIndex:
    def __init__(self, store, dense=None):
        self.store, self.con, self.dense = store, store.con, dense
        self.con.executescript(SCHEMA)
        self._cache: dict[tuple, tuple] = {}

    # ---------- write ----------
    def replace_session(self, bank: str, scope: str, session: str, facts: list[dict], *, model: str = '',
                        source: str = '', last_seq: int | None = None, error: str | None = None) -> int:
        """facts: [{'text': str, 'episodes': [episode ids]}]. Replaces the session's facts (re-extraction)."""
        with self.store.lock:
            old = [r[0] for r in self.con.execute('SELECT id FROM facts WHERE bank=? AND scope=? AND session=?',
                                                  (bank, scope, session))]
            if old:
                q = ','.join('?' * len(old))
                self.con.execute(f'DELETE FROM fact_vectors WHERE fact_id IN ({q})', old)
                self.con.execute(f'DELETE FROM facts WHERE id IN ({q})', old)
            n = 0
            for f in facts:
                eps = [int(e) for e in f.get('episodes') or []]
                if not f.get('text') or not eps:
                    continue
                self.con.execute('INSERT INTO facts(bank, scope, session, text, episodes, model, source) '
                                 'VALUES (?,?,?,?,?,?,?)', (bank, scope, session, f['text'], json.dumps(eps), model,
                                                            source))
                n += 1
            self.con.execute('INSERT OR REPLACE INTO fact_sessions(bank, scope, session, extractor, n_facts, last_seq, '
                             "done_at, error) VALUES (?,?,?,?,?,?,strftime('%Y-%m-%dT%H:%M:%fZ','now'),?)",
                             (bank, scope, session, model, n, last_seq, error))
            self.con.commit()
        self._cache.clear()
        return n

    def embed_pending(self, batch: int = 64) -> int:
        if self.dense is None:
            return 0
        rows = self.con.execute('SELECT f.id, f.text FROM facts f LEFT JOIN fact_vectors v ON v.fact_id = f.id AND '
                                'v.model = ? WHERE v.fact_id IS NULL LIMIT ?', (self.dense.name, batch)).fetchall()
        if not rows:
            return 0
        vecs = self.dense.embed([r[1] for r in rows], 'doc')
        with self.store.lock:
            self.con.executemany('INSERT OR REPLACE INTO fact_vectors(fact_id, model, vec) VALUES (?,?,?)',
                                 [(r[0], self.dense.name, np.asarray(v, dtype=np.float16).tobytes())
                                  for r, v in zip(rows, vecs)])
            self.con.commit()
        self._cache.clear()
        return len(rows)

    # ---------- read ----------
    def _matrix(self, bank: str, scope: str):
        key = (bank, scope, self.dense.name)
        n = self.con.execute('SELECT count(*) FROM facts WHERE bank=? AND scope=?', (bank, scope)).fetchone()[0]
        hit = self._cache.get(key)
        if hit and hit[0] == n:
            return hit[1:4]
        rows = self.con.execute('SELECT f.id, f.episodes, v.vec, f.text FROM facts f JOIN fact_vectors v ON v.fact_id = '
                                'f.id AND v.model = ? WHERE f.bank=? AND f.scope=?', (self.dense.name, bank, scope)).fetchall()
        ids = [r[0] for r in rows]
        eps = [json.loads(r[1]) for r in rows]
        mat = (np.vstack([np.frombuffer(r[2], dtype=np.float16) for r in rows]).astype(np.float32)
               if rows else np.zeros((0, 1), dtype=np.float32))
        ents = [entities(r[3]) for r in rows]
        inv = defaultdict(list)
        for j, es in enumerate(ents):
            for e in es:
                inv[e].append(j)
        self._cache[key] = (n, ids, eps, mat, ents, inv)
        return ids, eps, mat

    def episodes_for(self, bank: str, scope: str, qvec: np.ndarray, k: int, *, hop: int = 0,
                     question: str = '', min_sim: float = 0.0) -> list[int]:
        """Source episodes of the k facts nearest to the question, best fact first (deduplicated, in order).
        hop > 0 (entity graph): then up to `hop` more facts that share an entity with those k facts AND with the
        question, nearest first (offline LoCoMo check, held-out c5-c9, 4K: evidence fully in context 0.856 -> 0.873).
        min_sim > 0: only facts at least that similar to the question count (weak matches pull in nothing)."""
        if self.dense is None or k <= 0:
            return []
        ids, eps, mat = self._matrix(bank, scope)
        if not ids:
            return []
        sims = mat @ qvec
        top = [j for j in np.argsort(-sims)[:k] if sims[j] >= min_sim]
        if hop > 0:
            _, _, _, ents, inv = self._cache[(bank, scope, self.dense.name)][1:]
            shared = set().union(*(ents[j] for j in top)) & entities(question)
            cand = {j for e in shared for j in inv[e]} - set(top)
            top += sorted(cand, key=lambda j: -sims[j])[:hop]
        out: list[int] = []
        for j in top:
            out += [e for e in eps[j] if e not in out]
        return out

    def graph_episodes(self, bank: str, scope: str, qvec: np.ndarray, *, hops: int = 2, seeds: int = 16,
                       limit: int = 100, knn: int = 5, knn_min: float = 0.80, common: float = 0.10,
                       decay: float = 0.8) -> list[int]:
        """Graph recall (offline LoCoMo check; docs/EVALUATION.md E5): source episodes of the facts reached
        from the `seeds` facts nearest the question by up to `hops` steps, best first, at most `limit`.
        A step goes to facts sharing a named entity (entities in more than `common` of the scope's facts, e.g. the
        speakers, are skipped) or to the `knn` nearest facts by vector (cosine >= knn_min); a fact reached at step d from
        a seed of similarity s scores s * decay**d. These are candidates for the reranker, never context themselves."""
        if self.dense is None or seeds <= 0:
            return []
        ids, eps, mat = self._matrix(bank, scope)
        if not ids:
            return []
        _, _, _, ents, inv = self._cache[(bank, scope, self.dense.name)][1:]
        skip = {e for e, js in inv.items() if len(js) > common * len(ids)}
        sims = mat @ qvec
        best = {int(j): float(sims[j]) for j in np.argsort(-sims)[:seeds]}
        frontier = dict(best)
        for _ in range(hops):
            nxt: dict[int, float] = {}
            for j, sc in frontier.items():
                row = mat @ mat[j]                       # nearest facts, computed only for reached facts
                row[j] = -1.0
                nb = {int(k) for k in np.argsort(-row)[:knn] if row[k] >= knn_min}
                nb |= {k for e in ents[j] - skip for k in inv[e]}
                v = sc * decay
                for k in nb:
                    if v > best.get(k, -1.0) and v > nxt.get(k, -1.0):
                        nxt[k] = v
            best.update(nxt)
            frontier = nxt
        out: list[int] = []
        for j in sorted(best, key=lambda j: -best[j]):
            out += [e for e in eps[j] if e not in out]
            if len(out) >= limit:
                break
        return out[:limit]

    def count(self, bank: str, scope: str) -> int:
        return self.con.execute('SELECT count(*) FROM facts WHERE bank=? AND scope=?', (bank, scope)).fetchone()[0]
