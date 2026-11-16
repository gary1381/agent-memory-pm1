"""Hybrid retrieval over memory-tree items (design doc §2): BM25 + dense -> RRF -> reranker, the same recipe as
for raw records, applied to Tree.items() (version chains with history, events, relations, loops, digests, roll-ups).

Used by Recaller(tree_factory=...) to add a compact 'From the memory tree' block above the raw records, and by the
tree's own search op once a dense embedder is available.
"""
from __future__ import annotations

import math
import re
from collections import Counter

import numpy as np

_TOK = re.compile(r'[a-z0-9]+(?:[.,][0-9]+)*')
_STOP = frozenset('the a an of for to in on my our is was are be and or with at by it this that i me you'.split())


def toks(s: str) -> list[str]:
    return [w for w in _TOK.findall((s or '').lower()) if w not in _STOP]


class TreeIndex:
    def __init__(self, items: list[dict], dense=None):
        self.items = items
        self.docs = [toks(f"{it['layer']} {it['text']}") for it in items]
        self.df = Counter(w for d in self.docs for w in set(d))
        self.avg = sum(map(len, self.docs)) / max(1, len(self.docs))
        self.dense = dense
        self.vec = dense.embed([it['text'] for it in items], 'passage') if dense is not None and items else None

    def bm25(self, query: str, k1=1.2, b=0.75) -> np.ndarray:
        n = len(self.docs)
        q = toks(query)
        out = np.zeros(n, np.float32)
        for j, d in enumerate(self.docs):
            tf = Counter(d)
            for w in q:
                if w in tf:
                    idf = math.log(1 + (n - self.df[w] + 0.5) / (self.df[w] + 0.5))
                    out[j] += idf * tf[w] * (k1 + 1) / (tf[w] + k1 * (1 - b + b * len(d) / max(1e-9, self.avg)))
        return out

    def search(self, query: str, qv=None, k: int = 4, depth: int = 30, min_sim: float = 0.0, layers=None,
               rerank=None) -> list[tuple[dict, float]]:
        """-> [(item, dense_sim)] best first. rerank(query, texts) -> order (indices) or None."""
        if not self.items:
            return []
        keep = [j for j, it in enumerate(self.items) if not layers or any(it['layer'].startswith(l) for l in layers)]
        bm = self.bm25(query)
        lex = sorted((j for j in keep if bm[j] > 0), key=lambda j: -bm[j])[:depth]
        sims = (self.vec @ qv) if (self.vec is not None and qv is not None) else np.zeros(len(self.items))
        den = sorted(keep, key=lambda j: -sims[j])[:depth] if self.vec is not None and qv is not None else []
        score = {}
        for r in (lex, den):
            for pos, j in enumerate(r):
                score[j] = score.get(j, 0.0) + 1.0 / (60 + pos)
        cand = sorted(score, key=lambda j: -score[j])[:depth]
        if rerank is not None and len(cand) > 1:
            order = rerank(query, [self.items[j]['text'] for j in cand])
            if order:
                cand = [cand[i] for i in order]
        out = [(self.items[j], float(sims[j])) for j in cand if self.vec is None or sims[j] >= min_sim]
        return out[:k]
