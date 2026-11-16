"""Online recall: lexical + dense first stage, RRF, optional late-interaction rerank, evidence packing.

No generative model is called. Modes:
  lexical : BM25 only          dense : e5 only
  rrf     : BM25 + e5 fused    hybrid: rrf top-K reranked by pplx MaxSim
  routed  : rrf top-K reranked by a language-routed cross-encoder / late-interaction model (rerank.Router),
            chosen per query by language and latency budget; default when a router is configured
"""
from __future__ import annotations

import math
import os
import re
import time
from collections import defaultdict
from contextlib import contextmanager
from dataclasses import dataclass, field

import numpy as np

from .embed import DenseEmbedder, TokenEmbedder, maxsim
from .lang import detect
from .paths import env_float, env_int
from .plan import Plan, plan as make_plan
from .rerank import PROFILES, Router
from .store import Store
from .text import tokenize

_CJK = re.compile(r'[㐀-鿿]')


def est_tokens(text: str) -> int:
    cjk = len(_CJK.findall(text))
    return cjk + (len(text) - cjk) // 4 + 1


_SENT = re.compile(r'(?<=[.!?。！？])\s+')


_PIECE = re.compile(r'[^.!?。！？；;\n]*(?:[.!?。！？；;\n]+\s*|$)')


def focus_excerpt(text: str, max_tokens: int, query: str = '') -> tuple[str, int]:
    """(excerpt, its offset in text): at most max_tokens (est_tokens, the same count the budget uses) around the
    sentence sharing most words with the query, growing forward then backward a sentence at a time; the leading
    sentences when no sentence shares a word. Cuts are marked with '…'. Unlike excerpt() this keeps the evidence the
    reranker scored instead of the start of the text, and does not overrun the budget on CJK text (one token per
    character, where max_chars = 3 x tokens let a Chinese excerpt run to 2.5x its budget: audit 10-27)."""
    if est_tokens(text) <= max_tokens:
        return text, 0
    from .text import tokenize
    budget = max(8, max_tokens - 2)                       # the cut marks
    pieces = [(m.start(), m.end()) for m in _PIECE.finditer(text) if m.end() > m.start()]
    q = set(tokenize(query)) if query else set()
    hits = [len(q & set(tokenize(text[a:b]))) for a, b in pieces] if q else [0] * len(pieces)
    best = max(range(len(pieces)), key=lambda i: (hits[i], -i)) if pieces else 0
    if not pieces or est_tokens(text[pieces[best][0]:pieces[best][1]]) > budget:
        a, b = pieces[best] if pieces else (0, len(text))
        if q and hits and hits[best]:                     # one long sentence: start a little before the first match
            low = text.lower()
            pos = min((p for p in (low.find(t, a, b) for t in q) if p >= 0), default=a)
            a = max(a, pos - 80)
        lo, hi = a + 1, len(text)                         # the longest end that still fits
        while lo < hi:
            mid = (lo + hi + 1) // 2
            lo, hi = (mid, hi) if est_tokens(text[a:mid]) <= budget else (lo, mid - 1)
        start, end = a, lo
    else:
        i = j = best
        used = est_tokens(text[pieces[i][0]:pieces[j][1]])
        while True:
            grown = False
            for nxt in (j + 1, i - 1):                    # what follows a point usually explains it
                if 0 <= nxt < len(pieces) and not (i <= nxt <= j):
                    a, b = (pieces[i][0], pieces[nxt][1]) if nxt > j else (pieces[nxt][0], pieces[j][1])
                    cost = est_tokens(text[a:b])
                    if cost <= budget:
                        i, j, used, grown = min(i, nxt), max(j, nxt), cost, True
            if not grown:
                break
        start, end = pieces[i][0], pieces[j][1]
    out = text[start:end].strip()
    return ('… ' if start > 0 else '') + out + (' …' if end < len(text) else ''), start


def excerpt(text: str, max_chars: int) -> str:
    """Leading sentences up to max_chars (a turn usually states its point first)."""
    if len(text) <= max_chars:
        return text
    cut = max((m.start() for m in _SENT.finditer(text, 0, max_chars) if m.start() > max_chars // 3), default=max_chars)
    return text[:cut].rstrip() + ' …'


def rrf(rankings: list[list[int]], k: int = 60) -> list[int]:
    score: dict[int, float] = defaultdict(float)
    for ranking in rankings:
        for r, i in enumerate(ranking):
            score[i] += 1 / (k + r + 1)
    return sorted(score, key=lambda i: (-score[i], i))


@dataclass
class RecallResult:
    items: list[dict]
    context: str
    mode: str
    diagnostics: dict = field(default_factory=dict)


class Recaller:
    def __init__(self, store: Store, *, dense: DenseEmbedder | None = None, token: TokenEmbedder | None = None,
                 k_first: int = 50, k_rerank: int = 32, encode_missing: int = 8,
                 router: Router | None = None, k_route: int = 16, profile: str = 'balanced',
                 planning: bool = True, n_directives: int = 3, timeline_chars: int = 700,
                 multi_cap: int = 99, multi_weight: float = 1.5, multi_user_bias: float = 0.5,
                 multi_chars: int = 700, n_conflicts: int = 2, conflict_query_sim: float = 0.88,
                 directive_min_sim: float | None = None, directive_keep_top: bool = True, ledger=None, ledger_state_k: int = 6,
                 ledger_event_k: int = 10, ledger_min_sim: float = 0.80, anchor_k: int = 8,
                 ledger_mode: str = 'block', tree_factory=None, tree_k: int = 4, tree_min_sim: float = 0.80,
                 facts=None):
        self.store, self.dense, self.token = store, dense, token
        self.anchor_k = anchor_k        # matches taken from the date window a relative phrase points at (0 = off)
        self.k_first, self.k_rerank, self.encode_missing = k_first, k_rerank, encode_missing
        self.grep_channel = env_int('PM1_GREP_CHANNEL', 0)   # literal-match channel in the first-stage fusion (test, off)
        self.router, self.k_route, self.profile = router, env_int('PM1_RERANK_K', k_route), profile   # rerank depth
        self.planning, self.n_directives, self.timeline_chars = planning, n_directives, timeline_chars
        # chunking test (docs/EVALUATION.md, not adopted), all off: spans per record, chars shown around a hit, header for rerank
        self.parts, self.return_chars = env_int('PM1_RECALL_PARTS', 1), env_int('PM1_RECALL_RETURN_CHARS', 0)
        self.rerank_header = env_int('PM1_RERANK_HEADER', 0)
        self.multi_cap, self.multi_weight = multi_cap, multi_weight
        self.multi_user_bias, self.multi_chars = multi_user_bias, multi_chars
        self.n_conflicts, self.conflict_query_sim = n_conflicts, conflict_query_sim
        # gate standing instructions by similarity of the query to the instruction's trigger topic
        # ("... when I ask about <topic>"): top-1 always, more only above the threshold; None = top n_directives by
        # similarity to the whole instruction (BEAM-1M: 3 mostly irrelevant instructions on every question)
        self.directive_min_sim = directive_min_sim
        # keep the best-matching instruction even below the threshold (benchmarks: instruction-following questions);
        # False for agents, where an unrelated instruction on every recall is noise
        self.directive_keep_top = directive_keep_top
        self._topiccache: dict[tuple, tuple[list[int], np.ndarray]] = {}
        # P2 state ledger (pm1.ledger.Ledger): extracted state with supersession + dated events, shown as a
        # compact block above the raw records (which stay authoritative)
        self.ledger, self.ledger_state_k, self.ledger_event_k = ledger, ledger_state_k, ledger_event_k
        self.ledger_min_sim = ledger_min_sim
        # 'block': a ledger section above the records (ablated 2026-10-14: knowledge update up, abstention down);
        # 'annotate': no section, inline notes on raw records whose stated value was later replaced
        self.ledger_mode = ledger_mode
        # 'match': only value histories that changed and match the question (2026-10-15, after B100K diagnosis)
        self.match_k = env_int('PM1_MATCH_K', 3)
        self.match_min_sim = env_float('PM1_MATCH_MIN_SIM', 0.83)
        # memory tree v2 (pm1.tree): tree_factory(bank, scope) -> Tree | None; its items go through the same hybrid
        # recipe (pm1.tree_index) and the best few are shown as a compact block above the raw records
        self.tree_factory, self.tree_k, self.tree_min_sim = tree_factory, tree_k, tree_min_sim
        self._tree_idx: dict[tuple, object] = {}
        self._qcache: dict[str, np.ndarray] = {}
        self._rowcache: dict[tuple, tuple[int, dict]] = {}
        self._dense_down_until = 0.0
        self.user_first = os.environ.get('PM1_USER_FIRST', '0').strip().lower() not in ('0', 'off', 'false', 'no')
        # Session neighbours (LoCoMo competitor run, docs/EVALUATION.md E2, offline evidence check
        # harness/pm1_ctx_eval.py): each of the first NEIGHBOR_HITS hits brings the turns +-NEIGHBOR_RADIUS around it in
        # its session (the question it answers, the follow-up), so a turn is not cut off from its own exchange.
        # Evidence fully in context, held-out c5-c9: 0.587 -> 0.701 at 1K, -> 0.799 at 4K (with the budget filled).
        # PM1_NEIGHBORS="hits,radius"; "0" turns it off.
        # Fact index (pm1.facts.FactIndex): turns pointed to by the facts nearest the question go into the candidate
        # list after the first few hits (offline check on LoCoMo, held-out c5-c9, evidence fully in context:
        # 1K 0.701 -> 0.750 with 8 facts after position 2; 4K 0.799 -> 0.856 with 16 facts after position 5).
        self.facts = facts
        # entity hop over the facts (pm1-v4; budgets >= 3000 only). Off by default: it tied pm1-v3 on LoCoMo (result doc §11).
        self.fact_hop = int(os.environ.get('PM1_FACT_HOP', '0') or 0)
        # facts less similar to the question than this pull in nothing (work data: weak facts crowded out better hits)
        self.fact_min_sim = env_float('PM1_FACT_MIN_SIM', 0.0)
        # graph recall (hops over the fact graph; 0 = off): the facts' turns join the candidates BEFORE the rerank, so
        # set PM1_RERANK_K high enough to rerank them (offline LoCoMo 4K: 0.755 -> 0.877 with all candidates reranked)
        self.fact_graph = env_int('PM1_FACT_GRAPH', 0)
        self.fact_graph_max = env_int('PM1_FACT_GRAPH_MAX', 100)
        # second round (2026-10-31, as PM1_SESS_PRF in search_sessions): the top N fused spans and the turns around them
        # make a new keyword and vector search whose hits join the fusion; off by default
        # planned recall (prereg amendment 16, off): planner(query) -> extra queries, each searched and reranked on its own
        self.planner, self.plan_k, self.plan_weight = None, env_int('PM1_RECALL_PLAN_K', 60), env_int('PM1_RECALL_PLAN_W', 1)
        # one search per seed, reranked against that seed (prereg amendment 17, off); plan_mix 'rrf' or 'slot': extra lists
        # keep their own places behind the question's list
        self.seeds, self.seed_from, self.seed_q = env_int('PM1_RECALL_SEEDS', 0), 'rrf', False
        self.plan_seed_q = False                # a planned query's list is reranked again: question + that list's top record
        self.plan_mix, self.slot_n, self.slot_at, self.slot_hold = 'rrf', 1, 8, 20
        # a second vector model whose top k_first join the first-stage RRF beside e5 (prereg amendment 21, off);
        # time windows and the fact graph keep using self.dense only
        self.dense2, self._q2cache = None, {}
        self.prf,self.prf_radius, self.prf_terms = (env_int('PM1_RECALL_PRF', 0), env_int('PM1_RECALL_PRF_NB', 1),
                                                     env_int('PM1_RECALL_PRF_TERMS', 12))
        nb = os.environ.get('PM1_NEIGHBORS', '5,2').strip()
        self.neighbor_hits, self.neighbor_radius = (0, 0) if nb in ('0', '', 'off') else map(int, nb.split(','))
        # hedges against the reranker pushing out what the first stage found (docs/EVALUATION.md E9;
        # both off): PM1_KEEP_FUSED=N puts the fused top N (those not already there)
        # right after the reranked top KEEP_FUSED_AT; PM1_NEXT_TURN=H: hits ranked up to H that got no neighbour block
        # bring the next user / assistant turn of their session, when it fits
        self.keep_fused, self.next_turn = env_int('PM1_KEEP_FUSED', 0), env_int('PM1_NEXT_TURN', 0)

    # ---------- shared helpers ----------
    DENSE_COOLDOWN_S = 60.0

    @contextmanager
    def _dense_guard(self, query: str):
        """The dense embedder is an enhancement, never a dependency: when its service is down the call runs without it
        (BM25 + rerank) instead of failing, and the embedder is skipped for a minute rather than timing out on every
        query. Yields None, or the error text when this call runs without dense."""
        dense = self.dense
        if dense is None:
            yield None
            return
        err = 'skipped: embedder failed less than a minute ago' if time.monotonic() < self._dense_down_until else None
        if err is None:
            try:
                self._qvec(query)
            except Exception as exc:
                self._dense_down_until = time.monotonic() + self.DENSE_COOLDOWN_S
                err = f'{type(exc).__name__}: {str(exc)[:120]}'
        if err is None:
            yield None
            return
        self.dense = None
        try:
            yield err
        finally:
            self.dense = dense

    def _qvec(self, query: str) -> np.ndarray:
        v = self._qcache.get(query)
        if v is None:
            if len(self._qcache) > 256:
                self._qcache.clear()
            v = self._qcache[query] = self.dense.embed([query], 'query')[0]
        return v

    def _dense_rows(self, bank: str, scope: str) -> tuple[np.ndarray, np.ndarray, dict, np.ndarray]:
        """(ids, matrix, {span_id: row}, day-as-int per row) for the dense index, cached per store version."""
        v = self.store._version
        ids, mat = self.store.dense_matrix(bank, scope, self.dense.name)
        key = (bank, scope, self.dense.name)
        hit = self._rowcache.get(key)
        if hit and hit[0] == v and hit[1]['n'] == len(ids):
            c = hit[1]
        else:
            meta = self.store.span_meta(bank, scope)
            day = np.array([int(re.sub(r'\D', '', meta['day'][meta['pos'][int(i)]])[:8] or 0)
                            if int(i) in meta['pos'] else 0 for i in ids], dtype=np.int64)
            user = np.array([bool(meta['user'][meta['pos'][int(i)]]) if int(i) in meta['pos'] else False
                             for i in ids], dtype=bool)
            c = {'n': len(ids), 'row': {int(i): j for j, i in enumerate(ids)}, 'day': day, 'user': user}
            self._rowcache[key] = (v, c)
        return ids, mat, c['row'], c['day']

    def _fused(self, bank: str, scope: str, query: str, k: int,
               window: tuple[str, str] | None = None) -> dict[int, float]:
        """RRF scores (BM25 + dense) over the top-k of each, optionally inside a time window."""
        lex = [i for i, _ in self.store.bm25(bank, scope, query, k, window)]
        den: list[int] = []
        if self.dense is not None:
            ids, mat, _, day = self._dense_rows(bank, scope)
            if len(ids):
                sims = mat @ self._qvec(query)
                if window:
                    lo, hi = int(window[0].replace('-', '')), int(window[1].replace('-', ''))
                    sims = np.where((day >= lo) & (day <= hi), sims, -np.inf)
                top = np.argsort(-sims)[:k]
                den = [int(ids[j]) for j in top if np.isfinite(sims[j])]
        score: dict[int, float] = defaultdict(float)
        for ranking in (lex, den):
            for r, i in enumerate(ranking):
                score[i] += 1 / (60 + r + 1)
        return dict(score)

    def _conflicts(self, bank: str, scope: str, query: str) -> list[int]:
        """The user's own "I have never ..." statements closest to the query (cosine >= conflict_query_sim).
        Surfaced in their own block so the reader checks them against the other records; pm1 does not decide
        which statement is true. (Pairing a never-statement with its affirming turn by embedding similarity was
        tried: right partner in 1/15 BEAM-10M cases, so the affirmation is left to normal retrieval.)"""
        if not self.n_conflicts or self.dense is None:
            return []
        negs = self.store.negations(bank, scope)
        if not negs:
            return []
        ids, mat, row, _ = self._dense_rows(bank, scope)
        nrows = [row[i] for i in negs if i in row]
        if not nrows:
            return []
        sims = mat[nrows] @ self._qvec(query)
        top = [j for j in np.argsort(-sims)[:self.n_conflicts] if sims[j] >= self.conflict_query_sim]
        return [int(ids[nrows[j]]) for j in top]

    _TOPIC = re.compile(r"\b(?:when(?:ever)?|if)\s+(?:I|we)(?:'m| am| are)?\s+(?:ask(?:ing)?|talk(?:ing)?|discuss(?:ing)?|"
                        r"work(?:ing)?|chat(?:ting)?|inquir(?:e|ing))\s+(?:about|on|for|regarding)?\s*(.+)", re.I)

    @classmethod
    def directive_topic(cls, text: str) -> str:
        """'Always include X when I ask about Y' -> 'Y' (the whole instruction when no trigger clause)."""
        t = text.split('->->')[0]
        m = cls._TOPIC.search(t)
        return (m.group(1) if m else t).strip().rstrip('.!')[:200]

    def _directives(self, bank: str, scope: str, query: str) -> list[int]:
        """The standing user instructions most related to the query. Ranked by dense similarity of the query to
        each instruction's trigger topic when gated (directive_min_sim), else to the whole instruction."""
        if not self.n_directives or self.dense is None:
            return []
        dids = self.store.directives(bank, scope)
        if not dids:
            return []
        ids, mat, row, _ = self._dense_rows(bank, scope)
        rows = [row[i] for i in dids if i in row]
        if not rows:
            return []
        qv = self._qvec(query)
        if self.directive_min_sim is None:
            sims = mat[rows] @ qv
            top = np.argsort(-sims)[:self.n_directives]
            return [int(ids[rows[j]]) for j in top]
        key = (bank, scope, len(rows))
        if key not in self._topiccache:
            sids = [int(ids[r]) for r in rows]
            texts = self.store.span_texts(sids)
            self._topiccache[key] = (sids, self.dense.embed([self.directive_topic(texts.get(s, '')) for s in sids], 'query'))
        sids, tv = self._topiccache[key]
        sims = tv @ qv
        order = np.argsort(-sims)[:self.n_directives]
        # the best-matching instruction always stays (instruction-following questions need it); further ones only
        # when their trigger topic is close to the query (BEAM-1M: 0.82 keeps 94% of gold, ~1.3 instead of 3 elsewhere)
        top = [j for k, j in enumerate(order) if (k == 0 and self.directive_keep_top) or sims[j] >= self.directive_min_sim]
        return [sids[j] for j in top]

    def _rr_text(self, r: dict) -> str:
        return self.store.span_header(r) + '\n' + r['text'] if self.rerank_header else r['text']

    def _rerank_texts(self, query: str, sids: list[int], rows: dict, budget_ms) -> tuple[list[int], dict]:
        if self.router is None or not sids:
            return sids, {}
        budget = PROFILES.get(self.profile) if budget_ms == -1.0 else budget_ms
        idx, info = self.router.rerank(query, [self._rr_text(rows[i]) for i in sids], detect(query),
                                       [rows[i]['lang'] for i in sids], budget)
        return ([sids[j] for j in idx] + [s for k, s in enumerate(sids) if k not in set(idx)]
                if idx is not None else sids), info

    def _tree_block(self, bank: str, scope: str, query: str, budget_ms) -> str | None:
        from .tree_index import TreeIndex
        key = (bank, scope)
        if key not in self._tree_idx:
            t = self.tree_factory(bank, scope)
            self._tree_idx[key] = TreeIndex(t.items(), self.dense) if t is not None else None
        ti = self._tree_idx[key]
        if ti is None or not ti.items:
            return None

        def rr(q, texts):
            if self.router is None:
                return None
            budget = PROFILES.get(self.profile) if budget_ms == -1.0 else budget_ms
            idx, _info = self.router.rerank(q, texts, detect(q), [detect(x) for x in texts], budget)
            return idx
        hits = ti.search(query, self._qvec(query), k=self.tree_k, min_sim=self.tree_min_sim, rerank=rr)
        if not hits:
            return None
        lines = [f"- ({it['layer']}) {excerpt(it['text'], 600)} [src {', '.join(it['evidence'][:3])}]" for it, _s in hits]
        return ('From the memory tree (maintained from the records; the records below are authoritative if they '
                'disagree):\n' + '\n'.join(lines))

    # ---------- planned strategies ----------
    def _timeline(self, bank: str, scope: str, query: str, p: Plan, budget_ms) -> tuple[list[int], dict]:
        """One representative per day-thread (best match + the thread's opening user turn), threads ranked by
        relevance and later shown in time order. Inside the query's date window when it names one."""
        t0 = time.perf_counter()
        meta = self.store.span_meta(bank, scope)
        scores = self._fused(bank, scope, query, 800, p.window)
        best: dict[str, tuple[float, int]] = {}
        second: dict[str, tuple[float, int]] = {}
        for sid, sc in scores.items():
            j = meta['pos'].get(sid)
            if j is None:
                continue
            th = meta['thread'][j]
            if th not in best or sc > best[th][0]:
                if th in best:
                    second[th] = best[th]
                best[th] = (sc, sid)
            elif th not in second or sc > second[th][0]:
                second[th] = (sc, sid)
        aspect_hits: list[list[int]] = []
        for sub in p.subs:          # listed aspects ("including A, B, C"): searched inside the same window
            sc = self._fused(bank, scope, sub, 60, p.window)
            aspect_hits.append(sorted(sc, key=lambda i: -sc[i])[:6])
        threads = sorted(best, key=lambda t: -best[t][0])
        head = threads[:48]
        reps = [best[t][1] for t in head]
        rows = self.store.spans(bank, scope, reps)
        reps = [s for s in reps if s in rows]
        ranked, info = self._rerank_texts(query, reps, rows, budget_ms)
        th_of = {best[t][1]: t for t in head}
        order_threads = [th_of[s] for s in ranked] + threads[48:]
        if not p.window:          # no date range: keep the most relevant threads only
            order_threads = order_threads[:max(p.want or 0, 24)]
        # Breadth first: every thread's opening turn (or best match), then each thread's best match, then the
        # next opening user turns; packing stops at the token budget, so coverage across time comes first.
        rounds: list[list[int]] = [[], [], [], [], []]
        for th in order_threads:
            lead = meta['lead'].get(th, []) if p.openers else []
            rounds[0].append(best[th][1])
            if lead:
                rounds[1].append(lead[0])
            if th in second:
                rounds[3].append(second[th][1])
            rounds[4].extend(lead[1:3])
        for r in range(6):                                  # aspect hits, round-robin
            for hits in aspect_hits:
                if r < len(hits):
                    rounds[2].append(hits[r])
        # Then fill whatever budget is left with the most relevant remaining spans (few threads -> few rounds;
        # without this a 10-session history used only ~half the budget).
        fill = sorted(scores, key=lambda i: -scores[i])
        order = list(dict.fromkeys(rounds[0] + rounds[1] + rounds[2] + rounds[3] + rounds[4] + fill))
        return order, {'threads': len(threads), 'window': p.window, 'plan_ms': (time.perf_counter() - t0) * 1e3,
                       **{k: v for k, v in info.items() if k in ('rerank_model', 'route', 'fallback')}}

    def _multi(self, bank: str, scope: str, query: str, p: Plan, budget_ms) -> tuple[list[int], dict]:
        """One ranked search per sub-question (plus the whole question), merged round-robin, at most three spans
        per day-thread; with `openers`, the opening turns of each sub-question's top threads come first."""
        t0 = time.perf_counter()
        meta = self.store.span_meta(bank, scope)
        lists, info = [], {}
        for q in [query] + p.subs:
            order, _, stage = self.rank(bank, scope, q, budget_ms=budget_ms)
            lists.append(order[:80])
            info = info or stage                       # the whole question's own search describes the route
        out: list[int] = []
        seen: set[int] = set()
        per_thread: dict[str, int] = defaultdict(int)

        def take(sid: int, cap: int = 4) -> None:
            j = meta['pos'].get(sid)
            th = meta['thread'][j] if j is not None else ''
            if sid in seen or per_thread[th] >= cap:
                return
            seen.add(sid); per_thread[th] += 1; out.append(sid)

        if p.openers:
            for lst in lists[1:]:
                ths = []
                for sid in lst:
                    j = meta['pos'].get(sid)
                    if j is not None and meta['thread'][j] not in ths:
                        ths.append(meta['thread'][j])
                    if len(ths) >= 2:
                        break
                for th in ths:
                    op = meta['opener'].get(th)
                    if op is not None:
                        take(op, cap=4)
        # Weighted RRF: the whole question counts most, each sub-question adds evidence; spans that several
        # searches agree on rise. (A plain round-robin buried the whole question's own mid-ranked hits.)
        fused: dict[int, float] = defaultdict(float)
        for w, lst in zip([self.multi_weight] + [1.0] * len(p.subs), lists):
            for r, sid in enumerate(lst):
                fused[sid] += w / (60 + r + 1)
        if self.multi_user_bias != 1.0:        # facts to combine are the user's own statements, mostly
            for sid in fused:
                j = meta['pos'].get(sid)
                if j is not None and not meta['user'][j]:
                    fused[sid] *= self.multi_user_bias
        cap = self.multi_cap
        for sid in lists[0][:5]:
            take(sid, cap=cap)
        for lst in lists[1:]:                      # each sub-question's top hit is guaranteed a slot
            if lst:
                take(lst[0], cap=cap)
        for sid in sorted(fused, key=lambda i: -fused[i]):
            take(sid, cap=cap)
        return out, {'subs': p.subs, 'plan_ms': (time.perf_counter() - t0) * 1e3,
                     **{k: v for k, v in info.items() if k in ('rerank_model', 'route', 'fallback', 'dense_error')}}

    def rank(self, bank: str, scope: str, query: str, mode: str | None = None,
             budget_ms: float | None = -1.0, only: set[int] | None = None) -> tuple[list[int], dict, dict]:
        """Ordered span ids, per-span scores of the last stage, and stage timings (ms).
        budget_ms: rerank latency budget for mode 'routed' (-1 = the recaller's profile, None = no limit).
        only: rank just these span ids (the filter comes before the first-stage cut, not after it)."""
        with self._dense_guard(query) as err:
            order, scores, t = self._rank(bank, scope, query, mode, budget_ms, only)
        return order, scores, (t | {'dense_error': err}) if err else t

    def _material(self, bank: str, scope: str, ids: list[int]) -> str:
        """the spans' records and the turns around them"""
        con = self.store.con
        texts = []
        for i in ids:
            r = con.execute('SELECT e.text, e.session, e.seq, e.id FROM spans s JOIN episodes e ON e.id = s.episode_id '
                            'WHERE s.id = ?', (i,)).fetchone()
            if r is None:
                continue
            texts.append(r[0])
            if r[2] is not None:
                texts += [n[0][:self.NEIGHBOR_CHARS] for n in con.execute(
                    'SELECT text FROM episodes WHERE bank=? AND scope=? AND session=? AND seq BETWEEN ? AND ? AND id<>? '
                    'ORDER BY seq', (bank, scope, r[1], r[2] - self.prf_radius, r[2] + self.prf_radius, r[3]))]
        return '\n'.join(texts)

    def _search_from(self, bank: str, scope: str, query: str, blob: str, k_lex: int, k_den: int) -> tuple[list[int], list[int]]:
        """(keyword hits, vector hits) of a search made from a text: its rarer words added to the query, and its vector"""
        con = self.store.con
        have = set(tokenize(query))
        tf: dict[str, int] = defaultdict(int)
        for w in tokenize(blob):
            if w not in have:
                tf[w] += 1
        n = max(con.execute('SELECT max(rowid) FROM span_fts').fetchone()[0] or 0, 2)
        df = {w: con.execute('SELECT count(*) FROM (SELECT 1 FROM span_fts WHERE span_fts MATCH ? LIMIT ?)',
                             ('"' + w.replace('"', '""') + '"', n)).fetchone()[0]
              for w in sorted(tf, key=lambda w: -tf[w])[:60]}
        terms = sorted(df, key=lambda w: -tf[w] * math.log(n / (df[w] + 1)))[:self.prf_terms]
        plex = [i for i, _ in self.store.bm25(bank, scope, query + ' ' + ' '.join(terms), k_lex)]
        ids, mat = self.store.dense_matrix(bank, scope, self.dense.name)
        if not len(ids):
            return plex, []
        sims = mat @ self.dense.embed([self._cut(blob)], 'query')[0]
        return plex, [int(ids[j]) for j in np.argsort(-sims)[:k_den]]

    @staticmethod
    def _cut(text: str, tokens: int = 300) -> str:
        return text[:max(int(len(text) * min(1.0, tokens / max(est_tokens(text), 1))), 200)]

    def _second_round(self, bank: str, scope: str, query: str, order: list[int]) -> tuple[list[int], list[int]]:
        """(keyword hits, vector hits) of a search made from the first round's top spans and the turns around them"""
        return self._search_from(bank, scope, query, self._material(bank, scope, order[:self.prf]), self.k_first, self.k_first)

    def _seed_lists(self, bank: str, scope: str, query: str, order: list[int], budget_ms) -> list[list[int]]:
        """one list per seed: its own search, reranked against the seed's text"""
        seeds = order if self.seed_from == 'order' else self._rank_one(bank, scope, query, 'rrf')[0]
        out = []
        for seed in seeds[:self.seeds]:
            blob = self._material(bank, scope, [seed])
            rows = self.store.spans(bank, scope, [seed])
            text = rows[seed]['text'] if seed in rows else ''
            rq = self._cut(text if len(text) >= 80 else blob)
            if self.seed_q:
                rq = query + '\n' + rq
            head = rrf(list(self._search_from(bank, scope, query, blob, 50, 20)))
            out.append([i for i in self._rerank_against(bank, scope, rq, head, budget_ms) if i != seed])
        return out

    def _rerank_against(self, bank: str, scope: str, rq: str, ids: list[int], budget_ms) -> list[int]:
        rows = self.store.spans(bank, scope, ids)
        head = [i for i in ids if i in rows]
        idx, _ = self.router.rerank(rq, [rows[i]['text'] for i in head], detect(rq), [rows[i]['lang'] for i in head],
                                    PROFILES.get(self.profile) if budget_ms == -1.0 else budget_ms)
        return [head[j] for j in idx] if idx is not None else head

    def _mix(self, order: list[int], lists: list[list[int]]) -> list[int]:
        if self.plan_mix == 'rrf':
            return rrf([order] * self.plan_weight + lists)
        hold, picks = set(order[:self.slot_hold]), []
        for ranked in lists:
            picks += [i for i in ranked if i not in hold and i not in picks][:self.slot_n]
        front, rest = order[:self.slot_at], [i for i in order[self.slot_at:] if i not in picks]
        return front + picks + rest

    def _rank(self, bank: str, scope: str, query: str, mode: str | None = None,
              budget_ms: float | None = -1.0, only: set[int] | None = None) -> tuple[list[int], dict, dict]:
        order, scores, t = self._rank_one(bank, scope, query, mode, budget_ms, only)
        if (self.planner is None and not self.seeds) or only is not None or t['mode'] != 'routed' or not order:
            return order, scores, t
        lists = []
        if self.planner is not None:
            t0 = time.perf_counter()
            subs = [q for q in dict.fromkeys(self.planner(query)) if q.strip().lower() != query.strip().lower()]
            t['plan_ms'], t['plan_queries'] = (time.perf_counter() - t0) * 1e3, len(subs)
            t0 = time.perf_counter()
            for q in subs:
                ranked = self._rank_one(bank, scope, q, t['mode'], budget_ms, None, head_k=self.plan_k)[0][:self.plan_k]
                if self.plan_seed_q and ranked:
                    rows = self.store.spans(bank, scope, ranked[:1])
                    if ranked[0] in rows:
                        ranked = self._rerank_against(bank, scope, query + '\n' + self._cut(rows[ranked[0]]['text']), ranked, budget_ms)
                lists.append(ranked)
            t['plan_search_ms'] = (time.perf_counter() - t0) * 1e3
        if self.seeds:
            t0 = time.perf_counter()
            lists += self._seed_lists(bank, scope, query, order, budget_ms)
            t['seed_ms'] = (time.perf_counter() - t0) * 1e3
        return (self._mix(order, lists) if lists else order), scores, t

    KEEP_FUSED_AT = 8

    def _keep_fused(self, order: list[int], fused: list[int]) -> list[int]:
        front = order[:self.KEEP_FUSED_AT]
        have = set(front)
        picks = [i for i in fused[:self.keep_fused] if i not in have]
        have.update(picks)
        return front + picks + [i for i in order[self.KEEP_FUSED_AT:] if i not in have]

    GX_POOL = 300

    def _grep_list(self, bank: str, scope: str, query: str, only: set[int] | None = None) -> list[int]:
        """Literal-match channel (docs/EVALUATION.md E13, not adopted): the keyword hits re-ordered the way grep is used, by how
        many of the query's words a span contains whole (idf-weighted; a rare word counts more), plus a bonus for
        query words that stand next to each other in the span as a phrase. Keyword ties keep their bm25 order."""
        toks = list(dict.fromkeys(tokenize(query)))
        if not toks:
            return []
        pool = [i for i, _ in self.store.bm25(bank, scope, query, self.GX_POOL, ids=None if only is None else sorted(only))]
        if not pool:
            return []
        rows = self.store.spans(bank, scope, pool)
        n = max(self.store.con.execute('SELECT max(rowid) FROM span_fts').fetchone()[0] or 0, 2)
        idf = {t: math.log(n / (self.store.con.execute(
            'SELECT count(*) FROM (SELECT 1 FROM span_fts WHERE span_fts MATCH ? LIMIT ?)',
            ('"' + t.replace('"', '""') + '"', n)).fetchone()[0] + 1)) for t in toks}
        bigrams = [(a, b) for a, b in zip(toks, toks[1:])]
        order_in_pool = {i: r for r, i in enumerate(pool)}
        scored = []
        for i in pool:
            if i not in rows:
                continue
            words = tokenize(rows[i]['text'])
            have = set(words)
            cov = sum(idf[t] for t in toks if t in have)
            pos = {}
            for k, w in enumerate(words):
                pos.setdefault(w, []).append(k)
            ph = sum(idf[a] + idf[b] for a, b in bigrams if a in pos and b in pos and any(k + 1 in pos[b] for k in pos[a]))
            scored.append((-(cov + ph), order_in_pool[i], i))
        return [i for _, _, i in sorted(scored)]

    def _rank_one(self, bank: str, scope: str, query: str, mode: str | None = None, budget_ms: float | None = -1.0,
                  only: set[int] | None = None, head_k: int | None = None) -> tuple[list[int], dict, dict]:
        """head_k: a planned query's search: rerank this many, no second round or fact graph."""
        mode = mode or ('routed' if self.router else 'hybrid' if self.token and self.dense
                        else 'rrf' if self.dense else 'lexical')
        t: dict[str, float] = {}
        t0 = time.perf_counter()
        lex = [i for i, _ in self.store.bm25(bank, scope, query, self.k_first,
                                             ids=None if only is None else sorted(only))] if mode != 'dense' else []
        t['bm25_ms'] = (time.perf_counter() - t0) * 1e3
        den: list[int] = []
        if mode != 'lexical' and self.dense is not None:
            t0 = time.perf_counter()
            ids, mat = self.store.dense_matrix(bank, scope, self.dense.name)
            if len(ids):
                sims = mat @ self._qvec(query)
                if only is not None:
                    sims = np.where(np.isin(ids, np.fromiter(only, np.int64, len(only))), sims, -np.inf)
                top = np.argsort(-sims)[:self.k_first]
                den = [int(ids[j]) for j in top if np.isfinite(sims[j])]
            t['dense_ms'] = (time.perf_counter() - t0) * 1e3
        den2: list[int] = []
        if mode not in ('lexical', 'dense') and self.dense is not None and self.dense2 is not None:
            t0 = time.perf_counter()
            ids, mat = self.store.dense_matrix(bank, scope, self.dense2.name)
            if len(ids):
                if query not in self._q2cache:
                    self._q2cache = {} if len(self._q2cache) > 256 else self._q2cache
                    self._q2cache[query] = self.dense2.embed([query], 'query')[0]
                sims = mat @ self._q2cache[query]
                if only is not None:
                    sims = np.where(np.isin(ids, np.fromiter(only, np.int64, len(only))), sims, -np.inf)
                top = np.argsort(-sims)[:self.k_first]
                den2 = [int(ids[j]) for j in top if np.isfinite(sims[j])]
            t['dense2_ms'] = (time.perf_counter() - t0) * 1e3
        dens = [den, den2] if den2 else [den]
        t0 = time.perf_counter()
        gx = self._grep_list(bank, scope, query, only) if self.grep_channel and mode not in ('lexical', 'dense') else []
        if gx:
            t['gx_ms'] = (time.perf_counter() - t0) * 1e3
        order = lex if mode == 'lexical' else den if mode == 'dense' else rrf([lex, *dens, *([gx] if gx else [])])
        if self.prf and head_k is None and mode == 'routed' and only is None and order and self.dense is not None:
            t0 = time.perf_counter()
            order = rrf([lex, *dens, *self._second_round(bank, scope, query, order)])
            t['second_round_ms'] = (time.perf_counter() - t0) * 1e3
        if self.facts is not None and self.fact_graph > 0 and self.dense is not None and mode == 'routed' \
                and only is None and head_k is None:
            t0 = time.perf_counter()
            qv = self._qvec(query) if self.facts.dense is self.dense else self.facts.dense.embed([query], 'query')[0]
            eps = self.facts.graph_episodes(bank, scope, qv, hops=self.fact_graph,
                                            limit=self.fact_graph_max)
            have = set(order)
            first = self.store.first_spans(bank, scope, eps) if eps else {}
            extra = [first[e] for e in eps if e in first and first[e] not in have]
            order = order + list(dict.fromkeys(extra))
            t['graph_ms'] = (time.perf_counter() - t0) * 1e3
            t['graph_turns'] = len(extra)
        scores: dict[int, float] = {}
        if mode == 'routed' and self.router is not None and order:
            depth = head_k or self.k_route
            head, tail = order[:depth], order[depth:]
            rows = self.store.spans(bank, scope, head)
            head = [i for i in head if i in rows]
            budget = PROFILES.get(self.profile) if budget_ms == -1.0 else budget_ms
            t0 = time.perf_counter()
            idx, info = self.router.rerank(query, [self._rr_text(rows[i]) for i in head], detect(query),
                                           [rows[i]['lang'] for i in head], budget)
            t['rerank_ms'] = (time.perf_counter() - t0) * 1e3
            t |= {k: v for k, v in info.items() if k != 'rerank_ms'}
            if idx is not None:
                pre, order = order, [head[j] for j in idx] + tail
                if self.keep_fused and head_k is None:
                    order = self._keep_fused(order, pre)
        elif mode in ('hybrid', 'fuse3') and self.token is not None and order:
            t0 = time.perf_counter()
            head, tail = order[:self.k_rerank], order[self.k_rerank:]
            try:
                vecs = self.store.tokvecs(head, self.token.name)
                missing = [i for i in head if i not in vecs][:self.encode_missing]
                if missing:  # spans the worker has not indexed yet: encode a bounded few on the fly
                    rows = self.store.spans(bank, scope, missing)
                    mats = self.token.encode([rows[i]['text'] for i in missing if i in rows], 'document')
                    for i, m in zip([i for i in missing if i in rows], mats):
                        vecs[i] = m
                        self.store.put_tokvecs(i, self.token.name, m)
                [qv] = self.token.encode([query], 'query')
                scored = {i: maxsim(qv, vecs[i]) for i in head if i in vecs}
                scores = scored
                # Rerank only among spans that have token vectors; unindexed spans keep their fused slot
                # instead of being pushed down for a reason unrelated to relevance.
                by_maxsim = sorted(scored, key=lambda i: -scored[i])
                if mode == 'fuse3':
                    # MaxSim as a third voice: fuse its order with the BM25+dense order instead of replacing it.
                    order = rrf([head, by_maxsim + [i for i in head if i not in scored]]) + tail
                else:
                    ranked = iter(by_maxsim)
                    order = [next(ranked) if i in scored else i for i in head] + tail
                if len(scored) < len(head):
                    t['rerank_unindexed'] = len(head) - len(scored)
            except Exception as exc:
                # The reranker is an enhancement, never a dependency: keep the fused order.
                t['rerank_error'] = f'{type(exc).__name__}: {str(exc)[:120]}'
            t['rerank_ms'] = (time.perf_counter() - t0) * 1e3
        return order, scores, t | {'mode': mode}

    # adaptive context budget (decided 2026-10-14): 2.5K tokens by default, up to 8K when the plan says the question
    # aggregates across sessions (timeline / multi-part / counting & totals). An explicit budget_tokens always wins.
    BASE_BUDGET, WIDE_BUDGET = 2500, 8000
    # "the user's own words first" (BEAM-1M memory-value experiment: U - base +0.028,
    # 95% CI [+0.007, +0.050], 700 questions, no question type below -0.05). LongMemEval-S test (402 q, 512 tokens) showed
    # it must NOT apply to questions about what the assistant said / recommended / "our previous conversation" (single-session-
    # assistant 0.889 -> 0.578 without the exception); ASSISTANT_ASKED is the exception, designed on the dev split. Within the reranked head, the user's spans go
    # before the assistant's; a question about what the ASSISTANT said keeps the plain order. PM1_USER_FIRST=1 turns it ON (default off: it hurt on agent-work data).
    USER_FIRST_POOL = 48
    ASSISTANT_ASKED = re.compile(
        r"\byou(?:'ve| have)? (?:said|told|suggested|recommended|mentioned|gave|provided|wrote|listed|shared|described|explained)\b"
        r"|\byour (?:answer|suggestion|recommendation|advice|response|list)\b"
        r"|\bour (?:previous|last|earlier|past) (?:conversation|chat|session|discussion)\b"
        r"|\bwe (?:talked|discussed|chatted|went over|covered|spoke)\b|\bremind me\b"
        r"|\b(?:can|could|would) you (?:recommend|suggest)\b|\b(?:any|some) (?:recommendations?|suggestions?|ideas?)\b"
        r"|\bdo you have any (?:ideas|suggestions|recommendations)\b|\bwhat do you (?:recommend|suggest|think)\b"
        r"|你(?:之前|刚才|曾经)?(?:说|建议|推荐|提到|告诉)|你的(?:回答|建议|推荐)|我们(?:之前|上次)?(?:聊|讨论|谈)过|提醒我", re.I)

    NEIGHBOR_CHARS = 600                       # a neighbour is context, not the hit: long turns shown as an excerpt

    def recall(self, bank: str, scope: str, query: str, *, budget_tokens: int | None = None,
               max_items: int | None = None, mode: str | None = None, budget_ms: float | None = -1.0,
               now=None, until=None) -> RecallResult:
        """until: only records at or before this time are searched (as-of recall; benchmarks whose questions are
        asked part-way through a history).
        now: when the question is asked (date / datetime / 'YYYY/MM/DD ...'); lets relative phrases
        ("a week ago", "on Valentine's Day", "last month") pull in records from the date window they point at.
        max_items: None = fill the token budget (the budget is the limit). It used to default to 12, which capped
        short-turn stores at ~600 tokens whatever the budget (LoCoMo 10-26: 681 tokens used of 4,096)."""
        with self._dense_guard(query) as err, self.store.as_of(until):
            res = self._recall(bank, scope, query, budget_tokens=budget_tokens, max_items=max_items, mode=mode,
                               budget_ms=budget_ms, now=now)
        if err:
            res.diagnostics['dense_error'] = err
        return res

    def _recall(self, bank: str, scope: str, query: str, *, budget_tokens: int | None = None,
                max_items: int | None = None, mode: str | None = None, budget_ms: float | None = -1.0,
                now=None) -> RecallResult:
        t_all = time.perf_counter()
        if max_items is None:
            max_items = 10 ** 6
        p = make_plan(query, now) if self.planning and mode is None else Plan()
        if budget_tokens is None:
            budget_tokens = self.WIDE_BUDGET if (p.intent in ('timeline', 'multi') or p.aggregate) else self.BASE_BUDGET
        scores: dict = {}
        if p.intent == 'timeline':
            order, diag = self._timeline(bank, scope, query, p, budget_ms)
            diag['mode'] = 'timeline'
        elif p.intent == 'multi':
            order, diag = self._multi(bank, scope, query, p, budget_ms)
            diag['mode'] = 'multi'
        else:
            order, scores, diag = self.rank(bank, scope, query, mode, budget_ms)
        if p.anchor and self.anchor_k and not (p.intent == 'timeline' and p.window):
            # Time-anchored search: the best matches inside the date window the question points at. Relevance
            # alone misses "what did I do a week ago" (the event turn rarely shares words with the question).
            win = self._fused(bank, scope, query, 200, p.anchor)
            wo = sorted(win, key=lambda i: -win[i])[:self.anchor_k]
            if wo:
                ws = set(wo)
                if p.intent == 'fact':
                    head = rrf([order[:200], wo])
                    order = head + [i for i in order[200:] if i not in ws]
                else:                          # planned orders carry guaranteed slots: insert after the first few
                    top = order[:5]
                    order = top + [i for i in wo if i not in top] + [i for i in order[5:] if i not in ws]
            diag['anchor'] = list(p.anchor)
            diag['anchor_hits'] = len(wo)
        if self.facts is not None and mode is None and self.dense is not None and self.fact_graph <= 0:
            k, pos = (16, 5) if budget_tokens >= 3000 else (8, 2)
            hop = self.fact_hop if budget_tokens >= 3000 else 0
            eps = self.facts.episodes_for(bank, scope, self._qvec(query), k, hop=hop, question=query,
                                          min_sim=self.fact_min_sim)
            if eps:
                first = self.store.first_spans(bank, scope, eps)
                head = set(order[:pos])
                x = [first[e] for e in eps if e in first and first[e] not in head]
                xs = set(x)
                order = order[:pos] + x + [i for i in order[pos:] if i not in xs]
                diag['fact_turns'] = len(x)
        planned = p.intent != 'fact'
        cap = max_items * 3 if not planned else max(max_items * 3, 400)
        dirs = self._directives(bank, scope, query) if self.planning and mode is None else []
        negs = self._conflicts(bank, scope, query) if self.planning and mode is None else []
        rows = self.store.spans(bank, scope, list(dict.fromkeys(dirs + negs + order[:cap])))
        items, used, seen = [], 0, set()
        for sid in dirs:                       # standing instructions first, from the same budget
            r = rows.get(sid)
            if r is None or r['episode_id'] in seen or used + est_tokens(r['text']) + 16 > budget_tokens // 8:
                continue
            seen.add(r['episode_id'])
            used += est_tokens(r['text']) + 16
            items.append({'span_id': sid, 'episode_id': r['episode_id'], 'rank': -1, 'score': None,
                          'session': r['session'], 'seq': r['seq'], 'role': r['role'],
                          'occurred_at': r['occurred_at'], 'source': r['source'], 'kind': 'directive',
                          'trust': 'T0-record', 'text': r['text']})
        for sid in negs:                       # possible contradictions: the user's own never-statements
            r = rows.get(sid)
            if r is None or r['episode_id'] in seen:
                continue
            m = self.store._NEGATION.search(r['text'])
            if m and len(r['text']) > 500:     # keep the sentence that says "never" (plus a little context)
                a = max(r['text'].rfind('. ', 0, m.start()) + 2, m.start() - 200, 0)
                r = {**r, 'text': ('… ' if a else '') + excerpt(r['text'][a:], 400)}
            if used + est_tokens(r['text']) + 16 > budget_tokens // 6:
                continue
            seen.add(r['episode_id'])
            used += est_tokens(r['text']) + 16
            items.append({'span_id': sid, 'episode_id': r['episode_id'], 'rank': -1, 'score': None,
                          'session': r['session'], 'seq': r['seq'], 'role': r['role'],
                          'occurred_at': r['occurred_at'], 'source': r['source'], 'kind': 'never',
                          'trust': 'T0-record', 'text': r['text']})
        led_block = None
        led_tokens = 0
        if self.ledger is not None and self.ledger_mode == 'block' and self.planning and mode is None \
                and self.dense is not None:
            qv = self._qvec(query)
            sl = self.ledger.state_lines(bank, scope, qv, self.dense, k=self.ledger_state_k, min_sim=self.ledger_min_sim)
            el = self.ledger.event_lines(bank, scope, qv, self.dense, k=self.ledger_event_k, min_sim=self.ledger_min_sim)
            parts = []
            if sl:
                parts.append('Current facts (latest value first, with earlier values):\n' + '\n'.join(sl))
            if el:
                parts.append('Dated events (chronological):\n' + '\n'.join(el))
            if parts:
                block = ('From the memory ledger (extracted from the records; the raw records below are authoritative '
                         'if they disagree):\n' + '\n'.join(parts))
                cap_t = budget_tokens // 5
                while parts and est_tokens(block) > cap_t and (sl or el):     # trim to a fifth of the budget
                    if len(el) >= len(sl) and el:
                        el = el[:-1]
                    elif sl:
                        sl = sl[:-1]
                    parts = ([('Current facts (latest value first, with earlier values):\n' + '\n'.join(sl))] if sl else []) + \
                            ([('Dated events (chronological):\n' + '\n'.join(el))] if el else [])
                    block = ('From the memory ledger (extracted from the records; the raw records below are '
                             'authoritative if they disagree):\n' + '\n'.join(parts))
                if parts:
                    led_block = block
                    led_tokens = est_tokens(block)
                    used += led_tokens      # counts against the same budget as the raw records
        if self.ledger is not None and self.ledger_mode == 'match' and self.planning and mode is None \
                and self.dense is not None:
            evl = self.ledger.evolution_lines(bank, scope, query, self._qvec(query), self.dense,
                                              k=self.match_k, min_sim=self.match_min_sim)
            if evl:
                led_block = ('How these values changed over time (from memory, oldest → newest; the records below '
                             'are authoritative if they disagree):\n' + '\n'.join(evl))
                led_tokens = est_tokens(led_block)
                used += led_tokens
        if self.tree_factory is not None and self.planning and mode is None and self.dense is not None:
            tb = self._tree_block(bank, scope, query, budget_ms)
            if tb:
                led_block = tb if not led_block else led_block + '\n\n' + tb
                led_tokens += est_tokens(tb)
                used += est_tokens(tb)
        n_dir = len(items)
        if self.user_first and not planned and not self.ASSISTANT_ASKED.search(query):
            head = order[:min(cap, self.USER_FIRST_POOL)]
            u = [sid for sid in head if rows.get(sid) is not None and rows[sid]['role'] == 'user']
            ou = set(u)
            order = u + [sid for sid in head if sid not in ou] + order[len(head):]
            diag['user_first'] = True
        clip = self.timeline_chars if p.intent == 'timeline' else self.multi_chars if p.intent == 'multi' else 0
        with_nbrs = not planned and mode is None and self.neighbor_hits > 0
        n_hits = 0
        parts: dict = defaultdict(int)
        for rank, sid in enumerate(order[:cap]):
            r = rows.get(sid)
            if r is None or (r['episode_id'] in seen and not 0 < parts[r['episode_id']] < self.parts):
                continue
            if clip:                           # timeline: many events, each needs only its gist
                r = {**r, 'text': excerpt(r['text'], clip)}
            elif self.return_chars:
                r = {**r, 'text': self.store.widen(r, self.return_chars)}
            cost = est_tokens(r['text']) + 16
            if items and used + cost > budget_tokens:
                if planned:
                    continue                   # planned recall: skip a long span, keep filling with others
                break
            block = []
            if with_nbrs and n_hits < self.neighbor_hits:
                for nb in self.store.neighbors(bank, scope, r['session'], r['seq'], self.neighbor_radius):
                    if nb['episode_id'] != r['episode_id'] and nb['episode_id'] not in seen:
                        block.append({**nb, 'text': excerpt(nb['text'], self.NEIGHBOR_CHARS)})
                b_cost = sum(est_tokens(nb['text']) + 16 for nb in block)
                if used + cost + b_cost > budget_tokens:
                    block = []                 # the block does not fit: the hit alone
            elif self.next_turn and not planned and mode is None and n_hits < self.next_turn:
                block = [{**nb, 'text': excerpt(nb['text'], self.NEIGHBOR_CHARS)}
                         for nb in self.store.next_turn(bank, scope, r['session'], r['seq'])
                         if nb['episode_id'] != r['episode_id'] and nb['episode_id'] not in seen]
                if used + cost + sum(est_tokens(nb['text']) + 16 for nb in block) > budget_tokens:
                    block = []
            n_hits += 1
            parts[r['episode_id']] += 1
            seen.add(r['episode_id'])
            used += cost
            items.append({'span_id': sid, 'episode_id': r['episode_id'], 'rank': rank,
                          'score': round(scores[sid], 4) if sid in scores else None,
                          'session': r['session'], 'seq': r['seq'], 'role': r['role'],
                          'occurred_at': r['occurred_at'], 'source': r['source'],
                          'trust': 'T0-record', 'text': r['text']})
            for nb in block:                   # same session, adjacent turns: context for the hit
                seen.add(nb['episode_id'])
                used += est_tokens(nb['text']) + 16
                items.append({'span_id': None, 'episode_id': nb['episode_id'], 'rank': rank, 'score': None,
                              'session': r['session'], 'seq': nb['seq'], 'role': nb['role'],
                              'occurred_at': nb['occurred_at'], 'source': nb['source'], 'kind': 'neighbor',
                              'trust': 'T0-record', 'text': nb['text']})
            if len(items) - n_dir >= (max_items if not planned else 10 ** 6):
                break
        # Present chronologically within each session, sessions by first occurrence time.
        by_session: dict[str, list[dict]] = defaultdict(list)
        blocks = []
        dir_items = [it for it in items if it.get('kind') == 'directive']
        if dir_items:
            blocks.append('Standing instructions the user gave earlier (apply them when relevant):\n' + '\n'.join(
                f"[{it['occurred_at']} · {it['session']} · {it['role']}] {it['text']}" for it in dir_items))
        if led_block:
            blocks.append(led_block)
        neg_items = [it for it in items if it.get('kind') == 'never']
        if neg_items:
            blocks.append('The user also said they had NEVER done the following; check whether the records below '
                          'conflict with it:\n' + '\n'.join(
                              f"[{it['occurred_at']} · {it['session']} · {it['role']}] {it['text']}" for it in neg_items))
        for it in items:
            if it.get('kind') not in ('directive', 'never'):
                by_session[it['session']].append(it)
        notes = self.ledger.update_notes(bank, scope) if self.ledger is not None and self.ledger_mode == 'annotate' else {}
        n_notes = 0
        for sess in sorted(by_session, key=lambda s: min(i['occurred_at'] for i in by_session[s])):
            shown = set()
            for it in sorted(by_session[sess], key=lambda i: i['seq']):
                line = f"[{it['occurred_at']} · {sess} · {it['role']}] {it['text']}"
                day = str(it['occurred_at'])[:10].replace('/', '-')
                if notes and it['role'] == 'user' and (sess, day) not in shown and notes.get((sess, day)):
                    shown.add((sess, day))     # once per session-day, after its first user record
                    nl = notes[(sess, day)][:4]
                    n_notes += len(nl)
                    line += '\n  (memory note: ' + '; '.join(nl) + ')'
                blocks.append(line)
        diag |= {'total_ms': round((time.perf_counter() - t_all) * 1e3, 2), 'tokens': used, 'budget': budget_tokens,
                 'candidates': len(order), 'intent': p.intent, 'directives': len(dirs), 'never': len(neg_items), 'ledger_tokens': led_tokens, 'notes': n_notes}
        return RecallResult(items=items, context='\n\n'.join(blocks), mode=diag['mode'], diagnostics=diag)
