"""Language-aware reranker routing.

Two paths (language routing):
  en    — the query AND every candidate are English: any reranker may be used, including English-only ones
          (colbertv2, gte-reranker-modernbert, Qwen3-Reranker).
  multi — anything else (Chinese, other languages, mixed pools): only multilingual rerankers (Qwen3-Reranker).
          A mixed pool is never split by language, because scores from different models are not comparable.
Within a path the router picks the highest-quality reranker whose *expected* latency fits the caller's budget,
using an online latency estimate per reranker. If none fits, or every eligible reranker fails, the fused
first-stage order is kept ('rrf' fallback) — never an English-only model on non-English text.

Quality numbers default to measured T-MRR on LongMemEval-S dev (English, RRF top-32, 2026-10-12) and can be
overridden. Ties: with a latency budget the cheaper model wins; with no budget ('best') the larger one wins.
"""
from __future__ import annotations

import json
import os
import threading
import time
import urllib.request
from dataclasses import dataclass, field
from typing import Callable, Sequence

import numpy as np

from .lang import english_compatible

# latency budget per rerank call, ms. 'wait' is the agent-facing default (MCP, PM1_RERANK_PROFILE=wait): agents
# care about precision far more than a few seconds, so the reranker gets up to 8 s (Qwen over the full 64-candidate
# pool takes ~5 s); unlike 'best' it never sleeps through a reranker's error cooldown, and under GPU contention it
# reranks a shorter head instead of giving up.
PROFILES = {'fast': 250.0, 'balanced': 1000.0, 'wait': 8000.0, 'best': None}


@dataclass
class Reranker:
    name: str
    scorer: Callable[[str, Sequence[str]], list[float]]
    multilingual: bool
    quality: float                       # expected T-MRR (higher is better)
    ms_per_doc: float                    # prior latency estimate; replaced by an online EMA
    cooldown_s: float = 60.0
    _down_until: float = field(default=0.0, repr=False)
    _lock: threading.Lock = field(default_factory=threading.Lock, repr=False)

    def healthy(self) -> bool:
        return time.time() >= self._down_until

    def expected_ms(self, n_docs: int) -> float:
        return self.ms_per_doc * max(1, n_docs)

    def score(self, query: str, docs: Sequence[str]) -> list[float]:
        t0 = time.perf_counter()
        try:
            out = self.scorer(query, docs)
        except Exception:
            with self._lock:
                self._down_until = time.time() + self.cooldown_s
            raise
        per_doc = (time.perf_counter() - t0) * 1e3 / max(1, len(docs))
        with self._lock:
            self.ms_per_doc = 0.7 * self.ms_per_doc + 0.3 * per_doc
        if len(out) != len(docs):
            raise ValueError(f'{self.name}: {len(out)} scores for {len(docs)} docs')
        return [float(x) for x in out]


class Router:
    def __init__(self, rerankers: Sequence[Reranker], min_docs: int = 8):
        self.rerankers = list(rerankers)
        self.min_docs = min_docs

    @staticmethod
    def path(query_lang: str, cand_langs: Sequence[str]) -> str:
        ok = english_compatible(query_lang) and all(english_compatible(t) for t in cand_langs)
        return 'en' if ok else 'multi'

    def plan(self, path: str, n_docs: int, budget_ms: float | None) -> list[tuple[Reranker, int]]:
        """Eligible rerankers with the depth each can afford, best first.

        With a budget, a reranker too slow for all n_docs reranks only the top `fit` candidates (the first stage
        already put most evidence near the top), as long as fit >= min_docs; otherwise it is skipped."""
        eligible = [r for r in self.rerankers if (path == 'en' or r.multilingual) and r.healthy()]
        if budget_ms is None:
            return [(r, n_docs) for r in sorted(eligible, key=lambda r: (-round(r.quality, 2), -r.ms_per_doc))]
        out = []
        for r in eligible:
            fit = min(n_docs, int(budget_ms // max(r.ms_per_doc, 1e-3)))
            if fit >= min(self.min_docs, n_docs):
                out.append((r, fit))
        # quality first; a model that can see more of the pool wins a tie; then the cheaper one
        return sorted(out, key=lambda t: (-round(t[0].quality, 2), -t[1], t[0].ms_per_doc))

    def rerank(self, query: str, docs: Sequence[str], query_lang: str, cand_langs: Sequence[str],
               budget_ms: float | None) -> tuple[list[int] | None, dict]:
        """Returns (order of doc indices or None for 'keep first-stage order', diagnostics).
        When only the top `depth` docs are reranked, the rest keep their first-stage positions after them."""
        path = self.path(query_lang, cand_langs)
        info: dict = {'route': path, 'query_lang': query_lang, 'budget_ms': budget_ms}
        candidates = self.plan(path, len(docs), budget_ms)
        if not candidates and budget_ms is None:
            # No latency budget ('best'): a reranker in its error cooldown is waited for, not skipped, so one
            # failed call cannot leave the next questions unreranked (G3 export, 2026-10-16).
            down = [r._down_until - time.time() for r in self.rerankers if path == 'en' or r.multilingual]
            wait = min(down) if down else 0.0
            if 0 < wait <= max((r.cooldown_s for r in self.rerankers), default=0.0):
                time.sleep(wait + 0.05)
                info['waited_ms'] = round(wait * 1e3)
                candidates = self.plan(path, len(docs), budget_ms)
        if not candidates:
            info['rerank_model'], info['fallback'] = None, 'no reranker fits budget/language'
            return None, info
        errors = []
        for r, depth in candidates:
            t0 = time.perf_counter()
            try:
                s = r.score(query, docs[:depth])
            except Exception as exc:
                errors.append(f'{r.name}: {type(exc).__name__}: {str(exc)[:100]}')
                continue
            info |= {'rerank_model': r.name, 'rerank_depth': depth,
                     'rerank_ms': round((time.perf_counter() - t0) * 1e3, 1)}
            if errors:
                info['rerank_errors'] = errors
            head = [int(i) for i in np.argsort(-np.asarray(s), kind='stable')]
            return head + list(range(depth, len(docs))), info
        info |= {'rerank_model': None, 'fallback': 'all eligible rerankers failed', 'rerank_errors': errors}
        return None, info


# ---------- scorers ----------
def llama_scorer(url: str, timeout: float = 30.0):
    """llama.cpp `--rerank` server (/v1/rerank): Qwen3-Reranker, gte-reranker-modernbert, ..."""
    def f(query, docs):
        req = urllib.request.Request(url.rstrip('/') + '/v1/rerank',
                                     data=json.dumps({'query': query, 'documents': list(docs),
                                                      'top_n': len(docs)}).encode(),
                                     headers={'Content-Type': 'application/json'})
        with urllib.request.urlopen(req, timeout=timeout) as r:
            res = json.loads(r.read())['results']
        s = [float('-inf')] * len(docs)
        for x in res:
            s[x['index']] = x['relevance_score']
        return s
    return f


QWEN_RERANK_INSTRUCT = (
    'Given a question a user asks about their own past conversations with an AI assistant, judge whether this '
    'conversation turn contains information needed to answer it, such as facts or preferences the user stated, '
    'events and when they happened, or what the assistant said')
_QWEN_PREFIX = ('<|im_start|>system\nJudge whether the Document meets the requirements based on the Query and the '
                'Instruct provided. Note that the answer can only be "yes" or "no".<|im_end|>\n<|im_start|>user\n')
_QWEN_SUFFIX = '<|im_end|>\n<|im_start|>assistant\n<think>\n\n</think>\n\n'


def vllm_qwen_scorer(url: str, model: str = 'qwen3-reranker', instruct: str = QWEN_RERANK_INSTRUCT,
                     timeout: float = 30.0):
    """Qwen3-Reranker served by vLLM (/score, pooling runner). vLLM does not apply the reranker template, so the
    prompt is built here exactly as the llama.cpp '-memory' GGUF template does (same instruction, same empty think
    block); scores are P(yes) and rank identically to the original model (eval/rerank_engines_20261018)."""
    def f(query, docs):
        body = {'model': model, 'text_1': _QWEN_PREFIX + f'<Instruct>: {instruct}\n<Query>: {query}\n',
                'text_2': [f'<Document>: {d}' + _QWEN_SUFFIX for d in docs]}
        req = urllib.request.Request(url.rstrip('/') + '/score', data=json.dumps(body).encode(),
                                     headers={'Content-Type': 'application/json'})
        with urllib.request.urlopen(req, timeout=timeout) as r:
            res = json.loads(r.read())['data']
        s = [float('-inf')] * len(docs)
        for x in res:
            s[x['index']] = x['score']
        return s
    return f


def colbert_scorer(model: str = 'colbert-ir/colbertv2.0', gpu: bool = True):
    """Late interaction (MaxSim) with a fastembed model, documents encoded on the fly (rerank mode)."""
    import onnxruntime
    if gpu and hasattr(onnxruntime, 'preload_dlls'):
        onnxruntime.preload_dlls()
    from fastembed import LateInteractionTextEmbedding
    m = LateInteractionTextEmbedding(model, providers=(['CUDAExecutionProvider'] if gpu else []) + ['CPUExecutionProvider'])

    def f(query, docs):
        q = np.asarray(next(iter(m.query_embed([query]))), dtype=np.float32)
        return [float((q @ np.asarray(d, dtype=np.float32).T).max(1).sum()) for d in m.embed(list(docs), batch_size=8)]
    return f


def router_from_env() -> Router | None:
    """PM1_RERANK_QWEN=<url>  PM1_RERANK_GTE=<url>  PM1_RERANK_COLBERT=<fastembed model name>"""
    rs: list[Reranker] = []
    # PM1_RERANK_QWEN_VLLM=<url>: the same Qwen3 reranker on vLLM (fp16, ~16 ms/doc vs ~85 on llama.cpp Q8), rated
    # +0.01 because fp16 matches the original model exactly (top-10 1.00 vs 0.96 for Q8, 2026-10-18) - which also
    # makes it the first choice under every profile; the llama.cpp server stays as the fallback.
    if url := os.environ.get('PM1_RERANK_QWEN_VLLM'):
        rs.append(Reranker('qwen3-reranker-0.6b@vllm', vllm_qwen_scorer(url), multilingual=True,
                           quality=float(os.environ.get('PM1_Q_QWEN', 0.80)) + 0.01, ms_per_doc=16.0))
    if url := os.environ.get('PM1_RERANK_QWEN'):
        rs.append(Reranker('qwen3-reranker-0.6b', llama_scorer(url), multilingual=True,
                           quality=float(os.environ.get('PM1_Q_QWEN', 0.80)), ms_per_doc=85.0))
    if url := os.environ.get('PM1_RERANK_GTE'):
        rs.append(Reranker('gte-reranker-modernbert', llama_scorer(url), multilingual=False,
                           quality=float(os.environ.get('PM1_Q_GTE', 0.80)), ms_per_doc=18.0))
    # CPU instances (e.g. a second machine without a GPU): same models, ~10-50x slower, so the latency budget keeps
    # them out of fast/balanced; they take over for 'best' when the GPU instance fails or is cooling down.
    if url := os.environ.get('PM1_RERANK_QWEN_CPU'):
        rs.append(Reranker('qwen3-reranker-0.6b@cpu', llama_scorer(url, timeout=120.0), multilingual=True,
                           quality=float(os.environ.get('PM1_Q_QWEN', 0.80)) - 0.005, ms_per_doc=900.0))
    if url := os.environ.get('PM1_RERANK_BGE_CPU'):
        rs.append(Reranker('bge-reranker-v2-m3@cpu', llama_scorer(url, timeout=120.0), multilingual=True,
                           quality=float(os.environ.get('PM1_Q_BGE', 0.66)), ms_per_doc=320.0))
    if name := os.environ.get('PM1_RERANK_COLBERT'):
        try:
            rs.append(Reranker('colbertv2', colbert_scorer(name), multilingual=False,
                               quality=float(os.environ.get('PM1_Q_COLBERT', 0.78)), ms_per_doc=12.0))
        except Exception as exc:  # optional dependency (fastembed + onnxruntime) missing
            import sys
            print(f'pm1: colbert reranker disabled: {exc}', file=sys.stderr)
    return Router(rs) if rs else None
