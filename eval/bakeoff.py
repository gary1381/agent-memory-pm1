#!/usr/bin/env python3
"""pm1 retrieval model bake-off on the LongMemEval-S DEV split (never the test split).

Unit = one message. Per question the haystack is indexed in isolation (standard
LongMemEval protocol). First stage: BM25, e5 dense, and their RRF fusion.
Second stage: rerank the fused top-K with each reranker endpoint, or with pplx
late-interaction MaxSim. Reports session- and turn-level recall plus latency.

  .venv/bin/python eval/bakeoff.py --stage first
  .venv/bin/python eval/bakeoff.py --stage rerank --reranker bge-q8=http://127.0.0.1:18140
"""
from __future__ import annotations

import argparse
import hashlib
import json
import math
import os
import re
import sys
import statistics
import time
import urllib.request
from collections import Counter, defaultdict
from pathlib import Path

import numpy as np

DATA = Path.home() / '.cache/pm1/data/longmemeval_s_cleaned.json'
CACHE = Path.home() / '.cache/pm1/cache'
E5_URL = 'http://127.0.0.1:8767/v1/embeddings'
E5_MODEL = 'intfloat/multilingual-e5-small'
PPLX_URL = 'http://127.0.0.1:8770/v1/encode'
DOC_CHARS = 1500  # text sent to embedders / rerankers per message
sys.path.insert(0, str(Path(__file__).parent))


def split_of(qid: str) -> str:
    """Fixed hash split: ~20% dev, rest test. Never tune on test."""
    return 'dev' if int(hashlib.sha256(qid.encode()).hexdigest(), 16) % 5 == 0 else 'test'


def load(split: str = 'dev', limit: int | None = None):
    data = json.loads(DATA.read_text())
    qs = [q for q in data if split_of(q['question_id']) == split]
    return qs[:limit] if limit else qs


def messages(q) -> list[dict]:
    out = []
    for sid, date, sess in zip(q['haystack_session_ids'], q['haystack_dates'], q['haystack_sessions']):
        for i, t in enumerate(sess):
            out.append({'id': f'{sid}#{i}', 'session': sid, 'date': date, 'role': t['role'],
                        'text': t['content'], 'evidence': bool(t.get('has_answer'))})
    return out


def doc_text(m) -> str:
    return f"[{m['date']}] {m['role']}: {m['text']}"[:DOC_CHARS]


# ---------- BM25 (multilingual tokenizer) ----------
_STOP = frozenset('a an the and or of to in on at for with by from is are was were be been i you he she it we they '
                  'me my your our their this that these those do does did have has had what which who how when '
                  'where why can could would should will just so as if not no yes'.split())
_LATIN = re.compile(r'[a-z0-9]+(?:[\'.-][a-z0-9]+)*')
_CJK = re.compile(r'[㐀-鿿]+')


def tokenize(text: str) -> list[str]:
    text = text.lower()
    toks = [t for t in _LATIN.findall(text) if t not in _STOP]
    cjk = _CJK.findall(text)
    if cjk:
        import jieba  # lazy: English-only corpora never pay for it
        for run in cjk:
            toks += [w for w in jieba.lcut(run) if w.strip()]
            toks += [run[i:i + 2] for i in range(len(run) - 1)]  # char bigrams as a safety net
    return toks


class BM25:
    def __init__(self, docs: list[str], k1: float = 1.2, b: float = 0.75):
        self.toks = [Counter(tokenize(d)) for d in docs]
        self.len = np.array([sum(c.values()) for c in self.toks], dtype=float)
        self.avg = float(self.len.mean() or 1)
        df = Counter(t for c in self.toks for t in c)
        n = len(docs)
        self.idf = {t: math.log(1 + (n - f + 0.5) / (f + 0.5)) for t, f in df.items()}
        self.k1, self.b = k1, b

    def scores(self, query: str) -> np.ndarray:
        q = set(tokenize(query))
        out = np.zeros(len(self.toks))
        for i, c in enumerate(self.toks):
            norm = self.k1 * (1 - self.b + self.b * self.len[i] / self.avg)
            out[i] = sum(self.idf[t] * c[t] * (self.k1 + 1) / (c[t] + norm) for t in q if t in c)
        return out


# ---------- HTTP model clients ----------
def post(url: str, payload: dict, timeout: float = 120) -> dict:
    req = urllib.request.Request(url, data=json.dumps(payload).encode(),
                                 headers={'Content-Type': 'application/json'}, method='POST')
    with urllib.request.urlopen(req, timeout=timeout) as r:
        return json.loads(r.read())


def e5_embed(texts: list[str], kind: str, batch: int = 32) -> np.ndarray:
    """Local CPU e5-small (fastembed ONNX) by default, so the bake-off never competes
    with the host's GPU services; PM1_E5=service uses the shared HTTP service instead."""
    if os.environ.get('PM1_E5', 'local') == 'local':
        import local_e5
        return local_e5.embed(texts, kind)
    prefix = 'query: ' if kind == 'query' else 'passage: '
    vecs = []
    for i in range(0, len(texts), batch):
        out = post(E5_URL, {'model': E5_MODEL, 'input': [prefix + t for t in texts[i:i + batch]]})
        vecs += [d['embedding'] for d in sorted(out['data'], key=lambda d: d['index'])]
    v = np.asarray(vecs, dtype=np.float32)
    return v / np.linalg.norm(v, axis=1, keepdims=True).clip(1e-9)


def e5_docs(qid: str, docs: list[str]) -> np.ndarray:
    path = CACHE / 'e5' / f'{qid}.npy'
    if path.exists():
        v = np.load(path)
        if len(v) == len(docs):
            return v
    v = e5_embed(docs, 'passage')
    path.parent.mkdir(parents=True, exist_ok=True)
    np.save(path, v)
    return v


def pplx_encode(texts: list[str], kind: str, batch: int = 8) -> list[np.ndarray]:
    out = []
    for i in range(0, len(texts), batch):
        res = post(PPLX_URL, {'texts': texts[i:i + batch], 'kind': kind})
        for d in sorted(res['data'], key=lambda d: d['index']):
            m = np.asarray(d['embedding'], dtype=np.float32)
            out.append(m / np.linalg.norm(m, axis=1, keepdims=True).clip(1e-9))
    return out


def pplx_doc(qid: str, mid: str, text: str) -> np.ndarray:
    """Document token vectors are computed once (offline indexing) and cached as fp16."""
    path = CACHE / 'pplx' / qid / (hashlib.sha1(mid.encode()).hexdigest() + '.npy')
    if path.exists():
        return np.load(path).astype(np.float32)
    [m] = pplx_encode([text], 'document')
    path.parent.mkdir(parents=True, exist_ok=True)
    np.save(path, m.astype(np.float16))
    return m


def maxsim(q: np.ndarray, d: np.ndarray) -> float:
    return float((q @ d.T).max(axis=1).sum())


def rerank(url: str, query: str, docs: list[str]) -> list[float]:
    res = post(url.rstrip('/') + '/v1/rerank', {'query': query, 'documents': docs, 'top_n': len(docs)})
    scores = [0.0] * len(docs)
    for r in res['results']:
        scores[r['index']] = float(r['relevance_score'])
    return scores


# ---------- metrics ----------
def rrf(rankings: list[list[int]], k: int = 60) -> list[int]:
    score: dict[int, float] = defaultdict(float)
    for ranking in rankings:
        for r, i in enumerate(ranking):
            score[i] += 1 / (k + r + 1)
    return sorted(score, key=lambda i: -score[i])


def judge(order: list[int], msgs: list[dict], answer_sessions: set[str]) -> dict:
    sessions: list[str] = []
    for i in order:
        s = msgs[i]['session']
        if s not in sessions:
            sessions.append(s)
    out = {f'S@{k}': any(s in answer_sessions for s in sessions[:k]) for k in (1, 3, 5, 10)}
    ev = [r for r, i in enumerate(order) if msgs[i]['evidence']]
    if any(m['evidence'] for m in msgs):
        out.update({f'T@{k}': bool(ev and ev[0] < k) for k in (5, 10, 20)})
        out['T-MRR'] = 1 / (ev[0] + 1) if ev else 0.0
    return out


def summarize(rows: list[dict]) -> dict:
    keys = sorted({k for r in rows for k in r if k[0] in 'ST'})
    return {k: round(statistics.mean(float(r[k]) for r in rows if k in r), 4) for k in keys} | {'n': len(rows)}


def auc(pos: list[float], neg: list[float]) -> float | None:
    if not pos or not neg:
        return None
    allv = sorted([(v, 1) for v in pos] + [(v, 0) for v in neg])
    ranks, rank_sum = {}, 0.0
    i = 0
    while i < len(allv):  # average ranks for ties
        j = i
        while j < len(allv) and allv[j][0] == allv[i][0]:
            j += 1
        for t in range(i, j):
            ranks[t] = (i + j + 1) / 2
        i = j
    rank_sum = sum(ranks[t] for t, (_, lab) in enumerate(allv) if lab)
    return round((rank_sum - len(pos) * (len(pos) + 1) / 2) / (len(pos) * len(neg)), 4)


def pct(xs: list[float], p: float) -> float:
    xs = sorted(xs)
    return round(xs[min(len(xs) - 1, int(p * len(xs)))], 1) if xs else 0.0


# ---------- stages ----------
def first_stage(qs, topk: int = 100) -> dict:
    rows = {'bm25': [], 'e5': [], 'rrf': []}
    lat = {'bm25': [], 'e5': []}
    for n, q in enumerate(qs, 1):
        msgs = messages(q)
        docs = [doc_text(m) for m in msgs]
        e5d = e5_docs(q['question_id'], docs)  # indexing cost, excluded from latency
        bm = BM25(docs)
        t0 = time.perf_counter(); bs = bm.scores(q['question']); lat['bm25'].append((time.perf_counter() - t0) * 1e3)
        t0 = time.perf_counter(); qv = e5_embed([q['question']], 'query')[0]; es = e5d @ qv
        lat['e5'].append((time.perf_counter() - t0) * 1e3)
        orders = {'bm25': list(np.argsort(-bs)[:topk]), 'e5': list(np.argsort(-es)[:topk])}
        orders['rrf'] = rrf([orders['bm25'], orders['e5']])[:topk]
        cand = CACHE / 'first' / f"{q['question_id']}.json"
        cand.parent.mkdir(parents=True, exist_ok=True)
        cand.write_text(json.dumps({k: [int(i) for i in v] for k, v in orders.items()}))
        if not q['question_id'].endswith('_abs'):
            for k, order in orders.items():
                rows[k].append(judge(order, msgs, set(q['answer_session_ids'])))
        if n % 10 == 0:
            print(f'  first stage {n}/{len(qs)}', flush=True)
    return {'methods': {k: summarize(v) for k, v in rows.items()},
            'latency_ms': {k: {'p50': pct(v, .5), 'p95': pct(v, .95)} for k, v in lat.items()}}


def rerank_stage(qs, rerankers: dict[str, str], k: int = 40) -> dict:
    res = {}
    for name, url in rerankers.items():
        rows, lat, pos, neg, top_ans, top_abs, index_ms = [], [], [], [], [], [], []
        for q in qs:
            msgs = messages(q)
            fused = json.loads((CACHE / 'first' / f"{q['question_id']}.json").read_text())['rrf']
            cand = fused[:k]
            texts = [doc_text(msgs[i]) for i in cand]
            if name.startswith('pplx'):
                t0 = time.perf_counter()
                dvecs = [pplx_doc(q['question_id'], msgs[i]['id'], texts[j]) for j, i in enumerate(cand)]
                index_ms.append((time.perf_counter() - t0) * 1e3)
                t0 = time.perf_counter()
                [qv] = pplx_encode([q['question']], 'query')
                scores = [maxsim(qv, d) for d in dvecs]
            else:
                t0 = time.perf_counter()
                scores = rerank(url, q['question'], texts)
            lat.append((time.perf_counter() - t0) * 1e3)
            order = [cand[j] for j in np.argsort(-np.asarray(scores), kind='stable')] + fused[k:]
            if q['question_id'].endswith('_abs'):
                top_abs.append(max(scores))
                continue
            top_ans.append(max(scores))
            rows.append(judge(order, msgs, set(q['answer_session_ids'])))
            has_ev = any(m['evidence'] for m in msgs)
            for j, i in enumerate(cand):
                rel = msgs[i]['evidence'] if has_ev else msgs[i]['session'] in q['answer_session_ids']
                (pos if rel else neg).append(scores[j])
        res[name] = {'metrics': summarize(rows),
                     'latency_ms': {'p50': pct(lat, .5), 'p95': pct(lat, .95), 'candidates': k},
                     'relevance_auc': auc(pos, neg),
                     'abstention_auc': auc(top_ans, top_abs),
                     'score_range': [round(min(pos + neg), 3), round(max(pos + neg), 3)] if pos else None}
        if index_ms:
            res[name]['doc_indexing_ms_per_query_first_run'] = pct(index_ms, .5)
        print(name, json.dumps(res[name]), flush=True)
    return res


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument('--stage', choices=['first', 'rerank'], required=True)
    ap.add_argument('--limit', type=int)
    ap.add_argument('--reranker', action='append', default=[], help='name=url, or pplx=pplx')
    ap.add_argument('--k', type=int, default=40)
    ap.add_argument('--report', default=None)
    a = ap.parse_args()
    qs = load('dev', a.limit)
    out = {'split': 'dev', 'questions': len(qs), 'date': time.strftime('%Y-%m-%d %H:%M')}
    if a.stage == 'first':
        out |= first_stage(qs)
    else:
        out['rerank'] = rerank_stage(qs, dict(s.split('=', 1) for s in a.reranker), a.k)
    path = Path(a.report or Path(__file__).parent / 'reports' / f'bakeoff_{a.stage}.json')
    path.parent.mkdir(parents=True, exist_ok=True)
    prev = json.loads(path.read_text()) if path.exists() and a.stage == 'rerank' else {}
    if a.stage == 'rerank':
        out['rerank'] = prev.get('rerank', {}) | out['rerank']
    path.write_text(json.dumps(out, indent=2) + '\n')
    print(json.dumps({k: v for k, v in out.items() if k != 'rerank'}, indent=1)[:3000])


if __name__ == '__main__':
    main()
