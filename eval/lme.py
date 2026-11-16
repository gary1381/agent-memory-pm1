#!/usr/bin/env python3
"""LongMemEval-S retrieval evaluation THROUGH pm1's own Store + Recaller (what ships is what is measured).

Each question's haystack is ingested into an isolated in-memory pm1 Store; doc vectors
are computed once per span text and cached (~/.cache/pm1/vec_cache.sqlite), so online
latency counts only query-time work. DEV split only unless --split test is given
explicitly after the configuration is frozen.

  .venv/bin/python eval/lme.py --limit 40 --modes lexical,dense,rrf,hybrid
"""
from __future__ import annotations

import argparse
import hashlib
import json
import os
import sqlite3
import statistics
import sys
import time
from pathlib import Path

import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / 'src'))
from pm1 import Recaller, Store  # noqa: E402

DATA = Path.home() / '.cache/pm1/data/longmemeval_s_cleaned.json'
VCACHE = Path(os.environ.get('PM1_VCACHE') or Path.home() / '.cache/pm1/vec_cache.sqlite')
FAILURES: dict[str, int] = {}  # spans a model could not encode (reported, never hidden)


def split_of(qid: str) -> str:
    return 'dev' if int(hashlib.sha256(qid.encode()).hexdigest(), 16) % 5 == 0 else 'test'


class VecCache:
    def __init__(self, path: Path = VCACHE):
        path.parent.mkdir(parents=True, exist_ok=True)
        self.con = sqlite3.connect(str(path))
        self.con.execute('CREATE TABLE IF NOT EXISTS v (k TEXT PRIMARY KEY, shape TEXT, blob BLOB)')

    @staticmethod
    def key(model: str, kind: str, text: str) -> str:
        return hashlib.sha1(f'{model}|{kind}|{text}'.encode()).hexdigest()

    def get_many(self, keys: list[str]) -> dict[str, np.ndarray]:
        out = {}
        for i in range(0, len(keys), 500):
            chunk = keys[i:i + 500]
            for k, shape, blob in self.con.execute(
                    f"SELECT k, shape, blob FROM v WHERE k IN ({','.join('?' * len(chunk))})", chunk):
                out[k] = np.frombuffer(blob, dtype=np.float16).reshape(json.loads(shape)).astype(np.float32)
        return out

    def put_many(self, items: dict[str, np.ndarray]) -> None:
        self.con.executemany('INSERT OR REPLACE INTO v VALUES (?,?,?)',
                             [(k, json.dumps(list(m.shape)), np.asarray(m, dtype=np.float16).tobytes())
                              for k, m in items.items()])
        self.con.commit()


def session_label(qid: str, sid: str) -> str:
    """Opaque session tag stored in pm1 (and therefore shown in every packed context line).

    LongMemEval session ids encode the gold labels: evidence sessions are named 'answer_<hex>' and the evidence
    session of an abstention question 'answer_<hex>_abs'; filler sessions are 'sharegpt_*' / 'ultrachat_*' / hex.
    Storing raw ids let the answerer see which excerpts are evidence (and that a question is unanswerable).
    Scoring keeps the raw id in `meta`, so retrieval metrics are unchanged."""
    return 's' + hashlib.sha1(f'{qid}|{sid}'.encode()).hexdigest()[:8]


RAW_SESSION_IDS = os.environ.get('PM1_LME_RAW_SESSION_IDS') == '1'   # only to reproduce pre-2026-10-16 exports


def ingest(q, raw_session_ids: bool = RAW_SESSION_IDS) -> tuple[Store, dict[int, dict]]:
    store = Store(':memory:')
    meta: dict[int, dict] = {}
    for sid, date, sess in zip(q['haystack_session_ids'], q['haystack_dates'], q['haystack_sessions']):
        label = sid if raw_session_ids else session_label(q['question_id'], sid)
        for i, t in enumerate(sess):
            if not t['content'].strip():
                continue
            ep = store.add_episode(bank='lme', scope=q['question_id'], text=t['content'], role=t['role'],
                                   session=label, seq=i, occurred_at=date)
            for span in ep['span_ids']:
                meta[span] = {'session': sid, 'evidence': bool(t.get('has_answer')), 'episode': ep['episode_id']}
    return store, meta


def span_texts(store: Store, ids: list[int] | None = None) -> dict[int, str]:
    sql = 'SELECT s.id, s.start, s.end, e.text FROM spans s JOIN episodes e ON e.id=s.episode_id'
    rows = store.con.execute(sql).fetchall()
    return {r[0]: r[3][r[1]:r[2]] for r in rows if ids is None or r[0] in ids}


def fill(store: Store, emb, kind: str, cache: VecCache, ids: list[int] | None = None, batch: int = 64) -> float:
    """Ensure vectors for spans; returns seconds spent computing (offline indexing cost)."""
    texts = span_texts(store, set(ids) if ids is not None else None)
    keys = {sid: VecCache.key(emb.name, kind, t) for sid, t in texts.items()}
    have = cache.get_many(list(keys.values()))
    todo = [sid for sid in texts if keys[sid] not in have]
    t0 = time.perf_counter()
    for i in range(0, len(todo), batch):
        chunk = todo[i:i + batch]
        if kind == 'dense':
            new = dict(zip([keys[s] for s in chunk], emb.embed([texts[s] for s in chunk], 'passage')))
        else:
            new = {}
            try:
                new = dict(zip([keys[s] for s in chunk], emb.encode([texts[s] for s in chunk], 'document')))
            except Exception:
                # The shared GPU service can run out of memory; retry one by one, then shortened.
                for s in chunk:
                    for text in (texts[s], texts[s][:600]):
                        try:
                            new[keys[s]] = emb.encode([text], 'document')[0]
                            break
                        except Exception:
                            continue
                    else:
                        FAILURES[kind] = FAILURES.get(kind, 0) + 1
        cache.put_many(new)
        have |= new
    spent = time.perf_counter() - t0
    for sid, k in keys.items():
        if k not in have:
            continue  # not encoded: the span keeps its first-stage position
        if kind == 'dense':
            store.put_vector(sid, emb.name, have[k])
        else:
            store.put_tokvecs(sid, emb.name, have[k])
    return spent


def judge(order: list[int], meta: dict[int, dict], answer_sessions: set[str]) -> dict:
    sessions, eps = [], []
    for sid in order:
        s = meta[sid]['session']
        if s not in sessions:
            sessions.append(s)
    out = {f'S@{k}': any(s in answer_sessions for s in sessions[:k]) for k in (1, 3, 5, 10)}
    if any(m['evidence'] for m in meta.values()):
        ev = [r for r, sid in enumerate(order) if meta[sid]['evidence']]
        out |= {f'T@{k}': bool(ev and ev[0] < k) for k in (5, 10, 20)}
        out['T-MRR'] = 1 / (ev[0] + 1) if ev else 0.0
    return out


def mean_rows(rows: list[dict]) -> dict:
    keys = sorted({k for r in rows for k in r})
    return {k: round(statistics.mean(float(r[k]) for r in rows if k in r), 4) for k in keys} | {'n': len(rows)}


def pct(xs, p):
    xs = sorted(xs)
    return round(xs[min(len(xs) - 1, int(p * len(xs)))], 1) if xs else None


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument('--split', default='dev', choices=['dev', 'test'])
    ap.add_argument('--limit', type=int)
    ap.add_argument('--modes', default='lexical,dense,rrf,hybrid')
    ap.add_argument('--k-rerank', type=int, default=32)
    ap.add_argument('--dense', default='split', choices=['split', 'local'],
                    help='split = e5 Q8 GGUF (docs on a GPU host, queries local CPU); local = int8 ONNX on CPU')
    ap.add_argument('--report', default=str(Path(__file__).parent / 'reports' / 'lme_retrieval_dev.json'))
    a = ap.parse_args()
    modes = a.modes.split(',')
    from pm1.embed import LocalE5, PplxLate, SplitE5
    dense = SplitE5() if a.dense == 'split' else LocalE5()
    token = PplxLate() if {'hybrid', 'fuse3'} & set(modes) else None
    cache = VecCache()
    qs = [q for q in json.loads(DATA.read_text()) if split_of(q['question_id']) == a.split][:a.limit]
    rows = {m: [] for m in modes}
    by_type = {m: {} for m in modes}
    lat = {m: [] for m in modes}
    index_s = {'dense': 0.0, 'tok': 0.0}
    started = time.time()
    for n, q in enumerate(qs, 1):
        store, meta = ingest(q)
        index_s['dense'] += fill(store, dense, 'dense', cache)
        rec = Recaller(store, dense=dense, token=token, k_rerank=a.k_rerank, encode_missing=0)
        for mode in modes:
            if mode in ('hybrid', 'fuse3'):
                head, _, _ = rec.rank('lme', q['question_id'], q['question'], 'rrf')
                index_s['tok'] += fill(store, token, 'tok', cache, head[:a.k_rerank], batch=8)
            order, _, diag = rec.rank('lme', q['question_id'], q['question'], mode)
            if diag.get('rerank_unindexed'):
                FAILURES['rerank_unindexed_spans'] = FAILURES.get('rerank_unindexed_spans', 0) + diag['rerank_unindexed']
            if 'rerank_error' in diag:
                FAILURES['rerank_fallback'] = FAILURES.get('rerank_fallback', 0) + 1
            lat[mode].append(sum(v for k, v in diag.items() if k.endswith('_ms')))
            if q['question_id'].endswith('_abs'):
                continue
            j = judge(order, meta, set(q['answer_session_ids']))
            rows[mode].append(j)
            by_type[mode].setdefault(q['question_type'], []).append(j)
        store.close()
        if n % 5 == 0 or n in (1, len(qs)):
            report = {'split': a.split, 'questions': n, 'k_rerank': a.k_rerank, 'dense_model': dense.name,
                      'dense_fallbacks': getattr(dense, 'fallbacks', None),
                      'elapsed_min': round((time.time() - started) / 60, 1),
                      'offline_indexing_s': {k: round(v, 1) for k, v in index_s.items()},
                      'encode_failures': dict(FAILURES),
                      'modes': {m: {'metrics': mean_rows(rows[m]),
                                    'online_latency_ms': {'p50': pct(lat[m], .5), 'p95': pct(lat[m], .95)},
                                    'by_type': {t: mean_rows(v) for t, v in sorted(by_type[m].items())}}
                                for m in modes}}
            Path(a.report).parent.mkdir(parents=True, exist_ok=True)
            Path(a.report).write_text(json.dumps(report, indent=2) + '\n')
            print(json.dumps({'n': n, **{m: {k: report['modes'][m]['metrics'].get(k) for k in ('S@5', 'T@10')}
                                         | report['modes'][m]['online_latency_ms'] for m in modes}}), flush=True)


if __name__ == '__main__':
    main()
