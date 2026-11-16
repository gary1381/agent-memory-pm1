"""Late-interaction (ColBERT-family) models as FIRST stage on the LME dev split, against the shipped RRF.

Runs in .venv-gpu (fastembed-gpu). Every span of every haystack is encoded (no cache: token matrices are
large); MaxSim over all spans is one matmul + segment max. Modes per model:
  cb        exhaustive MaxSim alone
  rrf3      RRF(BM25 top-50, e5 top-50, MaxSim top-50)
  cb_rerank MaxSim over the shipped RRF top-32 (same shape as the pplx 'hybrid' mode)

  .venv-gpu/bin/python eval/colbert_eval.py colbert-ir/colbertv2.0 answerdotai/answerai-colbert-small-v1
"""
from __future__ import annotations

import json
import os
import statistics
import sys
import time
from pathlib import Path

import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parent))
from lme import DATA, VecCache, fill, ingest, span_texts, split_of  # noqa: E402

from pm1 import Recaller  # noqa: E402
from pm1.embed import SplitE5  # noqa: E402
from pm1.recall import rrf  # noqa: E402


def judge(order, meta, ans):
    sessions = []
    for sid in order:
        if meta[sid]['session'] not in sessions:
            sessions.append(meta[sid]['session'])
    ev = next((r for r, sid in enumerate(order) if meta[sid]['evidence']), None)
    out = {f'T@{k}': ev is not None and ev < k for k in (1, 5, 10, 20, 32)}
    out['S@5'] = any(s in ans for s in sessions[:5])
    out['T-MRR'] = 1 / (ev + 1) if ev is not None else 0.0
    return out


def main():
    import onnxruntime
    onnxruntime.preload_dlls()
    from fastembed import LateInteractionTextEmbedding
    names = sys.argv[1:]
    class _BgeM3Adapter:
        # BGE-M3 ColBERT head over a llama.cpp GGUF server (pm1.bgem3); same interface as fastembed.
        def __init__(self, url):
            from pm1.bgem3 import BgeM3
            self.m = BgeM3(url=url)

        def embed(self, texts, batch_size=8):
            return iter(self.m.encode(texts))

        def query_embed(self, texts):
            return iter(self.m.encode(texts))

    def _make(n):
        if n.startswith('bge-m3='):
            return _BgeM3Adapter(n.split('=', 1)[1])
        if n.startswith('llama-colbert='):   # llama-colbert=<label>|<url>|<model dir>[|noexpand]
            from pm1.llama_colbert import LlamaColbert
            _label, url, d, *flags = n.split('=', 1)[1].split('|')
            return LlamaColbert(url, d, expand_query='noexpand' not in flags)
        return LateInteractionTextEmbedding(n, providers=['CUDAExecutionProvider'])

    models = {n: _make(n) for n in names}
    dense, cache = SplitE5(), VecCache()
    qs = [q for q in json.loads(DATA.read_text())
          if split_of(q['question_id']) == 'dev' and not q['question_id'].endswith('_abs')][:int(os.environ.get('PM1_CB_LIMIT', '1000'))]
    res: dict[str, list] = {'rrf': []}
    lat: dict[str, list] = {}
    enc_rate: dict[str, list] = {n: [] for n in names}
    for n in names:
        for m in ('cb', 'rrf3', 'cb_rerank'):
            res[f'{n}|{m}'] = []
    for qi, q in enumerate(qs, 1):
        store, meta = ingest(q)
        fill(store, dense, 'dense', cache)
        scope, ans = q['question_id'], set(q['answer_session_ids'])
        full = Recaller(store, dense=dense, k_first=100000)
        lex, _, _ = full.rank('lme', scope, q['question'], 'lexical')
        den, _, _ = full.rank('lme', scope, q['question'], 'dense')
        shipped = rrf([lex[:50], den[:50]])
        res['rrf'].append(judge(shipped, meta, ans))
        texts = span_texts(store)
        ids = list(texts)
        for n, m in models.items():
            t0 = time.perf_counter()
            docs = list(m.embed([texts[i] for i in ids], batch_size=8))
            enc_rate[n].append(len(ids) / (time.perf_counter() - t0))
            lens = np.array([d.shape[0] for d in docs])
            starts = np.concatenate([[0], np.cumsum(lens)[:-1]])
            D = np.concatenate(docs).astype(np.float32)
            t0 = time.perf_counter()
            qv = np.asarray(next(iter(m.query_embed([q['question']]))), dtype=np.float32)
            t_enc = time.perf_counter() - t0
            t0 = time.perf_counter()
            sims = qv @ D.T                                   # (q_tokens, all doc tokens)
            per_doc = np.maximum.reduceat(sims, starts, axis=1).sum(axis=0)
            cb = [ids[j] for j in np.argsort(-per_doc)]
            t_scan = time.perf_counter() - t0
            score = dict(zip(ids, per_doc))
            head = shipped[:32]
            rerank = sorted(head, key=lambda i: -score[i]) + shipped[32:]
            res[f'{n}|cb'].append(judge(cb, meta, ans))
            res[f'{n}|rrf3'].append(judge(rrf([lex[:50], den[:50], cb[:50]]), meta, ans))
            res[f'{n}|cb_rerank'].append(judge(rerank, meta, ans))
            lat.setdefault(f'{n}|query_encode_ms', []).append(t_enc * 1e3)
            lat.setdefault(f'{n}|maxsim_scan_ms', []).append(t_scan * 1e3)
        store.close()
        if qi % 10 == 0 or qi == len(qs):
            report = {'questions': qi,
                      'modes': {k: {m: round(statistics.mean(float(r[m]) for r in v), 3) for m in v[0]}
                                for k, v in res.items()},
                      'latency_ms': {k: {'p50': round(float(np.percentile(v, 50)), 1),
                                         'p95': round(float(np.percentile(v, 95)), 1)} for k, v in lat.items()},
                      'encode_docs_per_s': {n: round(statistics.mean(v), 1) for n, v in enc_rate.items()}}
            Path(__file__).with_name('reports').joinpath(os.environ.get('PM1_CB_REPORT', 'colbert_dev.json')).write_text(
                json.dumps(report, indent=2) + '\n')
            print(json.dumps({'n': qi, **{k: (v['T@10'], v['T-MRR']) for k, v in report['modes'].items()}}),
                  flush=True)


if __name__ == '__main__':
    main()
