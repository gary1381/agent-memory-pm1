"""Late-interaction model as a RERANKER of the shipped RRF top-K (only K docs encoded per question).

For models too slow to encode every span (jina-colbert-v2 int8 on a busy CPU, BGE-M3 over JSON).
  .venv-gpu/bin/python eval/li_rerank_eval.py jina-int8-cpu [--k 32] [--limit N]
"""
from __future__ import annotations

import argparse
import json
import statistics
import sys
import time
from pathlib import Path

import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parent))
from lme import DATA, VecCache, fill, ingest, span_texts, split_of  # noqa: E402
from rerank_eval import judge  # noqa: E402

from pm1 import Recaller  # noqa: E402
from pm1.embed import SplitE5  # noqa: E402
from pm1.recall import rrf  # noqa: E402


def make(name):
    if name.startswith('jina-int8'):
        import onnxruntime
        onnxruntime.preload_dlls()
        from fastembed import LateInteractionTextEmbedding
        prov = 'CUDAExecutionProvider' if name.endswith('gpu') else 'CPUExecutionProvider'
        return LateInteractionTextEmbedding('jinaai/jina-colbert-v2',
                                            specific_model_path=str(Path('~/models/jina-colbert-v2-int8').expanduser()),
                                            providers=[prov], threads=8)
    if name == 'bge-m3':
        from pm1.bgem3 import BgeM3
        m = BgeM3()

        class A:
            def embed(self, t, batch_size=8):
                return iter(m.encode(t))

            def query_embed(self, t):
                return iter(m.encode(t))
        return A()
    raise SystemExit(f'unknown model {name}')


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument('model')
    ap.add_argument('--k', type=int, default=32)
    ap.add_argument('--limit', type=int)
    a = ap.parse_args()
    model, dense, cache = make(a.model), SplitE5(), VecCache()
    qs = [q for q in json.loads(DATA.read_text())
          if split_of(q['question_id']) == 'dev' and not q['question_id'].endswith('_abs')][:a.limit]
    res = {'rrf': [], f'{a.model}|rerank': [], f'{a.model}|fused': []}
    lat = []
    out = Path(__file__).parent / 'reports' / f'li_rerank_{a.model}.json'
    for qi, q in enumerate(qs, 1):
        store, meta = ingest(q)
        fill(store, dense, 'dense', cache)
        ans = set(q['answer_session_ids'])
        shipped, _, _ = Recaller(store, dense=dense).rank('lme', q['question_id'], q['question'], 'rrf')
        head, tail = shipped[:a.k], shipped[a.k:]
        texts = span_texts(store, set(head))
        t0 = time.perf_counter()
        qv = np.asarray(next(iter(model.query_embed([q['question']]))), dtype=np.float32)
        docs = list(model.embed([texts[s] for s in head], batch_size=8))
        score = {s: float((qv @ np.asarray(d, dtype=np.float32).T).max(1).sum()) for s, d in zip(head, docs)}
        lat.append((time.perf_counter() - t0) * 1e3)
        ranked = sorted(head, key=lambda s: -score[s])
        res['rrf'].append(judge(shipped, meta, ans))
        res[f'{a.model}|rerank'].append(judge(ranked + tail, meta, ans))
        res[f'{a.model}|fused'].append(judge(rrf([head, ranked]) + tail, meta, ans))
        store.close()
        if qi % 10 == 0 or qi == len(qs):
            rep = {'questions': qi, 'k': a.k,
                   'modes': {k: {m: round(statistics.mean(float(r[m]) for r in v), 3) for m in v[0]} for k, v in res.items()},
                   'encode_k_docs_ms': {'p50': round(float(np.percentile(lat, 50)), 1),
                                        'p95': round(float(np.percentile(lat, 95)), 1)}}
            out.write_text(json.dumps(rep, indent=2) + '\n')
            print(json.dumps({'n': qi, **{k: (v['T@1'], v['T@10'], v['T-MRR']) for k, v in rep['modes'].items()}}), flush=True)


if __name__ == '__main__':
    main()
