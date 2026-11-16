"""fastembed cross-encoders (ONNX) as rerankers of the shipped RRF top-K on LME dev, plus a Chinese sanity check.

  .venv-gpu/bin/python eval/ce_rerank_eval.py Xenova/ms-marco-MiniLM-L-12-v2 [--k 32] [--limit N] [--cpu]
"""
from __future__ import annotations

import argparse
import json
import statistics
import sys
import time
from pathlib import Path

import numpy as np
import onnxruntime

onnxruntime.preload_dlls()
from fastembed.rerank.cross_encoder import TextCrossEncoder  # noqa: E402

sys.path.insert(0, str(Path(__file__).resolve().parent))
from lme import DATA, VecCache, fill, ingest, span_texts, split_of  # noqa: E402
from rerank_eval import judge  # noqa: E402

from pm1 import Recaller  # noqa: E402
from pm1.embed import SplitE5  # noqa: E402
from pm1.recall import rrf  # noqa: E402

ZH = [('我的本科是在哪里读的？', ['我2015年在复旦大学读完了计算机本科。', '我周末喜欢去爬山。', '我女儿的生日是五月十二日。']),
      ('我的狗叫什么名字？', ['我周末喜欢去爬山。', '我家的狗叫旺财，是一只金毛。', '下个月我要去东京出差。']),
      ('我下个月去哪里出差？', ['我家的狗叫旺财，是一只金毛。', '我周末喜欢去爬山。', '下个月我要去东京出差，记得带转换插头。'])]


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument('model')
    ap.add_argument('--k', type=int, default=32)
    ap.add_argument('--limit', type=int)
    ap.add_argument('--cpu', action='store_true')
    a = ap.parse_args()
    ce = TextCrossEncoder(a.model, providers=['CPUExecutionProvider' if a.cpu else 'CUDAExecutionProvider'])
    zh_hits = 0
    for q, docs in ZH:
        s = list(ce.rerank(q, docs))
        gold = next(i for i, d in enumerate(docs) if any(w in d for w in ('复旦', '旺财', '东京')))
        zh_hits += int(np.argmax(s) == gold)
        print('zh', q, [round(float(x), 2) for x in s])
    print(f'Chinese sanity: {zh_hits}/{len(ZH)} correct top-1', flush=True)
    dense, cache = SplitE5(), VecCache()
    qs = [q for q in json.loads(DATA.read_text())
          if split_of(q['question_id']) == 'dev' and not q['question_id'].endswith('_abs')][:a.limit]
    tag = a.model.split('/')[-1]
    res = {'rrf': [], f'{tag}|rerank': [], f'{tag}|fused': []}
    lat = []
    for qi, q in enumerate(qs, 1):
        store, meta = ingest(q)
        fill(store, dense, 'dense', cache)
        ans = set(q['answer_session_ids'])
        shipped, _, _ = Recaller(store, dense=dense).rank('lme', q['question_id'], q['question'], 'rrf')
        head, tail = shipped[:a.k], shipped[a.k:]
        texts = span_texts(store, set(head))
        t0 = time.perf_counter()
        scores = list(ce.rerank(q['question'], [texts[s] for s in head], batch_size=8))
        lat.append((time.perf_counter() - t0) * 1e3)
        ranked = [head[j] for j in np.argsort(-np.asarray(scores), kind='stable')]
        res['rrf'].append(judge(shipped, meta, ans))
        res[f'{tag}|rerank'].append(judge(ranked + tail, meta, ans))
        res[f'{tag}|fused'].append(judge(rrf([head, ranked]) + tail, meta, ans))
        store.close()
        if qi % 10 == 0 or qi == len(qs):
            rep = {'questions': qi, 'k': a.k, 'chinese_sanity_top1': f'{zh_hits}/{len(ZH)}',
                   'modes': {k: {m: round(statistics.mean(float(r[m]) for r in v), 3) for m in v[0]} for k, v in res.items()},
                   'rerank_ms': {'p50': round(float(np.percentile(lat, 50)), 1), 'p95': round(float(np.percentile(lat, 95)), 1)}}
            (Path(__file__).parent / 'reports' / f'ce_rerank_{tag}.json').write_text(json.dumps(rep, indent=2) + '\n')
            print(json.dumps({'n': qi, **{k: (v['T@1'], v['T@10'], v['T-MRR']) for k, v in rep['modes'].items()}}), flush=True)


if __name__ == '__main__':
    main()
