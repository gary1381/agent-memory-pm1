"""Any ONNX sequence-classification cross-encoder (model.onnx + tokenizer.json) as reranker of RRF top-K.

  .venv-gpu/bin/python eval/ce_onnx_eval.py <label> <model.onnx> <tokenizer.json> [--cpu] [--max-len 512]
Includes the 3-question Chinese sanity check (gold judged by doc index, not keywords).
"""
from __future__ import annotations

import argparse
import json
import statistics
import sys
import time
from pathlib import Path

import numpy as np
import onnxruntime as ort

ort.preload_dlls()
from tokenizers import Tokenizer  # noqa: E402

sys.path.insert(0, str(Path(__file__).resolve().parent))
from lme import DATA, VecCache, fill, ingest, span_texts, split_of  # noqa: E402
from rerank_eval import judge  # noqa: E402

from pm1 import Recaller  # noqa: E402
from pm1.embed import SplitE5  # noqa: E402
from pm1.recall import rrf  # noqa: E402

ZH = [('我的本科是在哪里读的？', ['我2015年在复旦大学读完了计算机本科。', '我周末喜欢去爬山。', '我女儿的生日是五月十二日。'], 0),
      ('我的狗叫什么名字？', ['我周末喜欢去爬山。', '我家的狗叫旺财，是一只金毛。', '下个月我要去东京出差。'], 1),
      ('我下个月去哪里出差？', ['我家的狗叫旺财，是一只金毛。', '我周末喜欢去爬山。', '下个月我要去东京出差，记得带转换插头。'], 2)]


class OnnxCE:
    def __init__(self, model, tok, cpu=False, max_len=512, batch=8):
        prov = ['CPUExecutionProvider'] if cpu else ['CUDAExecutionProvider', 'CPUExecutionProvider']
        so = ort.SessionOptions()
        so.log_severity_level = 3
        self.s = ort.InferenceSession(model, so, providers=prov)
        self.inputs = {i.name for i in self.s.get_inputs()}
        self.tok = Tokenizer.from_file(tok)
        self.tok.enable_truncation(max_len)
        self.tok.enable_padding()
        self.batch = batch

    def score(self, query, docs):
        out = []
        for i in range(0, len(docs), self.batch):
            enc = self.tok.encode_batch([(query, d) for d in docs[i:i + self.batch]])
            feed = {'input_ids': np.array([e.ids for e in enc], dtype=np.int64),
                    'attention_mask': np.array([e.attention_mask for e in enc], dtype=np.int64)}
            if 'token_type_ids' in self.inputs:
                feed['token_type_ids'] = np.array([e.type_ids for e in enc], dtype=np.int64)
            logits = self.s.run(None, feed)[0]
            out += list(logits[:, 0] if logits.ndim == 2 else logits)
        return [float(x) for x in out]


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument('label')
    ap.add_argument('model')
    ap.add_argument('tokenizer')
    ap.add_argument('--cpu', action='store_true')
    ap.add_argument('--k', type=int, default=32)
    ap.add_argument('--max-len', type=int, default=512)
    ap.add_argument('--limit', type=int)
    a = ap.parse_args()
    ce = OnnxCE(a.model, a.tokenizer, a.cpu, a.max_len)
    print('providers', ce.s.get_providers(), flush=True)
    zh = 0
    for q, docs, gold in ZH:
        s = ce.score(q, docs)
        zh += int(int(np.argmax(s)) == gold)
        print('zh', q, [round(x, 2) for x in s], 'gold', gold)
    print(f'Chinese sanity: {zh}/{len(ZH)}', flush=True)
    dense, cache = SplitE5(), VecCache()
    qs = [q for q in json.loads(DATA.read_text())
          if split_of(q['question_id']) == 'dev' and not q['question_id'].endswith('_abs')][:a.limit]
    res = {'rrf': [], f'{a.label}|rerank': [], f'{a.label}|fused': []}
    lat = []
    for qi, q in enumerate(qs, 1):
        store, meta = ingest(q)
        fill(store, dense, 'dense', cache)
        ans = set(q['answer_session_ids'])
        shipped, _, _ = Recaller(store, dense=dense).rank('lme', q['question_id'], q['question'], 'rrf')
        head, tail = shipped[:a.k], shipped[a.k:]
        texts = span_texts(store, set(head))
        t0 = time.perf_counter()
        scores = ce.score(q['question'], [texts[s] for s in head])
        lat.append((time.perf_counter() - t0) * 1e3)
        ranked = [head[j] for j in np.argsort(-np.asarray(scores), kind='stable')]
        res['rrf'].append(judge(shipped, meta, ans))
        res[f'{a.label}|rerank'].append(judge(ranked + tail, meta, ans))
        res[f'{a.label}|fused'].append(judge(rrf([head, ranked]) + tail, meta, ans))
        store.close()
        if qi % 10 == 0 or qi == len(qs):
            rep = {'questions': qi, 'k': a.k, 'chinese_sanity_top1': f'{zh}/{len(ZH)}', 'providers': ce.s.get_providers(),
                   'modes': {k: {m: round(statistics.mean(float(r[m]) for r in v), 3) for m in v[0]} for k, v in res.items()},
                   'rerank_ms': {'p50': round(float(np.percentile(lat, 50)), 1), 'p95': round(float(np.percentile(lat, 95)), 1)}}
            (Path(__file__).parent / 'reports' / f'ce_onnx_{a.label}.json').write_text(json.dumps(rep, indent=2) + '\n')
            print(json.dumps({'n': qi, **{k: (v['T@1'], v['T@10'], v['T-MRR']) for k, v in rep['modes'].items()}}), flush=True)


if __name__ == '__main__':
    main()
