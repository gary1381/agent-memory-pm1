"""Rerank a frozen candidate pool (eval/data/lme_dev_pool_{en,zh}.json) with any scorer, per language setting.

Settings: en (EN query, EN docs), zh (ZH query, ZH docs), x (ZH query, EN docs: a Chinese user asking about
English memories). The pool order (English RRF) is the 'none' baseline. Scorers:
  llama=<url>            llama.cpp /v1/rerank (Qwen3-Reranker, gte-modernbert, bge-reranker-v2-m3 ...)
  onnx=<model>|<tok>     ONNX sequence-classification cross-encoder
  fe-li=<name>           fastembed late interaction (colbertv2 ...), MaxSim
  fe-ce=<name>           fastembed cross-encoder (ms-marco-MiniLM ...)
  e5                     e5-small-q8 cosine (the dense first stage, as a reranker)
  bm25                   jieba/latin BM25 inside the pool

  .venv-gpu/bin/python eval/pool_rerank_eval.py --settings en,zh,x qwen=llama=http://127.0.0.1:18161 ...
"""
from __future__ import annotations

import argparse
import json
import math
import statistics
import sys
import time
import urllib.request
from collections import Counter
from pathlib import Path

import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parent))
DATADIR = Path(__file__).parent / 'data'


def judge(order, cands):
    ev = next((r for r, i in enumerate(order) if cands[i]['evidence']), None)
    out = {f'T@{k}': ev is not None and ev < k for k in (1, 3, 5, 10)}
    out['T-MRR'] = 1 / (ev + 1) if ev is not None else 0.0
    return out


def llama_scorer(url):
    def f(q, docs):
        req = urllib.request.Request(url.rstrip('/') + '/v1/rerank',
                                     data=json.dumps({'query': q, 'documents': docs, 'top_n': len(docs)}).encode(),
                                     headers={'Content-Type': 'application/json'})
        with urllib.request.urlopen(req, timeout=600) as r:
            res = json.loads(r.read())['results']
        s = [0.0] * len(docs)
        for x in res:
            s[x['index']] = x['relevance_score']
        return s
    return f


def onnx_scorer(spec):
    from ce_onnx_eval import OnnxCE
    model, tok = spec.split('|')
    ce = OnnxCE(model, tok)
    return ce.score


def fe_li_scorer(name):
    import onnxruntime
    onnxruntime.preload_dlls()
    from fastembed import LateInteractionTextEmbedding
    m = LateInteractionTextEmbedding(name, providers=['CUDAExecutionProvider'])

    def f(q, docs):
        qv = np.asarray(next(iter(m.query_embed([q]))), dtype=np.float32)
        return [float((qv @ np.asarray(d, dtype=np.float32).T).max(1).sum()) for d in m.embed(docs, batch_size=8)]
    return f


def fe_ce_scorer(name):
    import onnxruntime
    onnxruntime.preload_dlls()
    from fastembed.rerank.cross_encoder import TextCrossEncoder
    ce = TextCrossEncoder(name, providers=['CUDAExecutionProvider', 'CPUExecutionProvider'])
    return lambda q, docs: [float(x) for x in ce.rerank(q, docs, batch_size=8)]


def e5_scorer():
    from pm1.embed import SplitE5
    e = SplitE5()

    def f(q, docs):
        return list(e.embed(docs, 'passage') @ e.embed([q], 'query')[0])
    return f


def bm25_scorer():
    from pm1.text import tokenize

    def f(q, docs):
        toks = [tokenize(d) for d in docs]
        df = Counter(t for ts in toks for t in set(ts))
        avg = sum(map(len, toks)) / max(1, len(toks))
        qt = tokenize(q)
        out = []
        for ts in toks:
            tf = Counter(ts)
            out.append(sum(math.log(1 + (len(toks) - df[t] + .5) / (df[t] + .5)) * tf[t] * 2.2
                           / (tf[t] + 1.2 * (.25 + .75 * len(ts) / avg)) for t in qt if t in tf))
        return out
    return f


def make(spec):
    kind, _, arg = spec.partition('=')
    return {'llama': lambda: llama_scorer(arg), 'onnx': lambda: onnx_scorer(arg), 'fe-li': lambda: fe_li_scorer(arg), 'fe-ce': lambda: fe_ce_scorer(arg),
            'e5': e5_scorer, 'bm25': bm25_scorer}[kind]()


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument('scorers', nargs='+', help='label=kind=arg')
    ap.add_argument('--settings', default='en,zh,x')
    ap.add_argument('--out', default=None)
    a = ap.parse_args()
    zh = json.loads((DATADIR / 'lme_dev_pool_zh.json').read_text())
    zh = [x for x in zh if x.get('complete')]     # same question set for every setting
    print(f'{len(zh)} complete questions, {len(zh[0]["cands"])} candidates each', flush=True)
    settings = a.settings.split(',')
    rows = {'none': [judge(list(range(len(x['cands']))), x['cands']) for x in zh]}
    lat = {}
    for spec in a.scorers:
        label, _, rest = spec.partition('=')
        score = make(rest)
        for st in settings:
            key = f'{label}|{st}'
            rows[key], lat[key] = [], []
            for x in zh:
                q = x['question_en'] if st == 'en' else x['question']
                docs = [c['text_en'] if st in ('en', 'x') else (c['text'] or c['text_en']) for c in x['cands']]
                t0 = time.perf_counter()
                s = score(q, docs)
                lat[key].append((time.perf_counter() - t0) * 1e3)
                order = [int(j) for j in np.argsort(-np.asarray(s), kind='stable')]
                rows[key].append(judge(order, x['cands']))
            m = {k: round(statistics.mean(float(r[k]) for r in rows[key]), 3) for k in rows[key][0]}
            print(json.dumps({key: m, 'p50_ms': round(float(np.percentile(lat[key], 50)), 1)}), flush=True)
    rep = {'questions': len(zh), 'modes': {k: {m: round(statistics.mean(float(r[m]) for r in v), 3) for m in v[0]}
                                            for k, v in rows.items()},
           'latency_ms_p50': {k: round(float(np.percentile(v, 50)), 1) for k, v in lat.items()}}
    out = Path(a.out) if a.out else Path(__file__).parent / 'reports' / ('pool_rerank_' + '_'.join(
        s.partition('=')[0] for s in a.scorers) + '.json')
    out.write_text(json.dumps(rep, indent=2, ensure_ascii=False) + '\n')


if __name__ == '__main__':
    main()
