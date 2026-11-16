"""Chinese questions over English records, full first stage + rerank (2026-10-29): do the e5 vectors already bridge
the languages, or does translating the query (zh -> en) still help, and merged vs separate keyword lanes?

xling_locomo.py's translated LoCoMo questions; a fixed sample (seed 0). Per question: keyword top 50 (pm1.text tokens,
FTS5 BM25, one conversation) + e5 top 50 (the pm1v3 store's own vectors, pm1's query embedder), deduplicated into one
pool, reranked by the production Qwen3 reranker with the question as asked. Reported: evidence in the pool, and
R@1 / R@5 / R@10 after the rerank.

  en      English question everywhere (upper bound)
  zh      Chinese question, no translation
  merged  keyword query = Chinese + English translation tokens; vectors and rerank on the Chinese question
  lanes   keyword top 50 for the Chinese and top 50 for the translation, both into the pool
  dense2  merged + e5 top 50 of the translation too

  PM1_DENSE=split PM1_RERANK_QWEN_VLLM=http://127.0.0.1:18166 ../.venv/bin/python xling_full.py [--n 300]
"""
from __future__ import annotations

import os
import argparse
import collections
import json
import random
import sqlite3
import sys
from pathlib import Path

import numpy as np

EV = Path(__file__).resolve().parent
sys.path[:0] = [str(EV.parent / 'src'), str(EV)]
from pm1.rerank import router_from_env  # noqa: E402
from pm1.store import Store  # noqa: E402
from pm1.text import tokenize  # noqa: E402
from pm1.worker import build_embedders  # noqa: E402
from tokenizer_cmp_en import LOCOMO, load  # noqa: E402
from xling_locomo import CACHE  # noqa: E402

STORES = Path(os.environ.get('PM1_EVAL_RUNS', str(Path.home() / '.cache/pm1-eval/runs'))) / 'main/pm1v3'


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument('--n', type=int, default=300)
    a = ap.parse_args()
    docs, qs = load(LOCOMO)
    t = json.load(open(CACHE))
    random.Random(0).shuffle(qs)
    qs = qs[:a.n]
    dense, _ = build_embedders('split', 'none')
    reranker = router_from_env().rerankers[0]
    text_of = {d: text for d, _, text in docs}
    conv_docs = collections.defaultdict(list)
    for d, conv, _ in docs:
        conv_docs[conv].append(d)
    idx = {}
    for conv, ds in conv_docs.items():                  # per conversation: FTS index + e5 matrix aligned to doc ids
        st = Store(str(STORES / f'c{conv}' / 'pm1.sqlite'))
        sid = {r[1]: r[0] for r in st.con.execute('SELECT s.id, e.text FROM spans s JOIN episodes e ON e.id = s.episode_id')}
        ids, mat = st.dense_matrix('personal', f'proj:locomo-c{conv}', dense.name)
        row = {int(s): i for i, s in enumerate(ids)}
        keep = [d for d in ds if text_of[d] in sid and sid[text_of[d]] in row]
        con = sqlite3.connect(':memory:')
        con.execute("CREATE VIRTUAL TABLE f USING fts5(toks, tokenize='unicode61 remove_diacritics 0')")
        con.executemany('INSERT INTO f(rowid, toks) VALUES (?, ?)', ((i, ' '.join(tokenize(text_of[d])))
                                                                    for i, d in enumerate(keep)))
        idx[conv] = (keep, con, mat[[row[sid[text_of[d]]] for d in keep]])
        st.close()

    def bm25(conv, toks, n=50):
        keep, con, _ = idx[conv]
        toks = sorted(set(toks))
        if not toks:
            return []
        m = ' OR '.join('"' + x.replace('"', '""') + '"' for x in toks)
        return [keep[r[0]] for r in con.execute('SELECT rowid FROM f WHERE f MATCH ? ORDER BY bm25(f) LIMIT ?', (m, n))]

    def vec(conv, q, n=50):
        keep, _, mat = idx[conv]
        s = mat @ dense.embed([q], 'query')[0]
        return [keep[i] for i in np.argsort(-s)[:n]]

    def pool(*lists):
        seen = []
        for lst in lists:
            seen += [d for d in lst if d not in seen]
        return seen

    arms = {
        'en': lambda q, zh, en: (pool(bm25(q['conv'], tokenize(q['q'])), vec(q['conv'], q['q'])), q['q']),
        'zh': lambda q, zh, en: (pool(bm25(q['conv'], tokenize(zh)), vec(q['conv'], zh)), zh),
        'merged': lambda q, zh, en: (pool(bm25(q['conv'], tokenize(zh) + tokenize(en)), vec(q['conv'], zh)), zh),
        'lanes': lambda q, zh, en: (pool(bm25(q['conv'], tokenize(zh)), bm25(q['conv'], tokenize(en)),
                                         vec(q['conv'], zh)), zh),
        'dense2': lambda q, zh, en: (pool(bm25(q['conv'], tokenize(zh) + tokenize(en)), vec(q['conv'], zh),
                                          vec(q['conv'], en)), zh),
    }
    res = {k: {'pool': [], 'rank': [], 'size': []} for k in arms}
    for j, q in enumerate(qs):
        zh, en = t[q['q']]['zh'], t[q['q']]['back']
        for name, f in arms.items():
            p, rq = f(q, zh, en)
            r = res[name]
            r['pool'].append(any(d in q['ev'] for d in p))
            r['size'].append(len(p))
            scores = reranker.score(rq, [text_of[d] for d in p]) if p else []
            order = [p[i] for i in np.argsort(-np.asarray(scores))]
            r['rank'].append(next((k for k, d in enumerate(order) if d in q['ev']), None))
        if j % 50 == 49:
            print(f'{j + 1} questions', flush=True)
    print(f'{len(qs)} questions (seed 0)')
    for name, r in res.items():
        at = lambda k: sum(x is not None and x < k for x in r['rank']) / len(r['rank'])
        print(f'{name:7s} evidence in pool {sum(r["pool"]) / len(qs):.3f} (pool {np.mean(r["size"]):.0f})  '
              f'after rerank R@1 {at(1):.3f}  R@5 {at(5):.3f}  R@10 {at(10):.3f}', flush=True)


if __name__ == '__main__':
    main()
