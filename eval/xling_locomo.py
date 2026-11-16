"""Chinese questions over English records (2026-10-29): does translating the query help keyword search, and should the
translation be searched on its own and fused rather than merged into one query? BM25 only, no LLM.

LoCoMo's records are English. Its 1,531 questions (with evidence) are machine-translated to Chinese (quickmt en-zh,
pm1.xlate service), and the Chinese is translated back to English (quickmt zh-en) the way a live Chinese query would
be. Each turn is one document; rank of the first evidence turn per question, as tokenizer_cmp_en.py.

  en        the original English question (upper bound)
  zh        the Chinese question alone
  zh->en    its English translation alone
  merged    Chinese + English translation tokens in one OR query (what PM1_SESS_XLATE does today)
  lanes     Chinese and English translation searched separately, top 50 each, fused by reciprocal rank (k=60)

  PM1_XLATE=http://127.0.0.1:18192 ../.venv/bin/python xling_locomo.py
"""
from __future__ import annotations

import collections
import json
import os
import sqlite3
import sys
import urllib.request
from pathlib import Path

EV = Path(__file__).resolve().parent
sys.path[:0] = [str(EV.parent / 'src'), str(EV)]
from pm1.text import tokenize  # noqa: E402
from tokenizer_cmp_en import LOCOMO, load  # noqa: E402

URL = os.environ.get('PM1_XLATE', 'http://127.0.0.1:18192') + '/translate'
CACHE = Path.home() / '.cache/pm1-eval/xling/locomo_questions.json'


def tr(text: str, to: str) -> str:
    req = urllib.request.Request(URL, json.dumps({'text': text, 'to': to}).encode(), {'Content-Type': 'application/json'})
    return json.load(urllib.request.urlopen(req, timeout=30))['text']


def translations(qs) -> dict:
    out = json.load(open(CACHE)) if CACHE.exists() else {}
    for i, q in enumerate(qs):
        if q['q'] not in out:
            zh = tr(q['q'], 'zh')
            out[q['q']] = {'zh': zh, 'back': tr(zh, 'en')}
            if i % 100 == 0:
                CACHE.parent.mkdir(parents=True, exist_ok=True)
                json.dump(out, open(CACHE, 'w'), ensure_ascii=False)
                print(f'translated {i}/{len(qs)}', flush=True)
    json.dump(out, open(CACHE, 'w'), ensure_ascii=False)
    return out


def main() -> None:
    docs, qs = load(LOCOMO)
    t = translations(qs)
    con = sqlite3.connect(':memory:')
    con.execute("CREATE VIRTUAL TABLE f USING fts5(toks, conv UNINDEXED, tokenize='unicode61 remove_diacritics 0')")
    con.executemany('INSERT INTO f(rowid, toks, conv) VALUES (?, ?, ?)',
                    ((i, ' '.join(tokenize(text)), conv) for i, (_, conv, text) in enumerate(docs)))

    def bm25(toks, conv, n=50):
        toks = sorted(set(toks))
        if not toks:
            return []
        m = ' OR '.join('"' + x.replace('"', '""') + '"' for x in toks)
        return [r[0] for r in con.execute('SELECT rowid FROM f WHERE f MATCH ? AND conv = ? ORDER BY bm25(f) LIMIT ?',
                                          (m, conv, n))]

    def lanes(a, b):
        s = collections.defaultdict(float)
        for lst in (a, b):
            for r, i in enumerate(lst):
                s[i] += 1 / (60 + r)
        return sorted(s, key=lambda i: -s[i])[:50]

    arms = {
        'en': lambda q: bm25(tokenize(q['q']), q['conv']),
        'zh': lambda q: bm25(tokenize(t[q['q']]['zh']), q['conv']),
        'zh->en': lambda q: bm25(tokenize(t[q['q']]['back']), q['conv']),
        'merged': lambda q: bm25(tokenize(t[q['q']]['zh']) + tokenize(t[q['q']]['back']), q['conv']),
        'lanes': lambda q: lanes(bm25(tokenize(t[q['q']]['zh']), q['conv']),
                                 bm25(tokenize(t[q['q']]['back']), q['conv'])),
    }
    print(f'{len(qs)} questions; example: {qs[0]["q"]} -> {t[qs[0]["q"]]}', flush=True)
    for name, f in arms.items():
        rs = []
        for q in qs:
            rows = f(q)
            rs.append(next((k for k, i in enumerate(rows) if docs[i][0] in q['ev']), None))
        at = lambda k: sum(r is not None and r < k for r in rs) / len(rs)
        mrr = sum(1 / (r + 1) for r in rs if r is not None) / len(rs)
        print(f'{name:8s} R@1 {at(1):.3f}  R@5 {at(5):.3f}  R@10 {at(10):.3f}  R@50 {at(50):.3f}  MRR {mrr:.3f}',
              flush=True)


if __name__ == '__main__':
    main()
