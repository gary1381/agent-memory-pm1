"""English tokenisation for pm1's keyword index (BM25), compared on LoCoMo (2026-10-29), no LLM.

Each LoCoMo turn is one document (scope = conversation); every question with evidence is scored by the rank of the
first evidence turn (D<session>:<turn>) in a BM25-only search over its own conversation, same FTS5 setup as
tokenizer_cmp.py (OR of the question's tokens).

  today        pm1.text.tokenize (lower-cased words, English function words dropped)
  porter       today's tokens through FTS5's porter stemmer (climbing / climbed -> climb), index and query
  nostop       lower-cased words, function words kept
  nostop+porter

  ../.venv/bin/python tokenizer_cmp_en.py [LOCOMO_JSON]
"""
from __future__ import annotations

import os
import collections
import json
import re
import sqlite3
import sys
from pathlib import Path

EV = Path(__file__).resolve().parent
sys.path[:0] = [str(EV.parent / 'src')]
from pm1.text import _LATIN, tokenize  # noqa: E402

LOCOMO = Path(os.environ.get('PM1_LOCOMO_JSON', str(Path(__file__).resolve().parent / 'data/locomo10_prepared.json')))
CATS = {1: 'multi-hop', 2: 'temporal', 3: 'open-domain', 4: 'single-hop', 5: 'adversarial'}
KINDS = {'today': (tokenize, 'unicode61'), 'porter': (tokenize, 'porter unicode61'),
         'nostop': (lambda t: _LATIN.findall(t.lower()), 'unicode61'),
         'nostop+porter': (lambda t: _LATIN.findall(t.lower()), 'porter unicode61')}


def load(path: Path):
    docs, qs = [], []
    for c in json.load(open(path)):
        for s in c['sessions']:
            n = int(re.search(r'\d+', s['key']).group())
            docs += [(f'{c["conv"]}|D{n}:{j}', c['conv'], m['content']) for j, m in enumerate(s['messages'], 1)]
        qs += [{'conv': c['conv'], 'cat': CATS.get(q['category'], q['category']), 'q': q['question'],
                'ev': {f'{c["conv"]}|{e.strip()}' for e in q.get('evidence') or []}} for q in c['qa']]
    ids = {d for d, _, _ in docs}
    qs = [q for q in qs if q['ev'] & ids]                 # a few evidence ids are malformed in LoCoMo
    return docs, qs


def main() -> None:
    docs, qs = load(Path(sys.argv[1]) if len(sys.argv) > 1 else LOCOMO)
    print(f'{len(docs)} turns, {len(qs)} questions: {dict(collections.Counter(q["cat"] for q in qs))}', flush=True)
    for kind, (tok, fts_tok) in KINDS.items():
        con = sqlite3.connect(':memory:')
        con.execute(f"CREATE VIRTUAL TABLE f USING fts5(toks, conv UNINDEXED, tokenize='{fts_tok} remove_diacritics 0')")
        con.executemany('INSERT INTO f(rowid, toks, conv) VALUES (?, ?, ?)',
                        ((i, ' '.join(tok(text)), conv) for i, (_, conv, text) in enumerate(docs)))
        ranks = collections.defaultdict(list)
        for q in qs:
            toks = sorted(set(tok(q['q'])))
            match = ' OR '.join('"' + t.replace('"', '""') + '"' for t in toks)
            rows = con.execute('SELECT rowid FROM f WHERE f MATCH ? AND conv = ? ORDER BY bm25(f) LIMIT 50',
                               (match, q['conv'])).fetchall() if toks else []
            rank = next((k for k, (i,) in enumerate(rows) if docs[i][0] in q['ev']), None)
            ranks[q['cat']].append(rank)
            ranks['all'].append(rank)
        for cat in [c for c in list(CATS.values()) + ["all"] if ranks[c]]:
            rs = ranks[cat]
            at = lambda k: sum(r is not None and r < k for r in rs) / len(rs)
            mrr = sum(1 / (r + 1) for r in rs if r is not None) / len(rs)
            print(f'{kind:14s} {cat:12s} n {len(rs):5d}  R@1 {at(1):.3f}  R@5 {at(5):.3f}  R@10 {at(10):.3f}  '
                  f'R@50 {at(50):.3f}  MRR {mrr:.3f}', flush=True)


if __name__ == '__main__':
    main()
