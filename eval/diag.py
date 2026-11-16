"""Where is the evidence lost? First-stage recall vs. ordering, on the LME dev split (cached vectors only).

For every scorable question: rank of the first evidence span in the full BM25 ranking, the full e5 ranking,
the shipped RRF order (union of top-50s), and a full-depth RRF. Misses at T@10 are dumped with what outranked
the evidence, so failures can be read, not just counted.

  .venv/bin/python eval/diag.py [--limit N] [--out eval/reports/diag_dev.json]
"""
from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent))
from lme import DATA, VecCache, fill, ingest, span_texts, split_of  # noqa: E402

from pm1 import Recaller  # noqa: E402
from pm1.embed import SplitE5  # noqa: E402
from pm1.recall import rrf  # noqa: E402

DEPTHS = (1, 5, 10, 20, 32, 50, 100)


def first_ev(order, meta):
    return next((r for r, sid in enumerate(order) if meta[sid]['evidence']), None)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument('--limit', type=int)
    ap.add_argument('--out', default=str(Path(__file__).parent / 'reports' / 'diag_dev.json'))
    a = ap.parse_args()
    dense, cache = SplitE5(), VecCache()
    qs = [q for q in json.loads(DATA.read_text())
          if split_of(q['question_id']) == 'dev' and not q['question_id'].endswith('_abs')][:a.limit]
    rows, misses = [], []
    for q in qs:
        store, meta = ingest(q)
        fill(store, dense, 'dense', cache)
        scope = q['question_id']
        full = Recaller(store, dense=dense, k_first=100000)
        lex_full, _, _ = full.rank('lme', scope, q['question'], 'lexical')
        den_full, _, _ = full.rank('lme', scope, q['question'], 'dense')
        shipped, _, _ = Recaller(store, dense=dense).rank('lme', scope, q['question'], 'rrf')
        fused_full = rrf([lex_full, den_full])
        n_ev = sum(m['evidence'] for m in meta.values())
        r = {'qid': scope, 'type': q['question_type'], 'n_spans': len(meta), 'n_evidence_spans': n_ev,
             'bm25': first_ev(lex_full, meta), 'e5': first_ev(den_full, meta),
             'rrf_shipped': first_ev(shipped, meta), 'rrf_full': first_ev(fused_full, meta),
             'shipped_pool': len(shipped)}
        rows.append(r)
        if r['rrf_shipped'] is None or r['rrf_shipped'] >= 10:
            texts = span_texts(store)
            ans = set(q['answer_session_ids'])
            top = shipped[:10]
            ev_ids = [s for s in meta if meta[s]['evidence']]
            misses.append(r | {
                'question': q['question'], 'answer': str(q['answer'])[:200],
                'evidence': [texts[s][:300] for s in ev_ids[:3]],
                'top10_from_answer_session': sum(meta[s]['session'] in ans for s in top),
                'top10_same_episode_neighbour': sum(
                    any(meta[s]['session'] == meta[e]['session'] for e in ev_ids) for s in top),
                'top3': [{'answer_session': meta[s]['session'] in ans, 'text': texts[s][:200]} for s in top[:3]]})
        store.close()

    def recall_at(key, k):
        return round(sum(r[key] is not None and r[key] < k for r in rows) / len(rows), 3)

    summary = {'questions': len(rows),
               'recall_at_depth': {key: {f'@{k}': recall_at(key, k) for k in DEPTHS}
                                   for key in ('bm25', 'e5', 'rrf_shipped', 'rrf_full')},
               'never_found_in_shipped_pool': [r['qid'] for r in rows if r['rrf_shipped'] is None],
               'misses_at_10': len(misses)}
    Path(a.out).write_text(json.dumps({'summary': summary, 'rows': rows, 'misses': misses},
                                      ensure_ascii=False, indent=2) + '\n')
    print(json.dumps(summary, indent=1))


if __name__ == '__main__':
    main()
