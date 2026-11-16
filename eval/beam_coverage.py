"""Evidence coverage on BEAM-10M without any LLM: gold source_chat_ids vs turn ids pm1 put into the context.

usage: beam_coverage.py [recall_meta.json] [--scores]
  coverage = |gold ∩ recalled| / |gold| per question, averaged per category.
  --scores also joins the official per-question scores (evaluation-results_conv*.json) to show correlation.
"""
import json, statistics, sys
from collections import defaultdict
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent))
from beam_qa import CATS, OUT, questions  # noqa: E402
from beam_cfg import STORES, conv_ids  # noqa: E402


def _flat(x):
    if isinstance(x, dict):
        for v in x.values():
            yield from _flat(v)
    elif isinstance(x, (list, tuple, set)):
        for v in x:
            yield from _flat(v)
    elif x is not None and str(x).strip().lstrip('-').isdigit():
        yield int(x)


def gold_ids(q):
    return set(_flat(q.get('source_chat_ids')))


_SEQ: dict = {}


def coverage(meta, group=False):
    """group=True: a question's gold turns are grouped by (session, day); coverage = share of groups with >=1 hit
    (one group ~ one event / rubric item for ordering and summary questions)."""
    import os, sqlite3
    per = defaultdict(list)
    rows = {}
    for conv in conv_ids():
        qs = questions(str(conv))
        if not any(f'{conv}|' in k for k in meta):
            continue
        con = sqlite3.connect(str(STORES / f'conv{conv}.sqlite')) if group else None
        for cat in CATS:
            for i, q in enumerate(qs[cat]):
                qid = f'{conv}|{cat}|{i}'
                m = meta.get(qid)
                g = gold_ids(q)
                if m is None or not g:
                    continue
                got = set(m.get('source_chat_ids') or [])
                c = len(g & got) / len(g)
                if group:
                    if conv not in _SEQ:
                        _SEQ[conv] = {r[0]: (r[1], r[2]) for r in con.execute(
                            'SELECT seq, session, substr(occurred_at,1,10) FROM episodes')}
                    grp = defaultdict(set)
                    for sid in g:
                        grp[_SEQ[conv].get(sid)].add(sid)
                    c = sum(1 for v in grp.values() if v & got) / len(grp)
                per[cat].append(c)
                rows[qid] = {'cov': c, 'gold': len(g), 'hit': len(g & got), 'tokens': m.get('tokens'),
                             'ms': m.get('recall_ms')}
    return per, rows


if __name__ == '__main__':
    args = [a for a in sys.argv[1:] if not a.startswith('--')]
    meta = json.loads(Path(args[0] if args else OUT / 'recall_meta.json').read_text())
    per, rows = coverage(meta, group='--group' in sys.argv)
    allc = [c for v in per.values() for c in v]
    print(f'{"category":28s} cov    n')
    for cat in CATS:
        if per[cat]:
            print(f'{cat:28s} {statistics.mean(per[cat]):.3f}  {len(per[cat])}')
    print(f'{"ALL (mean of categories)":28s} {statistics.mean(statistics.mean(v) for v in per.values() if v):.3f}  {len(allc)}')
    if '--scores' in sys.argv:
        sc = {}
        for f in OUT.glob('evaluation-results_conv*.json'):
            conv = f.stem.replace('evaluation-results_conv', '')
            d = json.loads(f.read_text())
            ans = json.loads((OUT / f'answers_conv{conv}.json').read_text())
            for cat in CATS:
                k = 0
                for i, a in enumerate(ans.get(cat, [])):
                    if a and k < len(d.get(cat, [])):
                        r = d[cat][k]; k += 1
                        sc[f'{conv}|{cat}|{i}'] = r['tau_norm'] if cat == 'event_ordering' else r['llm_judge_score']
        lo = [sc[q] for q in rows if q in sc and rows[q]['cov'] < 0.5]
        hi = [sc[q] for q in rows if q in sc and rows[q]['cov'] >= 0.5]
        print(f'official score | coverage<0.5: {statistics.mean(lo):.3f} (n={len(lo)})   '
              f'coverage>=0.5: {statistics.mean(hi):.3f} (n={len(hi)})')
