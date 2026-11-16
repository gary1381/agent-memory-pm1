"""Paired comparison of two judged BEAM runs (official metric per question), with a paired bootstrap CI.
usage: beam_ab_report.py <dirA> <dirB> [out.json]"""
import json, random, statistics, sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent))
from beam_qa import CATS  # noqa: E402


def per_question(d):
    d = Path(d)
    out, extra = {}, {}
    for f in d.glob('evaluation-results_conv*.json'):
        conv = f.stem.replace('evaluation-results_conv', '')
        ev, ans = json.loads(f.read_text()), json.loads((d / f'answers_conv{conv}.json').read_text())
        for cat in CATS:
            rows = ev.get(cat, [])
            if rows and all('q' in r for r in rows):        # imports record the question index (some may be
                pairs = [(r['q'], r) for r in rows]             # left out); older files are positional
            else:
                idx = [i for i, a in enumerate(ans.get(cat, [])) if a]
                pairs = list(zip(idx, rows))
            for i, r in pairs:
                if True:
                    out[f'{conv}|{cat}|{i}'] = r['tau_norm'] if cat == 'event_ordering' else r['llm_judge_score']
                    if cat == 'event_ordering':
                        extra[f'{conv}|{cat}|{i}'] = {x: r[x] for x in ('f1', 'final_score', 'llm_judge_score')}
    return out, extra


A, EA = per_question(sys.argv[1])
B, EB = per_question(sys.argv[2])
ids = sorted(set(A) & set(B))
CATS = [c for c in CATS if any(f'|{c}|' in i for i in ids)]     # subsets (e.g. an A/B on 4 categories)
rep = {'n': len(ids), 'per_category': {}}
print(f"{'category':28s}{'A':>8s}{'B':>8s}{'B-A':>8s}  up/down/same")
for cat in CATS:
    q = [i for i in ids if f'|{cat}|' in i]
    a, b = statistics.mean(A[i] for i in q), statistics.mean(B[i] for i in q)
    up = sum(B[i] > A[i] for i in q); dn = sum(B[i] < A[i] for i in q)
    rep['per_category'][cat] = {'A': round(a, 3), 'B': round(b, 3), 'up': up, 'down': dn, 'n': len(q)}
    print(f'{cat:28s}{a:8.3f}{b:8.3f}{b - a:+8.3f}  {up}/{dn}/{len(q) - up - dn}')


def overall(S, sample):
    return statistics.mean(statistics.mean(S[i] for i in sample if f'|{c}|' in i) for c in CATS)


oa, ob = overall(A, ids), overall(B, ids)
rng = random.Random(7)
by_cat = {c: [i for i in ids if f'|{c}|' in i] for c in CATS}
diffs = []
for _ in range(5000):          # stratified paired bootstrap (resample questions within each category)
    smp = [rng.choice(by_cat[c]) for c in CATS for _ in by_cat[c]]
    diffs.append(overall(B, smp) - overall(A, smp))
diffs.sort()
lo, hi = diffs[int(.025 * len(diffs))], diffs[int(.975 * len(diffs))]
rep |= {'overall_A': round(oa, 4), 'overall_B': round(ob, 4), 'diff': round(ob - oa, 4), 'ci95': [round(lo, 4), round(hi, 4)],
        'p_B_le_A': round(sum(d <= 0 for d in diffs) / len(diffs), 4)}
eo = [i for i in ids if '|event_ordering|' in i]
if eo:
    rep['event_ordering_detail'] = {k: [round(statistics.mean(E[i][k] for i in eo), 3) for E in (EA, EB)]
                                    for k in ('f1', 'final_score', 'llm_judge_score')}
print(f"{f'OVERALL (mean of {len(CATS)})':28s}{oa:8.3f}{ob:8.3f}{ob - oa:+8.3f}   95% CI [{lo:+.3f}, {hi:+.3f}]  "
      f"P(B<=A)={rep['p_B_le_A']}")
print('event ordering [A, B]:', rep.get('event_ordering_detail'))
if len(sys.argv) > 3:
    Path(sys.argv[3]).write_text(json.dumps(rep, indent=1))
