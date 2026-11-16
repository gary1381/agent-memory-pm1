"""Prereg s5 guardrail: LongMemEval-S G3 test (402 q), budget 512, base vs U (same code, PM1_USER_FIRST 0/1), luna answers, composer-2.5 judges.
Per type and overall: accuracy per arm, paired difference U-base with a bootstrap (4000, seed 2026)."""
import glob, json, os, random, statistics
from collections import defaultdict
HERE = os.path.dirname(os.path.abspath(__file__))
mp = json.load(open(f'{HERE}/data/pm1_g5_judge.map.json'))
ok = {}
for f in sorted(glob.glob(os.path.join(os.path.expanduser(os.environ.get('PM1_LABEL_DIR', '~/pm1-labeling')), 'outbox/composer/pm1_g5_judge_*.out.jsonl'))):
    for l in open(f):
        r = json.loads(l); ok[r['id']] = 'yes' in str(r.get('response', '')).lower()[:10]
import sys; sys.path.insert(0, HERE); sys.path.insert(0, os.path.join(HERE, '..', 'src'))
from lme_qa import questions
qs = {q['question_id']: q for q in questions('all')}
acc = defaultdict(dict)
for j, (arm, qid) in mp.items():
    acc[arm][qid] = ok.get(j)
missing = sum(1 for a in acc.values() for v in a.values() if v is None)
print('judged:', {a: len(v) for a, v in acc.items()}, 'missing', missing)
def typ(qid): return 'abstention' if qid.endswith('_abs') else qs[qid]['question_type']
by = defaultdict(list)
for qid in acc['g5base512']:
    by[typ(qid)].append(qid); by['ALL'].append(qid)
def boot(d):
    rng = random.Random(2026); n = len(d)
    bs = sorted(sum(d[rng.randrange(n)] for _ in range(n)) / n for _ in range(4000))
    return statistics.mean(d), bs[100], bs[3899]
print(f'{"type":28s} {"n":>4s} {"base":>6s} {"U":>6s} {"U-base":>8s}  95% CI')
for t in sorted(by, key=lambda t: (t == 'ALL', t)):
    ids = by[t]; b = [float(acc['g5base512'][q]) for q in ids]; u = [float(acc['g5u512'][q]) for q in ids]
    d = [x - y for x, y in zip(u, b)]; m, lo, hi = boot(d)
    flag = '   <-- below -0.05' if t == 'single-session-assistant' and m < -0.05 else ''
    print(f'{t:28s} {len(ids):4d} {statistics.mean(b):6.3f} {statistics.mean(u):6.3f} {m:+8.3f}  [{lo:+.3f}, {hi:+.3f}]{flag}')
