import sys, json, re, collections, random, statistics, glob, os
sys.path.insert(0, '.'); sys.path.insert(0, '../src')
from lme_qa import questions
NEW = re.compile(
    r"\byou(?:'ve| have)? (?:said|told|suggested|recommended|mentioned|gave|provided|wrote|listed|shared|described|explained|did say)\b"
    r"|\byour (?:answer|suggestion|recommendation|advice|response|list)\b"
    r"|\bour (?:previous|last|earlier|past) (?:conversation|chat|session|discussion)\b|\bwe (?:talked|discussed|chatted|went over|covered|spoke)\b"
    r"|\bremind me\b|\b(?:can|could|would) you (?:recommend|suggest)\b|\b(?:any|some) (?:recommendations?|suggestions?|ideas?)\b|\bdo you have any (?:ideas|suggestions|recommendations)\b"
    r"|\bwhat do you (?:recommend|suggest|think)\b", re.I)
def check(name, qs):
    c = collections.Counter(); t = collections.Counter()
    for q in qs:
        ty = 'abstention' if q['question_id'].endswith('_abs') else q['question_type']; t[ty] += 1; c[ty] += bool(NEW.search(q['question']))
    print(name, {k: f'{c[k]}/{t[k]}' for k in sorted(t)})
dev = questions('dev'); allq = questions('all'); test_ids = {json.loads(l)['id'] for l in open('data/lme_answer_prompts_g5base512.jsonl')}
test = [q for q in allq if q['question_id'] in test_ids]
check('DEV ', dev); check('TEST', test)
# counterfactual on the test answers already judged: flagged -> base answer, else U answer
mp = json.load(open('data/pm1_g5_judge.map.json')); ok = {}
for f in glob.glob(os.path.join(os.path.expanduser(os.environ.get('PM1_LABEL_DIR', '~/pm1-labeling')), 'outbox/composer/pm1_g5_judge_*.out.jsonl')):
    for l in open(f):
        r = json.loads(l); ok[r['id']] = 'yes' in str(r.get('response', '')).lower()[:10]
acc = collections.defaultdict(dict)
for j, (arm, qid) in mp.items(): acc[arm][qid] = float(ok[j])
qd = {q['question_id']: q for q in test}
def typ(qid): return 'abstention' if qid.endswith('_abs') else qd[qid]['question_type']
rows = collections.defaultdict(list)
for qid in acc['g5base512']:
    flagged = bool(NEW.search(qd[qid]['question']))
    new = acc['g5base512'][qid] if flagged else acc['g5u512'][qid]
    for t in (typ(qid), 'ALL'): rows[t].append((acc['g5base512'][qid], acc['g5u512'][qid], new))
def boot(d):
    rng = random.Random(2026); n = len(d)
    bs = sorted(sum(d[rng.randrange(n)] for _ in range(n)) / n for _ in range(4000)); return statistics.mean(d), bs[100], bs[3899]
print(f'{"type":28s} {"n":>4s} {"base":>6s} {"U":>6s} {"U+rule":>7s} {"rule-base":>10s}')
for t in sorted(rows, key=lambda t: (t == 'ALL', t)):
    b, u, n = zip(*rows[t]); m, lo, hi = boot([x - y for x, y in zip(n, b)])
    print(f'{t:28s} {len(b):4d} {statistics.mean(b):6.3f} {statistics.mean(u):6.3f} {statistics.mean(n):7.3f} {m:+7.3f} [{lo:+.3f},{hi:+.3f}]')
