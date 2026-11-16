"""Should a standing instruction be shown for this question? Calibrate a similarity gate on BEAM.

For every question: cosine(query, each directive) for the top-3 directives the recaller would pick, and whether that
directive turn is one of the question's gold source turns. Gold-relevant directives cluster in instruction/preference
following; elsewhere the block is noise (it changed answers on temporal / multi-session questions in BEAM-1M).

  PM1_BEAM_SET=1m beam_directive_calib.py  -> table of keep-rates per category for several thresholds
"""
import json, os, sys
from collections import defaultdict
from pathlib import Path

import numpy as np

HERE = Path(__file__).resolve().parent
sys.path.insert(0, str(HERE)); sys.path.insert(0, str(HERE.parent / 'src'))
from beam_cfg import STORES, conv_ids  # noqa: E402
from beam_qa import CATS, questions  # noqa: E402
from beam_coverage import gold_ids  # noqa: E402
from pm1.store import Store  # noqa: E402
from pm1.recall import Recaller  # noqa: E402
from pm1.embed import FastE5  # noqa: E402

dense = FastE5(gpu=os.environ.get('PM1_GPU', '0') != '0')
rows = []
TOPIC = __import__('re').compile(r"\b(?:when(?:ever)?|if)\s+(?:I|we)(?:'m| am| are)?\s+(?:ask(?:ing)?|talk(?:ing)?|discuss(?:ing)?|"
                                 r"work(?:ing)?|chat(?:ting)?|inquir(?:e|ing))\s+(?:about|on|for|regarding)?\s*(.+)", __import__('re').I)


def topic(t):
    t = t.split('->->')[0]
    m = TOPIC.search(t)
    return (m.group(1) if m else t).strip().rstrip('.!')[:200]


for conv in conv_ids():
    store = Store(STORES / f'conv{conv}.sqlite')
    rec = Recaller(store, dense=dense, router=None)
    dids = store.directives('beam', str(conv))
    if not dids:
        continue
    ids, mat, row, _ = rec._dense_rows('beam', str(conv))
    seq = {r[0]: r[1] for r in store.con.execute('SELECT s.id, e.seq FROM spans s JOIN episodes e ON e.id = s.episode_id')}
    drows = [row[i] for i in dids if i in row]
    dsid = [int(ids[r]) for r in drows]
    texts = {r[0]: r[1] for r in store.con.execute(
        'SELECT s.id, substr(e.text, s.start + 1, s.end - s.start) FROM spans s JOIN episodes e ON e.id = s.episode_id')}
    topics = [topic(texts.get(sid, '')) for sid in dsid]
    tvec = dense.embed(topics, 'query')
    qs = questions(str(conv))
    for cat in CATS:
        for i, q in enumerate(qs[cat]):
            qv = rec._qvec(q['question'])
            sims = mat[drows] @ qv
            tsims = tvec @ qv
            g = gold_ids(q)
            for r, j in enumerate(np.argsort(-tsims)[:3]):
                sid = dsid[j]
                rows.append({'cat': cat, 'rank': r, 'sim': float(sims[j]), 'tsim': float(tsims[j]),
                             'gold': seq.get(sid) in g, 'topic': topics[j]})
    store.close()
out = HERE / 'reports' / f"directive_calib_{os.environ.get('PM1_BEAM_SET', '10m')}.json"
out.write_text(json.dumps(rows))
by = defaultdict(list)
for r in rows:
    by[r['cat']].append(r)
TH = (0.80, 0.82, 0.84, 0.86, 0.88)
print('gold kept (topic sim): ' + '  '.join(f'@{t:.2f} {sum(r["tsim"] >= t for r in rows if r["gold"]) / max(1, sum(r["gold"] for r in rows)):.2f}' for t in TH))
print(f"{'category':26s} n_gold  " + '  '.join(f'dirs/q@{t:.2f}' for t in TH))
for cat in CATS:
    rs = by[cat]
    if not rs:
        continue
    print(f'{cat:26s} {sum(r["gold"] for r in rs):5d}  ' + '  '.join(
        f'{sum(r["tsim"] >= t for r in rs) / (len(rs) / 3):11.2f}' for t in TH))
gold = [r['sim'] for r in rows if r['gold']]
print('gold directive sims: n', len(gold), 'quantiles', np.round(np.quantile(gold, [0.05, 0.1, 0.25, 0.5]), 3) if gold else '-')
