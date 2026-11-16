"""Merge Kaggle rubric verdicts ('<tag>::<id>') with labeler alignment verdicts into per-run judge-response files.
  beam1m_merge_judge.py <kaggle judge.jsonl> <align labeler> <out_suffix> <tag> [<tag> ...]
     -> data/beam_judge_responses_1m_<tag><out_suffix>.jsonl
"""
import json, sys
from pathlib import Path

DATA = Path(__file__).resolve().parent / 'data'
judge, lab, sfx, *tags = sys.argv[1:]
by = {t: [] for t in tags}
for l in Path(judge).read_text(encoding='utf-8').split('\n'):
    if l.strip():
        r = json.loads(l)
        t, i = r['id'].split('::', 1)
        if t in by:
            by[t].append({'id': i, 'response': r['response']})
for t in tags:
    rows = by[t] + [json.loads(l) for l in (DATA / f'beam_align_resp_{t}_{lab}.jsonl').read_text().split('\n') if l.strip()]
    p = DATA / f'beam_judge_responses_1m_{t}{sfx}.jsonl'
    p.write_text(''.join(json.dumps(r, ensure_ascii=False) + '\n' for r in rows), encoding='utf-8')
    print(p, len(rows))
