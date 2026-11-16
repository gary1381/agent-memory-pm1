"""Closed-book screen for benchmark questions (T039 / T040): can a model answer without the history?

make:    a question file (id, question, must) -> labeling task inbox/<name>.jsonl + .md (answer from your own knowledge,
         no files, no web, no tools; UNKNOWN when not sure)
collect: both labelers' answers -> hit = every `must` regex of the question matches the answer (case-insensitive)
         -> <out>.jsonl (id, answers, hit per labeler) and a summary

  ../.venv/bin/python closedbook.py make T039-closedbook $PM1_EXT_OUTPUTS/T039/questions.jsonl
  label_dual.py T039-closedbook --dir $PM1_LABEL_DIR --labeler composer=grok:composer-2.5 --labeler grok47=grok:grok-4.7
  ../.venv/bin/python closedbook.py collect T039-closedbook <questions.jsonl> <out.jsonl>
"""
from __future__ import annotations

import os
import collections
import json
import re
import sys
from pathlib import Path

LAB = Path(os.environ.get('PM1_LABEL_DIR', str(Path.home() / 'pm1-labeling')))
SPEC = """# {task} — answer each question from your own knowledge only (closed book)

Each line's `prompt` is one question about a public project (a research codebase, a collaborative maths project, a
published novel, ...). Answer it from what you already know. Do NOT open files, search the web, run code or use any
tool: this checks what a model knows without being shown the project's history. If you do not know, or would be
guessing, answer exactly UNKNOWN. Lines are independent.

Output one line per id: {{"id": "<same id>", "response": "<short answer, or UNKNOWN>"}}
"""


def make(name: str, qfile: str) -> None:
    qs = [json.loads(x) for x in open(qfile)]
    rows = [{'id': q['id'], 'prompt': f"Question (answer briefly, or UNKNOWN): {q['question']}"} for q in qs]
    (LAB / 'inbox' / f'{name}.jsonl').write_text(''.join(json.dumps(r, ensure_ascii=False) + '\n' for r in rows),
                                                encoding='utf-8')
    (LAB / 'inbox' / f'{name}.md').write_text(SPEC.format(task=name), encoding='utf-8')
    print(len(rows), 'items ->', LAB / 'inbox' / f'{name}.jsonl')


def collect(name: str, qfile: str, out: str) -> None:
    qs = {q['id']: q for q in map(json.loads, open(qfile))}
    ans = collections.defaultdict(dict)
    for d in (LAB / 'outbox').iterdir():
        for p in d.glob(f'{name}*.out.jsonl') if d.is_dir() else []:
            for line in open(p):
                try:
                    r = json.loads(line)
                except ValueError:
                    continue
                v = r.get('response')
                ans[r.get('id')][d.name] = v if isinstance(v, str) else json.dumps(v, ensure_ascii=False)
    rows, hits = [], collections.Counter()
    for qid, q in qs.items():
        res = {}
        for who, a in ans.get(qid, {}).items():
            known = a.strip().upper() != 'UNKNOWN' and bool(q.get('must')) and \
                all(re.search(p, a, re.I) for p in q['must'])
            res[who] = known
            hits[who] += known
        rows.append({'id': qid, 'category': q.get('category'), 'project': q.get('project'),
                     'answers': ans.get(qid, {}), 'known': res, 'drop': any(res.values())})
    Path(out).write_text(''.join(json.dumps(r, ensure_ascii=False) + '\n' for r in rows), encoding='utf-8')
    print(f'{len(rows)} questions; answered by: {dict(collections.Counter(w for r in rows for w in r["answers"]))}')
    print(f'known closed-book (all must patterns matched): {dict(hits)}; flagged (either model): '
          f'{sum(r["drop"] for r in rows)}')
    for r in rows:
        if r['drop']:
            print('  ', r['id'], r['category'], {w: a[:80] for w, a in r['answers'].items()})


if __name__ == '__main__':
    {'make': make, 'collect': collect}[sys.argv[1]](*sys.argv[2:])
