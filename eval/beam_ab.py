"""Blind A/B batch for BEAM: two prompt sets answered (and judged) in ONE batch, ids hidden and shuffled,
so answer-model run-to-run style drift cannot masquerade as a retrieval effect.

  beam_ab.py make-answers  <tagA> <tagB>          -> data/beam_ab_answer_prompts.jsonl  (+ data/beam_ab_map.json)
  beam_ab.py split-answers <answers.jsonl>        -> data/beam_answers_<tag>_ab.jsonl per tag
     then per tag: PM1_BEAM_TAG=<tag>ab beam_qa.py import-answers data/beam_answers_<tag>_ab.jsonl <model>
                   PM1_BEAM_TAG=<tag>ab beam_qa.py export-judge
  beam_ab.py make-judge    <tagA> <tagB>          -> data/beam_ab_judge_prompts.jsonl (merged, hidden ids)
  beam_ab.py split-judge   <responses.jsonl>      -> data/beam_judge_responses_<tag>ab.jsonl per tag
     then per tag: PM1_BEAM_TAG=<tag>ab beam_qa.py import-judge ... <judge model>
A tag '' means the v1 files (no suffix).
"""
import json, random, sys
from pathlib import Path

DATA = Path(__file__).resolve().parent / 'data'
MAP = DATA / 'beam_ab_map.json'


def _sfx(tag):
    return f'_{tag}' if tag else ''


def _rows(path):
    return [json.loads(l) for l in Path(path).read_text(encoding='utf-8').split('\n') if l.strip()]


def _write(path, rows):
    Path(path).write_text(''.join(json.dumps(r, ensure_ascii=False) + '\n' for r in rows), encoding='utf-8')
    print(f'{path}: {len(rows)} rows')


def make(kind, tags):
    src = 'beam_answer_prompts' if kind == 'answers' else 'beam_judge_prompts'
    mp = json.loads(MAP.read_text()) if MAP.exists() else {}
    rows = []
    for tag in tags:
        name = tag if kind == 'answers' else f'{tag}ab'
        for r in _rows(DATA / f'{src}{_sfx(name)}.jsonl'):
            rows.append((tag, r))
    random.Random(20261013 + len(kind)).shuffle(rows)
    out, m = [], {}
    pre = 'q' if kind == 'answers' else 'j'
    for n, (tag, r) in enumerate(rows):
        hid = f'{pre}{n:05d}'
        m[hid] = [tag, r['id']]
        out.append({'id': hid, 'prompt': r['prompt']})
    mp[kind] = m
    MAP.write_text(json.dumps(mp))
    _write(DATA / f'beam_ab_{kind[:-1] if kind.endswith("s") else kind}_prompts.jsonl', out)


def split(kind, path, field):
    m = json.loads(MAP.read_text())[kind]
    by = {}
    for r in _rows(path):
        tag, rid = m[r['id']]
        by.setdefault(tag, []).append({'id': rid, field: r.get(field) or r.get('response') or r.get('answer') or ''})
    for tag, rows in by.items():
        name = (f'beam_answers{_sfx(tag)}_ab.jsonl' if kind == 'answers'
                else f'beam_judge_responses_{tag or "v1"}ab.jsonl')
        _write(DATA / name, rows)


if __name__ == '__main__':
    cmd, *a = sys.argv[1:]
    tags = ['' if x in ('v1', "''") else x for x in a]
    if cmd == 'make-answers':
        make('answers', tags)
    elif cmd == 'make-judge':
        make('judges', tags)
    elif cmd == 'split-answers':
        split('answers', a[0], 'answer')
    elif cmd == 'split-judge':
        split('judges', a[0], 'response')
