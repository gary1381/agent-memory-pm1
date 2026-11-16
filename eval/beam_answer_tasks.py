"""BEAM answering by external labelers (blind A/B of recall variants, hidden ids, small task files).

  beam_answer_tasks.py make <prefix> <size> <cats|all> <tag> [<tag> ...]
        answer prompts data/beam_answer_prompts_<tag>.jsonl (questions of the given comma-separated categories)
        -> inbox/<prefix>_NNN.jsonl/.md (+ data/<prefix>.map.json); items of all tags shuffled together
  beam_answer_tasks.py collect <prefix> <labeler>  -> data/beam_answers_<labeler>_<tag>.jsonl {"id","response"}
"""
import os
import json, random, sys
from pathlib import Path

EV = Path(__file__).resolve().parent
DATA = EV / 'data'
LAB = Path(os.environ.get('PM1_LABEL_DIR', str(Path.home() / 'pm1-labeling')))

SPEC = """# {task} — answer questions from a memory benchmark using ONLY the given context

Each line's `prompt` is a complete instruction: a CONTEXT (excerpts of a user's past conversations with an
assistant, each line tagged [date · session · role]) and a QUESTION at the end. Answer the question exactly as the
prompt instructs, using only that line's context. Lines are independent and shuffled: never reuse or adapt an answer
from another line, even when two questions look the same — their contexts differ and so may the right answer.

Read the whole context of each line before answering (the relevant facts can be anywhere in it). Do not write or run
code to produce answers; read and answer each item yourself.

Output one line per id: {{"id": "<same id>", "response": "<your answer as plain text>"}}
Answer the way the prompt asks (concise; a list when the question asks for an order or several items).
"""


def rows(p):
    return [json.loads(l) for l in Path(p).read_text(encoding='utf-8').split('\n') if l.strip()]


def make(prefix, size, cats, *tags):
    size = int(size)
    want = None if cats == 'all' else set(cats.split(','))
    src = lambda t: DATA / (f'{t}.jsonl' if t.startswith('lme_') else f'beam_answer_prompts_{t}.jsonl')
    items = [(t, r) for t in tags for r in rows(src(t))
             if want is None or ('|' in r['id'] and r['id'].split('|')[1] in want)]
    random.Random(20261015).shuffle(items)
    mp = {}
    for k in range(0, len(items), size):
        task = f'{prefix}_{k // size + 1:03d}'
        out = []
        for n, (t, r) in enumerate(items[k:k + size]):
            hid = f'q{k + n:05d}'
            mp[hid] = [t, r['id']]
            out.append({'id': hid, 'prompt': r['prompt']})
        (LAB / 'inbox' / f'{task}.jsonl').write_text(''.join(json.dumps(o) + '\n' for o in out),
                                                     encoding='utf-8')
        (LAB / 'inbox' / f'{task}.md').write_text(SPEC.format(task=task), encoding='utf-8')
    (DATA / f'{prefix}.map.json').write_text(json.dumps(mp))
    print(f'{prefix}: {len(items)} items in {(len(items) + size - 1) // size} tasks')


def collect(prefix, labeler):
    mp = json.loads((DATA / f'{prefix}.map.json').read_text())
    got = {}
    for p in sorted((LAB / 'outbox' / labeler).glob(f'{prefix}_*.out.jsonl'), key=lambda q: ('_fix_' in q.name, q.name)):
        for r in rows(p):
            v = r.get('response')
            v = v if isinstance(v, str) else json.dumps(v, ensure_ascii=False)
            if not v.strip():
                err = str(r.get('error') or '')
                if not err or _failed(err):
                    continue
                v = err
            got[r['id']] = v
    by, missing = {}, 0
    for hid, (t, rid) in mp.items():
        if hid not in got:
            missing += 1
            continue
        by.setdefault(t, []).append({'id': rid, 'response': got[hid]})
    for t, rs in by.items():
        p = DATA / f'beam_answers_{labeler}_{t}.jsonl'
        p.write_text(''.join(json.dumps(r, ensure_ascii=False) + '\n' for r in rs), encoding='utf-8')
        print(p, len(rs))
    print('missing', missing, 'of', len(mp))


FAIL_MARKS = ('unable to review', 'could not review', 'unable to process', 'not able to review', 'too long', 'could not complete', 'turn capacity', 'unable to determine this answer reliably', 'could not recover')


def _failed(err):
    return any(m in err.lower() for m in FAIL_MARKS)


def refix(prefix, labeler, size=30):
    """Re-ask items that are missing or failed (empty response with a processing error) in smaller task files."""
    mp = json.loads((DATA / f'{prefix}.map.json').read_text())
    ok = set()
    for p in (LAB / 'outbox' / labeler).glob(f'{prefix}_*.out.jsonl'):
        for r in rows(p):
            v = r.get('response')
            if (isinstance(v, str) and v.strip()) or (r.get('error') and not _failed(str(r['error']))):
                ok.add(r['id'])
    src = {}
    for t in {t for t, _ in mp.values()}:
        f = DATA / (f'{t}.jsonl' if t.startswith('lme_') else f'beam_answer_prompts_{t}.jsonl')
        src[t] = {r['id']: r['prompt'] for r in rows(f)}
    todo = [(h, src[t][rid]) for h, (t, rid) in sorted(mp.items()) if h not in ok]
    n0 = len(list((LAB / 'inbox').glob(f'{prefix}_fix_*.jsonl')))
    size = int(size)
    for k in range(0, len(todo), size):
        task = f'{prefix}_fix_{n0 + k // size + 1:03d}'
        (LAB / 'inbox' / f'{task}.jsonl').write_text(''.join(json.dumps({'id': h, 'prompt': pr}) + '\n'
                                                             for h, pr in todo[k:k + size]), encoding='utf-8')
        (LAB / 'inbox' / f'{task}.md').write_text(SPEC.format(task=task), encoding='utf-8')
    print(f'{prefix}: {len(todo)} items to re-ask in {(len(todo) + size - 1) // size} tasks')


if __name__ == '__main__':
    cmd, *a = sys.argv[1:]
    {'make': make, 'collect': collect, 'refix': refix}[cmd](*a)
