"""Gold rubric judging of BEAM runs by external labelers (blind A/B, hidden ids, <=150 items per task file).

  beam_gold_tasks.py make <prefix> <tag> [<tag> ...]     rubric (non-align) prompts of data/beam_judge_prompts_<tag>.jsonl
                                                          -> inbox/<prefix>_NNN.jsonl/.md + data/<prefix>.map.json
  beam_gold_tasks.py collect <prefix> <labeler>          -> data/beam_gold_<labeler>_<tag>.jsonl {"id","response"}
                                                            per tag (rubric verdicts; merge with align rows after)
"""
import os
import json, random, sys
from pathlib import Path

EV = Path(__file__).resolve().parent
DATA = EV / 'data'
LAB = Path(os.environ.get('PM1_LABEL_DIR', str(Path.home() / 'pm1-labeling')))
N = 150

SPEC = """# {task} — grade one answer against one rubric item (memory benchmark judging)

Each line's `prompt` is a complete grading instruction: a rubric criterion, the response to evaluate, and the scoring
scale. Follow the prompt exactly and grade each line on its own — lines are shuffled and independent; never reuse a
verdict from another line, even when two lines look alike.

Output one line per id: {{"id": "<same id>", "response": "<the JSON object the prompt asks for, as a string>"}}
i.e. `response` is the raw text you would reply with, e.g. "{{\\"score\\": 0.5, \\"reason\\": \\"...\\"}}".
Scores are only 1.0, 0.5 or 0.0. Keep the reason to one or two sentences.
"""


def rows(p):
    return [json.loads(l) for l in Path(p).read_text(encoding='utf-8').split('\n') if l.strip()]


def make(prefix, *tags):
    items = [(t, r) for t in tags for r in rows(DATA / f'beam_judge_prompts_{t}.jsonl')
             if not r['id'].startswith('align|')]
    random.Random(20261014).shuffle(items)
    mp = {}
    for k in range(0, len(items), N):
        task = f'{prefix}_{k // N + 1:03d}'
        out = []
        for n, (t, r) in enumerate(items[k:k + N]):
            hid = f'g{k + n:05d}'
            mp[hid] = [t, r['id']]
            out.append({'id': hid, 'prompt': r['prompt']})
        (LAB / 'inbox' / f'{task}.jsonl').write_text(''.join(json.dumps(o) + '\n' for o in out),
                                                     encoding='utf-8')
        (LAB / 'inbox' / f'{task}.md').write_text(SPEC.format(task=task), encoding='utf-8')
    (DATA / f'{prefix}.map.json').write_text(json.dumps(mp))
    print(f'{prefix}: {len(items)} items in {(len(items) + N - 1) // N} tasks')


def collect(prefix, labeler):
    mp = json.loads((DATA / f'{prefix}.map.json').read_text())
    by, missing = {}, 0
    got = {}
    for p in sorted((LAB / 'outbox' / labeler).glob(f'{prefix}_*.out.jsonl')):
        for r in rows(p):
            v = r.get('response')
            got[r['id']] = v if isinstance(v, str) else json.dumps(v)
    for hid, (t, rid) in mp.items():
        if hid not in got:
            missing += 1
            continue
        by.setdefault(t, []).append({'id': rid, 'response': got[hid]})
    for t, rs in by.items():
        p = DATA / f'beam_gold_{labeler}_{t}.jsonl'
        p.write_text(''.join(json.dumps(r, ensure_ascii=False) + '\n' for r in rs), encoding='utf-8')
        print(p, len(rs))
    print('missing', missing, 'of', len(mp))


if __name__ == '__main__':
    cmd, *a = sys.argv[1:]
    {'make': make, 'collect': collect}[cmd](*a)
