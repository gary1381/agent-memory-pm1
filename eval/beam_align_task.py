"""Event-ordering alignment for BEAM by external labelers, one item per question (not one per snippet pair).

The official metric asks an LLM, for each (system line, reference event) pair, "same event? YES/NO" and replays a
greedy matching. Local Qwen judges miss most YES (calibration), so these go to the labelers -- grouped per question
so a labeler sees the reference list and the answer list together. Items are shuffled with hidden ids (blind A/B).

  beam_align_task.py make <task> <tag> [<tag> ...]      reads data/beam_judge_prompts_<tag>.jsonl (align|... rows)
                                                         -> $PM1_LABEL_DIR/inbox/<task>.jsonl/.md + data/<task>.map.json
  beam_align_task.py responses <task> <labeler> <tag>   -> data/beam_align_resp_<tag>_<labeler>.jsonl
                                                            rows {"id": "align|conv|i|si|r", "response": "YES"/"NO"}
"""
import os
import json, random, re, sys
from collections import defaultdict
from pathlib import Path

EV = Path(__file__).resolve().parent
DATA = EV / 'data'
LAB = Path(os.environ.get('PM1_LABEL_DIR', str(Path.home() / 'pm1-labeling')))
PAT = re.compile(r'First snippet: (.*?) \n\nSecond snippet: (.*)\Z', re.S)

SPEC = """# {task} — match answer lines to reference events (same event or not)

Each line is one question from a memory benchmark. `prompt` shows two numbered lists:
- REFERENCE events (R1, R2, ...) — the correct events.
- ANSWER lines (S1, S2, ...) — lines from a system's answer.

For every answer line, decide whether it describes the SAME event/fact as one of the reference events. Same event
means the same underlying thing happened/was discussed (wording, detail level and dates may differ; a line that is
more specific or more general about the same item still counts). Different topics, or merely related topics, do NOT
count. An answer line can match at most one reference event and each reference event at most one answer line; if a
line could match several, pick the closest.

Output one line per id: {{"id": "<same id>", "response": {{"matches": [["S1", "R3"], ["S2", "R1"], ...],
"confidence": <0.0-1.0>}}}}
List only matching pairs (empty list if none). Use only the text in the prompt. `response` is a JSON object.
"""


def rows(p):
    return [json.loads(l) for l in Path(p).read_text(encoding='utf-8').split('\n') if l.strip()]


def make(task, *tags):
    items = []
    for tag in tags:
        refs, syss = defaultdict(dict), defaultdict(dict)
        for r in rows(DATA / f'beam_judge_prompts_{tag}.jsonl'):
            if not r['id'].startswith('align|'):
                continue
            _, conv, i, si, ri = r['id'].split('|')
            m = PAT.search(r['prompt'])
            refs[(conv, i)][int(ri)] = m.group(1).strip()
            syss[(conv, i)][int(si)] = m.group(2).strip()
        for q in refs:
            items.append((tag, q, refs[q], syss[q]))
    random.Random(20261014).shuffle(items)
    mp, out = {}, []
    for n, (tag, (conv, i), R, S) in enumerate(items):
        hid = f'a{n:04d}'
        rk, sk = sorted(R), sorted(S)
        mp[hid] = {'tag': tag, 'conv': conv, 'i': i, 'R': rk, 'S': sk}
        txt = ('REFERENCE events:\n' + '\n'.join(f'R{k + 1}. {R[r]}' for k, r in enumerate(rk)) +
               '\n\nANSWER lines:\n' + '\n'.join(f'S{k + 1}. {S[s]}' for k, s in enumerate(sk)))
        out.append({'id': hid, 'prompt': txt})
    (LAB / 'inbox' / f'{task}.jsonl').write_text(''.join(json.dumps(o) + '\n' for o in out),
                                                 encoding='utf-8')
    (LAB / 'inbox' / f'{task}.md').write_text(SPEC.format(task=task), encoding='utf-8')
    (DATA / f'{task}.map.json').write_text(json.dumps(mp))
    print(f'{task}: {len(out)} questions ({", ".join(tags)})')


def responses(task, labeler, tag):
    mp = json.loads((DATA / f'{task}.map.json').read_text())
    got = {}
    for r in rows(LAB / 'outbox' / labeler / f'{task}.out.jsonl'):
        v = r.get('response')
        if isinstance(v, str):
            try:
                v = json.loads(v)
            except ValueError:
                v = {}
        got[r['id']] = v or {}
    out, missing = [], 0
    for hid, m in mp.items():
        if m['tag'] != tag:
            continue
        if hid not in got:
            missing += 1
        pairs = set()
        for p in (got.get(hid, {}).get('matches') or []):
            try:
                s, r = int(str(p[0]).lstrip('Ss')) - 1, int(str(p[1]).lstrip('Rr')) - 1
                pairs.add((m['S'][s], m['R'][r]))
            except (ValueError, IndexError, TypeError):
                pass
        for s in m['S']:
            for r in m['R']:
                out.append({'id': f"align|{m['conv']}|{m['i']}|{s}|{r}", 'response': 'YES' if (s, r) in pairs else 'NO'})
    p = DATA / f'beam_align_resp_{tag}_{labeler}.jsonl'
    p.write_text(''.join(json.dumps(o) + '\n' for o in out))
    print(p, len(out), 'pairs; questions missing:', missing)


if __name__ == '__main__':
    cmd, *a = sys.argv[1:]
    {'make': make, 'responses': responses}[cmd](*a)
