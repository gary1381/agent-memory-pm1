"""A first judgement on reflection candidates by two Cursor models (composer-2.5 + grok-4.7), through the shared
labeling tool (PM1_LABEL_DUAL, not part of this repository: every labeler in its own isolated folder).

Measured 2026-10-19 (eval/deciders, 23 conflict pairs + 156 lesson candidates, labels by Claude):
  lesson   both say lesson: P 0.92 R 0.80 (reranker >= 0.05 today: P 0.73 R 0.27); local deciders (GLiNER2.5,
           ModernJEV, Laya) and local Qwen3.5-4B were below that or below the majority baseline
  conflict "does this need the user?" grok-4.7 P 1.00 R 1.00, composer 0.92 / 0.92; local models <= 0.72 F1
The judge only filters and labels what is shown to the user; it never activates, revokes or edits anything.
Any failure (receiver down, quota, timeout) returns None and the caller keeps today's behaviour.
Every verdict is logged (judge_log) so it can be checked later against what the user decided.
"""
from __future__ import annotations

import datetime as dt
import glob
import json
import os
import sqlite3
import subprocess
from pathlib import Path

from .redact import redact

LABEL_DUAL = Path(os.path.expanduser(os.environ.get('PM1_LABEL_DUAL', '~/pm1-labeling/label_dual.py')))
LABEL_DIR = Path(os.path.expanduser(os.environ.get('PM1_JUDGE_DIR', '~/pm1-labeling')))
LABELERS = (('composer', 'grok:composer-2.5'), ('grok47', 'grok:grok-4.7'))
SCHEMA = """
CREATE TABLE IF NOT EXISTS judge_log (id INTEGER PRIMARY KEY, ts TEXT NOT NULL, kind TEXT NOT NULL, ref TEXT,
  text TEXT, verdicts TEXT, outcome TEXT);
"""


def enabled() -> bool:
    off = os.environ.get('PM1_JUDGE', 'cursor').strip().lower() in ('off', 'false', 'no', 'none', '0', '')
    return not off and LABEL_DUAL.exists()


def run(kind: str, items: list[tuple[str, str]], question: str, labels: dict[str, str],
        timeout: int = 2400) -> dict[str, dict[str, str]] | None:
    """items: (id, text). Returns {id: {labeler: label}} or None when the judges could not run."""
    if not items or not enabled():
        return None
    task = f"pm1_judge_{kind}_{dt.datetime.now().strftime('%Y%m%d_%H%M%S')}"
    inbox = LABEL_DIR / 'inbox'
    inbox.mkdir(parents=True, exist_ok=True)
    with open(inbox / f'{task}.jsonl', 'w') as fh:
        for iid, text in items:
            fh.write(json.dumps({'id': iid, 'prompt': redact(text)}, ensure_ascii=False) + '\n')   # leaves for the cloud
    defs = '\n'.join(f'- `{k}`: {v}' for k, v in labels.items())
    # no output path here: label_dual tells each labeler its folder (naming one here sent outputs astray)
    (inbox / f'{task}.md').write_text(
        f'# {task}: one label per item\n\nQuestion for every item: **{question}**\n\nLabels (choose exactly one):\n'
        f'{defs}\n\nRules:\n- Use only the text of the item; do not open other files or other labelers\' outputs.\n'
        '- Text may be Chinese or English; judge the meaning.\n- When unsure, pick the better fit and lower your '
        'confidence.\n\nOutput: one JSON object per input line, same order and ids:\n'
        f'`{{"id": "<item id>", "response": {{"label": "<one of: {", ".join(labels)}>", "confidence": 0.0-1.0}}}}`\n')
    py = LABEL_DUAL.parent.parent / '.venv' / 'bin' / 'python'
    cmd = [str(py if py.exists() else 'python3'), str(LABEL_DUAL), task, '--dir', str(LABEL_DIR), '--gpt-review', 'none',
           '--timeout', str(timeout)]
    for name, spec in LABELERS:
        cmd += ['--labeler', f'{name}={spec}']
    try:
        p = subprocess.run(cmd, capture_output=True, text=True, timeout=timeout + 600)
    except (OSError, subprocess.TimeoutExpired):
        return None
    report = {}
    for line in (p.stdout or '').splitlines():
        if line.startswith('{') and f'"{task}"' in line:
            try:
                report = json.loads(line)
            except ValueError:
                pass
    out: dict[str, dict[str, str]] = {}
    for name, _ in LABELERS:
        files = [LABEL_DIR / 'outbox' / name / f'{task}.out.jsonl']
        iso = ((report.get('runs') or {}).get(name) or {}).get('isolated_dir')
        if iso:                                   # a labeler that wrote elsewhere inside its run folder
            files += [Path(f) for f in glob.glob(f'{iso}/outbox/**/{task}.out.jsonl', recursive=True)]
        f = next((f for f in files if f.exists()), None)
        if f is None:
            continue
        for line in open(f):
            try:
                r = json.loads(line)
            except ValueError:
                continue
            lab = r.get('response', {}).get('label') if isinstance(r.get('response'), dict) else r.get('label')
            if lab in labels:
                out.setdefault(str(r.get('id')), {})[name] = lab
    return out or None


FINETUNE_AT = 300          # user-decided verdicts before a local decider is worth training (deciders evaluation, 2026-10-19)


def stats(con: sqlite3.Connection) -> dict:
    """How the judges' verdicts compare with what the user then decided. Lessons: a kept candidate became a
    proposal the user approved (agree) or rejected (disagree). Conflicts: a flag the user dismissed (judges wrong)
    or settled by revoking / editing the rule (judges right). Verdicts the user never saw have no user label."""
    con.executescript(SCHEMA)
    out = {'verdicts': con.execute('SELECT count(*) FROM judge_log').fetchone()[0], 'user_decided': 0, 'agree': 0}
    tables = {r[0] for r in con.execute("SELECT name FROM sqlite_master WHERE type='table'")}
    if 'rules' in tables:
        for (st,) in con.execute("SELECT r.status FROM judge_log j JOIN rules r ON r.source_episode = CAST(j.ref AS "
                                 "INTEGER) AND r.proposed_by = 'lesson-miner' WHERE j.kind='lesson' AND "
                                 "j.outcome='kept' AND r.status IN ('active', 'rejected', 'revoked')"):
            out['user_decided'] += 1
            out['agree'] += st != 'rejected'
    if 'rule_conflicts' in tables:
        for ref, st in con.execute("SELECT j.ref, c.status FROM judge_log j JOIN rule_conflicts c ON j.ref = 'rule:' "
                                   "|| c.rule_id || ':ep:' || c.episode_id WHERE j.kind='conflict' AND "
                                   "j.outcome='open' AND c.status IN ('dismissed', 'resolved')"):
            out['user_decided'] += 1
            out['agree'] += st == 'resolved'
    out['agreement'] = round(out['agree'] / out['user_decided'], 2) if out['user_decided'] else None
    out['until_finetune'] = max(0, FINETUNE_AT - out['user_decided'])
    return out


def log(con: sqlite3.Connection, kind: str, ref: str, text: str, verdicts: dict, outcome: str) -> None:
    con.executescript(SCHEMA)
    con.execute('INSERT INTO judge_log(ts, kind, ref, text, verdicts, outcome) VALUES (?,?,?,?,?,?)',
                (dt.datetime.now(dt.timezone.utc).isoformat(timespec='seconds'), kind, ref, redact((text or '')[:600]),
                 json.dumps(verdicts, ensure_ascii=False), outcome))
    con.commit()                  # never leave a write transaction open while the caller goes on scanning


# the wording measured in eval/deciders (keep the two in step)
LESSON_Q = 'Is this sentence a reusable technical lesson worth remembering for other projects?'
LESSON_LABELS = {
    'lesson': 'A reusable technical lesson: a specific tool, library, model, service or configuration behaves in a '
              'surprising way (a limitation, a misleading error, a deprecation, a default that breaks things), '
              'with its cause, useful again in other projects.',
    'not_lesson': 'Not a reusable lesson: a request or question, a one-off incident, a project-specific detail, '
                  'general knowledge, a pasted log or a specification line.',
}
CONFLICT_Q = 'How does the later message from the user relate to the standing rule?'
CONFLICT_LABELS = {
    'replaces': 'The later message changes or reverses the rule: the user now wants something different.',
    'adds_condition': 'The later message keeps the rule but adds a condition, an exception or a temporary change '
                      '(only while quota lasts, for now, unless, wait until).',
    'different_scope': 'The later message is about a different project or system, so the rule itself is not affected.',
    'consistent': 'The later message repeats, applies or agrees with the rule.',
    'unrelated': 'The later message is about something else and does not affect the rule.',
}
NEEDS_USER = {'replaces', 'adds_condition'}
