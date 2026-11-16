#!/usr/bin/env python3
"""Fail if pm1 source contains a word list that looks copied from a benchmark.

Targets the v0.6 failure: hand-written vocabularies (sets/lists/tuples/dicts of words,
or regex alternations a|b|c) whose words are specific to a few benchmark items.
A word is "benchmark-specific" when it occurs in the benchmark's questions/answers of
at least 1 and at most --max-frac of the items. A literal with >= 5 words is flagged when
more than half of its words are benchmark-specific. Generic function words (present in
most items) and words absent from the benchmark are not counted.

  .venv/bin/python scripts/check_vocab_leak.py
  .venv/bin/python scripts/check_vocab_leak.py --extra path/to/fixture.json
"""
from __future__ import annotations

import argparse
import ast
import json
import re
import sys
from collections import Counter
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
WORD = re.compile(r'[a-z][a-z0-9-]{2,}|[㐀-鿿]{2,}')
TEXT_KEYS = ('question', 'answer', 'query', 'expected', 'gold', 'value', 'names')


def load_items(path: Path) -> list[str]:
    data = json.loads(path.read_text())
    items = data if isinstance(data, list) else (data.get('cases') or data.get('queries') or [])
    if isinstance(data, dict) and isinstance(data.get('memories'), list):
        items = items + data['memories']
    return [' '.join(str(it.get(k, '')) for k in TEXT_KEYS).lower() for it in items if isinstance(it, dict)]


def word_lists(tree: ast.AST):
    """(line, [words]) for collection literals of strings and for regex alternations."""
    for node in ast.walk(tree):
        if isinstance(node, (ast.Set, ast.List, ast.Tuple, ast.Dict)):
            elts = node.keys if isinstance(node, ast.Dict) else node.elts
            words = [e.value.lower() for e in elts if isinstance(e, ast.Constant) and isinstance(e.value, str)]
            if len(words) >= 5:
                yield node.lineno, words
        if isinstance(node, ast.Constant) and isinstance(node.value, str) and node.value.count('|') >= 4:
            alts = [a for a in re.split(r'[|()?:\\]', node.value.lower()) if WORD.fullmatch(a.strip() or '-')]
            if len(alts) >= 5:
                yield node.lineno, alts


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument('--bench', default=str(Path.home() / '.cache/pm1/data/longmemeval_s_cleaned.json'))
    ap.add_argument('--extra', action='append', default=[])
    ap.add_argument('--max-frac', type=float, default=0.25)
    ap.add_argument('--src', default=str(ROOT / 'src'))
    a = ap.parse_args()
    benches = [Path(p) for p in [a.bench, *a.extra] if Path(p).exists()]
    if not benches:
        print('no benchmark files found; nothing to check', file=sys.stderr)
        return 0
    hits = []
    for bench in benches:
        items = load_items(bench)
        df = Counter(w for text in items for w in set(WORD.findall(text)))
        limit = max(3, int(a.max_frac * len(items)))
        for py in sorted(Path(a.src).rglob('*.py')):
            for line, words in word_lists(ast.parse(py.read_text(), str(py))):
                specific = [w for w in words if 1 <= df.get(w.strip(), 0) <= limit]
                if len(specific) * 2 > len(words):
                    hits.append(f'{py}:{line}: {len(specific)}/{len(words)} words specific to '
                                f'{bench.name}: {", ".join(sorted(set(specific))[:8])}')
    for h in hits:
        print(h)
    print(f'{len(hits)} suspected benchmark word list(s)' if hits else 'no benchmark word lists in source')
    return 1 if hits else 0


if __name__ == '__main__':
    sys.exit(main())
