"""Chinese tokenisation for pm1's keyword index (BM25), compared on PerLTQA (2026-10-26), no LLM.

pm1's FTS5 index stores each span as space-joined tokens (pm1.text.tokenize: jieba words + character bigrams for CJK
runs) and queries with an OR of the question's tokens. Here the same FTS5 setup is rebuilt in memory once per
tokeniser from the PerLTQA store (perltqa_eval.py ingest), and every question is scored: rank of the first span from
the record that holds its answer (Reference Memory), per section.

  jieba+bigram   today's pm1.text.tokenize
  bigram         character bigrams only
  jieba          jieba words only
  search+bigram  jieba search mode (also emits the sub-words of long words) + bigrams
  uni+bigram     single characters + bigrams
  lac / lac+bigram  Baidu LAC words (2026-10-29), from a separate LAC worker (own py3.10 venv, PM1_LAC_DIR), ± bigrams

  taskset -c 0,1 ../.venv/bin/python tokenizer_cmp.py [kind ...]
"""
from __future__ import annotations

import ast
import collections
import json
import os
import sqlite3
import subprocess
import sys
from pathlib import Path

EV = Path(__file__).resolve().parent
sys.path[:0] = [str(EV.parent / 'src'), str(EV)]
from perltqa_eval import DB, SECTIONS, people  # noqa: E402
from pm1.text import _CJK, _LATIN, STOP, tokenize  # noqa: E402


LAC_DIR = Path(os.environ.get('PM1_LAC_DIR', str(Path.home() / 'pm1-lac')))
LAC_WORDS: dict[str, list[str]] = {}


def lac_words(texts) -> None:
    """segment every CJK run once with the LAC worker (one batch per 2,000 runs) into LAC_WORDS"""
    runs = sorted({r for t in texts for r in _CJK.findall(t.lower())} - LAC_WORDS.keys())
    env = dict(os.environ, OMP_NUM_THREADS='2', CPU_NUM='2')
    p = subprocess.Popen([str(LAC_DIR / '.venvs/lac/bin/python'), str(LAC_DIR / 'pm1d_names/lac_worker.py')],
                         stdin=subprocess.PIPE, stdout=subprocess.PIPE, text=True, env=env)
    for i in range(0, len(runs), 2000):
        p.stdin.write(json.dumps(runs[i:i + 2000], ensure_ascii=False) + '\n')
        p.stdin.flush()
        for run, tagged in zip(runs[i:i + 2000], json.loads(p.stdout.readline())):
            LAC_WORDS[run] = [w for w, _ in tagged if w.strip()]
    p.stdin.close()
    p.wait()


def make(kind: str):
    import jieba

    def tok(text: str) -> list[str]:
        text = text.lower()
        out = [t for t in _LATIN.findall(text) if t not in STOP]
        for run in _CJK.findall(text):
            if kind in ('jieba', 'jieba+bigram'):
                out += [w for w in jieba.lcut(run) if w.strip()]
            if kind == 'search+bigram':
                out += [w for w in jieba.lcut_for_search(run) if w.strip()]
            if kind == 'uni+bigram':
                out += list(run)
            if kind.startswith('lac'):
                out += LAC_WORDS[run]
            if kind not in ('jieba', 'lac'):
                out += [run[i:i + 2] for i in range(len(run) - 1)]
        return out
    return tokenize if kind == 'jieba+bigram' else tok


def all_questions() -> list[dict]:
    _, qa = people()
    out = []
    for name, v in qa:
        for sec in SECTIONS:
            s = v.get(sec)
            for q in (s if isinstance(s, list) else [x for t in (s or {}).values() for x in t]):
                ref = q['Reference Memory']
                refs = ast.literal_eval(ref) if ref.startswith('[') else [f'profile:{ref}']
                out.append({'name': name, 'section': sec, 'q': q['Question'], 'refs': set(refs)})
    return out


def main() -> None:
    src = sqlite3.connect(f'file:{DB}?mode=ro', uri=True)
    spans = src.execute('SELECT s.id, s.scope, e.source, substr(e.text, s.start + 1, s.end - s.start) '
                        'FROM spans s JOIN episodes e ON e.id = s.episode_id').fetchall()
    qs = all_questions()
    print(f'{len(spans)} spans, {len(qs)} questions: {dict(collections.Counter(q["section"] for q in qs))}', flush=True)
    kinds = sys.argv[1:] or ['jieba+bigram', 'bigram', 'jieba', 'search+bigram', 'uni+bigram', 'lac', 'lac+bigram']
    if any(k.startswith('lac') for k in kinds):
        lac_words([t for *_, t in spans] + [q['q'] for q in qs])
        print(f'LAC: {len(LAC_WORDS)} CJK runs segmented', flush=True)
    for kind in kinds:
        tok = make(kind)
        con = sqlite3.connect(':memory:')
        con.execute("CREATE VIRTUAL TABLE f USING fts5(toks, scope UNINDEXED, tokenize='unicode61 remove_diacritics 0')")
        con.executemany('INSERT INTO f(rowid, toks, scope) VALUES (?, ?, ?)',
                        ((sid, ' '.join(tok(text)), scope) for sid, scope, _, text in spans))
        src_of = {sid: source for sid, _, source, _ in spans}
        ranks = collections.defaultdict(list)
        for q in qs:
            toks = sorted(set(tok(q['q'])))
            match = ' OR '.join('"' + t.replace('"', '""') + '"' for t in toks)
            rows = con.execute('SELECT rowid FROM f WHERE f MATCH ? AND scope = ? ORDER BY bm25(f) LIMIT 50',
                               (match, f'perltqa:{q["name"]}')).fetchall() if toks else []
            rank = next((k for k, (sid,) in enumerate(rows) if src_of[sid] in q['refs']), None)
            ranks[q['section']].append(rank)
            ranks['all'].append(rank)
        for sec in SECTIONS + ('all',):
            rs = ranks[sec]
            at = lambda k: sum(r is not None and r < k for r in rs) / len(rs)
            mrr = sum(1 / (r + 1) for r in rs if r is not None) / len(rs)
            print(f'{kind:14s} {sec:20s} n {len(rs):5d}  R@1 {at(1):.3f}  R@5 {at(5):.3f}  R@10 {at(10):.3f}  '
                  f'R@50 {at(50):.3f}  MRR {mrr:.3f}', flush=True)


if __name__ == '__main__':
    main()
