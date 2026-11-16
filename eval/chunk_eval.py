"""Chunking test (not adopted): how long records are cut, what context a span carries, and what
is returned, on the two benchmarks with many long records (T039 research, T040 writing). No LLM answers here.

build:  one SQLite per (set, cut, header) under ~/.cache/pm1-eval/chunk; vectors are the production e5 (mean of windows),
  remembered by text hash so arms share what they can.
eval:   every question with evidence, recalled as of its asked_at, at 2K and 4K: must all in the context, evidence
  records all in the context. Rerank results are remembered per set, so 2K / 4K and return-only arms share them.
report: each arm against `cur`, paired bootstrap (10,000, seed 2026), per set, both sets, long-evidence questions.

  PYTHONPATH=src taskset -c 2,3 .venv/bin/python eval/chunk_eval.py build --set t039 --arm cur
  PM1_RERANK_QWEN_VLLM=http://127.0.0.1:18166 ... eval/chunk_eval.py eval --set t039 --arm cur
  ... eval/chunk_eval.py report
"""
from __future__ import annotations

import argparse
import hashlib
import json
import os
import pickle
import re
import sqlite3
import sys
from pathlib import Path

import numpy as np

EV = Path(__file__).resolve().parent
sys.path.insert(0, str(EV.parent / 'src'))
GROK = Path(os.environ.get('PM1_EXT_OUTPUTS', str(Path.home() / 'pm1-ext/outputs')))
OUT = Path.home() / '.cache/pm1-eval/chunk'
SETS = {'t039': (GROK / 'T039', 'questions.jsonl', 'research'), 't040': (GROK / 'T040', 'questions_v2.jsonl', 'writing'),
        't042': (GROK / 'T042', 'questions_v2.jsonl', 'tooldoc')}
ROLE = {'user': 'user', 'assistant': 'assistant', 'tool': 'tool', 'doc': 'event'}
ARMS = {'cur': {}, 'hdre': {'header': True}, 'hdr': {'header': True, 'env': {'PM1_RERANK_HEADER': '1'}},
        'ov': {'cut': (1200, 120)}, 'big': {'cut': (2000, 200)},
        'ret': {'env': {'PM1_RECALL_RETURN_CHARS': '2000'}}, 'two': {'env': {'PM1_RECALL_PARTS': '2'}},
        'ovtwo': {'cut': (1200, 120), 'env': {'PM1_RECALL_PARTS': '2'}},
        'hdrtwo': {'header': True, 'env': {'PM1_RERANK_HEADER': '1', 'PM1_RECALL_PARTS': '2'}},
        'hov': {'cut': (1200, 120), 'header': True, 'env': {'PM1_RERANK_HEADER': '1'}},
        'all': {'cut': (1200, 120), 'header': True, 'env': {'PM1_RERANK_HEADER': '1', 'PM1_RECALL_PARTS': '2'}},
        'rh': {'env': {'PM1_RERANK_HEADER': '1'}}, 'rhtwo': {'env': {'PM1_RERANK_HEADER': '1', 'PM1_RECALL_PARTS': '2'}},
        'ovrhtwo': {'cut': (1200, 120), 'env': {'PM1_RERANK_HEADER': '1', 'PM1_RECALL_PARTS': '2'}},
        'ovrh3': {'cut': (1200, 120), 'env': {'PM1_RERANK_HEADER': '1', 'PM1_RECALL_PARTS': '3'}},
        'rh3': {'env': {'PM1_RERANK_HEADER': '1', 'PM1_RECALL_PARTS': '3'}}}
ALL_ENV = sorted({k for s in ARMS.values() for k in s.get('env', {})})
_FENCE = re.compile(r'^(```|~~~)', re.M)
_HEAD = re.compile(r'^#{1,6} ', re.M)
_BLANK = re.compile(r'\n\s*\n')
_SENT = re.compile(r'(?<=[.!?。！？\n])\s+')


def cut_spans(text: str, max_chars: int, overlap: int) -> list[tuple[int, int]]:
    """Whole if it fits; else break in the second half of the window at the strongest boundary there: code fence,
    heading, blank line, sentence end. A broken record's next span starts `overlap` chars back, at a word start."""
    if len(text) <= max_chars:
        return [(0, len(text))]
    spans, start = [], 0
    while True:
        end = min(start + max_chars, len(text))
        if end == len(text):
            spans.append((start, end))
            return spans
        window = text[start:end]
        cut = 0
        for rx in (_FENCE, _HEAD, _BLANK, _SENT):
            at = [m.start() if rx is not _BLANK and rx is not _SENT else m.end() for m in rx.finditer(window)]
            at = [p for p in at if p > max_chars // 2]
            if at:
                cut = at[-1]
                break
        end = start + cut if cut else end
        spans.append((start, end))
        nxt = end - overlap
        while overlap and nxt < end and not text[nxt - 1].isspace():
            nxt += 1
        start = nxt if nxt > start else end


class VecCache:
    def __init__(self, path: Path, emb):
        self.emb, self.path = emb, path
        self.d = pickle.load(open(path, 'rb')) if path.exists() else {}

    def __call__(self, texts: list[str]) -> list[np.ndarray]:
        keys = [hashlib.sha1(t.encode()).digest()[:12] for t in texts]
        miss = list({k: t for k, t in zip(keys, texts) if k not in self.d}.items())
        for i in range(0, len(miss), 64):
            part = miss[i:i + 64]
            for (k, _), v in zip(part, self.emb.embed([t for _, t in part], 'passage')):
                self.d[k] = np.asarray(v, dtype=np.float32)
            if i % 640 == 0:
                print(f'vectors {i}/{len(miss)}', flush=True)
            if i % 12800 == 12736:
                self.save()
        self.save()
        return [self.d[k] for k in keys]

    def save(self) -> None:
        tmp = self.path.with_suffix(f'.{os.getpid()}.tmp')
        pickle.dump(self.d, open(tmp, 'wb'))
        os.replace(tmp, self.path)


def db_of(st: str, arm: str) -> Path:
    s = ARMS[arm]
    return OUT / f"{st}_{'x'.join(map(str, s['cut'])) if 'cut' in s else 'cur'}{'_h' if s.get('header') else ''}.sqlite"


def build(a) -> None:
    import pm1.store as ps
    from pm1 import Store
    from pm1.worker import build_embedders
    spec, db = ARMS[a.arm], db_of(a.set, a.arm)
    if db.exists():
        print(f'{db.name} is there', flush=True)
        return
    OUT.mkdir(parents=True, exist_ok=True)
    if 'cut' in spec:
        ps.split_spans = lambda text: cut_spans(text, *spec['cut'])
    dense, _ = build_embedders('split', 'none')
    tmp = db.with_suffix('.building')
    for f in OUT.glob(tmp.name + '*'):
        f.unlink()
    store = Store(str(tmp))
    root, _, kind = SETS[a.set]
    n = 0
    for f in sorted((root / 'corpus').glob('*.jsonl')):
        for r in map(json.loads, open(f)):
            store.add_episode(bank='personal', scope=f"{kind}:{r['project']}", session=r['session'], seq=r['seq'],
                              role=ROLE.get(r['role'], 'assistant'), text=f"{r['author']}: {r['text']}",
                              source=f"{r['session']}#{r['seq']}", occurred_at=r['at'])
            n += 1
    rows = store.con.execute('SELECT s.id, s.start, s.end, s.bank, s.scope, e.session, e.occurred_at, e.role, e.text '
                             'FROM spans s JOIN episodes e ON e.id = s.episode_id ORDER BY s.id').fetchall()
    texts = []
    for r in rows:
        t = r['text'][r['start']:r['end']]
        texts.append(store.span_header(dict(r)) + '\n' + t if spec.get('header') else t)
    print(f'{n} records, {len(rows)} spans, {sum(map(len, texts))} chars', flush=True)
    vecs = VecCache(OUT / 'vectors.pkl', dense)(texts)
    store.put_vectors(dense.name, zip([r['id'] for r in rows], vecs))
    store.con.commit()
    store.con.close()
    for f in OUT.glob(tmp.name + '*'):
        f.replace(str(db) + f.name[len(tmp.name):])
    print(f'built {db.name}', flush=True)


class Memo:
    """The router's answer for one exact (query, documents) list; failures are counted and never remembered."""

    def __init__(self, router, path: Path):
        self.router, self.path, self.fails, self.new = router, path, 0, 0
        self.d = pickle.load(open(path, 'rb')) if path.exists() else {}

    def rerank(self, query, docs, qlang, dlangs, budget):
        key = hashlib.sha1('\0'.join([query, *docs]).encode()).digest()[:12]
        if key not in self.d:
            idx, info = self.router.rerank(query, docs, qlang, dlangs, budget)
            if idx is None:
                self.fails += 1
                return idx, info
            self.d[key], self.new = (list(idx), info), self.new + 1
        return self.d[key]

    def save(self) -> None:
        tmp = self.path.with_suffix(f'.{os.getpid()}.tmp')
        pickle.dump(self.d, open(tmp, 'wb'))
        os.replace(tmp, self.path)


def contexts(a) -> None:
    """Contexts for the answer step: every question, absent ones too, as evaluate() recalls them."""
    rec, store, memo = recaller(a)
    root, qfile, kind = SETS[a.set]
    with open(OUT / f'ctx_{a.set}_{a.arm}.jsonl', 'w') as out:
        for q in map(json.loads, open(root / qfile)):
            for b in (2000, 4000):
                r = rec.recall('personal', f"{kind}:{q['project']}", q['question'], budget_tokens=b, budget_ms=None,
                               now=q['asked_at'], until=q['asked_at'])
                out.write(json.dumps({'id': q['id'], 'budget': b, 'context': r.context,
                                      'tokens': r.diagnostics.get('tokens')}, ensure_ascii=False) + '\n')
    memo.save()
    print(f'{a.set} {a.arm} contexts done, rerank failed {memo.fails}, new {memo.new}', flush=True)


def recaller(a):
    os.environ.setdefault('PM1_DENSE', 'split')
    os.environ['PM1_RERANK_K'] = '192'
    for k in ALL_ENV:
        os.environ.pop(k, None)
    os.environ.update(ARMS[a.arm].get('env', {}))
    from pm1 import Recaller, Store
    from pm1.rerank import router_from_env
    from pm1.worker import build_embedders
    dense, _ = build_embedders('split', 'none')
    store = Store(str(db_of(a.set, a.arm)))
    memo = Memo(router_from_env(), OUT / f'rerank_{a.set}.pkl')
    return Recaller(store, dense=dense, router=memo, profile='wait', k_route=192, directive_keep_top=False), store, memo


def evaluate(a) -> None:
    rec, store, memo = recaller(a)
    root, qfile, kind = SETS[a.set]
    ep = {(scope, src): (eid, n) for eid, scope, src, n in
          store.con.execute('SELECT id, scope, source, length(text) FROM episodes')}
    qs = [q for q in map(json.loads, open(root / qfile)) if q.get('evidence')]
    res = OUT / f'res_{a.set}_{a.arm}.jsonl'
    with open(res, 'w') as out:
        for i, q in enumerate(qs):
            scope = f"{kind}:{q['project']}"
            refs = [ep.get((scope, f"{e['session']}#{e['seq']}")) for e in q['evidence']]
            row = {'id': q['id'], 'category': q['category'], 'project': q['project'],
                   'long': any(x and x[1] > 1200 for x in refs)}
            gold = {x[0] if x else None for x in refs}
            for b in (2000, 4000):
                r = rec.recall('personal', scope, q['question'], budget_tokens=b, budget_ms=None,
                               now=q['asked_at'], until=q['asked_at'])
                got = {it['episode_id'] for it in r.items}
                ctx = re.sub(r'\s+', ' ', r.context or '')
                row[f'ev{b}'] = int(None not in gold and gold <= got)
                row[f'must{b}'] = int(all(re.search(p, ctx, re.I) for p in q['must']))
                row[f'items{b}'] = len(r.items)
            out.write(json.dumps(row) + '\n')
            out.flush()
            if i % 25 == 0:
                memo.save()
                print(f'{a.set} {a.arm} {i}/{len(qs)} failed {memo.fails}', flush=True)
    memo.save()
    print(f'{a.set} {a.arm} done, {len(qs)} questions, rerank failed {memo.fails}, new {memo.new}', flush=True)


def load(st: str, arm: str) -> dict:
    p = OUT / f'res_{st}_{arm}.jsonl'
    return {r['id']: r for r in map(json.loads, open(p))} if p.exists() else {}


def boot(d: list[float], n: int = 10000) -> tuple[float, float]:
    arr = np.asarray(d)
    m = np.sort(arr[np.random.default_rng(2026).integers(0, len(d), (n, len(d)))].mean(axis=1))
    return 100 * m[int(.025 * n)], 100 * m[int(.975 * n)]


def report(a) -> None:
    arms = [x for x in ARMS if x != 'cur']
    for key in ('must4000', 'must2000', 'ev4000', 'ev2000'):
        print(f'\n== {key}\narm   set     n    cur    arm   delta  95% CI              +   -')
        for arm in arms:
            parts = []
            for st in SETS:
                base, got = load(st, 'cur'), load(st, arm)
                parts.append((st, [(base[i], got[i]) for i in base if i in got]))
            used = [x for st, r in parts if st != 't042' for x in r]
            for name, rows in parts + [('both', used), ('long', [x for x in used if x[0]['long']])]:
                if rows:
                    d = [g[key] - b[key] for b, g in rows]
                    lo, hi = boot(d)
                    print(f"{arm:5s} {name:5s} {len(d):4d} {np.mean([b[key] for b, _ in rows]):.3f}  "
                          f"{np.mean([g[key] for _, g in rows]):.3f}  {100 * np.mean(d):+5.2f} [{lo:+6.2f}, {hi:+6.2f}] "
                          f"{sum(x > 0 for x in d):3d} {sum(x < 0 for x in d):3d}")
    print('\nsize of each index')
    for st in SETS:
        for arm in ARMS:
            db = db_of(st, arm)
            if db.exists():
                c = sqlite3.connect(f'file:{db}?mode=ro', uri=True)
                print(st, arm, c.execute('SELECT count(*), sum(end - start) FROM spans').fetchone())


if __name__ == '__main__':
    ap = argparse.ArgumentParser()
    ap.add_argument('cmd', choices=('build', 'eval', 'contexts', 'report'))
    ap.add_argument('--set', choices=tuple(SETS))
    ap.add_argument('--arm', choices=tuple(ARMS))
    a = ap.parse_args()
    {'build': build, 'eval': evaluate, 'contexts': contexts, 'report': report}[a.cmd](a)
