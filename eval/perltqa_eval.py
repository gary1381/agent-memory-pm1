"""PerLTQA (Chinese personal long-term memory; arXiv 2402.16288, CC BY-NC 4.0) as a retrieval test for pm1, no LLM.

Every question names the memory record holding its answer (a profile field, a relationship, an event, or a dialogue),
so pm1 can be scored on retrieval alone: the rank of the first recalled item from that record, and whether one is
packed into a 1K / 2K context.

ingest: one SQLite (~/.cache/pm1-eval/perltqa/zh.sqlite), scope perltqa:<protagonist> for each of the 32 people with
  questions. Profile: one record per field ("<name>的<field>: <value>"); relationships and events: one record each
  (role event; events dated by their creation time); dialogues: one record per turn (the protagonist's turns role user,
  the other speaker's assistant), session = the dialogue id. source = the memory id, so items map back to records.
eval:  a fixed sample of questions (seed 2026, stratified by section); recall with each rerank depth (PM1_RERANK_K).

  ../.venv/bin/python perltqa_eval.py ingest
  PM1_RERANK_QWEN_VLLM=http://127.0.0.1:18166 ../.venv/bin/python perltqa_eval.py eval [--n 1200] [--depths 16,60]
"""
from __future__ import annotations

import argparse
import ast
import collections
import json
import os
import random
import subprocess
import sys
from pathlib import Path

EV = Path(__file__).resolve().parent
sys.path.insert(0, str(EV.parent / 'src'))
DATA = Path(os.environ.get('PM1_PERLTQA_DIR', str(Path.home() / '.cache/pm1/data/perltqa/Dataset/zh')))
OUT = Path.home() / '.cache/pm1-eval/perltqa'
DB = Path(os.environ.get('PERLTQA_DB') or OUT / 'zh.sqlite')
SECTIONS = ('profile', 'social_relationship', 'events', 'dialogues')


def people() -> tuple[dict, list]:
    mem = {m['profile']['Protagonist']: m for m in json.load(open(DATA / 'perltmem.json'))}
    qa = [(name, v) for x in json.load(open(DATA / 'perltqa.json')) for name, v in x.items()]
    return mem, qa


def ingest(a) -> None:
    from pm1 import Store
    from pm1.worker import build_embedders
    dense, _ = build_embedders('split', 'none')
    mem, qa = people()
    OUT.mkdir(parents=True, exist_ok=True)
    for f in OUT.glob('zh.sqlite*'):
        f.unlink()
    store = Store(str(DB), index_models=(f'dense:{dense.name}',))
    n = 0
    for name, _ in qa:
        m, scope = mem[name], f'perltqa:{name}'
        add = lambda session, seq, role, text, src, at: store.add_episode(
            bank='personal', scope=scope, session=session, seq=seq, role=role, text=text, source=src, occurred_at=at)
        for j, (k, v) in enumerate(m['profile'].items()):
            add(f'{name}:profile', j, 'event', f'{name}的{k}: {v}', f'profile:{k}', '2020-01-01T00:00:00Z')
        for j, (k, v) in enumerate(m['social_relationship'].items()):
            add(f'{name}:social', j, 'event', f"{name}的{v.get('Relationship', '')}{v.get('Supporting Characters', '')}："
                f"{v.get('Description', '')}", k, '2020-01-02T00:00:00Z')
        for k, v in m['events'].items():
            at = str(v.get('Creation Time') or '2021-01-01')[:10] + 'T12:00:00Z'
            add(f'{name}:event:{k}', 0, 'event', v['content'], k, at)
        for k, v in m['dialogues'].items():
            seq = 0
            for t, lines in sorted(v['contents'].items()):
                for line in (x for l in lines for x in (l if isinstance(l, list) else [l])):   # a few are nested
                    who = line.split(':', 1)[0].split('：', 1)[0]
                    add(f'{name}:dialogue:{k}', seq, 'user' if who == name else 'assistant', line, k,
                        t.replace(' ', 'T') + f':{seq % 60:02d}Z')
                    seq += 1
        n += 1
    print(f'{n} people ingested; vectors next', flush=True)
    subprocess.run([sys.executable, '-m', 'pm1.worker', '--db', str(DB), '--dense', 'split', '--token', 'none',
                    '--exit-when-empty', '--batch', '64'], check=True, env={**os.environ, 'PYTHONPATH': str(EV.parent / 'src')})


def questions(n: int) -> list[dict]:
    _, qa = people()
    by = collections.defaultdict(list)
    for name, v in qa:
        for sec in SECTIONS:
            s = v.get(sec)
            for q in (s if isinstance(s, list) else [x for t in (s or {}).values() for x in t]):
                ref = q['Reference Memory']
                refs = ast.literal_eval(ref) if ref.startswith('[') else [f'profile:{ref}']
                by[sec].append({'name': name, 'section': sec, 'q': q['Question'], 'refs': refs})
    total = sum(map(len, by.values()))
    rng = random.Random(2026)
    out = []
    for sec, qs in by.items():
        out += rng.sample(qs, min(len(qs), round(n * len(qs) / total)))
    return out


def evaluate(a) -> None:
    os.environ.setdefault('PM1_DENSE', 'split')
    from pm1 import Recaller, Store
    from pm1.rerank import router_from_env
    from pm1.worker import build_embedders
    dense, _ = build_embedders('split', 'none')
    if a.dense:                                 # another vector model already in DB (harness/pool_exp.py embed)
        from pm1.embed import HttpDense
        dense = HttpDense(*a.dense.split(',', 2))
    store = Store(str(DB))
    qs = questions(a.n)
    print(f'{len(qs)} questions: {dict(collections.Counter(q["section"] for q in qs))}', flush=True)
    for depth in map(int, a.depths.split(',')):
        rec = Recaller(store, dense=dense, router=router_from_env(), profile='wait', k_route=depth,
                       directive_keep_top=False)
        if a.scorer:                            # another reranker scores the same candidates (11-04: LycheeMem, C10X/int8)
            import urllib.request

            def http_score(query, docs, url=a.scorer):
                req = urllib.request.Request(url, json.dumps({'query': query, 'docs': docs}).encode(),
                                             {'Content-Type': 'application/json'})
                return json.load(urllib.request.urlopen(req, timeout=600))['scores']
            for r in rec.router.rerankers:
                r.score = http_score
        if a.dense2:                            # beside e5 in the first-stage RRF (prereg amendment 21)
            from pm1.embed import HttpDense
            rec.dense2 = HttpDense(*a.dense2.split(',', 2))
        rows = []
        for q in qs:
            res = rec.recall('personal', f'perltqa:{q["name"]}', q['q'], budget_tokens=8000, budget_ms=None)
            items = sorted(res.items, key=lambda x: x['rank'])
            rank, used, fit = None, 0, {}
            for k, it in enumerate(items):
                used += max(1, len(it['text']) // 2)                     # Chinese: ~2 chars per token
                if (it.get('source') or '').split('#')[0] in q['refs'] or it.get('source') in q['refs']:
                    rank = k if rank is None else rank
                    for B in (1024, 2048):
                        fit[B] = fit.get(B) or used <= B
            rows.append({'section': q['section'], 'rank': rank, 'fit': fit})
        for sec in SECTIONS + ('all',):
            rs = [r for r in rows if sec in ('all', r['section'])]
            if not rs:
                continue
            at = lambda k: sum(r['rank'] is not None and r['rank'] < k for r in rs) / len(rs)
            print(f'depth {depth:3d} {sec:20s} n {len(rs):4d}  R@1 {at(1):.3f}  R@5 {at(5):.3f}  R@10 {at(10):.3f}  '
                  f'in 1K {sum(bool(r["fit"].get(1024)) for r in rs) / len(rs):.3f}  '
                  f'in 2K {sum(bool(r["fit"].get(2048)) for r in rs) / len(rs):.3f}', flush=True)


if __name__ == '__main__':
    ap = argparse.ArgumentParser()
    ap.add_argument('cmd', choices=('ingest', 'eval'))
    ap.add_argument('--n', type=int, default=1200)
    ap.add_argument('--depths', default='16,60')
    ap.add_argument('--dense', default='', help='URL,NAME[,QUERY_PREFIX]')
    ap.add_argument('--scorer', default='', help='URL of a POST {query, docs} -> {scores} reranker (competitors/harness/lyc_rerank.py)')
    ap.add_argument('--dense2', default='', help='URL,NAME[,QUERY_PREFIX]: a second vector model beside --dense')
    a = ap.parse_args()
    {'ingest': ingest, 'eval': evaluate}[a.cmd](a)
