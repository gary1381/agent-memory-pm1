"""Fact index for the BEAM-1M stores (docs/EVALUATION.md E3): per conversation, a copy of its store plus facts
written by luna (pm1.facts_job's prompt, ~12K-char chunks, through the shared proxy, tag beam1m:facts). Resumable per
session. Vectors are added later by mv_arms.py with the recall's own dense model.

  ../.venv/bin/python beam1m_facts.py --convs 1,2,3,4,5,6,7,8,9,10 [--workers 6]
"""
from __future__ import annotations

import argparse
import concurrent.futures as cf
import sqlite3
import sys
import time
from pathlib import Path

EV = Path(__file__).resolve().parent
sys.path.insert(0, str(EV.parent / 'src'))
STORES = Path.home() / '.cache/pm1/beam1m'
OUT = Path.home() / '.cache/pm1-eval/beam1m_facts'


def luna(prompt: str) -> dict:
    import httpx
    from pm1.facts_job import parse_json
    body = {'model': 'gpt-6-luna', 'temperature': 0.0, 'max_tokens': 8000, 'response_format': {'type': 'json_object'},
            'messages': [{'role': 'user', 'content': prompt}]}
    for attempt in range(4):
        try:
            r = httpx.post('http://127.0.0.1:4011/v1/chat/completions', json=body, timeout=900,
                           headers={'Authorization': 'Bearer sk-local-competitors', 'x-cmp-tag': 'beam1m:facts'})
            r.raise_for_status()
            return parse_json(r.json()['choices'][0]['message']['content'])
        except Exception:
            if attempt == 3:
                raise
            time.sleep(20 * (attempt + 1))


def main() -> None:
    from pm1 import Store
    from pm1.facts import FactIndex
    from pm1.facts_job import extract
    ap = argparse.ArgumentParser()
    ap.add_argument('--convs', default='1,2,3,4,5,6,7,8,9,10')
    ap.add_argument('--workers', type=int, default=6)
    a = ap.parse_args()
    OUT.mkdir(parents=True, exist_ok=True)
    jobs = []
    for c in a.convs.split(','):
        dst = OUT / f'conv{c}.sqlite'
        if not dst.exists():
            s = sqlite3.connect(f'file:{STORES / f"conv{c}.sqlite"}?mode=ro&immutable=1', uri=True)
            d = sqlite3.connect(dst)
            s.backup(d)
            s.close(); d.close()
        st = Store(str(dst))
        FactIndex(st, None)
        done = {r[0] for r in st.con.execute('SELECT session FROM fact_sessions WHERE error IS NULL')}
        for (sess,) in st.con.execute('SELECT DISTINCT session FROM episodes WHERE bank=? AND scope=?', ('beam', c)):
            if sess not in done:
                jobs.append((c, sess))
        st.close() if hasattr(st, 'close') else None
    print(f'{len(jobs)} sessions to extract', flush=True)

    def one(job):
        c, sess = job
        con = sqlite3.connect(OUT / f'conv{c}.sqlite')
        rows = con.execute('SELECT id, role, text, occurred_at FROM episodes WHERE bank=? AND scope=? AND session=? '
                           'ORDER BY seq', ('beam', c, sess)).fetchall()
        turns = [{'episode_id': r[0], 'role': r[1], 'text': r[2]} for r in rows]
        t0 = time.time()
        try:
            return job, extract(luna, turns, str(rows[0][3])[:10] if rows else ''), None, time.time() - t0
        except Exception as exc:
            return job, [], f'{type(exc).__name__}: {exc}'[:300], time.time() - t0

    with cf.ThreadPoolExecutor(a.workers) as ex:
        for (c, sess), facts, err, sec in ex.map(one, jobs):
            st = Store(str(OUT / f'conv{c}.sqlite'))
            FactIndex(st, None).replace_session('beam', c, sess, facts, model='gpt-6-luna', source='beam1m_facts', error=err)
            print(f'conv{c} {sess}: {len(facts)} facts {sec:.0f}s {err or ""}', flush=True)


if __name__ == '__main__':
    main()
