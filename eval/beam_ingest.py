"""Ingest BEAM-10M conversations into pm1 stores (one SQLite per conversation), all local, no generative model.

Each turn becomes an episode: session = "<plan>-b<batch>", seq = turn order, occurred_at = the most recent
time_anchor seen (BEAM marks dates on some user turns; later turns inherit it). Dense vectors: e5-small-q8 from
the local GPU llama-server (same GGUF/space as SplitE5), written straight into the store.

  .venv/bin/python eval/beam_ingest.py 1          # conversation id(s), or 'all'
"""
from __future__ import annotations

import datetime as dt
import os
import re
import sys
import time
from pathlib import Path

import pyarrow.parquet as pq

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / 'src'))
from pm1 import Store  # noqa: E402
from pm1.embed import FastE5, HttpE5  # noqa: E402

sys.path.insert(0, str(Path(__file__).resolve().parent))
from beam_cfg import D, GLOB, NAME, STORES as OUT  # noqa: E402
# fastembed fp32 on the GPU is ~4-10x faster than the llama.cpp bulk server on this shared card; its vectors live in
# their own space ('e5-small-onnx') and beam_qa embeds queries with the same model.
E5 = FastE5() if os.environ.get('PM1_BEAM_DENSE', 'onnx') == 'onnx' else \
    HttpE5(os.environ.get('PM1_E5_BULK_URL', 'http://127.0.0.1:18171/v1/embeddings'), batch=128, timeout=600,
           name='e5-small-q8')


def parse_anchor(s: str | None) -> str | None:
    if not s:
        return None
    for fmt in ('%B-%d-%Y', '%b-%d-%Y', '%Y-%m-%d', '%B %d, %Y'):
        try:
            return dt.datetime.strptime(s.strip(), fmt).date().isoformat()
        except ValueError:
            continue
    return None


def plan_key(k: str) -> int:
    return int(re.sub(r'\D', '', k) or 0)


def conversations(want: set[str] | None):
    for f in sorted(D.glob(GLOB)):
        for r in pq.read_table(f, columns=['conversation_id', 'chat']).to_pylist():
            if want is None or r['conversation_id'] in want:
                yield r


def ingest(r) -> Path:
    OUT.mkdir(parents=True, exist_ok=True)
    path = OUT / f"conv{r['conversation_id']}.sqlite"
    store = Store(path)
    scope, t0, n = r['conversation_id'], time.time(), 0
    current = '2024-01-01'
    resume = store.con.execute('SELECT count(*) FROM episodes WHERE scope=?', (scope,)).fetchone()[0] > 0
    chat = [] if resume else r['chat']
    if chat and isinstance(chat[0], list):      # 100K/500K/1M splits: a list of sessions, each a list of turns
        for si, sess in enumerate(chat, 1):
            for t in sess:
                current = parse_anchor(t.get('time_anchor')) or current
                if not (t.get('content') or '').strip():
                    continue
                store.add_episode(bank='beam', scope=scope, text=t['content'], role=t.get('role') or 'user',
                                  session=f's{si}', seq=int(t['id']), occurred_at=current,
                                  source=f"{NAME}:{t['id']}")
                n += 1
        chat = []
    for block in chat:
        for plan in sorted(block, key=plan_key):
            for b in block[plan] or []:
                current = parse_anchor(b.get('time_anchor')) or current
                for pair in b['turns'] or []:
                    for t in pair:
                        current = parse_anchor(t.get('time_anchor')) or current
                        if not (t['content'] or '').strip():
                            continue
                        store.add_episode(bank='beam', scope=scope, text=t['content'], role=t['role'] or 'user',
                                          session=f"{plan}-b{b['batch_number']}", seq=int(t['id']),
                                          occurred_at=current, source=f"{NAME}:{t['id']}")
                        n += 1
    t_lex = time.time() - t0
    if resume:
        n = store.con.execute('SELECT count(*) FROM episodes WHERE scope=?', (scope,)).fetchone()[0]
        print(f'conv {scope}: lexical index already built ({n} episodes), embedding the rest', flush=True)
    rows = store.con.execute('SELECT s.id, s.start, s.end, e.text FROM spans s JOIN episodes e ON e.id=s.episode_id '
                             'LEFT JOIN vectors v ON v.span_id=s.id AND v.model=? WHERE v.span_id IS NULL',
                             (E5.name,)).fetchall()
    t1 = time.time()
    for i in range(0, len(rows), 512):
        chunk = rows[i:i + 512]
        vecs = E5.embed([x[3][x[1]:x[2]] for x in chunk], 'passage')
        with store.lock:
            store.con.execute('BEGIN')
            for x, v in zip(chunk, vecs):
                store.put_vector(x[0], E5.name, v)
            store.con.execute('COMMIT')
        if (i // 512) % 40 == 0:
            print(f'  conv {scope}: {i + len(chunk)}/{len(rows)} spans embedded '
                  f'({(i + len(chunk)) / max(1e-9, time.time() - t1):.0f}/s)', flush=True)
    st = store.stats('beam', scope)
    print(f"conv {scope}: {n} turns, {st} | lexical {t_lex:.0f}s, dense {time.time() - t1:.0f}s", flush=True)
    store.close()
    return path


if __name__ == '__main__':
    want = None if sys.argv[1:] == ['all'] else set(sys.argv[1:])
    for conv in conversations(want):
        ingest(conv)
