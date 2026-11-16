"""Background indexer: turns queued span jobs into dense / token vectors.

  PM1_DB=~/.local/state/pm1/memory.sqlite PM1_DENSE=local PM1_TOKEN=pplx python -m pm1.worker

Bounded backfill (the nightly timer, deploy/pm1-backfill.*): stop at a local clock time, cap the rate, take the
newest spans first (recent sessions are the ones asked about), pause while the machine is busy, exit when the
queue is empty, and stop after repeated failures instead of hammering a service that is down:

  pm1-worker --until 07:00 --rate 12 --newest-first --max-load 12 --exit-when-empty --report <file.jsonl>
"""
from __future__ import annotations

import argparse
import datetime as dt
import json
import os
import signal
import time

from .paths import default_db
from .store import Store


def default_dense() -> str:
    """'local' (fastembed, CPU) when it is installed, else no dense vectors: BM25 + rerank still work."""
    import importlib.util
    return 'local' if importlib.util.find_spec('fastembed') else 'none'


def build_embedders(dense: str, token: str):
    d = t = None
    if dense == 'local':
        from .embed import LocalE5
        d = LocalE5()
    elif dense == 'split':  # docs on the shared GPU e5 (a remote GPU host), queries on local CPU, same GGUF
        from .embed import SplitE5
        d = SplitE5()
    elif dense == 'http':
        from .embed import HttpE5
        d = HttpE5(url=os.environ.get('PM1_E5_URL', 'http://127.0.0.1:8767/v1/embeddings'))
    if token == 'pplx':
        from .embed import PplxLate
        t = PplxLate(url=os.environ.get('PM1_PPLX_URL', 'http://127.0.0.1:8770/v1/encode'))
    return d, t


def run_once(store: Store, dense=None, token=None, batch: int = 32, newest_first: bool = False) -> dict:
    done = {'dense': 0, 'tok': 0, 'errors': 0}
    for kind, emb in (('dense', dense), ('tok', token)):
        if emb is None:
            continue
        jobs = store.claim_jobs(kind, emb.name, batch, newest_first=newest_first)
        if not jobs:
            continue
        ids, texts = [j['span_id'] for j in jobs], [j['text'] for j in jobs]
        try:
            if kind == 'dense':
                store.put_vectors(emb.name, zip(ids, emb.embed(texts, 'passage')))
            else:
                for sid, m in zip(ids, emb.encode(texts, 'document')):
                    store.put_tokvecs(sid, emb.name, m)
            store.finish_jobs(kind, emb.name, ids)
            done[kind] += len(ids)
        except Exception as exc:  # retried up to 3 attempts
            store.finish_jobs(kind, emb.name, ids, error=f'{type(exc).__name__}: {exc}'[:300])
            done['errors'] += len(ids)
    return done


def deadline(until: str | None, now: dt.datetime | None = None) -> dt.datetime | None:
    """'07:00' -> the next 07:00 local time (today if still ahead, else tomorrow)."""
    if not until:
        return None
    now = now or dt.datetime.now()
    h, m = (int(x) for x in until.split(':'))
    d = now.replace(hour=h, minute=m, second=0, microsecond=0)
    return d if d > now else d + dt.timedelta(days=1)


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument('--db', default=default_db())
    ap.add_argument('--dense', default=os.environ.get('PM1_DENSE') or default_dense(), choices=['local', 'split', 'http', 'none'])
    ap.add_argument('--token', default=os.environ.get('PM1_TOKEN', 'none'), choices=['pplx', 'none'])
    ap.add_argument('--once', action='store_true')
    ap.add_argument('--poll', type=float, default=2.0)
    ap.add_argument('--batch', type=int, default=32)
    ap.add_argument('--until', help='local HH:MM to stop at (e.g. 07:00)')
    ap.add_argument('--rate', type=float, default=0.0, help='max spans per second (0 = unlimited)')
    ap.add_argument('--newest-first', action='store_true')
    ap.add_argument('--max-load', type=float, default=0.0, help='pause while the 1-min load average is above this')
    ap.add_argument('--exit-when-empty', action='store_true')
    ap.add_argument('--max-errors', type=int, default=5, help='stop after this many failed batches in a row')
    ap.add_argument('--report', help='append a JSON summary line to this file')
    a = ap.parse_args()
    os.makedirs(os.path.dirname(a.db) or '.', exist_ok=True)
    store = Store(a.db)
    store.con.execute("UPDATE jobs SET state='pending' WHERE state='running'")  # recover after a crash
    dense, token = build_embedders(a.dense, a.token)
    def _term(*_):                                   # systemd stop -> same clean exit as Ctrl-C
        raise KeyboardInterrupt
    signal.signal(signal.SIGTERM, _term)
    stop_at, t0 = deadline(a.until), time.time()
    total = {'dense': 0, 'tok': 0, 'errors': 0, 'paused_s': 0}
    reason, bad = 'stopped', 0
    try:
        while True:
            if stop_at and dt.datetime.now() >= stop_at:
                reason = 'deadline'
                break
            if a.max_load and os.getloadavg()[0] > a.max_load:
                time.sleep(30)
                total['paused_s'] += 30
                continue
            t1 = time.time()
            done = run_once(store, dense, token, a.batch, a.newest_first)
            for k in ('dense', 'tok', 'errors'):
                total[k] += done[k]
            if a.once:
                print(done)
                return
            n = done['dense'] + done['tok']
            bad = bad + 1 if done['errors'] and not n else 0
            if bad >= a.max_errors:
                reason = 'errors'
                break
            if not n and not done['errors']:
                if a.exit_when_empty:
                    reason = 'empty'
                    break
                time.sleep(a.poll)
            elif bad:
                time.sleep(min(300, 15 * 2 ** bad))      # back off while the embedding service is failing
            elif a.rate:
                time.sleep(max(0.0, n / a.rate - (time.time() - t1)))
    except KeyboardInterrupt:
        reason = 'interrupted'
    finally:
        if a.report and not a.once:
            pending = store.con.execute("SELECT count(*) FROM jobs WHERE state='pending' AND attempts < 3").fetchone()[0]
            line = {'ended': dt.datetime.now().astimezone().isoformat(timespec='seconds'), 'reason': reason,
                    'minutes': round((time.time() - t0) / 60, 1), **total, 'pending_after': pending}
            with open(os.path.expanduser(a.report), 'a') as fh:
                fh.write(json.dumps(line) + '\n')
            print(json.dumps(line))


if __name__ == '__main__':
    main()
