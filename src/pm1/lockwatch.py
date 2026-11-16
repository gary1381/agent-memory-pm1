"""Watch the pm1 database for a write lock held too long, say who holds it, and end pm1's own batch jobs that do.

About 20 processes share one SQLite file. A lock cannot outlive its process (the kernel drops fcntl locks on exit and
SQLite rolls the transaction back), so a long hold means a live process kept a write transaction open: a script that
wrote and then went on with slow work, a paused process, an sqlite3 shell left inside BEGIN.

Every round (default 60 s) a probe asks for the write lock (BEGIN IMMEDIATE, then ROLLBACK, waiting up to PROBE_S).
When it cannot get it, the holder is read from /proc/locks (WAL: a WRITE lock on byte 120 of the -shm file;
rollback journal: the RESERVED/PENDING bytes of the database file).
- held ALERT_S (default 120 s): log, desktop notification, a row in lockwatch.jsonl;
- held KILL_S (default 600 s) by a pm1 batch job (nightly backfill, vector index rebuild, eval script under pm1/eval),
  owned by this user: SIGTERM, then SIGKILL after 30 s. SQLite rolls its transaction back; nothing stored is lost.
  Anything else (MCP servers, capture, hooks, other projects, unknown processes) is only reported, again every 30 min.
The kill rule was approved on 2026-10-27.

  python -m pm1.lockwatch watch [-i 60]    the loop (pm1-lockwatch.service)
  python -m pm1.lockwatch once             one probe, printed as JSON
"""
from __future__ import annotations

import argparse
import json
import os
import signal
import sqlite3
import subprocess
import sys
import time

from .paths import default_db

WAL_WRITE_BYTE = 120
RESERVED, PENDING = 0x40000001, 0x40000000
# pm1 batch jobs that may be stopped (matched on the command line of a process owned by this user)
BATCH = ('pm1-worker', 'pm1.worker', 'pm1-sessindex', 'pm1.sessindex', '/pm1/eval/')
NEVER = ('pm1-mcp', 'pm1.mcp', 'pm1-capture', 'pm1.capture', 'pm1-cc-hook', 'pm1.cc_hook', 'pm1.lockwatch')


def _key(path: str) -> str | None:
    try:
        st = os.stat(path)
    except OSError:
        return None
    return f'{os.major(st.st_dev):02x}:{os.minor(st.st_dev):02x}:{st.st_ino}'


def holders(db: str, locks_text: str | None = None) -> list[int]:
    """pids holding the write lock of ``db`` (from /proc/locks, or the given text)"""
    shm, main = _key(db + '-shm'), _key(db)
    if locks_text is None:
        with open('/proc/locks') as f:
            locks_text = f.read()
    out = []
    for line in locks_text.splitlines():
        p = line.split()
        if len(p) < 8 or p[1] == '->':          # '->' lines are waiters, not holders
            continue
        kind, pid, key, start, end = p[3], p[4], p[5], p[6], p[7]
        if kind != 'WRITE':
            continue
        lo, hi = int(start), (1 << 63) if end == 'EOF' else int(end)
        if (key == shm and lo <= WAL_WRITE_BYTE <= hi) or (key == main and (lo <= RESERVED <= hi or lo <= PENDING <= hi)):
            if int(pid) not in out:
                out.append(int(pid))
    return out


def proc_info(pid: int) -> dict:
    info: dict = {'pid': pid}
    try:
        with open(f'/proc/{pid}/cmdline', 'rb') as f:
            info['cmd'] = f.read().replace(b'\0', b' ').decode(errors='replace').strip()[:400]
        info['cwd'] = os.readlink(f'/proc/{pid}/cwd')
        info['uid'] = os.stat(f'/proc/{pid}').st_uid
    except OSError:
        info['gone'] = True
    return info


def may_stop(info: dict) -> bool:
    """only pm1's own batch jobs, owned by this user; never servers, capture, hooks or anything unknown"""
    cmd = info.get('cmd') or ''
    if info.get('gone') or info.get('uid') != os.getuid() or info.get('pid') in (os.getpid(), 1):
        return False
    if any(n in cmd for n in NEVER):
        return False
    if any(b in cmd for b in BATCH):
        return True
    # `python eval/x.py` started inside the pm1 checkout
    return '/pm1/' in (info.get('cwd') or '') + '/' and any(a.startswith('eval/') and a.endswith('.py')
                                                            for a in cmd.split())


def probe(db: str, wait_s: float) -> bool:
    """True when the write lock could be taken (and was released at once)"""
    con = sqlite3.connect(db, timeout=wait_s, isolation_level=None)
    try:
        con.execute('BEGIN IMMEDIATE')
        con.execute('ROLLBACK')
        return True
    except sqlite3.OperationalError as e:
        if 'locked' in str(e) or 'busy' in str(e):
            return False
        raise
    finally:
        con.close()


def notify(text: str) -> None:
    print(time.strftime('%H:%M:%S'), text, flush=True)
    try:
        subprocess.run(['notify-send', '-a', 'pm1', 'pm1 database lock', text], timeout=5,
                       stdout=subprocess.DEVNULL, stderr=subprocess.DEVNULL)
    except (OSError, subprocess.SubprocessError):
        pass


class Watch:
    def __init__(self, db: str, *, probe_s: float = 20, alert_s: float = 120, kill_s: float = 600,
                 realert_s: float = 1800, log: str | None = None, clock=time.time, probe_fn=None, holders_fn=None,
                 info_fn=proc_info, kill_fn=None, notify_fn=notify):
        self.db, self.probe_s, self.alert_s, self.kill_s, self.realert_s = db, probe_s, alert_s, kill_s, realert_s
        self.log = log or os.path.join(os.path.dirname(db) or '.', 'lockwatch.jsonl')
        self.clock, self.info_fn, self.notify = clock, info_fn, notify_fn
        self.probe_fn = probe_fn or (lambda: probe(self.db, self.probe_s))
        self.holders_fn = holders_fn or (lambda: holders(self.db))
        self.kill_fn = kill_fn or self._stop
        self.since: float | None = None      # first round the lock was unavailable
        self.pids: tuple = ()
        self.alerted: float | None = None
        self.stopped: set[int] = set()

    def _record(self, event: str, **kw) -> dict:
        row = {'ts': time.strftime('%Y-%m-%dT%H:%M:%S%z'), 'event': event, **kw}
        try:
            with open(self.log, 'a') as f:
                f.write(json.dumps(row, ensure_ascii=False) + '\n')
        except OSError:
            pass
        return row

    @staticmethod
    def _stop(pid: int) -> str:
        os.kill(pid, signal.SIGTERM)
        for _ in range(30):
            time.sleep(1)
            if not os.path.exists(f'/proc/{pid}'):
                return 'SIGTERM'
        os.kill(pid, signal.SIGKILL)
        return 'SIGKILL'

    def round(self) -> dict:
        t0 = self.clock()
        if self.probe_fn():
            if self.since is not None:
                row = self._record('released', held_s=round(t0 - self.since), pids=list(self.pids))
                if self.alerted is not None:
                    self.notify(f'write lock free again after {row["held_s"]} s')
            self.since, self.pids, self.alerted = None, (), None
            return {'locked': False}
        pids = tuple(self.holders_fn())
        if self.since is None or (pids and self.pids and set(pids).isdisjoint(self.pids)):
            self.since = t0 - self.probe_s          # a new holder: count from when this probe started waiting
            self.alerted = None
        self.pids = pids or self.pids
        held = self.clock() - self.since
        infos = [self.info_fn(p) for p in self.pids]
        out = {'locked': True, 'held_s': round(held), 'holders': infos}
        if held >= self.alert_s and (self.alerted is None or self.clock() - self.alerted >= self.realert_s):
            self.alerted = self.clock()
            who = '; '.join(f"pid {i['pid']} {i.get('cmd', '?')[:120]}" for i in infos) or 'holder not found'
            self._record('long_hold', held_s=round(held), holders=infos)
            self.notify(f'write lock held {round(held)} s by {who}')
        if held >= self.kill_s:
            for i in infos:
                if i['pid'] in self.stopped or not may_stop(i):
                    continue
                try:
                    how = self.kill_fn(i['pid'])
                except ProcessLookupError:
                    how = 'already gone'
                self.stopped.add(i['pid'])
                self._record('stopped', held_s=round(held), how=how, **i)
                self.notify(f"stopped pm1 batch job pid {i['pid']} ({how}) after holding the write lock "
                            f"{round(held)} s: {i.get('cmd', '')[:120]}")
                out.setdefault('stopped', []).append(i['pid'])
        return out


def wal_mb(db: str) -> float:
    try:
        return os.path.getsize(db + '-wal') / 1e6
    except OSError:
        return 0.0


def main(argv: list[str] | None = None) -> None:
    ap = argparse.ArgumentParser(prog='pm1-lockwatch', description=__doc__.split('\n')[0])
    ap.add_argument('cmd', choices=('watch', 'once'))
    ap.add_argument('--db', default=default_db())
    ap.add_argument('-i', '--interval', type=float, default=60)
    ap.add_argument('--alert', type=float, default=float(os.environ.get('PM1_LOCK_ALERT_S', '120')))
    ap.add_argument('--kill', type=float, default=float(os.environ.get('PM1_LOCK_KILL_S', '600')))
    a = ap.parse_args(argv)
    w = Watch(a.db, alert_s=a.alert, kill_s=a.kill)
    if a.cmd == 'once':
        print(json.dumps(w.round() | {'wal_mb': round(wal_mb(a.db), 1)}, ensure_ascii=False))
        return
    print(f'pm1 lockwatch: {a.db}, every {a.interval:g} s, alert {a.alert:g} s, stop batch jobs {a.kill:g} s',
          flush=True)
    wal_warned = 0.0
    while True:
        t = time.time()
        try:
            w.round()
        except Exception as exc:  # the watcher itself must keep running
            print(time.strftime('%H:%M:%S'), f'round failed: {exc!r}', file=sys.stderr, flush=True)
        mb = wal_mb(a.db)           # a WAL that never shrinks means a reader blocks checkpoints
        if mb > 2000 and time.time() - wal_warned > 6 * 3600:
            wal_warned = time.time()
            w._record('wal_large', wal_mb=round(mb))
            notify(f'pm1 WAL file is {mb / 1000:.1f} GB: a long-open reader is blocking checkpoints')
        time.sleep(max(1.0, a.interval - (time.time() - t)))


if __name__ == '__main__':
    main()
