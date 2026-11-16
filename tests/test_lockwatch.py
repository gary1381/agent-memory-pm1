"""lockwatch: finds the write-lock holder in /proc/locks, alerts after ALERT_S, stops only pm1 batch jobs after KILL_S."""
import os
import sqlite3
import subprocess
import sys
import time

from pm1 import lockwatch
from pm1.lockwatch import Watch, holders, may_stop


def test_holder_of_a_real_wal_write_lock(tmp_path):
    db = str(tmp_path / 'm.sqlite')
    c = sqlite3.connect(db, isolation_level=None)
    c.execute('PRAGMA journal_mode=WAL')
    c.execute('CREATE TABLE t(x)')
    c.close()
    p = subprocess.Popen([sys.executable, '-c', f"""
import sqlite3, sys, time
c = sqlite3.connect({db!r}, isolation_level=None); c.execute('BEGIN IMMEDIATE'); c.execute('INSERT INTO t VALUES (1)')
print('held', flush=True); time.sleep(30)"""], stdout=subprocess.PIPE, text=True)
    try:
        assert p.stdout.readline().strip() == 'held'
        assert holders(db) == [p.pid]
        assert not lockwatch.probe(db, 0.2)
    finally:
        p.kill()
        p.wait()
    assert holders(db) == [] and lockwatch.probe(db, 0.2)


def test_proc_locks_parsing_ignores_readers_and_waiters(tmp_path):
    db = str(tmp_path / 'm.sqlite')
    sqlite3.connect(db).execute('PRAGMA journal_mode=WAL').fetchall()
    sqlite3.connect(db).execute('CREATE TABLE t(x)')
    st = os.stat(db + '-shm')
    k = f'{os.major(st.st_dev):02x}:{os.minor(st.st_dev):02x}:{st.st_ino}'
    text = (f'1: POSIX  ADVISORY  READ 11 {k} 128 128\n'
            f'2: POSIX  ADVISORY  WRITE 22 {k} 120 120\n'
            f'2: -> POSIX  ADVISORY  WRITE 33 {k} 120 120\n'
            f'3: POSIX  ADVISORY  WRITE 44 {k} 121 121\n')
    assert holders(db, text) == [22]


def test_only_pm1_batch_jobs_may_be_stopped():
    me = os.getuid()
    ok = [{'pid': 5, 'uid': me, 'cmd': '/x/pm1/.venv/bin/python /x/pm1/.venv/bin/pm1-worker --until 07:00'},
          {'pid': 6, 'uid': me, 'cmd': 'python -m pm1.sessindex --db m.sqlite'},
          {'pid': 7, 'uid': me, 'cmd': '.venv/bin/python eval/replay.py --x', 'cwd': '/h/precision-memory/merge/pm1'}]
    no = [{'pid': 8, 'uid': me, 'cmd': '/x/pm1/.venv/bin/python /x/pm1/.venv/bin/pm1-mcp'},
          {'pid': 9, 'uid': me, 'cmd': '/x/pm1/.venv/bin/pm1-capture watch -i 60'},
          {'pid': 10, 'uid': me, 'cmd': 'sqlite3 memory.sqlite'},
          {'pid': 11, 'uid': me, 'cmd': 'python eval/run.py', 'cwd': '/h/other-project'},
          {'pid': 12, 'uid': me + 1, 'cmd': 'pm1-worker'},
          {'pid': 13, 'gone': True}]
    assert all(may_stop(i) for i in ok) and not any(may_stop(i) for i in no)


def run(watch, clock, locked_rounds, step=60):
    outs = []
    for _ in range(locked_rounds):
        outs.append(watch.round())
        clock[0] += step
    return outs


def test_alert_then_stop_batch_job_and_report_others(tmp_path):
    clock, killed, notes = [1000.0], [], []
    infos = {101: {'pid': 101, 'uid': os.getuid(), 'cmd': 'python -m pm1.sessindex'},
             102: {'pid': 102, 'uid': os.getuid(), 'cmd': 'pm1-mcp'}}
    for pid, expect_kill in ((101, True), (102, False)):
        killed.clear()
        notes.clear()
        w = Watch(str(tmp_path / 'm.sqlite'), probe_s=20, alert_s=120, kill_s=600, clock=lambda: clock[0],
                  probe_fn=lambda: False, holders_fn=lambda: [pid], info_fn=lambda p: dict(infos[p]),
                  kill_fn=lambda p: killed.append(p) or 'SIGTERM', notify_fn=notes.append)
        outs = run(w, clock, 12)
        assert outs[0]['held_s'] == 20
        assert sum('write lock held' in n for n in notes) == 1          # one alert, not one per round
        assert killed == ([pid] if expect_kill else [])
        assert ('stopped' in outs[-1] or any('stopped' in o for o in outs)) == expect_kill
    w.probe_fn = lambda: True
    assert w.round() == {'locked': False} and 'free again' in notes[-1]
    rows = (tmp_path / 'lockwatch.jsonl').read_text().splitlines()
    assert any('"released"' in r for r in rows) and any('"stopped"' in r for r in rows)


def test_short_holds_are_silent(tmp_path):
    clock, notes, state = [0.0], [], iter([False, True, False, True])
    w = Watch(str(tmp_path / 'm.sqlite'), clock=lambda: clock[0], probe_fn=lambda: next(state),
              holders_fn=lambda: [3], info_fn=lambda p: {'pid': p}, notify_fn=notes.append)
    run(w, clock, 4)
    assert notes == []
