"""Shared all-project dense index for search_sessions (PM1_SESS_FULL).

Each client session starts its own pm1 MCP server (33 were running on 2026-10-27); an in-process matrix of every
project span (~340 MB float32 at 220K spans) in each of them would not fit. Instead the nightly backfill writes one
file set and every server memory-maps it, so the matrix sits in the page cache once:

  <dir>/sessdense.<db name>.<model>.npy    float32 matrix (rows = spans), memory-mapped read-only
  <dir>/sessdense.<db name>.<model>.npz    span ids, per-row scope / host / role codes and time in seconds, the code names
                                 (a few MB, loaded fully)

Vectors written after the build (rowid > the file's last rowid) are appended in each process as before; spans
deleted since the build drop out when candidates are joined back to the database. Files are replaced atomically.

  pm1-sessindex [--db PATH] [--model e5-small-q8]
"""
from __future__ import annotations

import argparse
import os
import time

import sqlite3

import numpy as np

SQL = ('SELECT v.rowid, v.span_id, v.vec, s.scope, e.session, e.role, e.occurred_at FROM vectors v '
       'JOIN spans s ON s.id = v.span_id JOIN episodes e ON e.id = s.episode_id '
       "WHERE s.bank=? AND s.scope LIKE 'proj:%' AND v.model=? AND v.rowid > ? ORDER BY v.rowid")


def paths(db: str, model: str) -> tuple[str, str]:
    stem = os.path.splitext(os.path.basename(db))[0]      # one index per database, even when several share a folder
    base = os.path.join(os.path.dirname(os.path.abspath(db)), f'sessdense.{stem}.{model}')
    return base + '.npy', base + '.npz'


def host_of(session: str) -> str:
    return session.split(':', 1)[0] if ':' in session else ''


class Vocab:
    """string <-> small int codes for the per-row metadata (a process keeps one, shared by the file and new rows)"""
    def __init__(self, names=()):
        self.names = list(names)
        self.code = {n: i for i, n in enumerate(self.names)}

    def __call__(self, name: str) -> int:
        c = self.code.get(name)
        if c is None:
            c = self.code[name] = len(self.names)
            self.names.append(name)
        return c


def secs(at) -> 'np.ndarray':
    """ISO times (UTC, as stored) -> int64 seconds; '' -> 0. Second resolution, like comparing the strings."""
    a = np.array([(x or '')[:19] or '1970-01-01T00:00:00' for x in at], dtype='datetime64[s]')
    return a.astype(np.int64)


def encode(rows, vocab: dict) -> dict:
    """rows of SQL -> arrays: span ids, scope / host / role codes, time in seconds"""
    return {'ids': np.array([r[1] for r in rows], np.int64),
            'scope': np.array([vocab['scope'](r[3]) for r in rows], np.int32),
            'host': np.array([vocab['host'](host_of(r[4])) for r in rows], np.int16),
            'role': np.array([vocab['role'](r[5]) for r in rows], np.int16),
            'at': secs([r[6] for r in rows])}


def build(db: str, model: str, bank: str = 'personal') -> dict:
    t0 = time.time()
    con = sqlite3.connect(f'file:{db}?mode=ro', uri=True)
    npy, npz = paths(db, model)
    n = con.execute("SELECT count(*) FROM vectors v JOIN spans s ON s.id = v.span_id WHERE s.bank=? AND "
                    "s.scope LIKE 'proj:%' AND v.model=?", (bank, model)).fetchone()[0]
    dim = None
    mat = None
    meta, i, last = [], 0, 0
    for r in con.execute(SQL, (bank, model, 0)):
        v = np.frombuffer(r[2], dtype=np.float16)
        if mat is None:
            dim = len(v)
            mat = np.lib.format.open_memmap(npy + '.tmp.npy', mode='w+', dtype=np.float32, shape=(n, dim))
        if i >= n:                                    # rows added while building: the next build takes them
            break
        mat[i] = v
        meta.append((r[0], r[1], None, r[3], r[4], r[5], r[6]))       # vector bytes are in the matrix
        i += 1
        last = r[0]
    if mat is None:
        return {'rows': 0}
    mat.flush()
    del mat
    if i < n:                                         # rows deleted while building: shrink
        full = np.load(npy + '.tmp.npy', mmap_mode='r')[:i]
        np.save(npy + '.tmp2.npy', np.asarray(full))
        os.replace(npy + '.tmp2.npy', npy + '.tmp.npy')
    vocab = {k: Vocab() for k in ('scope', 'host', 'role')}
    arr = encode(meta, vocab)
    np.savez(npz + '.tmp.npz', last=np.array([last]), **arr,
             **{f'{k}_names': np.array(v.names) for k, v in vocab.items()})
    os.replace(npy + '.tmp.npy', npy)                 # matrix first: a reader pairing a new npz with an old npy
    os.replace(npz + '.tmp.npz', npz)                 # would mis-index, so readers check the row counts match
    return {'rows': i, 'dim': dim, 'last_rowid': last, 'seconds': round(time.time() - t0, 1), 'path': npy}


def load(db: str, model: str) -> dict | None:
    """The shared index as {mat (memmap), ids, scope, host, role, at (codes / seconds), last, vocab}, or None when
    absent or inconsistent."""
    npy, npz = paths(db, model)
    try:
        z = np.load(npz)
        mat = np.load(npy, mmap_mode='r')
    except (OSError, ValueError):
        return None
    if mat.shape[0] != len(z['ids']):
        return None
    return {'mat': mat, 'last': int(z['last'][0]), **{k: z[k] for k in ('ids', 'scope', 'host', 'role', 'at')},
            'vocab': {k: Vocab(z[f'{k}_names'].tolist()) for k in ('scope', 'host', 'role')}}


def main() -> None:
    from .paths import default_db
    ap = argparse.ArgumentParser(description=__doc__.split('\n\n')[0])
    ap.add_argument('--db', default=None)
    ap.add_argument('--model', default='e5-small-q8')
    a = ap.parse_args()
    print(build(a.db or default_db(), a.model))


if __name__ == '__main__':
    main()
