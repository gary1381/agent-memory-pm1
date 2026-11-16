"""Rebuild a store's keyword index (span_fts) with or without English stemming (2026-10-29).

span_fts keeps pm1.text.tokenize's space-joined tokens; FTS5's tokenizer then splits them at index and query time, so
switching it changes every keyword search with no code change. ``porter`` (FTS5's porter stemmer around unicode61)
merges climbing / climbed / climb; it leaves CJK tokens alone. LoCoMo BM25-only: R@1 0.316 -> 0.366
(eval/tokenizer_cmp_en.py); PerLTQA zh unchanged.

    python -m pm1.restem DB [--plain]        # porter by default; --plain restores unicode61
"""
from __future__ import annotations

import argparse
import sqlite3
import time

TOKENIZERS = {'porter': 'porter unicode61 remove_diacritics 0', 'plain': 'unicode61 remove_diacritics 0'}


def current(con: sqlite3.Connection) -> str:
    sql = con.execute("SELECT sql FROM sqlite_master WHERE name = 'span_fts'").fetchone()[0]
    return 'porter' if 'porter' in sql else 'plain'


def restem(con: sqlite3.Connection, kind: str = 'porter', batch: int = 20000) -> int:
    """recreate span_fts with tokenizer ``kind`` from its own stored tokens; returns the row count. The copy runs in
    batches so a live store's writers wait seconds, not the minutes of one transaction (305K rows: 97 s, past their
    30 s busy timeout); the swap catches up on whatever they changed meanwhile."""
    con.execute('DROP TABLE IF EXISTS span_fts_new')
    con.execute(f"CREATE VIRTUAL TABLE span_fts_new USING fts5(toks, tokenize='{TOKENIZERS[kind]}')")
    last = 0
    while True:
        rows = con.execute('SELECT rowid, toks FROM span_fts WHERE rowid > ? ORDER BY rowid LIMIT ?',
                           (last, batch)).fetchall()
        if not rows:
            break
        con.execute('BEGIN IMMEDIATE')
        try:
            con.executemany('INSERT INTO span_fts_new(rowid, toks) VALUES (?, ?)', rows)
            con.execute('COMMIT')
        except BaseException:
            con.execute('ROLLBACK')
            raise
        last = rows[-1][0]
        time.sleep(0.05)
    con.execute('BEGIN IMMEDIATE')
    try:
        con.execute('DELETE FROM span_fts_new WHERE rowid NOT IN (SELECT rowid FROM span_fts)')
        con.execute('INSERT INTO span_fts_new(rowid, toks) SELECT rowid, toks FROM span_fts '
                    'WHERE rowid NOT IN (SELECT rowid FROM span_fts_new)')
        n = con.execute('SELECT count(*) FROM span_fts_new').fetchone()[0]
        con.execute('DROP TABLE span_fts')
        con.execute('ALTER TABLE span_fts_new RENAME TO span_fts')
        con.execute('COMMIT')
    except BaseException:
        con.execute('ROLLBACK')
        raise
    return n


def main(argv: list[str] | None = None) -> None:
    ap = argparse.ArgumentParser(prog='pm1-restem')
    ap.add_argument('db')
    ap.add_argument('--plain', action='store_true', help='plain unicode61 (no stemming)')
    a = ap.parse_args(argv)
    con = sqlite3.connect(a.db, isolation_level=None, timeout=60)
    kind = 'plain' if a.plain else 'porter'
    if current(con) == kind:
        print(f'{a.db}: span_fts already {kind}')
        return
    t = time.perf_counter()
    n = restem(con, kind)
    print(f'{a.db}: span_fts rebuilt as {kind}, {n} rows in {time.perf_counter() - t:.0f} s')


if __name__ == '__main__':
    main()
