"""Keep the pm1 database and the session vector index in RAM (2026-10-28).

The live store sits on the /home disk, which also holds 85 GB of other programs' swap: only ~20% of the 1.4 GB
database stayed in the page cache, and a keyword search that missed it waited 3-13 s on reads that take 0.2 s when
cached. This process maps the files read-only and mlock()s them, so their page-cache pages cannot be evicted. Every
`interval` seconds it re-maps a file that grew or was replaced (new inode), then releases the old mapping.

    python -m pm1.pin [--interval 300] FILE...
"""
from __future__ import annotations

import argparse
import ctypes
import ctypes.util
import mmap
import os
import signal
import sys
import time

libc = ctypes.CDLL(ctypes.util.find_library('c'), use_errno=True)
libc.mmap.restype = ctypes.c_void_p
libc.mmap.argtypes = [ctypes.c_void_p, ctypes.c_size_t, ctypes.c_int, ctypes.c_int, ctypes.c_int, ctypes.c_long]
libc.munmap.argtypes = [ctypes.c_void_p, ctypes.c_size_t]
libc.mlock.argtypes = [ctypes.c_void_p, ctypes.c_size_t]
MAP_FAILED = ctypes.c_void_p(-1).value


class Pin:
    """one locked read-only mapping of a file; key = (inode, size) it was taken at"""

    def __init__(self, path: str):
        st = os.stat(path)
        self.key, self.size = (st.st_ino, st.st_size), st.st_size
        fd = os.open(path, os.O_RDONLY)
        try:
            addr = libc.mmap(None, self.size, mmap.PROT_READ, mmap.MAP_SHARED, fd, 0)
        finally:
            os.close(fd)                                  # the mapping keeps the file referenced
        if addr in (None, MAP_FAILED):
            raise OSError(ctypes.get_errno(), f'mmap {path}')
        if libc.mlock(addr, self.size) != 0:              # faults every page in, then keeps it resident
            err = ctypes.get_errno()
            libc.munmap(addr, self.size)
            raise OSError(err, f'mlock {path}: {os.strerror(err)}')
        self.addr = addr

    def release(self) -> None:
        libc.munmap(self.addr, self.size)                 # munmap also unlocks


def refresh(pins: dict[str, Pin], paths: list[str], log=print) -> None:
    for p in paths:
        try:
            st = os.stat(p)
        except FileNotFoundError:
            continue
        old = pins.get(p)
        if st.st_size == 0 or (old and old.key == (st.st_ino, st.st_size)):
            continue
        t = time.perf_counter()
        try:
            pins[p] = Pin(p)                              # lock the new mapping before dropping the old one
        except OSError as exc:
            log(f'pm1-pin: {exc}')
            continue
        if old:
            old.release()
        log(f'pm1-pin: {p} {st.st_size / 2**20:.0f} MiB locked in {time.perf_counter() - t:.1f} s')


def main(argv: list[str] | None = None) -> None:
    ap = argparse.ArgumentParser(prog='pm1-pin')
    ap.add_argument('files', nargs='+')
    ap.add_argument('--interval', type=float, default=300)
    a = ap.parse_args(argv)
    paths = [os.path.realpath(f) for f in a.files]
    pins: dict[str, Pin] = {}
    signal.signal(signal.SIGTERM, lambda *_: sys.exit(0))
    log = lambda m: print(m, flush=True)
    while True:
        refresh(pins, paths, log)
        time.sleep(a.interval)


if __name__ == '__main__':
    main()
