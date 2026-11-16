"""pm1.pin: files are locked in RAM and re-locked when they grow or are replaced."""
import os

from pm1.pin import refresh


def locked_kb() -> int:
    return next(int(l.split()[1]) for l in open('/proc/self/status') if l.startswith('VmLck:'))


def test_pins_grow_and_replace(tmp_path):
    f = tmp_path / 'db'
    f.write_bytes(b'x' * (1 << 20))
    pins, base = {}, locked_kb()
    refresh(pins, [str(f)], log=lambda m: None)
    assert locked_kb() - base >= 1024
    key = pins[str(f)].key
    refresh(pins, [str(f)], log=lambda m: None)            # unchanged: same mapping
    assert pins[str(f)].key == key
    with open(f, 'ab') as h:
        h.write(b'y' * (1 << 20))
    refresh(pins, [str(f)], log=lambda m: None)            # grown: re-locked at the new size, old one released
    assert pins[str(f)].size == 2 << 20 and 2048 <= locked_kb() - base < 3072
    g = tmp_path / 'new'
    g.write_bytes(b'z' * 4096)
    os.replace(g, f)                                       # replaced: new inode
    refresh(pins, [str(f)], log=lambda m: None)
    assert pins[str(f)].size == 4096 and locked_kb() - base < 1024
    refresh(pins, [str(tmp_path / 'missing')], log=lambda m: None)
    for p in pins.values():
        p.release()
