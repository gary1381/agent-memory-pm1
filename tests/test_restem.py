"""pm1.restem: rebuilding span_fts with porter makes inflected English forms match; plain restores it."""
from pm1.restem import current, restem
from pm1.store import Store


def test_restem_porter_and_back(tmp_path):
    s = Store(tmp_path / 'm.sqlite')
    s.add_episode(bank='b', scope='s', text='She went climbing with the kids last weekend.')
    s.add_episode(bank='b', scope='s', text='我们周末去爬山了')
    hit = lambda q: s.con.execute('SELECT count(*) FROM span_fts WHERE span_fts MATCH ?', (q,)).fetchone()[0]
    assert current(s.con) == 'porter' and hit('"climb"') == 1
    assert restem(s.con, 'plain') == 2
    assert current(s.con) == 'plain' and hit('"climb"') == 0
    assert restem(s.con, 'porter', batch=1) == 2
    assert current(s.con) == 'porter' and hit('"climb"') == 1 and hit('"climbed"') == 1 and hit('"爬山"') == 1
    restem(s.con, 'plain')
    assert current(s.con) == 'plain' and hit('"climb"') == 0 and hit('"climbing"') == 1
