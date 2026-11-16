"""memory_recall's second round: a term only the first round's top turn and its neighbour carry reaches a turn the
question's own words miss."""
import numpy as np

from pm1 import Recaller, Store


class Ids:
    name = 'stub'

    def embed(self, texts, kind):
        rng = np.random.default_rng(abs(hash(tuple(texts))) % 2 ** 32)
        v = rng.normal(size=(len(texts), 8)).astype(np.float32)
        return v / np.linalg.norm(v, axis=1, keepdims=True)


def build(tmp_path, monkeypatch, prf):
    monkeypatch.setenv('PM1_RECALL_PRF', str(prf))
    s = Store(tmp_path / f'm{prf}.sqlite')
    add = lambda seq, session, text: s.add_episode(bank='b', scope='s', session=session, seq=seq, role='user', text=text)
    add(0, 'a', 'I moved to Lisbon for the new job')
    add(1, 'a', 'the harbor office there is lovely')
    add(0, 'b', 'the harbor office wrote back about the visa')
    for i in range(30):
        add(i, 'c', f'unrelated chatter number {i} about gardens')
    return s, Recaller(s, dense=Ids(), router=None)


def test_second_round_adds_the_linked_turn(tmp_path, monkeypatch):
    s, rec = build(tmp_path, monkeypatch, 3)
    lex = [i for i, _ in s.bm25('b', 's', 'Lisbon job', 50)]
    plex, _ = rec._second_round('b', 's', 'Lisbon job', lex)
    got = {s.con.execute('SELECT e.text FROM spans x JOIN episodes e ON e.id = x.episode_id WHERE x.id=?', (i,)).fetchone()[0]
           for i in plex}
    assert 'the harbor office wrote back about the visa' in got
    assert all('the harbor office wrote back about the visa' not in
               s.con.execute('SELECT e.text FROM spans x JOIN episodes e ON e.id = x.episode_id WHERE x.id=?', (i,)).fetchone()[0]
               for i in lex)
