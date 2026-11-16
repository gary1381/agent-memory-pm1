import numpy as np

from pm1 import Store
from pm1.ledger import Ledger, norm_key


class BagDense:
    """deterministic bag-of-words embedder for tests"""
    name = 'bag'

    def embed(self, texts, kind):
        out = np.zeros((len(texts), 384), np.float32)
        for i, t in enumerate(texts):
            for w in t.lower().replace(':', ' ').split():
                out[i, hash(w) % 384] += 1
        n = np.linalg.norm(out, axis=1, keepdims=True)
        return out / np.where(n == 0, 1, n)


def test_norm_key_ignores_order_and_stopwords():
    assert norm_key('the user deadline', 'Current Date') == norm_key('deadline', 'date')


def test_supersession_and_lines(tmp_path):
    st = Store(tmp_path / 'm.sqlite')
    led = Ledger(st)
    led.add_extraction('b', 's', 's1|2024-03-01|0', '2024-03-01',
                       {'state': [{'subject': 'project', 'attribute': 'deadline', 'value': 'April 15'}],
                        'events': [{'date': '2024-03-01', 'what': 'kickoff of the project', 'kind': 'done'}]})
    led.add_extraction('b', 's', 's4|2024-04-02|0', '2024-04-02',
                       {'state': [{'subject': 'project', 'attribute': 'deadline', 'value': 'May 1',
                                   'previous': 'April 15'}],
                        'instructions': [{'text': 'always give dates in ISO format'}]})
    assert led.resolve('b', 's') == 1
    s = led.stats('b', 's')
    assert s['state'] == 2 and s['superseded'] == 1 and s['events'] == 1 and s['notes'] == 1
    d = BagDense()
    qv = d.embed(['what is the project deadline'], 'query')[0]
    lines = led.state_lines('b', 's', qv, d, min_sim=0.1)
    assert lines and 'May 1' in lines[0] and 'April 15' in lines[0] and lines[0].index('May 1') < lines[0].index('April 15')
    ev = led.event_lines('b', 's', d.embed(['project kickoff'], 'query')[0], d, min_sim=0.1)
    assert ev == ['- 2024-03-01: kickoff of the project']
