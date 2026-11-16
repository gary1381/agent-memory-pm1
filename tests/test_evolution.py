import numpy as np

from pm1.ledger import Ledger, norm_key
from pm1.store import Store


class FakeDense:
    """Bag-of-words embedder: cosine of word sets, enough to exercise the similarity gate."""
    def embed(self, texts, kind):
        vocab = 'holiday gift budget webinar date coffee color'.split()
        out = []
        for t in texts:
            v = np.array([1.0 if w in t.lower() else 0.0 for w in vocab]) + 1e-3
            out.append(v / np.linalg.norm(v))
        return np.array(out, np.float32)


def _ledger(tmp_path):
    led = Ledger(Store(tmp_path / 's.sqlite'))
    rows = [('c1', '2024-12-05', 'holiday gifts budget', '$400 total'),
            ('c2', '2024-12-05', 'holiday gift budget', '$450'),
            ('c3', '2025-01-02', 'favorite coffee', 'latte'),
            ('c4', '2025-01-03', 'favorite coffee', 'latte')]
    for ch, day, attr, val in rows:
        led.add_extraction('b', 's', ch, day, {'state': [{'subject': 'User', 'attribute': attr, 'value': val}]})
    led.resolve('b', 's')
    return led


def test_plural_attributes_chain():
    assert norm_key('User', 'holiday gifts budget') == norm_key('User', 'holiday gift budget')


def test_evolution_lines_match_only_changed_and_relevant(tmp_path):
    led, d = _ledger(tmp_path), FakeDense()
    q = 'What is my total budget for holiday gifts?'
    lines = led.evolution_lines('b', 's', q, d.embed([q], 'query')[0], d, min_sim=0.5)
    assert len(lines) == 1 and '$400 total' in lines[0] and lines[0].index('$400') < lines[0].index('$450')
    q2 = 'What coffee do I like?'                     # unchanged value -> no history line
    assert led.evolution_lines('b', 's', q2, d.embed([q2], 'query')[0], d, min_sim=0.5) == []
    q3 = 'When is the webinar?'                       # nothing relevant -> nothing shown
    assert led.evolution_lines('b', 's', q3, d.embed([q3], 'query')[0], d, min_sim=0.5) == []
