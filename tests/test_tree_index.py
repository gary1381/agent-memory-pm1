import numpy as np

from pm1.tree_index import TreeIndex


class Bag:
    vocab = 'gift budget holiday redis connection dog rex interview'.split()

    def embed(self, texts, kind):
        out = []
        for t in texts:
            v = np.array([1.0 if w in t.lower() else 0.0 for w in self.vocab]) + 1e-3
            out.append(v / np.linalg.norm(v))
        return np.array(out, np.float32)


ITEMS = [{'layer': 'version', 'id': 'v1', 'text': 'User — holiday gift budget: $400 (2024-12-05) → $450 (2024-12-06)',
          'evidence': ['t1', 't3'], 'day': '2024-12-06'},
         {'layer': 'loop', 'id': 'o1', 'text': '[task, open] Fix the Redis connection (due 2024-07-05)',
          'evidence': ['t9'], 'day': '2024-07-03'},
         {'layer': 'event', 'id': 'e1', 'text': '2024-12-07: User adopted a dog named Rex', 'evidence': ['t4'],
          'day': '2024-12-07'}]


def test_hybrid_search_and_gate():
    d = Bag()
    ti = TreeIndex(ITEMS, d)
    q = 'What is my holiday gift budget?'
    hits = ti.search(q, d.embed([q], 'query')[0], k=2, min_sim=0.3)
    assert hits and hits[0][0]['id'] == 'v1'
    q2 = 'What is the capital of France?'
    assert ti.search(q2, d.embed([q2], 'query')[0], k=2, min_sim=0.7) == []       # nothing close -> no block
    order_rev = lambda q, texts: list(range(len(texts)))[::-1]
    assert ti.search('redis dog', None, k=3, rerank=order_rev)                      # BM25-only + rerank path
