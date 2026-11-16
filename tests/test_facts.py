"""Fact index: facts point back to turns; recall pulls those turns into the candidates (the records stay the text)."""
from __future__ import annotations

import numpy as np

from pm1 import Recaller, Store
from pm1.facts import FactIndex


class WordDense:
    """A toy embedder: one dimension per keyword."""
    name = 'word-dense'
    WORDS = ('europe', 'england', 'turtle', 'weather', 'pottery')

    def embed(self, texts, kind):
        out = []
        for t in texts:
            t = t.lower()
            v = np.array([float(w in t) for w in self.WORDS] + [0.05], dtype=np.float32)
            out.append(v / np.linalg.norm(v))
        return np.vstack(out)


def _store():
    s = Store(index_models=('dense:word-dense',))
    ids = []
    for i in range(20):
        text = 'Maria: I got the idea from that trip a few years ago, the castles were magic' if i == 7 else \
            f'Maria: chatting about the weather again, day {i}'
        ids.append(s.add_episode(bank='b', scope='s', session='s1', seq=i, role='user', text=text,
                                 occurred_at=f'2023-05-08T13:{i:02d}:00Z')['episode_id'])
    return s, ids


def test_replace_and_search():
    s, ids = _store()
    fx = FactIndex(s, WordDense())
    assert fx.replace_session('b', 's', 's1', [{'text': 'Maria visited England (a European country)',
                                                 'episodes': [ids[7]]},
                                                {'text': 'no source', 'episodes': []}], model='m') == 1
    assert fx.embed_pending() == 1
    q = WordDense().embed(['Which countries in Europe and England has Maria been to?'], 'query')[0]
    assert fx.episodes_for('b', 's', q, 4) == [ids[7]]
    fx.replace_session('b', 's', 's1', [], model='m')         # re-extraction replaces
    assert fx.count('b', 's') == 0


def test_recall_pulls_fact_turns(monkeypatch):
    monkeypatch.setenv('PM1_NEIGHBORS', '0')
    s, ids = _store()
    dense = WordDense()
    fx = FactIndex(s, dense)
    fx.replace_session('b', 's', 's1', [{'text': 'Maria visited England (a European country)', 'episodes': [ids[7]]}])
    fx.embed_pending()
    q = 'Which European countries has Maria been to?'
    without = Recaller(s, dense=dense, planning=False).recall('b', 's', q, budget_tokens=110, budget_ms=None)
    with_f = Recaller(s, dense=dense, planning=False, facts=fx).recall('b', 's', q, budget_tokens=110, budget_ms=None)
    assert ids[7] not in [it['episode_id'] for it in without.items]
    assert ids[7] in [it['episode_id'] for it in with_f.items]
    assert 'castles' in with_f.context and 'European country' not in with_f.context   # the record, not the fact


def test_job_extracts_quiet_sessions_once():
    from pm1 import facts_job
    s, ids = _store()
    fx = FactIndex(s, WordDense())
    prompts = []

    def call(prompt):
        prompts.append(prompt)
        return {'facts': [{'fact': 'Maria visited England (a European country)', 'turns': [7]},
                          {'fact': 'bad turn number', 'turns': [99]}]}
    assert facts_job.run_once(s, fx, call, 'test', idle_min=0) == 1
    assert fx.count('b', 's') == 1 and '[7] user: Maria: I got the idea' in prompts[0]
    assert facts_job.run_once(s, fx, call, 'test', idle_min=0) == 0            # unchanged session: not again
    s.add_episode(bank='b', scope='s', session='s1', seq=20, role='user', text='Maria: one more thing',
                  occurred_at='2023-05-08T14:00:00Z')
    assert facts_job.run_once(s, fx, call, 'test', idle_min=0) == 1            # grown: re-extracted


def test_chunks_split_long_sessions():
    from pm1 import facts_job
    turns = [{'episode_id': i, 'role': 'user', 'text': 'x' * 5000} for i in range(5)]
    assert [len(c) for c in facts_job.chunks(turns)] == [2, 2, 1]


def test_entity_hop_adds_facts_about_question_entities():
    s, ids = _store()
    fx = FactIndex(s, WordDense())
    fx.replace_session('b', 's', 's1', [
        {'text': 'Maria visited England (a European country)', 'episodes': [ids[7]]},
        {'text': 'Maria loves pottery classes', 'episodes': [ids[3]]},
        {'text': 'John loves pottery classes', 'episodes': [ids[4]]}])
    fx.embed_pending()
    q = 'Which European countries has Maria been to?'
    qv = WordDense().embed([q], 'query')[0]
    assert fx.episodes_for('b', 's', qv, 1) == [ids[7]]
    hopped = fx.episodes_for('b', 's', qv, 1, hop=4, question=q)
    assert hopped[:2] == [ids[7], ids[3]] and ids[4] not in hopped      # Maria's other fact, not John's


def test_parse_json_salvages_cut_output():
    from pm1.facts_job import parse_json
    assert parse_json('```json\n{"facts": []}\n```') == {'facts': []}
    cut = '{"facts": [{"fact": "a", "turns": [0]}, {"fact": "b", "turns": [1]}, {"fact": "c", "tu'
    assert parse_json(cut) == {'facts': [{'fact': 'a', 'turns': [0]}, {'fact': 'b', 'turns': [1]}], 'truncated': True}


def test_chunks_respect_max_chars():
    from pm1.facts_job import chunks
    turns = [{'text': 'x' * 1500} for _ in range(10)]
    assert [len(c) for c in chunks(turns, 4000)] == [2, 2, 2, 2, 2]
    assert len(chunks(turns)) == 2


class SeeAllRouter:
    """Stand-in reranker: keeps the order and records which texts it was shown."""
    def __init__(self):
        self.seen = []

    def rerank(self, query, docs, qlang, langs, budget):
        self.seen = list(docs)
        return list(range(len(docs))), {'rerank_model': 'stub'}


def test_graph_recall_adds_second_hop_candidates(monkeypatch):
    monkeypatch.setenv('PM1_NEIGHBORS', '0')
    s = Store(index_models=('dense:word-dense',))
    ids = []
    texts = ['Maria: my friend Ottoline moved away last spring'] + \
            [f'Maria: chatting about the weather again, day {i}' for i in range(1, 30)] + \
            ['Maria: Ottoline sends pottery photos from her new place every week']
    for i, t in enumerate(texts):
        ids.append(s.add_episode(bank='b', scope='s', session='s1', seq=i, role='user', text=t,
                                 occurred_at=f'2023-05-08T13:{i:02d}:00Z')['episode_id'])
    dense = WordDense()
    fx = FactIndex(s, dense)
    fx.replace_session('b', 's', 's1', [
        {'text': "Maria's friend Ottoline moved to England", 'episodes': [ids[0]]},
        {'text': 'Ottoline took up pottery', 'episodes': [ids[-1]]}] +
        [{'text': f'Maria talked about the weather on day {i}', 'episodes': [ids[i]]} for i in range(1, 30)])
    fx.embed_pending()
    q = 'Where in England did Maria\'s friend move?'
    qv = dense.embed([q], 'query')[0]
    assert ids[-1] not in fx.graph_episodes('b', 's', qv, hops=0, seeds=1)
    assert ids[-1] in fx.graph_episodes('b', 's', qv, hops=1, seeds=1)      # shares the rare name Ottoline
    monkeypatch.setenv('PM1_FACT_GRAPH', '1')
    monkeypatch.setenv('PM1_RERANK_K', '200')
    router = SeeAllRouter()                    # k_first=5: the first stage alone keeps 10 candidates at most
    Recaller(s, dense=dense, router=router, planning=False, facts=fx, k_first=5).recall(
        'b', 's', q, budget_tokens=2000, budget_ms=None)
    assert any('pottery photos' in d for d in router.seen)                    # reached the reranker via the graph
    monkeypatch.setenv('PM1_FACT_GRAPH', '0')
    router2 = SeeAllRouter()
    Recaller(s, dense=dense, router=router2, planning=False, facts=fx, k_first=5).recall(
        'b', 's', q, budget_tokens=2000, budget_ms=None)
    assert not any('pottery photos' in d for d in router2.seen)               # without it, never a candidate
