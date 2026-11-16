"""Planned recall (prereg amendment 16): off by default, each planned query reranked alone, results mixed."""
from pm1 import Recaller


def test_mix_lets_a_planned_hit_in(tmp_path, monkeypatch):
    r = Recaller.__new__(Recaller)
    r.planner, r.plan_k, r.plan_weight = (lambda q: ['garden tools', q, 'garden tools']), 60, 1
    r.seeds, r.plan_mix, r.plan_seed_q = 0, 'rrf', False
    seen = []

    def one(bank, scope, q, mode=None, budget_ms=-1.0, only=None, head_k=None):
        seen.append((q, head_k))
        return ([1, 2] if head_k is None else [3, 1]), {}, {'mode': 'routed'}
    r._rank_one = one
    order, _, t = r._rank('personal', 's', 'where does alice live')
    assert seen == [('where does alice live', None), ('garden tools', 60)]     # own rerank, duplicates and the question dropped
    assert order[0] == 1 and 3 in order and t['plan_queries'] == 1


def test_off_and_filtered_keep_the_plain_path(tmp_path):
    r = Recaller.__new__(Recaller)
    r.planner, r.plan_k, r.plan_weight = None, 60, 1
    r.seeds, r.plan_mix, r.plan_seed_q = 0, 'rrf', False
    r._rank_one = lambda *a, **k: ([1], {}, {'mode': 'routed'})
    assert r._rank('p', 's', 'q')[0] == [1]
    r.planner = lambda q: ['x']
    assert r._rank('p', 's', 'q', only={1})[0] == [1]


def test_weight_keeps_the_question_first(tmp_path):
    r = Recaller.__new__(Recaller)
    r.planner, r.plan_k, r.plan_weight = (lambda q: ['a', 'b', 'c']), 60, 3
    r.seeds, r.plan_mix, r.plan_seed_q = 0, 'rrf', False
    r._rank_one = lambda b, s, q, m=None, bm=-1.0, o=None, head_k=None: (([1, 2], {}, {'mode': 'routed'}) if head_k is None
                                                                          else ([9, 2], {}, {'mode': 'routed'}))
    assert r._rank('p', 's', 'q')[0] == [2, 1, 9]        # 2 is in every list; 1 and 9 tie, three first places each


def test_slots_keep_the_question_list_and_add_new_records():
    r = Recaller.__new__(Recaller)
    r.plan_mix, r.slot_n, r.slot_at, r.slot_hold = 'slot', 1, 2, 3
    order = [1, 2, 3, 4, 5, 6]
    assert r._mix(order, [[2, 5, 9], [5, 8], [1, 3]]) == [1, 2, 5, 8, 3, 4, 6]     # 5 once, 2 and 1 already held


def test_seed_list_is_reranked_against_the_seed():
    class Router:
        def rerank(self, q, docs, *a):
            Router.q = q
            return [1, 0, 2], {}
    r = Recaller.__new__(Recaller)
    r.seeds, r.seed_from, r.seed_q, r.profile, r.router = 1, 'rrf', False, 'wait', Router()

    class Store:
        def spans(self, bank, scope, ids):
            return {i: {'text': f'record {i} ' * 20, 'lang': 'en'} for i in ids}
    r.store = Store()
    r._rank_one = lambda *a, **k: ([7, 8], {}, {'mode': 'rrf'})
    r._material = lambda *a: 'material'
    r._search_from = lambda *a: ([7, 11], [12])
    assert r._seed_lists('p', 's', 'question', [1], -1.0) == [[12, 11]]
    assert Router.q.startswith('record 7')


def test_planned_list_is_reranked_with_question_and_its_top_record():
    r = Recaller.__new__(Recaller)
    r.planner, r.plan_k, r.plan_weight = (lambda q: ['planned']), 60, 1
    r.seeds, r.plan_mix, r.plan_seed_q = 0, 'rrf', True
    r._rank_one = lambda b, s, q, m=None, bm=-1.0, o=None, head_k=None: (([1], {}, {'mode': 'routed'}) if head_k is None
                                                                          else ([5, 6], {}, {'mode': 'routed'}))

    class Store:
        def spans(self, bank, scope, ids):
            return {i: {'text': f'record {i}', 'lang': 'en'} for i in ids}
    r.store = Store()
    seen = []
    r._rerank_against = lambda b, s, rq, ids, bm: seen.append(rq) or ids[::-1]
    assert r._rank('p', 's', 'question')[0] == [1, 6, 5]
    assert seen == ['question\nrecord 5']
