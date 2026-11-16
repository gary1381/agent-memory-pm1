import pytest

from pm1 import Recaller, Store
from pm1.lang import detect, english_compatible
from pm1.rerank import PROFILES, Reranker, Router


@pytest.mark.parametrize('text,tag', [
    ('I adopted a golden retriever named Max last spring and he loves the beach.', 'en'),
    ('Where did I buy my tennis racket?', 'en'),
    ('我上周在杭州买了一台二手相机，花了三千块。', 'zh'),
    ('我家的狗叫 Max，是一只 golden retriever。', 'zh'),
    ('先週、東京でカメラを買いました。', 'ja'),
    ('저는 어제 서울에서 카메라를 샀어요.', 'ko'),
    ('Ich habe letzte Woche in Hamburg eine gebrauchte Kamera gekauft, für dreihundert Euro.', 'other'),
    ('Compré una cámara usada en Madrid la semana pasada por trescientos euros.', 'other'),
    ('Я купил камеру на прошлой неделе.', 'other'),
    ('2023-05-20 14:30', 'und'),
    ('def add(x, y):\n    return x + y  # simple helper used by the parser module', 'en'),
    ('* Fill in the table with tasks\n* Block time slots\n* Review weekly priorities and adjust', 'en'),
])
def test_detect(text, tag):
    assert detect(text) == tag


def test_english_compatible():
    assert english_compatible('en') and english_compatible('und')
    assert not any(english_compatible(t) for t in ('zh', 'ja', 'ko', 'other'))


def fake(name, multilingual, quality, ms, calls, fail=False):
    def scorer(q, docs):
        calls.append(name)
        if fail:
            raise ConnectionError('down')
        return [float(len(d)) for d in docs]          # longer doc = higher score
    return Reranker(name, scorer, multilingual=multilingual, quality=quality, ms_per_doc=ms)


def make(calls, **fail):
    return Router([fake('qwen', True, 0.80, 85, calls, fail.get('qwen', False)),
                   fake('gte', False, 0.80, 18, calls, fail.get('gte', False)),
                   fake('colbert', False, 0.78, 12, calls, fail.get('colbert', False))])


def test_english_path_picks_by_budget():
    docs = ['a' * i for i in range(1, 17)]                       # 16 docs
    # fresh routers: a real call updates the latency estimate (fakes are ~0 ms), which would change later picks
    _, info = make([]).rerank('q', docs, 'en', ['en'] * 16, None)        # best: no budget -> larger model on ties
    assert info['route'] == 'en' and info['rerank_model'] == 'qwen'
    _, info = make([]).rerank('q', docs, 'en', ['en'] * 16, PROFILES['balanced'])
    assert info['rerank_model'] == 'gte' and info['rerank_depth'] == 16     # tie with qwen; gte sees all 16
    order, info = make([]).rerank('q', docs, 'en', ['en'] * 16, PROFILES['fast'])
    assert info['rerank_model'] == 'gte' and info['rerank_depth'] == 13     # 250 ms / 18 ms: top-13 only
    assert order[13:] == [13, 14, 15]                                       # the rest keep first-stage order
    _, info = make([]).rerank('q', docs, 'en', ['en'] * 16, 120.0)
    assert info['rerank_model'] == 'colbert' and info['rerank_depth'] == 10 # gte fits only 6 < 8; colbert fits 10


def test_chinese_shrinks_depth_instead_of_skipping():
    docs = ['文档' * i for i in range(1, 17)]
    order, info = make([]).rerank('问题', docs, 'zh', ['zh'] * 16, PROFILES['balanced'])
    assert info['route'] == 'multi' and info['rerank_model'] == 'qwen' and info['rerank_depth'] == 11
    assert len(order) == 16


def test_non_english_never_uses_english_only_models():
    calls = []
    r = make(calls)
    docs = ['文档一', '文档二二', 'doc three']
    order, info = r.rerank('我的狗叫什么？', docs, 'zh', ['zh', 'zh', 'en'], 100.0)
    assert info['route'] == 'multi' and order is None and info['fallback']   # qwen cannot rerank even 3 in 100 ms
    assert calls == []                                                        # gte/colbert never called
    order, info = r.rerank('my dog?', docs, 'en', ['zh', 'zh', 'en'], None)   # English query, mixed pool
    assert info['route'] == 'multi' and info['rerank_model'] == 'qwen' and order == [2, 1, 0]


def test_failure_falls_through_then_back_to_rrf():
    calls = []
    r = make(calls, qwen=True)
    _, info = r.rerank('q', ['x', 'yy'], 'en', ['en', 'en'], None)
    assert info['rerank_model'] == 'gte' and info['rerank_errors']            # qwen failed, gte took over
    order, info = r.rerank('问题', ['x', 'yy'], 'zh', ['zh', 'zh'], None)       # qwen in cooldown, no multilingual left
    assert order is None and info['route'] == 'multi'


def test_latency_estimate_adapts():
    calls = []
    rr = fake('gte', False, 0.8, 1000.0, calls)                  # absurd prior
    rr.score('q', ['a', 'b'])
    assert rr.ms_per_doc < 1000.0


def test_recaller_routed_mode_end_to_end():
    s = Store()
    s.add_episode(bank='b', scope='s', text='I bought a used camera in Hangzhou for 3000 yuan last week.', session='1')
    s.add_episode(bank='b', scope='s', text='我上周在杭州买了一台二手相机。', session='2')
    s.add_episode(bank='b', scope='s', text='The weather was nice, we went hiking in the hills.', session='3')
    calls = []
    rec = Recaller(s, router=make(calls), profile='best')
    res = rec.recall('b', 's', 'camera Hangzhou')
    d = res.diagnostics
    assert res.mode == 'routed' and d['query_lang'] == 'en'
    langs = {it['text'][:2] for it in res.items}
    assert d['route'] == ('multi' if '我上' in langs else 'en')
    assert d['rerank_model'] in ('qwen', 'gte', 'colbert')


def test_best_profile_waits_for_cooldown_instead_of_skipping():
    import time as _t
    from pm1.rerank import Reranker, Router
    calls = []
    def ok(q, docs):
        calls.append(len(docs)); return [float(-i) for i in range(len(docs))]
    r = Reranker('m', ok, multilingual=True, quality=0.5, ms_per_doc=1.0, cooldown_s=0.3)
    r._down_until = _t.time() + 0.2
    order, info = Router([r]).rerank('q', ['a', 'b', 'c'], 'zh', ['zh'] * 3, None)
    assert order is not None and info['rerank_model'] == 'm' and info.get('waited_ms', 0) > 0
    r._down_until = _t.time() + 0.2      # with a latency budget the down reranker is still skipped
    order, info = Router([r]).rerank('q', ['a', 'b', 'c'], 'zh', ['zh'] * 3, 1000.0)
    assert order is None
