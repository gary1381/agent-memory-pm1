"""pm1 core: raw episodes, multilingual BM25, isolation, forgetting, recall pipeline."""
from __future__ import annotations

import numpy as np
import pytest

from pm1 import Recaller, Store
from pm1.text import split_spans, tokenize
from pm1.worker import run_once


class FakeDense:
    name = 'fake-dense'

    def embed(self, texts, kind):
        out = []
        for t in texts:
            t = t.lower()
            v = np.array([('podman' in t) + ('容器' in t), ('postgres' in t) + 0.01, 0.1], dtype=np.float32)
            out.append(v / np.linalg.norm(v))
        return np.vstack(out)


class FakeToken:
    name = 'fake-tok'

    def encode(self, texts, kind):
        return [FakeDense().embed(t.split()[:4] or [t], kind) for t in texts]


def test_tokenize_multilingual_and_spans():
    assert 'podman' in tokenize('Deploy scripts use Podman.')
    zh = tokenize('部署脚本改用容器')
    assert '部署' in zh and '容器' in zh and '署脚' in zh  # jieba words + char bigrams
    text = 'One. ' * 400
    spans = split_spans(text, 600)
    assert spans[0][0] == 0 and spans[-1][1] == len(text) and all(e - s <= 600 for s, e in spans)


def test_raw_episode_dedupe_and_bm25_chinese():
    s = Store()
    a = s.add_episode(bank='b', scope='p', text='定了：部署脚本统一改用 podman，不再用 docker。', session='s1', seq=0)
    again = s.add_episode(bank='b', scope='p', text='定了：部署脚本统一改用 podman，不再用 docker。', session='s1', seq=0)
    assert again['duplicate'] and again['episode_id'] == a['episode_id']
    s.add_episode(bank='b', scope='p', text='We keep PostgreSQL 15 this quarter.', session='s1', seq=1)
    hits = s.bm25('b', 'p', '部署脚本用什么？')
    assert hits and hits[0][0] == a['span_ids'][0]


def test_scope_isolation_in_sql():
    s = Store()
    s.add_episode(bank='b', scope='private', text='secret plan uses podman')
    assert s.bm25('b', 'other', 'podman') == []
    assert s.bm25('x', 'private', 'podman') == []


def test_forget_removes_all_derived_rows():
    s = Store(index_models=('dense:fake-dense',))
    ep = s.add_episode(bank='b', scope='p', text='podman everywhere')
    run_once(s, dense=FakeDense())
    out = s.forget_episode('b', 'p', ep['episode_id'])
    assert out['verified'] and s.bm25('b', 'p', 'podman') == []
    assert s.dense_matrix('b', 'p', 'fake-dense')[0].size == 0


def test_recall_hybrid_returns_evidence_in_budget():
    s = Store(index_models=('dense:fake-dense', 'tok:fake-tok'))
    s.add_episode(bank='b', scope='p', text='Decision: deploy scripts now use podman.', session='s1', seq=0,
                  occurred_at='2026-10-01T10:00:00+00:00')
    s.add_episode(bank='b', scope='p', text='We keep postgres 15.', session='s1', seq=1,
                  occurred_at='2026-10-01T10:01:00+00:00')
    for _ in range(3):
        run_once(s, dense=FakeDense(), token=FakeToken())
    r = Recaller(s, dense=FakeDense(), token=FakeToken())
    res = r.recall('b', 'p', 'what do deploy scripts use, podman?', budget_tokens=200)
    assert res.mode == 'hybrid' and res.items[0]['text'].startswith('Decision')
    assert res.items[0]['trust'] == 'T0-record' and res.diagnostics['tokens'] <= 200
    assert '2026-10-01' in res.context and 'user' in res.context
    lex = r.recall('b', 'p', 'postgres', mode='lexical')
    assert lex.items[0]['text'].startswith('We keep')


def test_split_e5_routes_and_falls_back():
    import numpy as np
    from pm1.embed import SplitE5

    class Fake:
        def __init__(self, tag, fail=False):
            self.tag, self.fail, self.calls = tag, fail, []

        def embed(self, texts, kind):
            self.calls.append(kind)
            if self.fail:
                raise ConnectionRefusedError('down')
            return np.ones((len(texts), 4), dtype=np.float32) / 2

    s = SplitE5('http://x', 'http://y')
    s.doc, s.query = Fake('doc'), Fake('query')
    s.embed(['a'], 'passage'); s.embed(['q'], 'query')
    assert s.doc.calls == ['passage'] and s.query.calls == ['query']
    s.query.fail = True
    assert s.embed(['q'], 'query').shape == (1, 4)
    assert s.fallbacks == {'doc': 0, 'query': 1} and s.doc.calls[-1] == 'query'


def test_caches_see_other_process_writes(tmp_path):
    """The MCP server's caches must pick up vectors and records that the worker / capture service commit."""
    db = tmp_path / 'm.sqlite'
    reader = Store(db, index_models=('dense:fake-dense',))
    writer = Store(db, index_models=('dense:fake-dense',))
    writer.add_episode(bank='b', scope='s', text='Deploy scripts use podman.')
    ids, _ = reader.dense_matrix('b', 's', 'fake-dense')
    assert len(ids) == 0
    v0 = reader._version
    assert reader._version == v0                          # nothing new: stable key, no rebuild
    run_once(writer, FakeDense())                         # another connection fills the vectors
    ids, _ = reader.dense_matrix('b', 's', 'fake-dense')
    assert len(ids) == 1
    writer.add_episode(bank='b', scope='s', text='Always run the tests before pushing.')
    assert len(reader.directives('b', 's')) == 1


def test_directives_gated_without_keep_top():
    s = Store(index_models=('dense:fake-dense',))
    s.add_episode(bank='b', scope='s', text='Always mention postgres when I ask about databases.')
    run_once(s, FakeDense())
    q = 'how do I configure podman containers?'
    assert Recaller(s, dense=FakeDense(), directive_min_sim=0.9)._directives('b', 's', q)       # top-1 kept
    assert not Recaller(s, dense=FakeDense(), directive_min_sim=0.9, directive_keep_top=False)._directives('b', 's', q)


class DownDense(FakeDense):
    calls = 0

    def embed(self, texts, kind):
        if kind == 'query':
            DownDense.calls += 1
            raise OSError('connection refused')
        return super().embed(texts, kind)


def test_recall_survives_embedder_down():
    """Embedding service down: recall and rank fall back to lexical instead of failing, and stop retrying for a while."""
    s = Store(index_models=('dense:fake-dense',))
    s.add_episode(bank='b', scope='s', text='Deploy scripts use podman now.')
    run_once(s, FakeDense())
    d = DownDense()
    d.name = 'fake-dense'
    r = Recaller(s, dense=d)
    res = r.recall('b', 's', 'what do deploy scripts use?')
    assert res.items and 'podman' in res.items[0]['text'] and 'dense_error' in res.diagnostics
    n = DownDense.calls
    order, _, t = r.rank('b', 's', 'deploy podman')
    assert order and 'dense_error' in t and DownDense.calls == n          # cooldown: no new attempt
    assert r.dense is d                                                    # restored after the call


def test_claim_jobs_is_exclusive(tmp_path):
    db = tmp_path / 'm.sqlite'
    a, b = Store(db, index_models=('dense:x',)), Store(db, index_models=('dense:x',))
    for i in range(6):
        a.add_episode(bank='b', scope='s', text=f'record number {i}')
    ja, jb = a.claim_jobs('dense', 'x', 4), b.claim_jobs('dense', 'x', 4)
    assert len(ja) == 4 and len(jb) == 2 and not {j['span_id'] for j in ja} & {j['span_id'] for j in jb}
