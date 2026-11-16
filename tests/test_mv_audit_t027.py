"""T027 (2026-10-19): pre-answer audit of eval/mv_arms.py against its pre-registration (s1-2, s6).

Tests only. Each test pins one gap between the runner and the prereg (or a T020 failure mode); no production code is changed.
Where several fixes would be fine (refuse vs. record), the test accepts either.
"""
from __future__ import annotations

import http.server
import json
import re
import sqlite3
import sys
import threading
import types
from collections import Counter
from pathlib import Path

import numpy as np
import pytest

EVAL = Path(__file__).resolve().parents[1] / 'eval'
sys.path.insert(0, str(EVAL))
import mv_arms  # noqa: E402

QID = '1|information_extraction|0'


# ---------- a tiny local /v1/rerank server (llama.cpp response shape) ----------
class _Rerank(http.server.BaseHTTPRequestHandler):
    model, drop_last = 'qwen-q8-mem', False

    def do_POST(self):  # noqa: N802
        body = json.loads(self.rfile.read(int(self.headers['Content-Length'])))
        docs = body['documents']
        keep = docs[:-1] if self.drop_last and len(docs) > 1 else docs
        out = {'model': self.model, 'object': 'list',
               'results': [{'index': i, 'relevance_score': 0.9 - 0.1 * i} for i in range(len(keep))]}
        raw = json.dumps(out).encode()
        self.send_response(200)
        self.send_header('Content-Type', 'application/json')
        self.send_header('Content-Length', str(len(raw)))
        self.end_headers()
        self.wfile.write(raw)

    def log_message(self, *a):
        pass


@pytest.fixture
def rerank_server():
    servers = []

    def start(model='qwen-q8-mem', drop_last=False):
        h = type('H', (_Rerank,), {'model': model, 'drop_last': drop_last})
        s = http.server.HTTPServer(('127.0.0.1', 0), h)
        threading.Thread(target=s.serve_forever, daemon=True).start()
        servers.append(s)
        return f'http://127.0.0.1:{s.server_address[1]}'
    yield start
    for s in servers:
        s.shutdown()


# ---------- prereg s1: "no silent fallback" for the reranker ----------
def test_doc_missing_from_rerank_response_is_counted_as_failure(rerank_server, monkeypatch):
    """llama_scorer fills a document the server did not score with -inf; Qwen.score passes that on as a normal score,
    counts 0 failures, and the doc sorts last among 'scored' docs. Prereg s1: a failed score is counted, never silent."""
    monkeypatch.setattr(mv_arms.time, 'sleep', lambda s: None)
    q = mv_arms.Qwen(rerank_server(drop_last=True))
    sc, _models, fails = q.score('apples?', ['a', 'b', 'c'])
    assert fails == 1 and sc[2] is None, f'scores={sc} failures={fails}'


def test_model_name_per_score_is_the_served_model(rerank_server):
    """Prereg s1: 'every score records its model name'. Qwen.score returns the constant 'qwen3-reranker-0.6b' whatever
    PM1_RERANK_QWEN points at (llama.cpp reports the served model in every response; it is discarded). A GTE server
    on that URL (gte-mb runs on :18165 next to Qwen on :18162) would be logged as Qwen. Refusing it is also fine."""
    q = mv_arms.Qwen(rerank_server(model='gte-mb'))
    try:
        _sc, models, _f = q.score('apples?', ['a', 'b'])
    except Exception:  # noqa: BLE001
        return
    assert models == ['gte-mb', 'gte-mb'], models


# ---------- run_conv harness (fake recaller / stores, real fill/render/TreeIndex) ----------
def _harness(tmp_path, monkeypatch, *, dense_error=None):
    stores = tmp_path / 'stores'
    stores.mkdir()
    con = sqlite3.connect(stores / 'conv1.sqlite')
    con.execute('CREATE TABLE episodes (id INTEGER PRIMARY KEY, session TEXT, seq INT, role TEXT, occurred_at TEXT, text TEXT)')
    con.executemany('INSERT INTO episodes VALUES (?,?,?,?,?,?)',
                    [(i, 's1', i - 1, 'user' if i % 2 else 'assistant', '2024-03-01', f'turn {i} about apples. ' * 4)
                     for i in range(1, 7)])
    con.commit()
    con.close()
    ndir = tmp_path / 'notes'
    ndir.mkdir()
    con = sqlite3.connect(ndir / 'conv1.sqlite')
    con.execute('CREATE TABLE note (path TEXT, content TEXT, evidence TEXT, updated TEXT)')
    con.executemany('INSERT INTO note VALUES (?,?,?,?)',
                    [('/u/apples', 'kind: fact\nUser likes apples.', '["t1","t3"]', '2024-03-01'),
                     ('/a/advice', 'kind: advice_given\nt2: suggested green apples.', '["t2"]', '2024-03-01')])
    con.commit()
    con.close()
    monkeypatch.setattr(mv_arms, 'NOTES_DIR', ndir)

    class Dense:
        name = 'fake'

        def embed(self, texts, kind):
            return np.full((len(texts), 4), 0.5, np.float32)

    class Rec:
        k_first, dense = 50, Dense()

        def rank(self, bank, scope, q, mode):
            t = {'bm25_ms': 0.1}
            if dense_error:            # what pm1.recall.Recaller.rank reports when it ran BM25-only (f4e233f)
                t['dense_error'] = dense_error
            return list(range(1, 7)), {}, t

        def _qvec(self, q):
            return np.full(4, 0.5, np.float32)

    class StoreF:
        def spans(self, bank, scope, ids):
            return {s: {'episode_id': s} for s in ids}

        def close(self):
            pass

    qs = {c: [] for c in mv_arms.CATS}
    qs['information_extraction'] = [{'question': 'What about apples?', 'rubric': ['apples']}]
    bq = types.ModuleType('beam_qa')
    bq._recaller, bq.questions = (lambda conv: (StoreF(), Rec())), (lambda conv: qs)
    bc = types.ModuleType('beam_cfg')
    bc.STORES = stores
    bo, bop = types.ModuleType('beam_official'), types.ModuleType('beam_official.prompts')
    bop.answer_generation_for_rag = 'CONTEXT:\n<context>\n\nQUESTION:\n<question>\n'
    bo.prompts = bop
    for name, mod in (('beam_qa', bq), ('beam_cfg', bc), ('beam_official', bo), ('beam_official.prompts', bop)):
        monkeypatch.setitem(sys.modules, name, mod)

    class FakeQwen:
        def score(self, q, docs):
            return [1.0 - 0.01 * i for i in range(len(docs))], [mv_arms.Qwen.NAME] * len(docs), 0
    out = tmp_path / 'out'
    out.mkdir()
    return FakeQwen(), out


def test_harness_builds_all_arms(tmp_path, monkeypatch):
    """Sanity check of the harness itself (passes today)."""
    qwen, out = _harness(tmp_path, monkeypatch)
    mv_arms.run_conv('1', qwen, 2, out)
    m = json.loads((out / 'meta_conv1.jsonl').read_text().splitlines()[0])
    assert set(m['budgets']['2000']) == set(mv_arms.ARMS)
    assert len((out / 'prompts_conv1_U_2000_s2.jsonl').read_text().splitlines()) == 1


def test_bm25_only_first_stage_is_not_silent(tmp_path, monkeypatch):
    """Prereg s1: first stage is BM25 + dense (k_first=100). Since f4e233f, Recaller.rank runs BM25-only when the
    embedder fails (or failed < 60 s ago) and only says so in the stage dict, which run_conv discards
    (`sp, _, st = rec.rank(...)`; st unused). Such a question must be refused or counted in meta."""
    qwen, out = _harness(tmp_path, monkeypatch, dense_error='ConnectionError: embedder down')
    try:
        mv_arms.run_conv('1', qwen, 2, out)
    except Exception:  # noqa: BLE001
        return
    meta = (out / 'meta_conv1.jsonl').read_text()
    assert 'dense_error' in meta or 'dense_failed' in meta, 'BM25-only first stage left no trace in meta'


def test_export_refuses_a_dense_model_without_vectors(tmp_path, monkeypatch):
    """beam_qa._recaller uses HttpE5 (name 'e5-small-q8') unless PM1_BEAM_DENSE=onnx; the beam1m stores only hold
    'e5-small-onnx' vectors, so dense_matrix() is empty and rank() silently runs BM25-only (no error at all).
    export() checks PM1_BEAM_SET and PM1_RERANK_* but not this."""
    monkeypatch.setattr(mv_arms, 'DATA', tmp_path)
    monkeypatch.setenv('PM1_BEAM_SET', '1m')
    monkeypatch.setenv('PM1_RERANK_QWEN', 'http://127.0.0.1:9')
    monkeypatch.delenv('PM1_RERANK_GTE', raising=False)
    monkeypatch.setenv('PM1_BEAM_DENSE', 'http')
    monkeypatch.setattr(mv_arms, 'Qwen', lambda url: types.SimpleNamespace(stat=Counter()))
    monkeypatch.setattr(mv_arms, 'run_conv', lambda *a, **k: (_ for _ in ()).throw(AssertionError('ran with no dense vectors')))
    with pytest.raises(SystemExit):
        mv_arms.export('t', '1', 2)


# ---------- prereg s1 / s6: logs ----------
def _fake_export_env(tmp_path, monkeypatch):
    monkeypatch.setattr(mv_arms, 'DATA', tmp_path)
    monkeypatch.setenv('PM1_BEAM_SET', '1m')
    monkeypatch.setenv('PM1_RERANK_QWEN', 'http://127.0.0.1:9')
    monkeypatch.delenv('PM1_RERANK_GTE', raising=False)
    monkeypatch.delenv('PM1_BEAM_DENSE', raising=False)
    monkeypatch.setattr(mv_arms, 'Qwen', lambda url: types.SimpleNamespace(stat=Counter()))

    def fake_run_conv(conv, qwen, samples, out, qf=None):
        for a in mv_arms.ARMS:
            for b in mv_arms.BUDGETS:
                for s in range(1, samples + 1):
                    (out / f'prompts_conv{conv}_{a}_{b}_s{s}.jsonl').write_text(json.dumps({'id': f'{conv}|x|0', 'prompt': 'p'}) + '\n')
    monkeypatch.setattr(mv_arms, 'run_conv', fake_run_conv)


def _in_git_checkout() -> bool:
    import subprocess
    try:
        return subprocess.run(['git', 'rev-parse', 'HEAD'], cwd=EVAL, capture_output=True, timeout=10).returncode == 0
    except (OSError, subprocess.SubprocessError):
        return False


@pytest.mark.skipif(not _in_git_checkout(), reason='needs a git checkout with at least one commit (source archive '
                                                    'or an uncommitted tree has no commit id to log)')
def test_export_logs_the_code_commit(tmp_path, monkeypatch):
    """Prereg s1 'code': the commit is written into the run log. env.json holds PM1_* vars, notes_dir and time only;
    T022 asked the operator to write it by hand."""
    _fake_export_env(tmp_path, monkeypatch)
    mv_arms.export('t', '1', 2)
    env = json.loads((tmp_path / 'mv_t' / 'env.json').read_text())
    assert any(re.fullmatch(r'[0-9a-f]{7,40}', str(v)) for k, v in env.items() if 'commit' in k or 'git' in k), sorted(env)


def test_second_export_into_a_run_does_not_drop_earlier_convs(tmp_path, monkeypatch):
    """Prereg s6: each run gets a new directory and logs are not overwritten. A second `export <run> <convs>` (e.g.
    resuming after a crash) reuses data/mv_<run>/, overwrites env.json and rerank_stats.json, and rewrites the merged
    answer-prompt files from THIS call's convs only, so earlier convs silently leave the answer batch. Refusing is fine."""
    _fake_export_env(tmp_path, monkeypatch)
    mv_arms.export('t', '1', 2)
    try:
        mv_arms.export('t', '2', 2)
    except SystemExit:
        return
    ids = {json.loads(l)['id'] for l in (tmp_path / 'beam_answer_prompts_mv_t_base_2000_s1.jsonl').read_text().splitlines()}
    assert ids == {'1|x|0', '2|x|0'}, ids


# ---------- T020 failure mode: duplicate / foreign ids in the judge batch ----------
def test_make_judge_reads_only_its_own_run(tmp_path, monkeypatch):
    """make_judge globs beam_answers_<lab>_mv_<run>_*_<B>_s*.jsonl, so run 'r1' also takes run 'r1_b' (the prereg's
    'new run id after a bug fix'), putting a second run's answers into the blind batch. scores() then files them under
    arm 'b_U' etc. (ignored by report, but the judge batch is no longer the run's own)."""
    data, lab = tmp_path / 'data', tmp_path / 'lab'
    data.mkdir()
    (lab / 'inbox').mkdir(parents=True)
    monkeypatch.setattr(mv_arms, 'DATA', data)
    monkeypatch.setattr(mv_arms, 'LAB', lab)
    monkeypatch.setattr(mv_arms, 'gold_of', lambda qid: ('information_extraction', 'Q?', '- g'))
    for run in ('r1', 'r1_b'):
        for a in mv_arms.ARMS:
            for s in (1, 2):
                (data / f'beam_answers_gpt_mv_{run}_{a}_2000_s{s}.jsonl').write_text(json.dumps({'id': QID, 'response': 'x'}) + '\n')
    mv_arms.make_judge('r1', 2000, 'gpt', 40)
    mp = json.loads((data / 'pm1_mv_r1_judge_2000.map.json').read_text())
    assert len(mp) == 8, sorted({t for t, _ in mp.values()})


def test_base_and_u_only_run_needs_no_notes(tmp_path, monkeypatch):
    """Amendment 3: the extension to the other BEAM-1M conversations builds only base and U (no teacher notes exist there)."""
    monkeypatch.setattr(mv_arms, 'ARMS', ('base', 'U'))
    monkeypatch.setattr(mv_arms, 'NOTES_ON', False)
    qwen, out = _harness(tmp_path, monkeypatch)
    mv_arms.run_conv('1', qwen, 2, out)
    m = json.loads((out / 'meta_conv1.jsonl').read_text().splitlines()[0])
    assert set(m['budgets']['2000']) == {'base', 'U'}
    assert not (out / 'prompts_conv1_N_2000_s1.jsonl').exists()
