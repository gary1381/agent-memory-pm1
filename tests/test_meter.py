"""Usage meter: call ids + stored results, pm1_feedback, quality floor, audit prepare/ingest, dashboard, worker."""
from __future__ import annotations

import datetime as dt
import json

from pm1 import Recaller, Store
from pm1 import meter
from pm1.mcp import Service, handle
from pm1.sessions import SessionSearch
from pm1.worker import deadline, run_once


def svc(client='codex'):
    s = Store()
    r = Recaller(s)
    return s, Service(s, r, bank='personal', default_scope='p', sessions=SessionSearch(s, r, 'personal'),
                      client=client)


def call(service, name, args):
    r = handle(service, {'jsonrpc': '2.0', 'id': 1, 'method': 'tools/call',
                         'params': {'name': name, 'arguments': args}})['result']
    return r['isError'], (r['content'][0]['text'] if r['isError'] else json.loads(r['content'][0]['text']))


def seed(s: Store):
    s.add_episode(bank='personal', scope='proj:/w/app', session='codex:1', role='user',
                  text='We decided to deploy with podman instead of docker on the build host.')
    s.add_episode(bank='personal', scope='proj:/w/app', session='codex:1', role='assistant',
                  text='Lunch options near the office: noodles or tacos. token=abcd1234efgh5678 here')


def test_results_carry_call_id_and_are_stored():
    s, v = svc()
    seed(s)
    err, out = call(v, 'search_sessions', {'query': 'podman docker deploy'})
    assert not err and isinstance(out['call_id'], int)
    row = s.con.execute('SELECT result FROM mcp_results WHERE call_id=?', (out['call_id'],)).fetchone()
    assert 'podman' in row[0]


def test_feedback_tool_records_and_validates():
    s, v = svc()
    seed(s)
    _, out = call(v, 'search_sessions', {'query': 'podman'})
    err, fb = call(v, 'pm1_feedback', {'call_id': out['call_id'], 'verdict': 'wrong', 'item_ids': ['1'],
                                       'note': 'outdated'})
    assert not err and fb['recorded']
    r = s.con.execute('SELECT client, verdict, item_ids FROM mcp_feedback').fetchone()
    assert tuple(r) == ('codex', 'wrong', '["1"]')
    assert call(v, 'pm1_feedback', {'verdict': 'great'})[0]
    assert call(v, 'pm1_feedback', {'verdict': 'helpful', 'call_id': 99999})[0]
    tools = {t['name']: t for t in handle(v, {'jsonrpc': '2.0', 'id': 2, 'method': 'tools/list'})['result']['tools']}
    assert tools['pm1_feedback']['annotations']['destructiveHint'] is False
    assert tools['search_sessions']['annotations']['readOnlyHint'] is True


def test_quality_floor(monkeypatch):
    _, v = svc()
    monkeypatch.delenv('PM1_QUALITY_FLOOR', raising=False)
    assert v._budget('fast') == 250.0 and v._budget(None) == -1.0
    monkeypatch.setenv('PM1_QUALITY_FLOOR', 'wait')
    assert v._budget('fast') == 8000.0 and v._budget('balanced') == 8000.0
    assert v._budget('best') is None and v._budget(None) == -1.0


def test_audit_prepare_redacts_and_ingest_takes_majority(tmp_path, monkeypatch):
    s, v = svc()
    seed(s)
    monkeypatch.setattr(meter, 'LABEL_DIR', tmp_path)
    _, out = call(v, 'search_sessions', {'query': 'podman docker lunch token'})
    call(svc('test')[1], 'search_sessions', {'query': 'x'})          # other store, ignored anyway
    day = s.con.execute('SELECT substr(ts,1,10) FROM mcp_calls WHERE id=?', (out['call_id'],)).fetchone()[0]
    task, n = meter.audit_prepare(s.con, day)
    assert n == 1
    line = json.loads((tmp_path / 'inbox' / f'{task}.jsonl').read_text())
    assert '[redacted]' in line['prompt'] and 'abcd1234efgh5678' not in line['prompt']
    ids = line['item_ids']
    (tmp_path / 'outbox' / 'composer').mkdir(parents=True)
    (tmp_path / 'outbox' / 'grok').mkdir(parents=True)
    (tmp_path / 'outbox' / 'gpt').mkdir(parents=True)

    def write(who, labels):
        (tmp_path / 'outbox' / who / f'{task}.out.jsonl').write_text(json.dumps(
            {'id': line['id'], 'response': {'labels': dict(zip(ids, labels)), 'confidence': 0.9}}) + '\n')
    write('composer', ['answers', 'irrelevant'])
    write('grok', ['answers', 'related'])
    write('gpt', ['answers', 'irrelevant'])       # GPT only ran on the disagreement; it breaks the tie
    res = meter.audit_ingest(s.con, task)
    assert res['items_labelled'] == 2
    rows = s.con.execute('SELECT item, label, agreed FROM mcp_audit ORDER BY rowid').fetchall()
    assert [tuple(r) for r in rows] == [(ids[0], 'answers', 1), (ids[1], 'irrelevant', 0)]
    assert meter.audit_prepare(s.con, day)[1] == 0                 # already audited


def test_summary_and_dashboard_exclude_eval_clients():
    s, v = svc('codex')
    seed(s)
    call(v, 'search_sessions', {'query': 'podman'})
    _, e = svc('codex-eval')
    e.sessions, e.store = v.sessions, s
    call(e, 'search_sessions', {'query': 'podman'})
    sm = meter.summary(s.con, 7)
    assert sm['real_calls'] == 1 and sm['by_client'] == {'codex': 1}
    page = meter.dashboard(s.con, 7)
    assert '<title>pm1 Usage</title>' in page and 'podman' in page


def test_worker_deadline_and_newest_first():
    now = dt.datetime(2026, 10, 18, 23, 0)
    assert deadline('07:00', now) == dt.datetime(2026, 10, 19, 7, 0)
    assert deadline('23:30', now) == dt.datetime(2026, 10, 18, 23, 30)

    class Emb:
        name = 'fake'
        seen: list = []

        def embed(self, texts, kind):
            self.seen.append(texts)
            return [[1.0, 0.0]] * len(texts)
    s = Store(index_models=('dense:fake',))
    for i in range(5):
        s.add_episode(bank='b', scope='p', session='s', role='user', text=f'message number {i} ' * 3)
    run_once(s, Emb(), None, batch=2, newest_first=True)
    assert 'number 4' in Emb.seen[0][0]
