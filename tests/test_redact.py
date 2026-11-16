"""Secrets are cut on every write and from every MCP result; scrub cleans records stored before that."""
from __future__ import annotations

from pm1 import Recaller, Store
from pm1.mcp import Service
from pm1.redact import MARK, redact, scrub

KEY = 'sk-' + 'A1b2C3d4E5f6G7h8I9j0K1l2'


def test_patterns_and_false_positives():
    assert redact(f'export OPENAI_API_KEY={KEY}') == f'export OPENAI_API_KEY={MARK}'
    assert redact('APCA_API_SECRET_KEY=abcdEFGH12345678xyz') == f'APCA_API_SECRET_KEY={MARK}'
    assert redact('Authorization: Bearer abcdefghijklmnop.qrstuv') == f'Authorization: Bearer {MARK}'
    assert redact('"password": "hunter2hunter2"') == f'"password": "{MARK}"'
    assert MARK in redact('-----BEGIN RSA PRIVATE KEY-----\nMIIE\n-----END RSA PRIVATE KEY-----')
    for keep in ('disk-usage-monitor-service-x', '"input_tokens": 123456789', 'max_tokens=2048', 'the secret is out',
                 'token = self.tokenizer(text2048)', "password=os.environ.get('PW1')", 'api_key=${OPENAI_API_KEY:-}',
                 'token: Optional[str]', 'secret_key=settings.SECRET_KEY2'):
        assert redact(keep) == keep
    assert redact(redact(f'token={KEY}')) == redact(f'token={KEY}')        # idempotent


def test_store_and_mcp_never_hold_or_return_secrets():
    s = Store()
    s.add_episode(bank='b', scope='p', text=f'the key is {KEY} for staging', role='tool', source='x#1')
    assert KEY not in s.con.execute('SELECT group_concat(text) FROM episodes').fetchone()[0]
    svc = Service(s, Recaller(s), bank='b', default_scope='p')
    s.con.execute("UPDATE episodes SET text=? WHERE id=1", (f'legacy row {KEY} staging',))   # written before redaction
    assert KEY not in str(svc.call('memory_recall', {'query': 'legacy staging'}))


def test_scrub_rewrites_and_reindexes():
    s = Store(index_models=('dense:fake',))
    eid = s.add_episode(bank='b', scope='p', text='placeholder text for staging deploy', source='x#1')['episode_id']
    s.con.execute('UPDATE episodes SET text=? WHERE id=?', (f'deploy staging with {KEY} then restart', eid))
    assert scrub(s, dry_run=True)['episodes'] == 1
    assert scrub(s)['episodes'] == 1 and scrub(s)['episodes'] == 0
    text = s.con.execute('SELECT text FROM episodes WHERE id=?', (eid,)).fetchone()[0]
    assert KEY not in text and MARK in text
    hits = s.bm25('b', 'p', 'restart')
    assert hits and s.spans('b', 'p', [hits[0][0]])[hits[0][0]]['text'].endswith('restart')
    assert s.con.execute("SELECT count(*) FROM jobs WHERE state='pending'").fetchone()[0] == len(s.span_ids(eid))
