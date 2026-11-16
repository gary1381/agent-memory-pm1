"""PM1_SESS_XLATE: an English query also searches the keyword index with its Chinese translation (pm1.xlate)."""
from __future__ import annotations

import json
import threading
from http.server import BaseHTTPRequestHandler, ThreadingHTTPServer

from pm1 import Recaller, Store
from pm1.sessions import SessionSearch
from pm1.xlate import Xlate, mostly_english


class FakeMT(BaseHTTPRequestHandler):
    def do_POST(self):
        text = json.loads(self.rfile.read(int(self.headers['Content-Length'])))['text']
        body = json.dumps({'text': '缓存文件保留多久' if 'cache file' in text else '无关'}).encode()
        self.send_response(200)
        self.send_header('Content-Length', str(len(body)))
        self.end_headers()
        self.wfile.write(body)

    def log_message(self, *a):
        pass


def _search(monkeypatch, url):
    s = Store()
    want = s.add_episode(bank='personal', scope='proj:/w/aglab', session='codex:s1', seq=0, role='assistant',
                         text='缓存文件最多临时保留 24 小时，超期后需要重新下载。')['episode_id']
    for i in range(1, 6):
        s.add_episode(bank='personal', scope='proj:/w/aglab', session='codex:s1', seq=i, role='user',
                      text=f'今天先看部署脚本，第 {i} 步')
    if url:
        monkeypatch.setenv('PM1_SESS_XLATE', url)
    else:
        monkeypatch.delenv('PM1_SESS_XLATE', raising=False)
    out = SessionSearch(s, Recaller(s), 'personal').search('cache file retention', budget_ms=None)
    return want, out


def test_mostly_english():
    assert mostly_english('cache file retention v2')
    assert not mostly_english('缓存文件 retention')
    assert not mostly_english('1234 --')


def test_english_query_finds_chinese_record(monkeypatch):
    srv = ThreadingHTTPServer(('127.0.0.1', 0), FakeMT)
    threading.Thread(target=srv.serve_forever, daemon=True).start()
    try:
        want, out = _search(monkeypatch, f'http://127.0.0.1:{srv.server_port}')
        assert want in [it['episode_id'] for it in out['items']]
        assert out['route']['keywords_also_from'] == '缓存文件保留多久'
        try:                                                      # off: the English words match nothing
            _, off = _search(monkeypatch, '')
            assert want not in [it['episode_id'] for it in off['items']]
        except ValueError:
            pass
    finally:
        srv.shutdown()


def test_direction_switch():
    x = Xlate('http://127.0.0.1:9', timeout=0.2, dirs='en-zh')
    assert x.other('缓存文件 retention') is None and x.other('1234') is None


def test_service_down_is_silent():
    assert Xlate('http://127.0.0.1:9', timeout=0.2).other('cache file') is None


def test_merge_mode_also_finds_it(monkeypatch):
    monkeypatch.setenv('PM1_SESS_XLATE_MODE', 'merge')
    srv = ThreadingHTTPServer(('127.0.0.1', 0), FakeMT)
    threading.Thread(target=srv.serve_forever, daemon=True).start()
    try:
        want, out = _search(monkeypatch, f'http://127.0.0.1:{srv.server_port}')
        assert want in [it['episode_id'] for it in out['items']]
    finally:
        srv.shutdown()


def test_glossary(tmp_path):
    from pm1.xlate import Glossary
    g = tmp_path / 'g.tsv'
    g.write_text('# comment\n重排\treranker|rerank\n重排服务\treranker service\n显存\tVRAM\n', encoding='utf-8')
    gl = Glossary((str(g),))
    text, found = gl.to_en('重排服务 显存 比例')
    assert 'reranker service' in text and 'VRAM' in text and found == ['reranker service', 'VRAM']
    assert gl.to_zh('Reranker VRAM share, re-rank') == ['重排', '显存']        # whole words only: not "re-rank"
    g.write_text('延迟\tlatency\n', encoding='utf-8')
    import os
    os.utime(g, (1, 1))                                                     # a changed file is read again
    assert gl.to_zh('p50 latency') == ['延迟']
