"""vLLM Qwen3 reranker: prompt sent to /score matches the llama.cpp '-memory' template; router prefers it."""
from __future__ import annotations

import json
import threading
from http.server import BaseHTTPRequestHandler, HTTPServer

from pm1.rerank import PROFILES, router_from_env, vllm_qwen_scorer

GGUF_MEMORY_TEMPLATE = (
    '<|im_start|>system\nJudge whether the Document meets the requirements based on the Query and the Instruct '
    'provided. Note that the answer can only be "yes" or "no".<|im_end|>\n<|im_start|>user\n<Instruct>: Given a '
    'question a user asks about their own past conversations with an AI assistant, judge whether this conversation '
    'turn contains information needed to answer it, such as facts or preferences the user stated, events and when '
    'they happened, or what the assistant said\n<Query>: {query}\n<Document>: {document}<|im_end|>\n'
    '<|im_start|>assistant\n<think>\n\n</think>\n\n')


def fake_vllm():
    seen = []

    class H(BaseHTTPRequestHandler):
        def do_POST(self):
            body = json.loads(self.rfile.read(int(self.headers['Content-Length'])))
            seen.append((self.path, body))
            data = [{'index': i, 'score': float(len(d))} for i, d in enumerate(body['text_2'])]
            out = json.dumps({'data': list(reversed(data))}).encode()
            self.send_response(200)
            self.send_header('Content-Type', 'application/json')
            self.send_header('Content-Length', str(len(out)))
            self.end_headers()
            self.wfile.write(out)

        def log_message(self, *a):
            pass
    srv = HTTPServer(('127.0.0.1', 0), H)
    threading.Thread(target=srv.serve_forever, daemon=True).start()
    return srv, seen


def test_prompt_matches_gguf_template_and_scores_keep_order():
    srv, seen = fake_vllm()
    try:
        f = vllm_qwen_scorer(f'http://127.0.0.1:{srv.server_port}')
        s = f('什么时候改的?', ['a', 'bbb'])
        path, body = seen[0]
        assert path == '/score' and body['model'] == 'qwen3-reranker'
        for doc, t2 in zip(['a', 'bbb'], body['text_2']):
            assert body['text_1'] + t2 == GGUF_MEMORY_TEMPLATE.format(query='什么时候改的?', document=doc)
        assert s[1] > s[0]                         # index-aligned despite the reversed response order
    finally:
        srv.shutdown()


def test_router_prefers_vllm_under_every_profile(monkeypatch):
    monkeypatch.setenv('PM1_RERANK_QWEN_VLLM', 'http://127.0.0.1:9')
    monkeypatch.setenv('PM1_RERANK_QWEN', 'http://127.0.0.1:9')
    monkeypatch.setenv('PM1_RERANK_GTE', 'http://127.0.0.1:9')
    r = router_from_env()
    for prof in PROFILES.values():
        for path in ('en', 'multi'):
            plan = r.plan(path, 64, prof)
            assert plan and plan[0][0].name == 'qwen3-reranker-0.6b@vllm', (prof, path)
