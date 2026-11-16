"""Embedding providers. All return L2-normalised float32 arrays.

dense : one vector per text (e5-small, local CPU via fastembed or the host HTTP service)
token : one matrix per text (pplx-embed-v2-late, late interaction / MaxSim)
"""
from __future__ import annotations

import json
import os
import urllib.error
import urllib.request
from typing import Protocol, Sequence

import numpy as np


def _norm(m: np.ndarray) -> np.ndarray:
    return m / np.linalg.norm(m, axis=-1, keepdims=True).clip(1e-9)


def _post(url: str, payload: dict, timeout: float) -> dict:
    req = urllib.request.Request(url, data=json.dumps(payload).encode(),
                                 headers={'Content-Type': 'application/json'}, method='POST')
    with urllib.request.urlopen(req, timeout=timeout) as r:
        return json.loads(r.read())


class DenseEmbedder(Protocol):
    name: str
    def embed(self, texts: Sequence[str], kind: str) -> np.ndarray: ...


class TokenEmbedder(Protocol):
    name: str
    def encode(self, texts: Sequence[str], kind: str) -> list[np.ndarray]: ...


class LocalE5:
    """multilingual-e5-small int8 ONNX on CPU (no GPU, no network after first download)."""
    name = 'e5-small'
    HF = 'intfloat/multilingual-e5-small'

    def __init__(self, threads: int | None = None,
                 model_file: str = 'onnx/model_qint8_avx512_vnni.onnx'):
        from fastembed import TextEmbedding
        from fastembed.common.model_description import ModelSource, PoolingType
        try:
            TextEmbedding.add_custom_model(model=self.HF, pooling=PoolingType.MEAN, normalization=True,
                                           sources=ModelSource(hf=self.HF), dim=384, model_file=model_file)
        except ValueError:
            pass
        self._m = TextEmbedding(self.HF, threads=threads or int(os.environ.get('PM1_THREADS', '8')))

    def embed(self, texts, kind):
        prefix = 'query: ' if kind == 'query' else 'passage: '
        return _norm(np.asarray(list(self._m.embed([prefix + t for t in texts], batch_size=32)), dtype=np.float32))


class FastE5:
    """multilingual-e5-small fp32 ONNX through fastembed on the GPU (bulk indexing; ~130 docs/s on a shared 3060).
    Its own vector space ('e5-small-onnx'): queries must use the same class."""
    name = 'e5-small-onnx'
    HF = 'intfloat/multilingual-e5-small'

    def __init__(self, gpu: bool = True, batch: int = 16):
        import onnxruntime
        if gpu and hasattr(onnxruntime, 'preload_dlls'):
            onnxruntime.preload_dlls()
        from fastembed import TextEmbedding
        from fastembed.common.model_description import ModelSource, PoolingType
        try:
            TextEmbedding.add_custom_model(model=self.HF, pooling=PoolingType.MEAN, normalization=True,
                                           sources=ModelSource(hf=self.HF), dim=384, model_file='onnx/model.onnx')
        except ValueError:
            pass
        prov = (['CUDAExecutionProvider'] if gpu else []) + ['CPUExecutionProvider']
        self._m, self.batch = TextEmbedding(self.HF, providers=prov), batch

    def embed(self, texts, kind):
        prefix = 'query: ' if kind == 'query' else 'passage: '
        return _norm(np.asarray(list(self._m.embed([prefix + t for t in texts], batch_size=self.batch)),
                                dtype=np.float32))


class HttpE5:
    """OpenAI-compatible /v1/embeddings (e.g. the host's e5 service). Small batches on purpose."""
    name = 'e5-small'

    def __init__(self, url: str = 'http://127.0.0.1:8767/v1/embeddings',
                 model: str = 'intfloat/multilingual-e5-small', batch: int = 8, timeout: float = 30,
                 name: str | None = None):
        self.url, self.model, self.batch, self.timeout = url, model, batch, timeout
        if name:
            self.name = name

    def _call(self, inputs: list[str]) -> list[list[float]]:
        res = _post(self.url, {'model': self.model, 'input': inputs}, self.timeout)
        return [d['embedding'] for d in sorted(res['data'], key=lambda d: d['index'])]

    def _one(self, text: str) -> list[float]:
        """One text the server refused as too long (llama.cpp: 400 exceed_context with the real token count): split it
        into as many pieces as that count needs and average their vectors, so nothing is cut off."""
        try:
            return self._call([text])[0]
        except urllib.error.HTTPError as exc:
            body = exc.read()
            if exc.code != 400 or b'exceed_context' not in body or len(text) < 64:
                raise
            n = json.loads(body)['error'].get('n_prompt_tokens') or 2 * self.max_tokens
        k = max(2, -(-n // (self.max_tokens - 40)))              # pieces needed, with some slack for the prefix
        step = -(-len(text) // k)
        parts = [text[i:i + step] for i in range(0, len(text), step)]
        parts = [parts[0]] + ['passage: ' + q if text.startswith('passage: ') else q for q in parts[1:]]
        vs = np.asarray([self._one(q) for q in parts], dtype=np.float32)
        return _norm(vs.mean(axis=0, keepdims=True))[0].tolist()

    max_tokens = 510
    window_tokens = 420          # a window is cut well under the 512-token context, by a conservative estimate

    @staticmethod
    def _w(c: str) -> float:
        """Estimated sentencepiece tokens of one character (measured against llama.cpp, 2026-10-21: Chinese text with
        spaces and full-width marks runs about 0.9-1.0 token per character; prose Latin one per 3-4 characters but code, paths
        and numbers one per 1.5-2, so this is deliberately high and `_one` splits exactly when the server still refuses)."""
        if '\u3000' <= c <= '\u9fff' or '\uac00' <= c <= '\ud7af':
            return 1.3
        if ord(c) > 127:
            return 1.0
        return 0.25 if c.isspace() else 0.5

    @classmethod
    def est_tokens(cls, text: str) -> float:
        return sum(cls._w(c) for c in text)

    @classmethod
    def windows(cls, text: str) -> list[str]:
        """The text as pieces that each fit the context, cut at line / sentence / space boundaries where possible.
        A short text is returned as is. A long one is embedded piece by piece and the vectors averaged, so no tail is
        lost and no batch is ever refused for one long input."""
        if cls.est_tokens(text) <= cls.window_tokens:
            return [text]
        out, start, n = [], 0, len(text)
        while start < n:
            budget, end = float(cls.window_tokens), start
            while end < n and budget > 0:
                budget -= cls._w(text[end])
                end += 1
            if end < n:                                   # back up to a natural boundary within the last fifth
                lo = start + (end - start) * 4 // 5
                cut = max(text.rfind(b, lo, end) for b in ('\n', '。', '. ', '！', '？', '; ', ' '))
                end = cut + 1 if cut > lo else end
            out.append(text[start:end])
            start = end
        return out

    def _chunk(self, chunk: list[str]) -> list[list[float]]:
        """One request for the batch. If llama.cpp refuses it because one input is over its context (it answers 500 for a
        batch, 400 exceed_context for a single input), halve the batch until the culprit stands alone and let `_one`
        split it: a few extra requests instead of one request per item."""
        try:
            return self._call(chunk)
        except urllib.error.HTTPError as exc:
            if exc.code not in (400, 500):
                raise
            if len(chunk) == 1:
                return [self._one(chunk[0])]
            mid = len(chunk) // 2
            return self._chunk(chunk[:mid]) + self._chunk(chunk[mid:])

    def embed(self, texts, kind):
        prefix = 'query: ' if kind == 'query' else 'passage: '
        pieces, owner = [], []
        for i, t in enumerate(texts):
            for w in self.windows(t):
                pieces.append(prefix + w)
                owner.append(i)
        out = []
        for i in range(0, len(pieces), self.batch):
            out += self._chunk(pieces[i:i + self.batch])
        vecs = _norm(np.asarray(out, dtype=np.float32))
        if len(pieces) == len(texts):
            return vecs
        merged = np.zeros((len(texts), vecs.shape[1]), dtype=np.float32)
        np.add.at(merged, np.asarray(owner), vecs)         # mean of a text's windows, renormalised
        return _norm(merged)


class HttpDense(HttpE5):
    """Any /v1/embeddings model with its own prefixes (BGE-M3 and Qwen3-Embedding-0.6B, 1024 dimensions)."""

    def __init__(self, url: str, name: str, query_prefix: str = '', doc_prefix: str = '', batch: int = 16,
                 timeout: float = 60):
        super().__init__(url, model=name, batch=batch, timeout=timeout, name=name)
        self.query_prefix, self.doc_prefix = query_prefix, doc_prefix

    def embed(self, texts, kind):
        prefix = self.query_prefix if kind == 'query' else self.doc_prefix
        pieces, owner = [], []
        for i, t in enumerate(texts):
            for w in self.windows(t):
                pieces.append(prefix + w)
                owner.append(i)
        out = []
        for i in range(0, len(pieces), self.batch):
            out += self._chunk(pieces[i:i + self.batch])
        vecs = _norm(np.asarray(out, dtype=np.float32))
        merged = np.zeros((len(texts), vecs.shape[1]), dtype=np.float32)
        np.add.at(merged, np.asarray(owner), vecs)
        return _norm(merged)


E5_Q8_REMOTE = 'http://127.0.0.1:18171/v1/embeddings'   # shared GPU e5 for bulk documents; set PM1_E5_DOC_URL
E5_Q8_LOCAL = 'http://127.0.0.1:18170/v1/embeddings'


class SplitE5:
    """One vector space, two placements: documents go to a shared GPU service in bulk (background,
    throughput matters), queries go to a local CPU server running the *same* GGUF (online, latency matters).

    Measured 2026-10-12: local-CPU vs remote-GPU cosine 0.99996, top-10 overlap 1.000. Each side falls back
    to the other on failure, so the space never mixes with a different quantisation (int8 ONNX top-10
    overlap is only ~0.81, hence the distinct cache name).
    """
    name = 'e5-small-q8'

    def __init__(self, doc_url: str | None = None, query_url: str | None = None):
        # the nightly backfill points PM1_E5_DOC_URL at the local server (same GGUF) when the remote GPU host is unavailable;
        # batch 64 there runs ~58 spans/s vs ~3/s through the batch-4 query fallback (2026-10-18)
        self.doc = HttpE5(doc_url or os.environ.get('PM1_E5_DOC_URL', E5_Q8_REMOTE),
                          batch=int(os.environ.get('PM1_E5_DOC_BATCH', 16)), timeout=60)
        self.query = HttpE5(query_url or os.environ.get('PM1_E5_QUERY_URL', E5_Q8_LOCAL), batch=4, timeout=60)
        self.fallbacks = {'doc': 0, 'query': 0}

    def embed(self, texts, kind):
        first, second = (self.query, self.doc) if kind == 'query' else (self.doc, self.query)
        try:
            return first.embed(texts, kind)
        except (OSError, urllib.error.URLError):
            self.fallbacks['query' if kind == 'query' else 'doc'] += 1
            return second.embed(texts, kind)


class PplxLate:
    """pplx-embed-v2-late-0.6b token vectors via the host service's /v1/encode."""
    name = 'pplx-0.6b'

    def __init__(self, url: str = 'http://127.0.0.1:8770/v1/encode', batch: int = 8, timeout: float = 60):
        self.url, self.batch, self.timeout = url, batch, timeout

    def encode(self, texts, kind):
        out = []
        for i in range(0, len(texts), self.batch):
            res = _post(self.url, {'texts': list(texts[i:i + self.batch]), 'kind': kind}, self.timeout)
            out += [_norm(np.asarray(d['embedding'], dtype=np.float32))
                    for d in sorted(res['data'], key=lambda d: d['index'])]
        return out


def maxsim(q: np.ndarray, d: np.ndarray) -> float:
    return float((q @ d.T).max(axis=1).sum())
