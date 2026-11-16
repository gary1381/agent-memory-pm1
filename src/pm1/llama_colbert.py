"""PyLate-style ColBERT on a llama.cpp GGUF server (`--embedding --pooling none`).

GGUF conversions keep only the BERT encoder. What PyLate adds is redone here, from the original repo files:
  * tokenisation with the model's tokenizer.json; prefix token "[Q] " / "[D] " right after [CLS]
  * queries padded with [MASK] to query_length (query augmentation), documents cut to document_length
  * the 1_Dense projection (hidden -> 128, no bias), then per-token L2 normalisation
  * punctuation (skiplist) tokens dropped from documents
Token ids are sent to llama-server directly, so its own tokenizer is never involved.
"""
from __future__ import annotations

import json
import struct
import urllib.request
from pathlib import Path

import numpy as np

try:  # big float arrays: orjson parses them several times faster
    from orjson import loads as _loads
except ImportError:  # pragma: no cover
    _loads = json.loads

SKIP = set('!"#$%&\'()*+,-./:;<=>?@[\\]^_`{|}~')


def load_safetensors(path: str | Path) -> dict[str, np.ndarray]:
    raw = Path(path).read_bytes()
    n = struct.unpack('<Q', raw[:8])[0]
    header = json.loads(raw[8:8 + n])
    dt = {'F32': np.float32, 'F16': np.float16, 'BF16': None}
    out = {}
    for k, v in header.items():
        if k == '__metadata__':
            continue
        a, b = v['data_offsets']
        out[k] = np.frombuffer(raw[8 + n + a:8 + n + b], dtype=dt[v['dtype']]).reshape(v['shape']).astype(np.float32)
    return out


class LlamaColbert:
    def __init__(self, url: str, model_dir: str | Path, name: str = 'llama-colbert',
                 query_length: int = 32, document_length: int = 300, expand_query: bool = True,
                 batch: int = 8, timeout: float = 120, workers: int = 4):
        from tokenizers import Tokenizer
        d = Path(model_dir).expanduser()
        self.tok = Tokenizer.from_file(str(d / 'tokenizer.json'))
        self.tok.no_truncation()
        self.tok.no_padding()
        w = load_safetensors(d / 'dense.safetensors')
        self.W = next(v for k, v in w.items() if k.endswith('weight'))       # (128, hidden)
        v = self.tok.get_vocab()
        self.cls, self.sep, self.mask = v['[CLS]'], v['[SEP]'], v['[MASK]']
        self.qp, self.dp = v['[Q] '], v['[D] ']
        self.skip_ids = {i for t, i in v.items() if t in SKIP}
        self.url, self.name, self.timeout, self.batch = url.rstrip('/'), name, timeout, batch
        self.workers = workers
        self.qlen, self.dlen, self.expand = query_length, document_length, expand_query

    def _ids(self, text: str, prefix: int, max_len: int) -> list[int]:
        body = self.tok.encode(text, add_special_tokens=False).ids[:max_len - 3]
        return [self.cls, prefix] + body + [self.sep]

    def _post(self, chunk: list[list[int]]) -> list[np.ndarray]:
        req = urllib.request.Request(self.url + '/embedding', data=json.dumps({'content': chunk}).encode(),
                                     headers={'Content-Type': 'application/json'})
        with urllib.request.urlopen(req, timeout=self.timeout) as r:
            res = _loads(r.read())
        return [np.asarray(x['embedding'], dtype=np.float32) for x in sorted(res, key=lambda x: x['index'])]

    def _hidden(self, seqs: list[list[int]]) -> list[np.ndarray]:
        """Batches go out concurrently (the server has several slots; JSON is the bottleneck, not the GPU)."""
        from concurrent.futures import ThreadPoolExecutor
        chunks = [seqs[i:i + self.batch] for i in range(0, len(seqs), self.batch)]
        if len(chunks) == 1:
            return self._post(chunks[0])
        with ThreadPoolExecutor(self.workers) as ex:
            return [h for part in ex.map(self._post, chunks) for h in part]

    def _project(self, h: np.ndarray) -> np.ndarray:
        m = h @ self.W.T
        return m / np.linalg.norm(m, axis=-1, keepdims=True).clip(1e-9)

    def embed(self, texts, batch_size=None):
        seqs = [self._ids(t, self.dp, self.dlen) for t in texts]
        out = []
        for ids, h in zip(seqs, self._hidden(seqs)):
            assert h.shape[0] == len(ids), f'server returned {h.shape[0]} vectors for {len(ids)} tokens'
            keep = [j for j, t in enumerate(ids) if t not in self.skip_ids]
            out.append(self._project(h[keep]))
        return iter(out)

    def query_embed(self, texts):
        seqs = []
        for t in texts:
            ids = self._ids(t, self.qp, self.qlen)
            if self.expand:
                ids += [self.mask] * (self.qlen - len(ids))
            seqs.append(ids)
        return iter([self._project(h) for h in self._hidden(seqs)])
