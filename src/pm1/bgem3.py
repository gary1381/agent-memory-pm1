"""BGE-M3 multi-vector (ColBERT) + dense + sparse from a llama.cpp GGUF server.

llama.cpp only knows BGE-M3's encoder. Started with `--embedding --pooling none` it returns the final hidden
state of every token; the two small heads BGE-M3 adds on top are applied here, exactly as FlagEmbedding does:

  dense   = normalize(h[CLS])
  colbert = normalize(colbert_linear(h[1:]))          (1024 -> 1024, drops the CLS position)
  sparse  = relu(sparse_linear(h)) max-pooled per token id (special tokens dropped)

The heads ship as PyTorch pickles (colbert_linear.pt, sparse_linear.pt); `load_torch_linear` reads them
without importing torch.
"""
from __future__ import annotations

import json
import pickle
import urllib.request
import zipfile
from pathlib import Path

import numpy as np

try:  # big float arrays: orjson parses them several times faster
    from orjson import loads as _loads
except ImportError:  # pragma: no cover
    _loads = json.loads

_DTYPES = {'HalfStorage': np.float16, 'FloatStorage': np.float32, 'BFloat16Storage': None}


def load_torch_linear(path: str | Path) -> dict[str, np.ndarray]:
    """Minimal reader for a torch.save'd state dict of plain tensors (no torch import)."""
    zf = zipfile.ZipFile(path)
    root = zf.namelist()[0].split('/')[0]

    class _Storage:
        def __init__(self, dtype, key):
            self.dtype, self.key = dtype, key

    def _rebuild(storage, offset, size, stride, *args):
        raw = np.frombuffer(zf.read(f'{root}/data/{storage.key}'), dtype=storage.dtype)
        n = int(np.prod(size)) if size else 1
        return raw[offset:offset + n].reshape(size).astype(np.float32)

    class _U(pickle.Unpickler):
        def find_class(self, module, name):
            if name == '_rebuild_tensor_v2':
                return _rebuild
            if name.endswith('Storage'):
                return name
            if module == 'collections' and name == 'OrderedDict':
                from collections import OrderedDict
                return OrderedDict
            raise pickle.UnpicklingError(f'unexpected {module}.{name}')

        def persistent_load(self, pid):
            _, storage_type, key, _loc, _n = pid
            dtype = _DTYPES.get(storage_type if isinstance(storage_type, str) else storage_type.__name__)
            if dtype is None:
                raise ValueError(f'unsupported storage {storage_type}')
            return _Storage(dtype, key)

    return dict(_U(zf.open(f'{root}/data.pkl')).load())


def _norm(m):
    return m / np.linalg.norm(m, axis=-1, keepdims=True).clip(1e-9)


class BgeM3:
    """`/embedding` of a llama-server running bge-m3 with --pooling none."""
    name = 'bge-m3-q8'

    def __init__(self, url: str = 'http://127.0.0.1:18180', heads: str | Path = '~/models/bge-m3',
                 batch: int = 8, timeout: float = 120):
        heads = Path(heads).expanduser()
        c = load_torch_linear(heads / 'colbert_linear.pt')
        s = load_torch_linear(heads / 'sparse_linear.pt')
        self.cw, self.cb = c['weight'], c['bias']
        self.sw, self.sb = s['weight'], s['bias']
        self.url, self.batch, self.timeout = url.rstrip('/'), batch, timeout

    def _hidden(self, texts):
        out = []
        for i in range(0, len(texts), self.batch):
            req = urllib.request.Request(self.url + '/embedding',
                                         data=json.dumps({'content': list(texts[i:i + self.batch])}).encode(),
                                         headers={'Content-Type': 'application/json'})
            with urllib.request.urlopen(req, timeout=self.timeout) as r:
                res = _loads(r.read())
            out += [np.asarray(d['embedding'], dtype=np.float32) for d in sorted(res, key=lambda d: d['index'])]
        return out

    def encode(self, texts, kind=None):
        """ColBERT matrices (n_tokens-1, 1024), L2-normalised per token."""
        return [_norm(h[1:] @ self.cw.T + self.cb) for h in self._hidden(texts)]

    def encode_all(self, texts):
        """(dense vectors, colbert matrices) from one forward pass."""
        hs = self._hidden(texts)
        return (_norm(np.stack([h[0] for h in hs])), [_norm(h[1:] @ self.cw.T + self.cb) for h in hs])
