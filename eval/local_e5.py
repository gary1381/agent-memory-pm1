"""CPU multilingual-e5-small via fastembed/ONNX (same model as the host's e5 service, no GPU use)."""
from __future__ import annotations

import os

import numpy as np

_MODEL = None
NAME = 'intfloat/multilingual-e5-small'


def model():
    global _MODEL
    if _MODEL is None:
        from fastembed import TextEmbedding
        from fastembed.common.model_description import ModelSource, PoolingType
        try:
            TextEmbedding.add_custom_model(
                model=NAME, pooling=PoolingType.MEAN, normalization=True,
                sources=ModelSource(hf=NAME), dim=384,
                model_file=os.environ.get('PM1_E5_FILE', 'onnx/model_qint8_avx512_vnni.onnx'))
        except ValueError:
            pass  # already registered
        _MODEL = TextEmbedding(NAME, threads=int(os.environ.get('PM1_THREADS', '8')))
    return _MODEL


def embed(texts: list[str], kind: str) -> np.ndarray:
    prefix = 'query: ' if kind == 'query' else 'passage: '
    v = np.asarray(list(model().embed([prefix + t for t in texts], batch_size=32)), dtype=np.float32)
    return v / np.linalg.norm(v, axis=1, keepdims=True).clip(1e-9)
