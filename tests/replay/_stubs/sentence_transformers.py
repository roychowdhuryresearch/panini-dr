"""Test stub for ``sentence_transformers``: a deterministic hashing encoder (CPU, no model files).

Each text becomes a normalised bag of hashed tokens in 64 dimensions, so the real
``DenseRetriever`` ranks chunks by token overlap. Under the cassette in record mode this
class is what the embed hook wraps; in replay the vectors come from the cassette.
"""

from __future__ import annotations

import hashlib
import re

import numpy as np

__version__ = "stub"
DIM = 64
_TOKEN = re.compile(r"[a-z0-9]+")


def _vec(text: str) -> np.ndarray:
    v = np.zeros(DIM, dtype=np.float32)
    for tok in _TOKEN.findall(text.lower()):
        h = int.from_bytes(hashlib.sha256(tok.encode()).digest()[:4], "little")
        v[h % DIM] += 1.0
    n = float(np.linalg.norm(v))
    return v / n if n else v


class SentenceTransformer:
    def __init__(self, model_name_or_path=None, *args, **kwargs) -> None:
        self.model_name = model_name_or_path

    def encode(self, sentences, batch_size: int = 32, normalize_embeddings: bool = False,
               convert_to_numpy: bool = True, show_progress_bar: bool = False, **kwargs):
        single = isinstance(sentences, str)
        rows = [_vec(s) for s in ([sentences] if single else sentences)]
        out = np.stack(rows) if rows else np.zeros((0, DIM), dtype=np.float32)
        return out[0] if single else out
