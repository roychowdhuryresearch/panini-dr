"""Test stub for ``faiss``: an exact inner-product index in numpy (``IndexFlatIP`` semantics).

Only put on ``PYTHONPATH`` of the subprocesses the replay end-to-end test starts; it lets
the real ``DenseRetriever`` build, save, load and search an index on CPU without faiss.
"""

from __future__ import annotations

from pathlib import Path

import numpy as np

__version__ = "stub"


class IndexFlatIP:
    def __init__(self, dim: int) -> None:
        self.d = int(dim)
        self._x = np.zeros((0, self.d), dtype=np.float32)

    @property
    def ntotal(self) -> int:
        return int(self._x.shape[0])

    def add(self, x) -> None:
        self._x = np.vstack([self._x, np.asarray(x, dtype=np.float32).reshape(-1, self.d)])

    def search(self, q, k: int):
        q = np.asarray(q, dtype=np.float32).reshape(-1, self.d)
        scores = q @ self._x.T
        order = np.argsort(-scores, axis=1, kind="stable")[:, :k]
        return np.take_along_axis(scores, order, axis=1).astype(np.float32), order.astype(np.int64)


def write_index(index: IndexFlatIP, path: str) -> None:
    with Path(path).open("wb") as f:
        np.savez(f, dim=np.array([index.d]), x=index._x)


def read_index(path: str) -> IndexFlatIP:
    payload = np.load(path)
    index = IndexFlatIP(int(payload["dim"][0]))
    index.add(payload["x"])
    return index


def omp_set_num_threads(n: int) -> None:  # noqa: ARG001
    return None
