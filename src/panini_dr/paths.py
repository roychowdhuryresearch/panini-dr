"""Where PANINI-DR looks for data when a path is not given explicitly.

Every location is resolved at call time from environment variables, so a shell
export, a test monkeypatch or a launcher script all work in any import order.
Nothing here points at a particular machine.

``PANINI_DATA_DIR``
    Root for prepared datasets and caches. Defaults to ``./data`` (relative to
    the working directory). The benchmark preparation scripts under
    ``datasets/`` write into it.
"""

from __future__ import annotations

import os
from pathlib import Path


def data_dir() -> Path:
    """Root data directory (``$PANINI_DATA_DIR``, default ``./data``)."""
    return Path(os.environ.get("PANINI_DATA_DIR") or "data").expanduser()


def dense_cache_root() -> Path:
    """Parent of the fingerprinted dense-index directories."""
    return data_dir() / "retrieval_cache"


__all__ = ["data_dir", "dense_cache_root"]
