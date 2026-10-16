"""Reading ``.env``: the one loader every entry point uses.

``panini-dr`` (``run``, ``prepare``, ``score``, ``verify``) and the scripts under
``datasets/`` call :func:`load_env_file` before they read any variable. The file
is the first ``.env`` found from the working directory upwards. A variable that
is already in the environment wins, so a shell export always beats the file
(``panini-dr verify`` relies on this: it starts its replays with the
behaviour-changing variables set to empty, and a ``.env`` cannot fill them in).

An empty value in the file (``HF_HOME=``, as a line copied from ``.env.example``
leaves it) is ignored: it never sets a variable. For some tools an empty
variable is not the same as an unset one (an empty ``HF_HOME`` moves the
Hugging Face cache to ``./hub``).
"""

from __future__ import annotations

import os
from pathlib import Path

ENV_FILE_NAME = ".env"


def find_env_file(start: str | Path | None = None) -> Path | None:
    """The first ``.env`` in ``start`` (default: the working directory) or a parent."""
    here = Path(start or os.getcwd()).resolve()
    for d in (here, *here.parents):
        p = d / ENV_FILE_NAME
        if p.is_file():
            return p
    return None


def load_env_file(path: str | Path | None = None) -> Path | None:
    """Load ``path`` (default: :func:`find_env_file`) into ``os.environ``.

    Variables already set in the environment are left alone, and empty values
    are skipped. Returns the file that was read, or None when there is none.
    """
    try:
        from dotenv import dotenv_values
    except ImportError:  # pragma: no cover - python-dotenv is a dependency
        return None
    p = Path(path) if path else find_env_file()
    if p is None or not p.is_file():
        return None
    for key, value in dotenv_values(p).items():
        if key in os.environ or value is None or not value.strip():
            continue
        os.environ[key] = value
    return p


__all__ = ["ENV_FILE_NAME", "find_env_file", "load_env_file"]
