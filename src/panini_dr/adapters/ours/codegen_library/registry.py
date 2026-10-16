"""Skill-library registry for gsw_codegen_compute.

A :class:`Library` maps an *operation key* -> a trusted pure function seeded
from :mod:`.seeds`. (Generated functions persisted to an on-disk store were
part of the removed generation path; the paper's frozen library held only the
seeds.)

Off-path this module is never imported, so it stays fully self-contained.
"""

from __future__ import annotations

from typing import Any, Callable, Optional

from . import seeds

# (op, tuple(arg_types), intent_tag)
OpKey = tuple[str, tuple[str, ...], str]


class Library:
    """In-memory op-key -> callable map of the seed functions."""

    def __init__(self, funcs: dict[OpKey, Callable[..., Any]]) -> None:
        self._funcs = funcs

    # -- construction --------------------------------------------------------
    @classmethod
    def load(cls) -> "Library":
        return cls({entry["key"]: entry["fn"] for entry in seeds.SEEDS})

    # -- query ---------------------------------------------------------------
    def lookup(self, key: OpKey) -> Optional[Callable[..., Any]]:
        return self._funcs.get(key)
