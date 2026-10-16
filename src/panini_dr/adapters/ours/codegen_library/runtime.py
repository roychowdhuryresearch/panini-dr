"""Per-question runtime handle threaded into the compute seam.

A :class:`CodegenRuntime` bundles what ``_compute_constraint`` needs to route a
compute step through the skill library: the shared :class:`Library` and the
current question (read by the years-vs-days intent cue). It is only
constructed when ``gsw_codegen_compute`` is on; off-path the ``codegen``
argument stays ``None`` and every consumer is byte-identical.
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import Any


@dataclass
class CodegenRuntime:
    enabled: bool
    lib: Any
    question: str = ""
