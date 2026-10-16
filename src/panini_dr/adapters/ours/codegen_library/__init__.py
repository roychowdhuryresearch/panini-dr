"""gsw_codegen_compute: computation through a library of seeded skill functions.

Self-contained package (no dependency on ``_planner_exec``) so it can be
imported lazily behind the default-off ``gsw_codegen_compute`` flag without
perturbing the off-path.

Public surface:
- :mod:`.seeds`     — trusted seed pure functions (day-granular date diff,
  birthday-aware age).
- :mod:`.registry`  — :class:`~.registry.Library` op-key -> callable map.
- :mod:`.generator` — library-hit resolver (misses fall back).
"""

from __future__ import annotations
