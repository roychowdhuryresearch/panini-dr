"""Library-hit resolver for the codegen compute step.

On a compute step (under ``gsw_codegen_compute``) we :func:`resolve` an
operation key against the library of seeded pure functions:

* **library hit** — run the function on the inputs. **No LLM call.**
  Deterministic. If it raises, returns ``None`` or the wrong type, the caller
  falls back to the hardcoded op-executor.
* **library miss** — ``(None, "fallback")``.

The paper ran the library frozen (``gsw_codegen_frozen=true``): a miss never
asked the LLM to write a new function. That generation path (LLM write, AST
sandbox, persisted store) was removed with the key.
"""

from __future__ import annotations

from typing import Any


def resolve(
    lib: Any,
    key: tuple,
    inputs: list,
    expected_type: str,
) -> tuple[Any, str]:
    """Return ``(value, provenance)`` with provenance in
    ``{"library_hit", "fallback"}``.
    """
    fn = lib.lookup(key)
    if fn is not None:
        try:
            value = fn(*inputs)
        except Exception:  # noqa: BLE001 - a failed hit -> hardcoded fallback
            value = None
        if value is not None and _type_ok(value, expected_type):
            return value, "library_hit"
        return None, "fallback"
    return None, "fallback"


def _type_ok(value: Any, expected_type: str) -> bool:
    if value is None:
        return False
    et = (expected_type or "").strip().lower()
    if et in ("number", "int", "integer", "float", "numeric"):
        return isinstance(value, (int, float)) and not isinstance(value, bool)
    if et in ("bool", "boolean"):
        return isinstance(value, bool)
    if et in ("str", "text", "string", "date", "entity", ""):
        return isinstance(value, str)
    # Unknown target type: accept any non-None value.
    return True
