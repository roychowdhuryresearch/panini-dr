"""The CUDA preload must survive, because its failure mode is a SILENT void run.

Without it, `libnvrtc.so.13` cannot be resolved, transformers dies while probing
torchcodec, the adapter catches it per question, and the harness records 50/50
questions answered and judged -- scoring F1 = 0.000 with turns=0. That is
indistinguishable from a real result unless someone checks the turn counts.
"""
from __future__ import annotations

import subprocess
import sys
from pathlib import Path

REPO = Path(__file__).resolve().parents[1]


def test_importing_the_package_preloads_cuda_libs():
    """`import panini_dr` must run the preload -- not merely define it."""
    src = REPO / "src" / "panini_dr" / "__init__.py"
    text = src.read_text()
    assert "preload_cuda_libs" in text, "__init__ must import the preload"
    assert "_preload_cuda_libs()" in text, (
        "__init__ must CALL the preload; importing it is not enough")


def test_preload_is_idempotent_and_never_raises():
    from panini_dr._cuda_preload import preload_cuda_libs

    first = preload_cuda_libs()
    second = preload_cuda_libs()
    assert isinstance(first, list)
    assert second == [], "second call must be a no-op"


def test_retrieval_stack_imports_without_ld_library_path():
    """The real regression: the transformers chain must import on a bare env.

    Runs in a subprocess with LD_LIBRARY_PATH stripped, which is exactly the
    condition that produced the void MoNaCo-50 arm on 2026-08-14.
    """
    code = (
        "import sys; sys.path.insert(0, %r);"
        "import panini_dr;"
        "from panini_dr.retrieval.rerank import maybe_wrap_reranker;"
        "print('ok')" % str(REPO / "src")
    )
    env_clear = {"PATH": "/usr/bin:/bin", "HOME": str(Path.home())}
    proc = subprocess.run(
        [sys.executable, "-c", code],
        capture_output=True, text=True, env=env_clear, timeout=600,
    )
    assert proc.returncode == 0, (
        "retrieval stack failed to import without LD_LIBRARY_PATH:\n"
        + proc.stderr[-2000:])
    assert "ok" in proc.stdout
