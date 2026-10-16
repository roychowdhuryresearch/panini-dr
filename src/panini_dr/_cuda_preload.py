"""Preload CUDA runtime libs the dynamic loader cannot find on its own.

THE FAILURE THIS PREVENTS. torch 2.11+cu130 ships ``libnvrtc.so.13`` inside the
venv at ``site-packages/nvidia/cu13/lib/``, but nothing puts that directory on
the loader path: ``ldconfig`` carries only nvrtc **12**, and torch's own
``_load_global_deps()`` preloads just ``libtorch_global_deps.so``, not the cu13
tree. Any later ``dlopen`` of a library that needs nvrtc 13 therefore fails with

    OSError: libnvrtc.so.13: cannot open shared object file

``transformers`` triggers exactly that by probing ``torchcodec`` on import.

WHY IT MATTERS FAR MORE THAN AN OPTIONAL VIDEO DECODER. The error surfaces deep
inside the adapter, is caught per question, and the harness records the question
as a normal wrong answer. On 2026-08-14 a full MoNaCo-50 arm came back with 50/50
answered, judged, and scored F1 = 0.000 -- indistinguishable from a real result
except for ``turns=0 tools=0``. A run can look complete and be entirely void.
Prefixing every launch with ``LD_LIBRARY_PATH`` works but is not durable: it has
to be remembered at every call site, and forgetting it fails silently.

The preload is idempotent, costs one ``dlopen``, and is a no-op on machines where
the loader can already resolve the library (or where the cu13 tree is absent).
``RTLD_GLOBAL`` is required: the point is to make the symbols available to the
*later* ``dlopen`` that torchcodec performs.
"""
from __future__ import annotations

import ctypes
import glob
import os

_DONE = False


def preload_cuda_libs() -> list[str]:
    """Preload bundled CUDA libs into the global symbol namespace.

    Returns the paths loaded (empty when nothing was needed or found). Never
    raises: a research run must not die because an optional decoder is unhappy.
    """
    global _DONE
    if _DONE:
        return []
    _DONE = True

    loaded: list[str] = []
    here = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))  # src/
    # site-packages sits beside the venv's python; resolve it from this module's
    # interpreter rather than assuming a layout.
    import sysconfig

    candidates = []
    purelib = sysconfig.get_paths().get("purelib")
    if purelib:
        candidates.append(os.path.join(purelib, "nvidia", "cu13", "lib"))
    # editable installs put the package outside site-packages; also try the
    # venv adjacent to the repo checkout.
    candidates.append(os.path.join(
        os.path.dirname(here), ".venv", "lib", "python3.12",
        "site-packages", "nvidia", "cu13", "lib"))

    for d in candidates:
        for pat in ("libnvrtc.so.13", "libnvrtc-builtins.so.13*"):
            for path in sorted(glob.glob(os.path.join(d, pat))):
                try:
                    ctypes.CDLL(path, mode=ctypes.RTLD_GLOBAL)
                    loaded.append(path)
                except OSError:
                    pass
        if loaded:
            break
    return loaded
