"""PANINI-DR: controlled deep research through Generative Semantic Workspaces.

The CUDA preload runs at import time, before anything pulls in transformers.
See ``_cuda_preload`` for why: without it a missing ``libnvrtc.so.13`` can make
every question in a run fail *silently* with turns=0, producing a
complete-looking run that scores 0.000. It is a no-op where it is not needed.
"""
from panini_dr._cuda_preload import preload_cuda_libs as _preload_cuda_libs

_preload_cuda_libs()

__version__ = "0.1.0"
