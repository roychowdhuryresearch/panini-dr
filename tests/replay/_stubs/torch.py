"""Test stub for ``torch``: just the names ``DenseRetriever`` touches on a CPU-only path."""

from types import SimpleNamespace

__version__ = "stub"
float16 = "float16"
bfloat16 = "bfloat16"
float32 = "float32"
cuda = SimpleNamespace(is_available=lambda: False, device_count=lambda: 0)
