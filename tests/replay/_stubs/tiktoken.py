"""Test stub: makes ``tiktoken`` unavailable in the replay end-to-end subprocesses.

The article reader then counts tokens with its heuristic in BOTH the recording and the
replay (and never tries to download an encoding, which the socket guard would refuse).
"""

raise ImportError("tiktoken is stubbed out in the replay end-to-end test")
