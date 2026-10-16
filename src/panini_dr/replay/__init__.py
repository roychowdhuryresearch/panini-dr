"""Record / replay of every external call, and the ``panini-dr verify`` equivalence check.

A *cassette* holds every response the harness received from the outside world during
one run: LLM chat completions, query embeddings and reranker calls. Replaying it
re-runs the harness code with those responses served from disk, so a replay that
reproduces the recorded run's outputs exactly shows that the code makes the same
decisions as the code that made the recording.

Modules
-------
``store``      the cassette: content-addressed keys, record / replay, miss reports
``gate``       per-question event-order gate (reproduces the recorded thread order)
``patches``    runtime hooks (LLM client, sentence-transformers, httpx.post, socket guard,
               thread lineage)
``launcher``   ``python -m panini_dr.replay --mode {record,replay} --cassette DIR -- <runner args>``
``normalise``  what a comparison ignores (wall-clock fields, paths, run tag) and why
``compare``    strict comparison of two run directories
``bundle``     verify bundles: cassettes + normalised reference outputs + manifest
``verify``     ``panini-dr verify --bundle DIR``

Nothing here is imported by ``panini-dr run``; the hooks exist only in a process
started through the launcher.
"""
