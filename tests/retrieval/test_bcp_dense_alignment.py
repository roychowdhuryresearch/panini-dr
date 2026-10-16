"""Aligning Tevatron's per-document vectors to our chunk ids.

Their index ships as 4 pickles of ``(reps, lookup)`` where ``reps[i]`` is the
embedding of the document ``lookup[i]``. Our DenseRetriever loads
``chunk_embeddings.npy`` + ``chunk_ids.pkl`` and pairs them BY POSITION. So the
whole conversion is: emit our chunk id for each row of their matrix, in their row
order, without ever reordering the matrix.

Getting this wrong is undetectable at runtime -- every query would return
plausible documents that are simply the wrong ones. Hence a test.
"""
import pickle

import numpy as np
import pytest

from panini_dr.eval.bcp_dense import align_chunk_ids, validate_prebuilt_dense_cache


def test_ids_follow_their_row_order():
    lookup = ["b", "a", "c"]
    keys = {"a": "Alpha [a]", "b": "Beta [b]", "c": "Gamma [c]"}
    assert align_chunk_ids(lookup, keys) == ["Beta [b]__0", "Alpha [a]__0", "Gamma [c]__0"]


def test_docid_types_are_normalised():
    """lookup entries may be ints from the pickle; the key map is str-keyed."""
    assert align_chunk_ids([1, 2], {"1": "One [1]", "2": "Two [2]"}) == [
        "One [1]__0", "Two [2]__0",
    ]


def test_missing_docid_fails_loudly():
    """A docid in their index with no document in our cache means the two
    artifacts disagree -- silently dropping it would shift every later row."""
    with pytest.raises(KeyError):
        align_chunk_ids(["a", "missing"], {"a": "Alpha [a]"})


def test_row_count_is_preserved():
    lookup = [str(i) for i in range(100)]
    keys = {str(i): f"T{i} [{i}]" for i in range(100)}
    ids = align_chunk_ids(lookup, keys)
    assert len(ids) == len(lookup) == 100
    assert len(set(ids)) == 100


def test_alignment_matches_matrix_rows():
    """End-to-end shape contract: ids[i] describes reps[i]."""
    reps = np.arange(12, dtype="float32").reshape(3, 4)
    lookup = ["x", "y", "z"]
    keys = {"x": "X [x]", "y": "Y [y]", "z": "Z [z]"}
    ids = align_chunk_ids(lookup, keys)
    assert reps.shape[0] == len(ids)
    assert ids[1] == "Y [y]__0"


class _Corpus:
    def __init__(self, ids):
        self.chunks = [object() for _ in ids]
        self._ids = set(ids)

    def get_chunk(self, chunk_id):
        return object() if chunk_id in self._ids else None


def _write_cache_files(path, ids):
    (path / "chunk_embeddings.npy").write_bytes(b"placeholder")
    (path / "faiss.index").write_bytes(b"placeholder")
    with (path / "chunk_ids.pkl").open("wb") as handle:
        pickle.dump(ids, handle)


def test_prebuilt_cache_alignment_fails_closed_on_count_mismatch(tmp_path):
    _write_cache_files(tmp_path, ["a__0"])

    with pytest.raises(ValueError, match="refusing an implicit embedding rebuild"):
        validate_prebuilt_dense_cache(_Corpus(["a__0", "b__0"]), tmp_path)


def test_prebuilt_cache_alignment_checks_id_membership(tmp_path):
    _write_cache_files(tmp_path, ["a__0", "wrong__0"])

    with pytest.raises(ValueError, match="silently misaligned"):
        validate_prebuilt_dense_cache(_Corpus(["a__0", "b__0"]), tmp_path)


def test_prebuilt_cache_alignment_accepts_matching_ids(tmp_path):
    _write_cache_files(tmp_path, ["b__0", "a__0"])

    validate_prebuilt_dense_cache(_Corpus(["a__0", "b__0"]), tmp_path)
