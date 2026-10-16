"""Cassette store: request keys, blob compaction, repeated-key consumption, miss reports."""

from __future__ import annotations

import json

import numpy as np
import pytest

from panini_dr.replay.store import (
    Cassette,
    canon,
    compact_request,
    decode_array,
    encode_array,
    expand_request,
    rebuild_exception,
    request_key,
    serialise_exception,
)


def _llm_req(messages, **kw):
    return {"kind": "llm", "model": "toy-model", "messages": messages, "tools": None, "tool_choice": None,
            "temperature": kw.get("temperature", 0.0), "max_tokens": 128, "seed": 42,
            "response_format": None, "stop": None}


# ---------------------------------------------------------------------------
# keying
# ---------------------------------------------------------------------------


def test_key_is_sha256_of_the_full_canonical_request():
    req = _llm_req([{"role": "user", "content": "Où est Zorvan Klett né ?"}])
    assert request_key(req) == request_key(json.loads(json.dumps(req)))     # stable across a JSON round trip
    assert len(request_key(req)) == 64
    # canonical JSON keeps non-ASCII and has no whitespace
    assert canon({"a": "é", "b": [1, 2]}) == '{"a":"é","b":[1,2]}'


def test_key_keeps_dict_insertion_order_on_purpose():
    # tool-schema property order reaches the model's prompt, so it is part of the request
    a = {"kind": "llm", "tools": [{"properties": {"x": 1, "y": 2}}]}
    b = {"kind": "llm", "tools": [{"properties": {"y": 2, "x": 1}}]}
    assert request_key(a) != request_key(b)


def test_only_integral_sampling_scalars_are_normalised():
    base = _llm_req([{"role": "user", "content": "q"}])
    assert request_key({**base, "temperature": 0}) == request_key({**base, "temperature": 0.0})
    assert request_key({**base, "max_tokens": 128.0}) == request_key({**base, "max_tokens": 128})
    assert request_key({**base, "temperature": 0.5}) != request_key({**base, "temperature": 0.4})
    # a float elsewhere is NOT normalised
    assert request_key({**base, "extra": {"top_k": 5.0}}) != request_key({**base, "extra": {"top_k": 5}})
    # any change to any message changes the key
    assert request_key(base) != request_key(_llm_req([{"role": "user", "content": "q "}]))


def test_blob_compaction_round_trips_and_dedups():
    blobs: dict[str, object] = {}

    def put(v):
        h = request_key({"v": v})
        blobs[h] = v
        return h

    sysmsg = {"role": "system", "content": "You are a planner."}
    req = {**_llm_req([sysmsg, {"role": "user", "content": "q1"}]), "tools": [{"name": "search"}]}
    c = compact_request(req, put)
    assert all(set(m) == {"$blob"} for m in c["messages"]) and set(c["tools"]) == {"$blob"}
    assert expand_request(c, blobs) == req
    req2 = _llm_req([sysmsg, {"role": "user", "content": "q2"}])
    c2 = compact_request(req2, put)
    assert c2["messages"][0] == c["messages"][0]               # the shared system message is one blob


def test_array_codec_is_exact():
    a = np.arange(12, dtype=np.float16).reshape(3, 4) / 7
    b = decode_array(json.loads(json.dumps(encode_array(a))))
    assert b.dtype == a.dtype and b.shape == a.shape and np.array_equal(a, b)


# ---------------------------------------------------------------------------
# record -> replay
# ---------------------------------------------------------------------------


def _record(tmp_path, calls):
    """calls: (lineage, lseq, req, resp, t0, t1)"""
    cas = Cassette(tmp_path / "cas", "record")
    for lineage, lseq, req, resp, t0, t1 in calls:
        cas.record(kind=req["kind"], req=req, key=request_key(req),
                   ctx={"lineage": lineage, "lseq": lseq, "role": "planner"},
                   t0=cas.t_origin + t0, t1=cas.t_origin + t1, resp=resp)
    cas.close()
    return tmp_path / "cas"


def test_recording_refuses_to_overwrite(tmp_path):
    d = _record(tmp_path, [])
    with pytest.raises(SystemExit):
        Cassette(d, "record")


def test_repeated_identical_requests_are_consumed_with_lineage_affinity(tmp_path):
    req = _llm_req([{"role": "user", "content": "same"}])
    k = request_key(req)
    d = _record(tmp_path, [
        ("q:1", 0, req, {"n": "q1-first"}, 0.0, 1.0),
        ("q:2", 0, req, {"n": "q2-first"}, 0.5, 1.5),
        ("q:1", 3, req, {"n": "q1-second"}, 2.0, 3.0),
        ("q:3", 0, req, {"n": "q3"}, 2.5, 3.5),
    ])
    cas = Cassette(d, "replay", order="free")
    assert len(cas.by_key[k]) == 4
    assert cas.take(k, "q:1", 3)["resp"] == {"n": "q1-second"}      # same lineage, same per-lineage index
    assert cas.take(k, "q:2", 7)["resp"] == {"n": "q2-first"}       # same lineage, other index
    assert cas.take(k, "q:9", 0)["resp"] == {"n": "q1-first"}       # nothing of q:9: oldest (by t0) left
    assert [e["resp"]["n"] for e in cas.leftovers()] == ["q3"]
    assert cas.take(k, "q:3", 0)["resp"] == {"n": "q3"}
    assert cas.take(k, "q:3", 1) is None                             # all consumed
    assert dict(cas.pick_stats) == {"exact_lineage_seq": 2, "same_lineage": 1, "fifo_fallback": 1}
    assert cas.leftovers() == []


def test_unknown_key_misses_and_the_report_shows_a_diff_against_the_nearest(tmp_path):
    msgs = [{"role": "system", "content": "You verify one blank."},
            {"role": "user", "content": "Blank: b_bird\nProposed value: Black francolin\nConstraints: none"}]
    other = [{"role": "system", "content": "You verify one blank."},
             {"role": "user", "content": "Blank: t_lake\nProposed value: Sea of Galilee"}]
    d = _record(tmp_path, [("q:1", 0, _llm_req(msgs), {"ok": 1}, 0.0, 1.0),
                           ("q:2", 0, _llm_req(other), {"ok": 2}, 0.0, 1.0)])
    cas = Cassette(d, "replay", report_dir=tmp_path / "rep")
    changed = [msgs[0], {"role": "user", "content": msgs[1]["content"].replace("francolin", "francolim")}]
    req = _llm_req(changed)
    assert cas.take(request_key(req), "q:1", 0) is None
    path = cas.report_miss(kind="llm", req=req, key=request_key(req), ctx={"lineage": "q:1", "lseq": 0, "role": "planner"})
    diff = path.read_text()
    assert "-    Proposed value: Black francolin" in diff and "+    Proposed value: Black francolim" in diff
    assert "recorded i=0 lineage=q:1" in diff                         # nearest = same question
    detail = json.loads((tmp_path / "rep" / "misses" / "miss_0000.json").read_text())
    assert detail["nearest"]["i"] == 0 and detail["nearest_request"]["messages"] == msgs
    assert cas.misses[0]["kind"] == "llm" and cas.misses[0]["lineage"] == "q:1"


class ToyServerError(Exception):
    def __init__(self, message, code=None):
        super().__init__(message)
        self.message, self.code = message, code


def test_recorded_exceptions_are_rebuilt_with_class_text_and_attributes():
    d = serialise_exception(ToyServerError("unexpected tokens remaining in message header", code=500))
    exc = rebuild_exception(json.loads(json.dumps(d)))
    assert type(exc) is ToyServerError
    assert str(exc) == "unexpected tokens remaining in message header" and exc.code == 500
    unknown = rebuild_exception({**d, "type": "no.such.module.Error"})
    assert type(unknown) is RuntimeError and str(unknown) == str(exc)
