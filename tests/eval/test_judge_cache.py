import json

from panini_dr.eval.judge_cache import JudgeCache, cache_key


def _k(**kw):
    base = dict(grader="g", model="m", passes=3, qid="q1", prediction="Paris", prompt="P")
    base.update(kw)
    return cache_key(**base)


def test_key_changes_with_prediction_model_passes_prompt():
    k = _k()
    assert _k() == k
    assert _k(prediction="Lyon") != k
    assert _k(model="other-judge") != k
    assert _k(passes=1) != k
    assert _k(prompt="P2") != k
    assert _k(qid="q2") != k
    assert _k(grader="h") != k


def test_save_is_atomic_and_only_when_dirty(tmp_path):
    path = tmp_path / "sub" / "cache.json"
    c = JudgeCache(path)
    assert c.save() is False and not path.exists()        # nothing to write
    c.put("a", {"correct": True})
    assert c.save() is True
    assert json.loads(path.read_text()) == {"a": {"correct": True}}
    assert not [p for p in path.parent.iterdir() if p.name.endswith(".tmp")]
    mtime = path.stat().st_mtime_ns
    c2 = JudgeCache(path)
    c2.put("a", {"correct": True})                          # identical record: not dirty
    assert c2.save() is False and path.stat().st_mtime_ns == mtime
    assert c2.get("a") == {"correct": True} and "a" in c2 and len(c2) == 1


def test_no_path_cache_never_writes(tmp_path):
    c = JudgeCache(None)
    c.put("a", 1)
    assert c.save() is False
