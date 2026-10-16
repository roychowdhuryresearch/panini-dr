from panini_dr.adapters.ours._planner_exec import GSWPlan
from panini_dr.adapters.ours._connection_manifest import build_connection_manifest


def _plan():
    return GSWPlan.model_validate({
        "entities": [
            {"id": "e_fam", "kind": "filled", "name": "famous", "category": True, "role": "scope-filter"},
            {"id": "b_island", "kind": "blank", "value_type": "entity", "role": "bridge-entity"},
            {"id": "b_year", "kind": "blank", "value_type": "number", "role": "bridge-entity"},
            {"id": "b_author", "kind": "blank", "value_type": "entity", "is_target": True, "role": "target"},
        ],
        "verb_phrases": [
            {"id": "vp1", "subject_id": "b_author", "phrase": "instance_of", "object_id": "e_fam"},
            {"id": "vp2", "subject_id": "b_author", "phrase": "owns_bookstore_in", "object_id": "b_island"},
            {"id": "vp3", "subject_id": "b_author", "phrase": "started_writing_in_year", "object_id": "b_year"},
        ],
        "constraints": [],
    })


def test_manifest_classifies_partner_status():
    m = build_connection_manifest(_plan(), "b_author", resolved_values={"b_island": "Key West"})
    assert m["blank_id"] == "b_author"
    assert m["is_target"] is True
    conns = {c["vp_id"]: c for c in m["connections"]}
    assert conns["vp1"]["partner_status"] == "filled"          # e_fam is a filled entity
    assert conns["vp1"]["partner"] == "famous"
    assert conns["vp2"]["partner_status"] == "resolved_bridge" # b_island resolved
    assert conns["vp2"]["partner"] == "Key West"
    assert conns["vp3"]["partner_status"] == "deferred"        # b_year not resolved
    assert all("direction" in c and "relation" in c for c in m["connections"])


def test_manifest_marks_evidence_required_per_taxonomy():
    m = build_connection_manifest(_plan(), "b_author", resolved_values={})
    conns = {c["vp_id"]: c for c in m["connections"]}
    assert conns["vp1"]["evidence_required"] is False   # instance_of -> scaffold_typing
    assert conns["vp2"]["evidence_required"] is True    # owns_bookstore_in -> factual
    assert conns["vp3"]["evidence_required"] is True    # started_writing_in_year -> factual
