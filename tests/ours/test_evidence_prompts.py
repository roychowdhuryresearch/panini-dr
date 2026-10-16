from panini_dr.adapters.ours._evidence_prompts import (
    RESOLVER_USER,
    VERIFIER_SYSTEM, VERIFIER_USER, parse_verifier_output,
    render_resolver_unit_instruction,
)


def test_verifier_prompts_are_format_strings():
    assert "{value}" in VERIFIER_USER and "{constraints}" in VERIFIER_USER
    assert "uniqu" in VERIFIER_SYSTEM.lower()


def test_parse_verifier_output_full():
    txt = ('{"satisfies_all": false, "unique": false,'
           ' "missing_constraints": ["cofounder_of Netflix"],'
           ' "deferred_constraints": [],'
           ' "ambiguous_candidates": ["Reed Hastings", "Marc Randolph"],'
           ' "reason": "Chouinard founded Patagonia, not Netflix"}')
    d = parse_verifier_output(txt)
    assert d["satisfies_all"] is False and d["unique"] is False
    assert d["missing_constraints"] == ["cofounder_of Netflix"]
    assert d["ambiguous_candidates"] == ["Reed Hastings", "Marc Randolph"]
    assert "Patagonia" in d["reason"]


def test_parse_verifier_output_garbage_is_safe_reject():
    d = parse_verifier_output("not json")
    assert d["satisfies_all"] is False and d["unique"] is False
    assert d["missing_constraints"] == [] and d["deferred_constraints"] == []
    assert d["ambiguous_candidates"] == []


# ---------------------------------------------------------------------------
# Hard rule 7 / q16 class -- RESOLVER_USER unit_instruction field.
# ---------------------------------------------------------------------------

def test_resolver_user_template_formats_with_empty_unit_instruction():
    s = RESOLVER_USER.format(subject="Serbia", relation="independent_in",
                             value_type="date", question="Q", context="(none)",
                             chunks="[c1] body", unit_instruction="")
    assert "Serbia" in s and "date" in s and "[c1] body" in s


def test_resolver_user_no_unit_is_byte_identical_to_pre_hint_template():
    """GUARD RAIL: a blank with no declared unit (unit_instruction="") must
    render EXACTLY the prompt this template produced before the
    unit_instruction field existed. Any drift here means unaffected
    questions would silently change behavior."""
    legacy = (
        "Blank to resolve:\n"
        "- subject: Serbia\n"
        "- relation: independent_in\n"
        "- expected value type: date\n"
        "\n"
        "Overall question (for disambiguation only): Q\n"
        "\n"
        "Resolved context (upstream values): (none)\n"
        "\n"
        "Retrieved chunks:\n"
        "[c1] body\n"
        "\n"
        "Return the JSON object now."
    )
    current = RESOLVER_USER.format(subject="Serbia", relation="independent_in",
                                   value_type="date", question="Q", context="(none)",
                                   chunks="[c1] body", unit_instruction="")
    assert current == legacy


def test_render_resolver_unit_instruction_empty_when_no_unit():
    assert render_resolver_unit_instruction(None, "some hint") == ""
    assert render_resolver_unit_instruction("", "some hint") == ""


def test_render_resolver_unit_instruction_names_unit_and_requires_conversion():
    text = render_resolver_unit_instruction("metres", "height ... in meters?")
    assert "metres" in text
    assert "convert" in text.lower()
    assert "REQUIRED UNIT" in text
    assert "height ... in meters?" in text
