import os
import sys
import types

import pytest

from panini_dr.eval import official
from tests.eval.fakes import MULTI_PROMPT, SINGLE_PROMPT

FAKE_EVALUATE = '''
import some_module_that_does_not_exist  # never executed: only the function is extracted

def compute_llm_judge_score_V2(llm_judgement, gold_answers_length):
    return {"judge_score": 0.5 if gold_answers_length > 1 else 1.0, "precision": 1.0}
'''


def _fake_monaco_checkout(root):
    (root / "prompts").mkdir(parents=True)
    (root / "evaluation").mkdir()
    (root / "evaluation" / "evaluate.py").write_text(FAKE_EVALUATE)
    (root / "prompts" / "judge_prompt.py").write_text(
        f"single_answer_llm_judge_prompt = {SINGLE_PROMPT!r}\nmulti_answer_llm_judge_prompt = {MULTI_PROMPT!r}\n")


def test_monaco_judge_is_fingerprinted(tmp_path):
    _fake_monaco_checkout(tmp_path)
    with pytest.raises(official.OfficialSourceError, match="fingerprint"):
        official.load_monaco_judge(tmp_path)
    j = official.load_monaco_judge(tmp_path, require_fingerprint=False)
    assert not j.fingerprint_ok
    assert j.single_prompt == SINGLE_PROMPT and j.multi_prompt == MULTI_PROMPT
    assert j.score("x", 3) == {"judge_score": 0.5, "precision": 1.0}


def test_monaco_judge_missing_is_an_explicit_error(tmp_path, monkeypatch):
    monkeypatch.delenv("MONACO_JUDGE_DIR", raising=False)
    with pytest.raises(official.OfficialSourceError, match="MONACO_JUDGE_DIR"):
        official.load_monaco_judge()
    with pytest.raises(official.OfficialSourceError, match="could not find"):
        official.load_monaco_judge(tmp_path)


@pytest.mark.skipif(not os.environ.get("MONACO_JUDGE_DIR"), reason="needs a MoNaCo checkout (MONACO_JUDGE_DIR)")
def test_real_monaco_judge_matches_paper_and_parses_markdown():
    from panini_dr.eval.grade_set import demarkdown

    j = official.load_monaco_judge()
    assert j.fingerprint_ok
    reply = "extracted_final_answers: a, b\nfinal answer length: **2**\noverlapping answers: a###b"
    assert j.score(demarkdown(reply), 2)["judge_score"] == pytest.approx(1.0)


def test_bcp_grader_loads_from_a_checkout(tmp_path, monkeypatch):
    se = tmp_path / "scripts_evaluation"
    se.mkdir()
    (se / "__init__.py").write_text("")
    (se / "evaluate_run.py").write_text(
        "import vllm  # stubbed by the loader\n"
        "GRADER_TEMPLATE = 'Q={question} R={response} A={correct_answer}'\n"
        "def parse_judge_response(t):\n    return {'correct': 'yes' in t}\n")
    for m in [m for m in sys.modules if m.startswith("scripts_evaluation")]:
        monkeypatch.delitem(sys.modules, m)
    monkeypatch.setattr(sys, "path", list(sys.path))
    monkeypatch.setitem(sys.modules, "vllm", types.SimpleNamespace(LLM=None, SamplingParams=None))
    g = official.load_bcp_grader(tmp_path)
    assert g.template.startswith("Q=") and g.parse("yes") == {"correct": True}
    for m in [m for m in sys.modules if m.startswith("scripts_evaluation")]:
        monkeypatch.delitem(sys.modules, m)


def test_bcp_grader_import_hides_the_upstream_syntax_warning(tmp_path, monkeypatch):
    """The upstream evaluate_run.py has an invalid escape sequence ("0|\\%|") that
    Python reports with a SyntaxWarning when it first compiles the file."""
    import warnings

    se = tmp_path / "scripts_evaluation"
    se.mkdir()
    (se / "__init__.py").write_text("")
    (se / "evaluate_run.py").write_text(
        "GRADER_TEMPLATE = 'between 0|\\%| and 100|\\%| Q={question}'\n"
        "def parse_judge_response(t):\n    return {}\n")
    for m in [m for m in sys.modules if m.startswith("scripts_evaluation")]:
        monkeypatch.delitem(sys.modules, m)
    monkeypatch.setattr(sys, "path", list(sys.path))
    monkeypatch.setitem(sys.modules, "vllm", types.SimpleNamespace(LLM=None, SamplingParams=None))
    with warnings.catch_warnings(record=True) as caught:
        warnings.simplefilter("always")
        g = official.load_bcp_grader(tmp_path)
        # the filter is scoped to the import: the same warning still shows elsewhere
        compile("x = '0|\\%|'", "elsewhere.py", "exec")
    assert g.template.startswith("between 0|\\%|")
    syntax = [w for w in caught if issubclass(w.category, SyntaxWarning)]
    assert [w.filename for w in syntax] == ["elsewhere.py"], [str(w.message) for w in syntax]
    for m in [m for m in sys.modules if m.startswith("scripts_evaluation")]:
        monkeypatch.delitem(sys.modules, m)


def test_bcp_grader_missing_is_an_explicit_error(tmp_path, monkeypatch):
    monkeypatch.delenv("BCP_REPO", raising=False)
    with pytest.raises(official.OfficialSourceError, match="BCP_REPO"):
        official.load_bcp_grader()
    with pytest.raises(official.OfficialSourceError, match="evaluate_run.py"):
        official.load_bcp_grader(tmp_path)
