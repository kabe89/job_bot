"""Repair what is mechanical; never ship a defective resume silently.

Tests verifying that mechanical defect repair fixes unverified additions (such as invented months),
while omissions (such as dropped awards) are preserved as reported defects.
"""
from unittest.mock import patch

import pytest

from jobbot import ollama_client as oc
from jobbot.source_fidelity import check_fidelity, repair_fidelity

_SOURCE = """Jane M. Doe
Education
- Ph.D. Systems Engineering (Expected 2027) - Example State University, Springfield, IL.
Awards & Honors
- Distinguished Engineering Award - Fall 2020
"""


# --- mechanical repair --------------------------------------------------------

@pytest.mark.parametrize("bad", ["Expected Dec 2027", "Expected December 2027",
                                 "Expected Jan 2027", "expected May 2027"])
def test_an_invented_month_is_stripped(bad):
    fixed, _ = repair_fidelity(_SOURCE, f"**Ph.D., Systems Engineering ({bad})** | State University\n")
    assert "2027" in fixed
    for m in ("Dec", "December", "Jan", "May"):
        assert m not in fixed, f"{m!r} survived in {fixed!r}"


def test_repair_actually_satisfies_the_checker():
    draft = "**Ph.D., Systems Engineering (Expected Dec 2027)** | State University\n- Distinguished Engineering Award - Fall 2020\n"
    assert not check_fidelity(_SOURCE, draft).ok
    fixed, remaining = repair_fidelity(_SOURCE, draft)
    assert check_fidelity(_SOURCE, fixed).ok
    assert remaining == []


def test_a_faithful_date_is_left_alone():
    draft = "**Ph.D., Systems Engineering (Expected 2027)** | State University\n- Distinguished Engineering Award\n"
    fixed, remaining = repair_fidelity(_SOURCE, draft)
    assert fixed == draft
    assert remaining == []


def test_a_month_the_source_itself_gives_is_not_stripped():
    """Only invented precision is removed. If the source says the month, keep it."""
    src = "- Ph.D. Systems Engineering (Expected May 2027)\n- Distinguished Engineering Award\n"
    draft = "**Ph.D. (Expected May 2027)**\n- Distinguished Engineering Award\n"
    fixed, _ = repair_fidelity(src, draft)
    assert "May" in fixed


def test_a_dropped_credential_is_reported_not_auto_inserted():
    """Code cannot know where an award belongs; a wrong insertion is worse than
    a reported gap. It must survive as a defect, not be silently patched in."""
    draft = "**Ph.D., Systems Engineering (Expected 2027)** | State University\n"
    fixed, remaining = repair_fidelity(_SOURCE, draft)
    assert "Distinguished Engineering Award" not in fixed
    assert any("Distinguished Engineering Award" in d for d in remaining)


# --- never ship silently ------------------------------------------------------

def _run(final_draft: str) -> dict:
    """Drive the loop with the LLM stubbed so it returns `final_draft` and a
    critique that blesses it."""
    blessed = ('{"score": 0.95, "issues": [], "missing_keywords": [], '
               '"fabrication_risks": [], "omissions": []}')

    def fake_generate(prompt, **kw):
        return blessed if '"score"' in prompt or "STRICT JSON" in prompt else final_draft

    with patch.object(oc, "_generate", side_effect=fake_generate), \
         patch.object(oc, "_profile_context", return_value=""):
        return oc.tailor_resume_iterative(_SOURCE, "Scientist", "Acme Corp", "JD",
                                          rounds=1)


def test_the_result_carries_the_verified_defects():
    out = _run("**Ph.D. (Expected 2027)** | State University\n")   # Distinguished Engineering Award dropped
    assert "defects" in out, "callers cannot warn about what they are not told"
    assert any("Distinguished Engineering Award" in d for d in out["defects"])


def test_a_clean_run_reports_no_defects():
    out = _run("**Ph.D. (Expected 2027)** | State University\n- Distinguished Engineering Award - Fall 2020\n")
    assert out["defects"] == []


def test_the_returned_resume_is_the_repaired_one():
    out = _run("**Ph.D. (Expected Dec 2027)** | State University\n- Distinguished Engineering Award - Fall 2020\n")
    assert "Dec" not in out["resume"]
    assert out["defects"] == []
