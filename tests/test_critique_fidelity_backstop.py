"""The critique must not be able to bless a draft that deleted a publication."""
from unittest.mock import patch

import pytest

from jobbot import ollama_client as oc

_SOURCE = """Jane M. Doe
Publications
- Journal of Systems 2022, 14, 2137. https://doi.org/10.1000/182-demo-systems
"""
_DRAFT_DROPPED = "# Jane M. Doe\n### Publications\n- (none listed)\n"
_DRAFT_OK = ("# Jane M. Doe\n### Publications\n"
             "- Journal of Systems 2022, 14, 2137. https://doi.org/10.1000/182-demo-systems\n")

_BLESSED = '{"score": 0.95, "issues": [], "missing_keywords": [], ' \
           '"fabrication_risks": [], "omissions": []}'


def _crit(draft, raw=_BLESSED):
    """Critique `draft` with the LLM stubbed to a glowing 0.95 with nothing flagged."""
    with patch.object(oc, "_generate", return_value=raw), \
         patch.object(oc, "_profile_context", return_value=""):
        return oc._critique_tailored(draft, "Software Engineer", "Acme Corp", "JD text",
                                     source_resume=_SOURCE)


def test_a_blessed_draft_with_a_deleted_paper_is_caught_anyway():
    c = _crit(_DRAFT_DROPPED)
    assert any("182-demo-systems" in o for o in c["omissions"])


def test_the_inflated_score_is_capped_below_target():
    """0.95 >= the 0.9 target would early-stop the loop on a broken draft."""
    c = _crit(_DRAFT_DROPPED)
    assert c["score"] <= 0.5


def test_the_verified_defect_is_surfaced_as_an_actionable_issue():
    c = _crit(_DRAFT_DROPPED)
    assert any("VERIFIED DEFECT" in i for i in c["issues"])


def test_a_clean_draft_keeps_the_models_score_and_findings():
    """The backstop must not punish a faithful draft."""
    c = _crit(_DRAFT_OK)
    assert c["score"] == 0.95
    assert c["omissions"] == []
    assert c["fabrication_risks"] == []


def test_the_backstop_survives_an_unparseable_llm_response():
    """A garbled critique must still get the deterministic audit."""
    c = _crit(_DRAFT_DROPPED, raw="not json at all")
    assert any("182-demo-systems" in o for o in c["omissions"])
    assert c["score"] == 0.5      # unknown score + severe defect -> capped


def test_an_unknown_score_on_a_clean_draft_stays_unknown():
    """None means UNKNOWN and must not be conflated with a real score."""
    c = _crit(_DRAFT_OK, raw="not json at all")
    assert c["score"] is None
