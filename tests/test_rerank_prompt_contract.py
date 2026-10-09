"""Regression guard for _RERANK_PROMPT, the stage-2 LLM fit rubric.
"""
import pytest

from jobbot.ranking import _RERANK_PROMPT as P

_LOW = P.lower()


def test_the_rubric_is_more_than_be_strict():
    """The one-line "Be strict." rubric is what produced the 0.92 QC tech."""
    assert len(P) > 900, (
        "the rubric is too short to define fit; this is the regression that "
        "scored an entry-level QC technician 0.92 for a PhD candidate")


def test_overqualification_is_explicitly_a_penalty():
    """The exact defect: the model must be told that EXCEEDING a role is bad."""
    assert "overqualified" in _LOW or "over-qualified" in _LOW
    assert "exceed" in _LOW


@pytest.mark.parametrize("anchor", ["0.9", "0.5", "0.2"])
def test_the_score_scale_has_named_anchors(anchor):
    """A float 0-1 with no anchors is uncalibrated; name what the numbers mean."""
    assert anchor in P


def test_seniority_is_a_scoring_dimension():
    assert "seniority" in _LOW or "level" in _LOW


def test_dealbreakers_are_a_hard_gate_not_a_hint():
    """Dealbreakers were interpolated into the prompt but nothing said what to
    do with one. Make a hit cap the score."""
    assert "dealbreaker" in _LOW
    assert "0.1" in P or "cap" in _LOW


def test_vocabulary_overlap_is_not_method_overlap():
    """The embedding stage already rewards shared vocabulary; the reranker
    exists to catch what cosine cannot. Aptamers vs mRNA share words, not
    method -- that is the Innotech 0.77 lesson."""
    assert "vocabulary" in _LOW or "keyword" in _LOW


def test_the_job_and_profile_fields_are_still_interpolated():
    """Guard the format() contract -- a missing field raises at judge time."""
    for field in ("titles", "skills", "domains", "dealbreakers", "summary",
                  "job_title", "company", "description"):
        assert "{" + field + "}" in P


def test_strict_json_contract_survives():
    assert "strict json" in _LOW
    assert "score" in _LOW and "rationale" in _LOW
