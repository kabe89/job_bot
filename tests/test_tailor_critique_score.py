"""_critique_tailored collapsed an unparseable/missing score to 0.0, which is
indistinguishable from "this resume genuinely scored zero". Observed live: the
same resume scored 0.42 on one run and 0.0 (with ZERO fabrication_risks) on
another -- a nonsense verdict that came from the fallback, not the model.

Consequences of a phantom 0.0:
  - the refinement loop's early-stop (`score >= target`) can never fire, so it
    burns every round and the LLM calls that go with them
  - the CLI reports "self-score 0.00" for a clean resume

Score is now Optional[float]: None means "unknown", which is not the same as bad.
"""
import pytest

from jobbot import ollama_client as oc


def _patch(monkeypatch, raw):
    monkeypatch.setattr(oc, "_generate", lambda *a, **k: raw)


def test_a_real_score_parses(monkeypatch):
    _patch(monkeypatch, '{"score": 0.42, "issues": [], "missing_keywords": [], '
                        '"fabrication_risks": []}')
    assert oc._critique_tailored("md", "T", "C", "jd")["score"] == 0.42


def test_a_genuine_zero_is_preserved(monkeypatch):
    _patch(monkeypatch, '{"score": 0.0, "issues": ["everything"], '
                        '"missing_keywords": [], "fabrication_risks": ["all of it"]}')
    assert oc._critique_tailored("md", "T", "C", "jd")["score"] == 0.0


def test_a_missing_score_is_unknown_not_zero(monkeypatch):
    _patch(monkeypatch, '{"issues": [], "missing_keywords": [], "fabrication_risks": []}')
    assert oc._critique_tailored("md", "T", "C", "jd")["score"] is None


def test_a_null_score_is_unknown_not_zero(monkeypatch):
    _patch(monkeypatch, '{"score": null, "issues": [], "fabrication_risks": []}')
    assert oc._critique_tailored("md", "T", "C", "jd")["score"] is None


def test_a_nonnumeric_score_is_unknown_not_zero(monkeypatch):
    _patch(monkeypatch, '{"score": "excellent", "issues": [], "fabrication_risks": []}')
    assert oc._critique_tailored("md", "T", "C", "jd")["score"] is None


def test_unparseable_output_is_unknown_not_zero(monkeypatch):
    _patch(monkeypatch, "the model rambled instead of returning JSON")
    out = oc._critique_tailored("md", "T", "C", "jd")
    assert out["score"] is None
    assert out["fabrication_risks"] == []


@pytest.mark.parametrize("raw", ["85", "1.4", "-0.2", "100"])
def test_an_out_of_contract_score_is_unknown_not_guessed(raw, monkeypatch):
    """85 could mean 0.85; 1.4 could mean 1.0. We cannot know, and guessing risks
    inventing a HIGH score that early-stops refinement. Unknown is the honest answer."""
    _patch(monkeypatch, '{"score": %s, "issues": [], "fabrication_risks": []}' % raw)
    assert oc._critique_tailored("md", "T", "C", "jd")["score"] is None


def test_the_contract_boundaries_are_valid(monkeypatch):
    for raw, want in (("0", 0.0), ("1", 1.0)):
        _patch(monkeypatch, '{"score": %s, "issues": [], "fabrication_risks": []}' % raw)
        assert oc._critique_tailored("md", "T", "C", "jd")["score"] == want


def test_refine_loop_does_not_early_stop_on_an_unknown_score(monkeypatch):
    """An unknown score must not be read as "good enough" NOR as 0 -- it just
    keeps refining, and the reported score stays None rather than a lie."""
    monkeypatch.setattr(oc, "tailor_resume", lambda *a, **k: "draft")
    monkeypatch.setattr(oc, "_critique_tailored",
                        lambda *a, **k: {"score": None, "issues": [],
                                         "missing_keywords": [], "fabrication_risks": []})
    monkeypatch.setattr(oc, "_refine_tailored", lambda *a, **k: "revised")
    out = oc.tailor_resume_iterative("r", "T", "C", "jd", rounds=2, target_score=0.9)
    assert out["rounds"] == 2          # ran both rounds, no phantom early stop
    assert out["score"] is None        # honest "unknown", not 0.0
    assert out["resume"] == "revised"


def test_refine_loop_early_stops_on_a_real_high_score(monkeypatch):
    monkeypatch.setattr(oc, "tailor_resume", lambda *a, **k: "draft")
    monkeypatch.setattr(oc, "_critique_tailored",
                        lambda *a, **k: {"score": 0.95, "issues": [],
                                         "missing_keywords": [], "fabrication_risks": []})
    monkeypatch.setattr(oc, "_refine_tailored", lambda *a, **k: "revised")
    out = oc.tailor_resume_iterative("r", "T", "C", "jd", rounds=3, target_score=0.9)
    assert out["rounds"] == 1
    assert out["resume"] == "draft"    # never revised - it was already good


def test_fabrication_blocks_early_stop_even_at_a_high_score(monkeypatch):
    monkeypatch.setattr(oc, "tailor_resume", lambda *a, **k: "draft")
    monkeypatch.setattr(oc, "_critique_tailored",
                        lambda *a, **k: {"score": 0.95, "issues": [],
                                         "missing_keywords": [],
                                         "fabrication_risks": ["invented qPCR"]})
    monkeypatch.setattr(oc, "_refine_tailored", lambda *a, **k: "revised")
    out = oc.tailor_resume_iterative("r", "T", "C", "jd", rounds=2, target_score=0.9)
    assert out["rounds"] == 2          # kept going: fabrication must be removed
