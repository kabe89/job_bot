"""The critic audits both additions (fabrications) and omissions (dropped credentials or altered dates).

The critic must audit in both directions: invented content AND lost/altered content.
"""
import pytest

from jobbot import ollama_client as oc


@pytest.fixture
def captured(monkeypatch):
    box = {}

    def fake_generate(prompt, **kw):
        box["prompt"] = prompt
        return ('{"score": 0.9, "issues": [], "missing_keywords": [], '
                '"fabrication_risks": [], "omissions": []}')

    monkeypatch.setattr(oc, "_generate", fake_generate)
    monkeypatch.setattr(oc, "_profile_context", lambda: "")
    return box


def test_critique_prompt_asks_for_omissions(captured):
    oc._critique_tailored("md", "T", "C", "jd", source_resume="RESUME")
    low = captured["prompt"].lower()
    assert "omission" in low
    assert "publication" in low, "dropping a paper is the defect that motivated this"


def test_critique_prompt_forbids_altering_facts(captured):
    oc._critique_tailored("md", "T", "C", "jd", source_resume="RESUME")
    low = captured["prompt"].lower()
    assert "altered" in low or "changed" in low
    # the exact observed drift: "Expected 2027" -> "Expected January 2027"
    assert "more specific" in low or "invent" in low


def test_omissions_is_returned_as_a_list(captured):
    out = oc._critique_tailored("md", "T", "C", "jd", source_resume="RESUME")
    assert out["omissions"] == []


def test_omissions_survives_a_model_that_ignores_the_key(monkeypatch):
    monkeypatch.setattr(oc, "_profile_context", lambda: "")
    monkeypatch.setattr(oc, "_generate", lambda *a, **k:
                        '{"score": 0.9, "issues": [], "fabrication_risks": []}')
    out = oc._critique_tailored("md", "T", "C", "jd", source_resume="RESUME")
    assert out["omissions"] == []          # normalised, never KeyError


def test_omissions_are_passed_to_the_refiner(monkeypatch):
    box = {}
    monkeypatch.setattr(oc, "_generate", lambda p, **k: box.setdefault("prompt", p) or "x")
    monkeypatch.setattr(oc, "_profile_context", lambda: "")
    monkeypatch.setattr(oc, "_template_block", lambda kind="resume": "")
    oc._refine_tailored("draft",
                        {"score": 0.5, "issues": [], "missing_keywords": [],
                         "fabrication_risks": [], "omissions": ["dropped Water 2022 paper"]},
                        "resume", "T", "C", "jd")
    assert "dropped Water 2022 paper" in box["prompt"]


def test_omissions_block_the_early_stop(monkeypatch):
    """A high score must not end the loop while real content is missing --
    exactly the 0.98-with-a-deleted-publication case."""
    monkeypatch.setattr(oc, "tailor_resume", lambda *a, **k: "draft")
    monkeypatch.setattr(oc, "_critique_tailored",
                        lambda *a, **k: {"score": 0.98, "issues": [],
                                         "missing_keywords": [], "fabrication_risks": [],
                                         "omissions": ["dropped Water 2022 publication"]})
    monkeypatch.setattr(oc, "_refine_tailored", lambda *a, **k: "revised")
    out = oc.tailor_resume_iterative("r", "T", "C", "jd", rounds=2, target_score=0.9)
    assert out["rounds"] == 2, "must keep refining until the record is restored"


def test_a_clean_high_score_still_early_stops(monkeypatch):
    monkeypatch.setattr(oc, "tailor_resume", lambda *a, **k: "draft")
    monkeypatch.setattr(oc, "_critique_tailored",
                        lambda *a, **k: {"score": 0.98, "issues": [],
                                         "missing_keywords": [], "fabrication_risks": [],
                                         "omissions": []})
    monkeypatch.setattr(oc, "_refine_tailored", lambda *a, **k: "revised")
    out = oc.tailor_resume_iterative("r", "T", "C", "jd", rounds=3, target_score=0.9)
    assert out["rounds"] == 1
    assert out["resume"] == "draft"


def test_history_records_omissions(monkeypatch):
    monkeypatch.setattr(oc, "tailor_resume", lambda *a, **k: "draft")
    monkeypatch.setattr(oc, "_critique_tailored",
                        lambda *a, **k: {"score": 0.5, "issues": [],
                                         "missing_keywords": [], "fabrication_risks": [],
                                         "omissions": ["lost the Water paper"]})
    monkeypatch.setattr(oc, "_refine_tailored", lambda *a, **k: "revised")
    out = oc.tailor_resume_iterative("r", "T", "C", "jd", rounds=1, target_score=0.9)
    assert out["history"][0]["omissions"] == ["lost the Water paper"]
