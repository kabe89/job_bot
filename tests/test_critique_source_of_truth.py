"""The critic and the drafter must agree on what ground truth encompasses.

SYSTEM_RESUME_TAILOR includes the verified profile context as ground truth.
The critic must audit against the same ground truth the drafter was given.
"""
import pytest

from jobbot import ollama_client as oc


@pytest.fixture
def captured(monkeypatch):
    """Capture the critique prompt without calling a model."""
    box = {}

    def fake_generate(prompt, **kw):
        box["prompt"] = prompt
        box["kwargs"] = kw
        return '{"score": 0.9, "issues": [], "missing_keywords": [], "fabrication_risks": []}'

    monkeypatch.setattr(oc, "_generate", fake_generate)
    return box


def test_critique_prompt_includes_the_profile_as_source_of_truth(captured, monkeypatch):
    monkeypatch.setattr(oc, "_profile_context",
                        lambda: "SKILLS: qPCR, transfection, Bradford/BCA assays")
    oc._critique_tailored("tailored md", "T", "C", "jd", source_resume="RESUME TEXT")
    prompt = captured["prompt"]
    assert "RESUME TEXT" in prompt
    assert "qPCR, transfection, Bradford/BCA assays" in prompt, \
        "the critic cannot audit fairly without the profile the drafter was given"


def test_critique_no_longer_calls_the_resume_the_only_source(captured, monkeypatch):
    monkeypatch.setattr(oc, "_profile_context", lambda: "SKILLS: qPCR")
    oc._critique_tailored("md", "T", "C", "jd", source_resume="RESUME")
    low = captured["prompt"].lower()
    assert "resume (the only source of truth)" not in low
    assert "original resume (the only source" not in low


def test_critique_prompt_names_the_profile_as_legitimate(captured, monkeypatch):
    monkeypatch.setattr(oc, "_profile_context", lambda: "SKILLS: qPCR")
    oc._critique_tailored("md", "T", "C", "jd", source_resume="RESUME")
    low = captured["prompt"].lower()
    assert "profile" in low
    # It must say a profile-only skill is NOT fabrication.
    assert "not fabrication" in low or "is not a fabrication" in low


def test_critique_still_works_without_a_profile(captured, monkeypatch):
    """No profile configured -> resume is the whole source of truth; no crash."""
    monkeypatch.setattr(oc, "_profile_context", lambda: "")
    out = oc._critique_tailored("md", "T", "C", "jd", source_resume="RESUME ONLY")
    assert out["score"] == 0.9
    assert "RESUME ONLY" in captured["prompt"]


def test_critique_survives_an_unreadable_profile(captured, monkeypatch):
    def boom():
        raise OSError("profile gone")
    monkeypatch.setattr(oc, "_profile_context", boom)
    out = oc._critique_tailored("md", "T", "C", "jd", source_resume="RESUME")
    assert out["score"] == 0.9        # fail-open, still critiques
    assert "RESUME" in captured["prompt"]


def test_fabrication_rule_is_still_present(captured, monkeypatch):
    """Widening the source of truth must not disarm the fabrication audit."""
    monkeypatch.setattr(oc, "_profile_context", lambda: "SKILLS: qPCR")
    oc._critique_tailored("md", "T", "C", "jd", source_resume="RESUME")
    low = captured["prompt"].lower()
    assert "fabrication" in low
    assert "below 0.5" in low
