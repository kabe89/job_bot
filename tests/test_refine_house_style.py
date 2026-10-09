"""_refine_tailored told the model to "Keep the house style template's structure,
section order, and bullet form" -- but never injected the template. It referenced
a document the model could not see, so every refinement round was free to drift
off the house standard that tailor_resume had just applied.

The refine pass also judged fabrication against the ORIGINAL RESUME alone, while
the drafter is given resume + profile (see test_critique_source_of_truth.py). It
would therefore strip real, profile-sourced skills as "fabrication".
"""
import pytest

from jobbot import ollama_client as oc


@pytest.fixture
def captured(monkeypatch):
    box = {}

    def fake_generate(prompt, **kw):
        box["prompt"] = prompt
        return "# revised resume"

    monkeypatch.setattr(oc, "_generate", fake_generate)
    return box


@pytest.fixture(autouse=True)
def _style(monkeypatch):
    monkeypatch.setattr(oc, "_template_block",
                        lambda kind="resume": "\n\nHOUSE STYLE TEMPLATE — SENTINEL_STYLE")


def _crit(**kw):
    base = {"score": 0.4, "issues": [], "missing_keywords": [], "fabrication_risks": []}
    base.update(kw)
    return base


def test_refine_injects_the_house_style_template(captured):
    oc._refine_tailored("draft", _crit(), "resume", "T", "C", "jd")
    assert "SENTINEL_STYLE" in captured["prompt"], \
        "refine tells the model to follow the house style but never showed it"


def test_refine_includes_the_profile_as_source_of_truth(captured, monkeypatch):
    monkeypatch.setattr(oc, "_profile_context", lambda: "SKILLS: qPCR, transfection")
    oc._refine_tailored("draft", _crit(), "RESUME TEXT", "T", "C", "jd")
    prompt = captured["prompt"]
    assert "RESUME TEXT" in prompt
    assert "qPCR, transfection" in prompt, \
        "refine would strip real profile-sourced skills as fabrication"


def test_refine_still_prioritises_removing_fabrication(captured, monkeypatch):
    monkeypatch.setattr(oc, "_profile_context", lambda: "")
    oc._refine_tailored("draft", _crit(fabrication_risks=["invented qPCR"]),
                        "resume", "T", "C", "jd")
    low = captured["prompt"].lower()
    assert "fabrication" in low
    assert "invented qpcr" in low


def test_refine_passes_the_critique_through(captured, monkeypatch):
    monkeypatch.setattr(oc, "_profile_context", lambda: "")
    oc._refine_tailored("draft",
                        _crit(issues=["SENTINEL_ISSUE"],
                              missing_keywords=["SENTINEL_KEYWORD"]),
                        "resume", "T", "C", "jd")
    assert "SENTINEL_ISSUE" in captured["prompt"]
    assert "SENTINEL_KEYWORD" in captured["prompt"]


def test_refine_survives_an_unreadable_profile(captured, monkeypatch):
    def boom():
        raise OSError("profile gone")
    monkeypatch.setattr(oc, "_profile_context", boom)
    assert oc._refine_tailored("draft", _crit(), "resume", "T", "C", "jd")


def test_refine_works_with_no_style_template(captured, monkeypatch):
    monkeypatch.setattr(oc, "_template_block", lambda kind="resume": "")
    monkeypatch.setattr(oc, "_profile_context", lambda: "")
    out = oc._refine_tailored("draft", _crit(), "resume", "T", "C", "jd")
    assert out == "# revised resume"
    assert "SENTINEL_STYLE" not in captured["prompt"]
