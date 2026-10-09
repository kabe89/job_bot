"""Regression tests for `_ai_answer` reliability on large option-heavy forms.

"""
from jobbot import apply_questions as aq
from jobbot.apply_questions import FormQuestion


def test_norm_key_strips_numbering_and_options_hint():
    a = aq._norm_key("4. Are you willing to relocate? (You MUST answer with "
                     "exactly one of: Yes | No)")
    b = aq._norm_key("Are you willing to relocate?")
    assert a == b


def test_default_for_only_fires_on_how_did_you_hear():
    heard = FormQuestion(text="How did you hear about this job?",
                         options=["LinkedIn", "Company website", "Other"])
    assert aq._default_for(heard) == "Company website"
    # A substantive question must NOT get a silent default.
    subst = FormQuestion(text="Are you authorized to work in the US?",
                         options=["Yes", "No"])
    assert aq._default_for(subst) == ""


def test_ai_answer_recovers_drifted_keys(monkeypatch):
    # The batch returns the answer under a drifted key; tolerant matching must
    # still land it on the right question.
    q = FormQuestion(text="Are you willing to relocate?", options=["Yes", "No"])

    def fake_batch(resume, title, company, jd, formatted):
        return {"1. Are you willing to relocate?": "Yes"}

    import jobbot.ai_client as gc
    monkeypatch.setattr(gc, "answer_application_questions", fake_batch)
    aq._ai_answer([q], "resume", "Scientist", "Recursion", "jd")
    assert q.answer == "Yes"
    assert q.answer_source == "ai"


def test_ai_answer_retries_blank_option_selects(monkeypatch):
    # Big batch returns blank for the select; a focused single-question retry
    # recovers a real choice.
    q = FormQuestion(text="Which site do you prefer?",
                     options=["Salt Lake City", "New York City"])
    calls = {"n": 0}

    def fake(resume, title, company, jd, formatted):
        calls["n"] += 1
        if calls["n"] == 1:
            return {q.text: ""}                     # batch call: blank
        return {formatted[0]: "New York City"}      # retry call: real answer

    import jobbot.ai_client as gc
    monkeypatch.setattr(gc, "answer_application_questions", fake)
    aq._ai_answer([q], "resume", "Scientist", "Recursion", "jd")
    assert q.answer == "New York City"
    assert calls["n"] == 2  # batch + one retry


def test_ai_answer_defaults_how_did_you_hear_when_unanswered(monkeypatch):
    q = FormQuestion(text="How did you hear about this position?",
                     required=True,
                     options=["LinkedIn", "Company website", "A friend"])

    def fake(resume, title, company, jd, formatted):
        return {}  # model never answers it, batch or retry

    import jobbot.ai_client as gc
    monkeypatch.setattr(gc, "answer_application_questions", fake)
    aq._ai_answer([q], "resume", "Scientist", "Recursion", "jd")
    assert q.answer == "Company website"
    assert q.answer_source == "ai-default"
