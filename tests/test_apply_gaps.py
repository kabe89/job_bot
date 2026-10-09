"""Auto-apply gap handling: never submit a wrong answer, always ask instead.

Covers the three failure modes that made the apply queue fill questions badly:

  1. an over-broad answer-bank pattern bleeding one answer onto a different
     question ("gender" matching "transgender"),
  2. an answer that doesn't match any of the form's real options being written
     anyway instead of being flagged,
  3. an unverified AI guess on a required question going out unreviewed.
"""
from __future__ import annotations

import pytest

from jobbot.apply_questions import (
    FormQuestion,
    answer_questions,
    needs_attention,
    personal_info_bank_entries_safe,
)


def _ans(questions, **kw):
    kw.setdefault("resume", "Jane is a computational biologist.")
    kw.setdefault("job_title", "Scientist")
    kw.setdefault("company", "Acme Bio")
    kw.setdefault("job_description", "Do science.")
    kw.setdefault("use_ai", False)
    return answer_questions(questions, **kw)


# ---------- 1. over-broad bank patterns ----------

def test_gender_answer_does_not_bleed_onto_transgender_question(monkeypatch):
    """Regression: the personal-info sheet's `gender` pattern was r"gender",
    which also matches "I identify as transgender" — so the user's gender
    ("Male") was written as the answer to a Yes/No transgender question."""
    monkeypatch.setattr(
        "jobbot.apply_questions.load_answer_bank", lambda: [])
    monkeypatch.setattr(
        "jobbot.apply_questions._personal_info_entries",
        lambda: [{"match": r"(?<!trans)\bgender\b", "answer": "Male"}])

    gender = FormQuestion(text="I identify my gender as:", qtype="select",
                          options=["Male", "Female", "I don't wish to answer"])
    trans = FormQuestion(text="I identify as transgender:", qtype="select",
                         options=["Yes", "No", "I don't wish to answer"])
    _ans([gender, trans])

    assert gender.answer == "Male"
    # The transgender question must NOT inherit the gender answer. It is an EEO
    # question, so it correctly lands on the form's own decline option.
    assert trans.answer != "Male"
    assert trans.answer == "I don't wish to answer"
    assert trans.answer_source == "eeo-default"


def test_personal_info_gender_pattern_excludes_transgender():
    """The shipped pattern itself must not match 'transgender'."""
    import re
    from jobbot.personal_info import _FIELD_QUESTION_RE
    pat = _FIELD_QUESTION_RE["gender"]
    assert re.search(pat, "I identify my gender as:", re.I)
    assert not re.search(pat, "I identify as transgender:", re.I)


# ---------- 2. answers that don't match a real option ----------

def test_banked_answer_that_matches_no_option_is_escalated_not_written(monkeypatch):
    """A banked answer that can't snap onto one of the form's options is a gap,
    not an answer. Writing it anyway produced un-submittable forms."""
    monkeypatch.setattr("jobbot.apply_questions.load_answer_bank",
                        lambda: [{"match": r"favou?rite colour", "answer": "Chartreuse"}])
    monkeypatch.setattr("jobbot.apply_questions._personal_info_entries", lambda: [])

    q = FormQuestion(text="Favourite colour", qtype="select",
                     options=["Red", "Blue"], required=True)
    _ans([q])

    assert q.answer == ""            # nothing invalid written
    assert q.needs_user is True      # surfaced to the user
    assert q.guess == "Chartreuse"   # but the rejected value is kept as a seed


def test_identity_answer_that_matches_no_option_is_escalated(monkeypatch):
    monkeypatch.setattr("jobbot.apply_questions.load_answer_bank", lambda: [])
    monkeypatch.setattr("jobbot.apply_questions._personal_info_entries", lambda: [])

    q = FormQuestion(text="Location", qtype="select",
                     options=["New York City", "Salt Lake City"], required=True)
    _ans([q], identity_fields={"location": "Springfield, IL"})

    assert q.answer == ""
    assert q.needs_user is True
    assert q.guess == "Springfield, IL"


def test_valid_banked_answer_still_fills_normally(monkeypatch):
    monkeypatch.setattr("jobbot.apply_questions.load_answer_bank",
                        lambda: [{"match": r"favou?rite colour", "answer": "blue"}])
    monkeypatch.setattr("jobbot.apply_questions._personal_info_entries", lambda: [])

    q = FormQuestion(text="Favourite colour", qtype="select", options=["Red", "Blue"])
    _ans([q])

    assert q.answer == "Blue"        # snapped to the real option
    assert q.answer_source == "answer_bank"
    assert q.needs_user is False


def test_free_text_banked_answer_is_unaffected_by_option_snapping(monkeypatch):
    """Questions with no options have nothing to snap to — keep the raw value."""
    monkeypatch.setattr("jobbot.apply_questions.load_answer_bank",
                        lambda: [{"match": r"favou?rite colour", "answer": "Chartreuse"}])
    monkeypatch.setattr("jobbot.apply_questions._personal_info_entries", lambda: [])

    q = FormQuestion(text="Favourite colour", qtype="text")
    _ans([q])

    assert q.answer == "Chartreuse"
    assert q.needs_user is False


# ---------- 3. unverified AI guesses on required questions ----------

def test_ai_answer_on_required_question_is_flagged_for_review(monkeypatch):
    monkeypatch.setattr("jobbot.apply_questions.load_answer_bank", lambda: [])
    monkeypatch.setattr("jobbot.apply_questions._personal_info_entries", lambda: [])

    def fake_ai(batch, *a, **kw):
        for q in batch:
            q.answer, q.answer_source = "Yes", "ai"

    monkeypatch.setattr("jobbot.apply_questions._ai_answer", fake_ai)

    required = FormQuestion(text="Are you willing to relocate to Utah?",
                            qtype="select", options=["Yes", "No"], required=True)
    optional = FormQuestion(text="Anything else we should know?", qtype="textarea")
    answer_questions([required, optional], resume="r", job_title="t",
                     company="c", job_description="d", use_ai=True)

    assert required.answer == "Yes"
    assert required.needs_review is True     # answered, but confirm before submit
    assert required.needs_user is False      # not blank — just unverified
    assert optional.needs_review is False    # optional free-text may stand


def test_verified_sources_are_never_flagged_for_review(monkeypatch):
    monkeypatch.setattr("jobbot.apply_questions.load_answer_bank",
                        lambda: [{"match": r"work in the United States", "answer": "Yes"}])
    monkeypatch.setattr("jobbot.apply_questions._personal_info_entries", lambda: [])

    q = FormQuestion(text="Are you authorized to work in the United States?",
                     qtype="select", options=["Yes", "No"], required=True)
    _ans([q])

    assert q.answer == "Yes"
    assert q.answer_source == "answer_bank"
    assert q.needs_review is False


# ---------- needs_attention helper ----------

def test_needs_attention_collects_both_blanks_and_unverified_answers():
    blank = FormQuestion(text="A", needs_user=True)
    unverified = FormQuestion(text="B", answer="Yes", needs_review=True)
    fine = FormQuestion(text="C", answer="Yes", answer_source="answer_bank")

    got = needs_attention([blank, unverified, fine])

    assert [q.text for q in got] == ["A", "B"]


def test_a_remembered_answer_beats_a_broader_bank_entry(monkeypatch):
    """Ticking 'remember' must actually take effect: the question-specific
    entry it saves has to win over any broad pattern already in the bank."""
    import re as _re
    question = "How did you hear about this job?"
    monkeypatch.setattr("jobbot.apply_questions.load_answer_bank", lambda: [
        {"match": r"how did you (hear|find|learn)", "answer": "Example State University"},
        {"match": _re.escape(question), "answer": "LinkedIn"},
    ])
    monkeypatch.setattr("jobbot.apply_questions._personal_info_entries", lambda: [])

    q = FormQuestion(text=question, qtype="select",
                     options=["LinkedIn", "Referral", "Other"])
    _ans([q])

    assert q.answer == "LinkedIn"


def test_the_more_specific_bank_pattern_wins_when_several_match(monkeypatch):
    monkeypatch.setattr("jobbot.apply_questions.load_answer_bank", lambda: [
        {"match": r"salary", "answer": "Negotiable"},
        {"match": r"desired base salary for this role", "answer": "$150,000"},
    ])
    monkeypatch.setattr("jobbot.apply_questions._personal_info_entries", lambda: [])

    q = FormQuestion(text="Desired base salary for this role", qtype="text")
    _ans([q])

    assert q.answer == "$150,000"


def test_personal_info_entries_helper_is_fail_open(monkeypatch):
    """A broken personal-info sheet must not take the whole pipeline down."""
    def boom(*a, **kw):
        raise OSError("disk on fire")
    monkeypatch.setattr("jobbot.personal_info.personal_info_bank_entries", boom)
    assert personal_info_bank_entries_safe() == []
