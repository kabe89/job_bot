"""Interactive gap-filling: the bot asks instead of guessing."""
from __future__ import annotations

import pytest

from jobbot.apply_fill import Gap, Reply, fill_gaps, gaps, validate
from jobbot.apply_questions import FormQuestion


def _q(**kw):
    kw.setdefault("text", "Q")
    return FormQuestion(**kw)


# ---------- gap detection ----------

def test_gaps_reports_blanks_and_confirmations_with_reasons():
    blank = _q(text="Blank", needs_user=True)
    confirm = _q(text="Confirm", answer="Yes", needs_review=True)
    done = _q(text="Done", answer="Yes", answer_source="answer_bank")

    found = gaps([blank, confirm, done])

    assert [g.question.text for g in found] == ["Blank", "Confirm"]
    assert found[0].reason == "blank"
    assert found[1].reason == "confirm"
    # index points back into the original list so answers can be written back
    assert [g.index for g in found] == [0, 1]


# ---------- answer validation ----------

def test_validate_snaps_a_loose_reply_onto_a_real_option():
    q = _q(qtype="select", options=["Yes", "No"])
    value, err = validate(q, "yes")
    assert value == "Yes"
    assert err == ""


def test_validate_accepts_a_numeric_option_pick():
    q = _q(qtype="select", options=["Red", "Green", "Blue"])
    value, err = validate(q, "3")
    assert value == "Blue"
    assert err == ""


def test_validate_prefers_a_literal_numeric_option_over_its_position():
    # "How many years?" dropdowns are numeric, so a bare "2" is the option
    # itself, not the second option. Reading it positionally submitted "1".
    q = _q(qtype="select", options=["0", "1", "2", "3"])
    value, err = validate(q, "2")
    assert value == "2"
    assert err == ""


def test_validate_accepts_a_year_option_typed_literally():
    q = _q(qtype="select", options=["2024", "2025", "2026"])
    value, err = validate(q, "2026")
    assert value == "2026"
    assert err == ""


def test_validate_rejects_a_reply_matching_no_option():
    q = _q(qtype="select", options=["Yes", "No"])
    value, err = validate(q, "chartreuse")
    assert value == ""
    assert "Yes" in err and "No" in err       # error lists the real options


def test_validate_accepts_any_text_when_there_are_no_options():
    q = _q(qtype="textarea")
    value, err = validate(q, "  a thoughtful answer  ")
    assert value == "a thoughtful answer"
    assert err == ""


def test_validate_rejects_an_empty_reply_on_a_required_question():
    q = _q(required=True)
    value, err = validate(q, "")
    assert value == ""
    assert "required" in err.lower()


# ---------- the fill loop ----------

def test_fill_gaps_writes_answers_and_clears_the_flags():
    blank = _q(text="Relocate?", qtype="select", options=["Yes", "No"],
               required=True, needs_user=True)
    questions = [blank]

    report = fill_gaps(questions, ask=lambda g: Reply(value="yes"))

    assert blank.answer == "Yes"
    assert blank.answer_source == "user"
    assert blank.needs_user is False
    assert blank.needs_review is False
    assert report["answered"] == 1


def test_fill_gaps_reprompts_until_the_reply_is_valid():
    q = _q(qtype="select", options=["Yes", "No"], required=True, needs_user=True)
    replies = iter([Reply(value="maybe"), Reply(value="banana"), Reply(value="no")])

    report = fill_gaps([q], ask=lambda g: next(replies))

    assert q.answer == "No"
    assert report["answered"] == 1


def test_fill_gaps_honours_skip_and_leaves_the_gap_open():
    q = _q(needs_user=True, required=True)
    report = fill_gaps([q], ask=lambda g: Reply(skip=True))

    assert q.answer == ""
    assert q.needs_user is True          # still a gap
    assert report["answered"] == 0
    assert report["skipped"] == 1


def test_fill_gaps_stops_early_on_quit():
    a = _q(text="A", needs_user=True)
    b = _q(text="B", needs_user=True)

    report = fill_gaps([a, b], ask=lambda g: Reply(quit=True))

    assert report["answered"] == 0
    assert a.needs_user is True and b.needs_user is True
    assert report["quit"] is True


def test_fill_gaps_remembers_answers_when_asked():
    q = _q(text="How did you hear about us?", needs_user=True)
    saved: list[tuple[str, str]] = []

    report = fill_gaps([q], ask=lambda g: Reply(value="Example State University", remember=True),
                       remember=lambda text, ans: saved.append((text, ans)))

    assert saved == [("How did you hear about us?", "Example State University")]
    assert report["remembered"] == 1


def test_fill_gaps_does_not_remember_unless_asked():
    q = _q(text="Why this role?", needs_user=True)
    saved: list = []

    fill_gaps([q], ask=lambda g: Reply(value="Because."),
              remember=lambda text, ans: saved.append((text, ans)))

    assert saved == []


def test_confirming_an_ai_guess_keeps_it_and_marks_it_reviewed():
    """Pressing enter on a held AI answer accepts it as-is."""
    q = _q(answer="Yes", answer_source="ai", required=True, needs_review=True)

    report = fill_gaps([q], ask=lambda g: Reply(accept=True))

    assert q.answer == "Yes"
    assert q.needs_review is False
    assert q.answer_source == "user"     # a human stands behind it now
    assert report["answered"] == 1


def test_the_rejected_guess_is_offered_to_the_prompt():
    """A value we couldn't use becomes the suggested default, so the user can
    accept it rather than retyping."""
    q = _q(qtype="select", options=["New York City", "Salt Lake City"],
           needs_user=True, guess="Springfield, IL")
    seen: list[Gap] = []

    fill_gaps([q], ask=lambda g: (seen.append(g), Reply(value="Salt Lake City"))[1])

    assert seen[0].suggestion == "Springfield, IL"


def test_recheck_demotes_a_stored_answer_that_is_not_a_valid_option():
    """Kits built before the snapping fix carry answers the form never
    offered — the gender-in-a-transgender-field case."""
    bad = _q(text="I identify as transgender:", qtype="select",
             options=["Yes", "No", "I don't wish to answer"],
             answer="Male", answer_source="answer_bank")
    good = _q(text="Gender", qtype="select", options=["Male", "Female"],
              answer="Male", answer_source="answer_bank")
    free = _q(text="Why us?", answer="Because.", answer_source="ai")

    from jobbot.apply_fill import recheck
    changed = recheck([bad, good, free])

    assert changed == 1
    assert bad.answer == ""
    assert bad.guess == "Male"        # kept, so the prompt can show it
    assert bad.needs_user is True
    assert good.answer == "Male"      # valid answers untouched
    assert free.answer == "Because."  # free text has nothing to validate against


def test_fill_gaps_is_a_no_op_when_nothing_needs_attention():
    q = _q(answer="Yes", answer_source="answer_bank")
    calls = []
    report = fill_gaps([q], ask=lambda g: calls.append(g) or Reply(skip=True))

    assert calls == []
    assert report["answered"] == 0
