"""Snapping a free-text answer onto a form's real options.

The rule the whole apply path depends on: an answer that does not genuinely
correspond to one of the offered options must come back empty, so the caller
turns it into a gap and asks the user. A confident-looking wrong option is
worse than a blank.
"""
from __future__ import annotations

from jobbot.apply_questions import _snap_to_options


# ---------- matches that must keep working ----------

def test_exact_option_matches_case_insensitively():
    assert _snap_to_options("yes", ["Yes", "No"]) == "Yes"


def test_qualified_yes_still_snaps_to_yes():
    assert _snap_to_options("Yes, I am authorized", ["Yes", "No"]) == "Yes"


def test_negative_answer_still_snaps_to_no():
    assert _snap_to_options("No I do not", ["Yes", "No"]) == "No"


def test_punctuation_differences_do_not_block_a_match():
    assert _snap_to_options("nonbinary", ["Male", "Female", "Non-binary"]) == "Non-binary"


def test_decline_answer_snaps_to_the_decline_option():
    assert _snap_to_options("prefer not to say",
                            ["Male", "Female", "I prefer not to disclose"]) \
        == "I prefer not to disclose"


# ---------- the ambiguity that must NOT be resolved silently ----------

def test_not_sure_is_not_read_as_no():
    # "no" is a substring of "not", which turned an explicit non-answer into a
    # definite No on the form.
    assert _snap_to_options("Not sure", ["Yes", "No"]) == ""


def test_unrelated_answer_returns_empty():
    assert _snap_to_options("chartreuse", ["Yes", "No"]) == ""
