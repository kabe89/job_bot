"""A sponsorship question must never be answered with the work-authorization default.

Regression: Korro Bio asks "Do you now or will you in the future require employer
sponsorship for work authorization in the United States?". Because the phrase
contains "work authorization", it matched the `authoriz` rule ahead of the
`sponsor|visa` rule and was answered from `work_authorized` ("Yes") — telling the
employer the applicant needs sponsorship when he does not. Biohub's near-identical question
("require visa sponsorship") lacks the word and was answered correctly ("No").
"""
import re

import pytest

from jobbot.apply_questions import _SCREENING_PATTERNS


def _key_for(text: str) -> str:
    """Mirror the resolution in fill_answers: first matching pattern wins."""
    for pat, key in _SCREENING_PATTERNS:
        if re.search(pat, text, re.I):
            return key
    return ""


SPONSORSHIP_QUESTIONS = [
    # The regression case: sponsorship AND work authorization in one sentence.
    "Do you now or will you in the future require employer sponsorship for "
    "work authorization in the United States?",
    # The phrasing that already resolved correctly — must not regress.
    "Do you now or in the future require visa sponsorship to continue working "
    "in the United States?",
    "Will you now or in the future require sponsorship for employment visa status?",
]

WORK_AUTH_QUESTIONS = [
    "Are you currently eligible to work in the United States of America?",
    "Do you have the legal right to work in the United States?",
    "Are you legally authorized to work in the United States?",
]


@pytest.mark.parametrize("text", SPONSORSHIP_QUESTIONS)
def test_sponsorship_questions_use_the_sponsorship_default(text):
    assert _key_for(text) == "requires_sponsorship", (
        f"{text!r} resolved to the wrong screening default — answering it from "
        f"work_authorized inverts the meaning."
    )


@pytest.mark.parametrize("text", WORK_AUTH_QUESTIONS)
def test_plain_work_authorization_questions_still_use_work_authorized(text):
    assert _key_for(text) == "work_authorized"
