"""Fidelity checks for credentials, asserted absence, and altered date ranges.

Tests verifying that credentials and awards from source resumes survive tailoring,
and that negative assertions or altered date endpoints are caught.
"""
import pytest

from jobbot.source_fidelity import check_fidelity

_SOURCE = """
Education
- B.S. Computer Science - Example State University, Springfield, IL, 8/2017-11/2020.
  Academic Honors.
  Honor Roll: Fall 2017, Spring 2018, Fall 2019, Spring 2020, Fall 2020.

Awards & Honors
- Academic Honors - Example State University, Springfield, IL
- Engineering Achievement Award - Fall 2020
"""


# --- asserted absence ---------------------------------------------------------

@pytest.mark.parametrize("line", [
    "- None listed",
    "- none listed",
    "- N/A",
    "- None",
    "None listed",          # no bullet
    "  - None to report",   # indented
])
def test_a_draft_may_not_assert_it_has_no_awards(line):
    draft = f"### Awards & Certifications\n{line}\n"
    r = check_fidelity(_SOURCE, draft)
    assert not r.ok, f"draft asserting {line!r} under Awards was not flagged"
    assert any("assert" in f.lower() or "none" in f.lower()
               for f in r.fabrications + r.omissions)


def test_asserted_absence_only_matters_when_the_source_has_the_thing():
    """If the candidate genuinely has no awards, "None listed" is honest."""
    r = check_fidelity("Education\n- B.S. Computer Science, 8/2017-11/2020.\n",
                       "### Awards\n- None listed\n")
    assert r.ok


def test_the_word_none_in_ordinary_prose_is_not_an_assertion():
    """Guard against a checker that fires on any occurrence of "none"."""
    draft = ("### Awards\n- Engineering Achievement Award - Fall 2020\n- Academic Honors\n"
             "- Honor Roll: Fall 2017, Spring 2018, Fall 2019, Spring 2020, Fall 2020\n"
             "\n### Summary\nNone of the models required retraining.\n")
    r = check_fidelity(_SOURCE, draft)
    assert r.ok, f"ordinary prose flagged: {r.fabrications} {r.omissions}"


# --- dropped credentials ------------------------------------------------------

def test_a_dropped_named_award_is_caught():
    draft = ("### Awards\n- Academic Honors\n"
             "- Honor Roll: Fall 2017, Spring 2018, Fall 2019, Spring 2020, Fall 2020\n")
    r = check_fidelity(_SOURCE, draft)
    assert not r.ok
    assert any("Engineering Achievement Award" in o for o in r.omissions)


def test_a_dropped_honor_roll_is_caught():
    draft = "### Awards\n- Engineering Achievement Award - Fall 2020\n- Academic Honors\n"
    r = check_fidelity(_SOURCE, draft)
    assert not r.ok
    assert any("Honor Roll" in o for o in r.omissions)


def test_a_draft_keeping_every_credential_is_clean():
    draft = ("### Education\n- B.S. Computer Science, 8/2017-11/2020. Academic Honors.\n"
             "  Honor Roll: Fall 2017, Spring 2018, Fall 2019, Spring 2020, Fall 2020\n"
             "### Awards\n- Engineering Achievement Award - Fall 2020\n- Academic Honors\n")
    r = check_fidelity(_SOURCE, draft)
    assert r.ok, f"clean draft flagged: {r.omissions} {r.fabrications}"


def test_credentials_are_matched_case_insensitively():
    draft = ("### Education\n- B.S. Computer Science, 8/2017-11/2020. ACADEMIC HONORS.\n"
             "  HONOR ROLL: Fall 2017, Spring 2018, Fall 2019, Spring 2020, Fall 2020\n"
             "### Awards\n- ENGINEERING ACHIEVEMENT AWARD - Fall 2020\n")
    r = check_fidelity(_SOURCE, draft)
    assert r.ok


# --- altered date ranges ------------------------------------------------------

def test_moving_the_graduation_date_is_caught():
    """The defect: source 8/2017-11/2020, draft 08/2017 - 05/2020."""
    draft = ("### Education\n- B.S. Computer Science, 08/2017 - 05/2020. Academic Honors.\n"
             "  Honor Roll: Fall 2017, Spring 2018, Fall 2019, Spring 2020, Fall 2020\n"
             "### Awards\n- Engineering Achievement Award - Fall 2020\n")
    r = check_fidelity(_SOURCE, draft)
    assert not r.ok
    assert any("2020" in f for f in r.fabrications)


def test_a_zero_padded_but_faithful_range_is_accepted():
    """08/2017 == 8/2017. Reformatting is not altering."""
    draft = ("### Education\n- B.S. Computer Science, 08/2017 - 11/2020. Academic Honors.\n"
             "  Honor Roll: Fall 2017, Spring 2018, Fall 2019, Spring 2020, Fall 2020\n"
             "### Awards\n- Engineering Achievement Award - Fall 2020\n")
    r = check_fidelity(_SOURCE, draft)
    assert r.ok, f"a faithful reformat was flagged: {r.fabrications}"
