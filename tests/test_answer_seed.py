from jobbot.answer_seed import (current_role, deterministic_entries,
                                expected_graduation, highest_degree,
                                inferred_entries, total_years_experience)

RESUME = """
# Jane Doe

## Education
Ph.D. in Computer Science, Apex Institute of Technology, expected 2027-01
B.S. in Software Engineering, Example State University, 2018-05

## Experience
Graduate Research Assistant, Apex Tech, 2020-08 - present
Laboratory Technician, Acme Research, 2018-06 - 2020-07

## Skills
Python, distributed systems, cloud computing
"""


def test_highest_degree_picks_the_doctorate():
    assert highest_degree(RESUME) == "PhD"


def test_highest_degree_returns_empty_when_absent():
    assert highest_degree("No schooling listed here.") == ""


def test_total_years_counts_from_the_earliest_role():
    # 2018-06 to now (2026) is 8 years. Verified as an int in a sane range so a
    # parse failure can never emit "58 years of experience" to an employer.
    years = total_years_experience(RESUME, now_year=2026)
    assert years == 8


def test_total_years_rejects_an_implausible_span():
    assert total_years_experience("Worked since 1802.", now_year=2026) == 0


def test_expected_graduation_is_the_future_dated_degree():
    assert expected_graduation(RESUME, now_year=2026) == "2027-01"


def test_expected_graduation_is_empty_when_every_degree_is_past():
    assert expected_graduation("B.S. Biochemistry, 2018-05", now_year=2026) == ""


def test_current_role_is_the_one_marked_present():
    assert current_role(RESUME) == ("Graduate Research Assistant", "Apex Tech")


def test_current_role_is_empty_when_nothing_is_current():
    assert current_role("Technician, Acme Research, 2018-06 - 2020-07") == ("", "")


def test_entries_are_bank_shaped_and_match_real_phrasings():
    import re
    entries = deterministic_entries(RESUME, skills=["Python"], now_year=2026)
    by_answer = {e["answer"]: e["match"] for e in entries}
    assert "PhD" in by_answer
    for phrasing in ["Highest degree earned", "What is your highest level of education?"]:
        assert re.search(by_answer["PhD"], phrasing, re.I), phrasing
    assert re.search(by_answer["2027-01"], "Expected graduation date", re.I)
    assert re.search(by_answer["Graduate Research Assistant"],
                     "What is your current job title?", re.I)


def test_a_skill_question_answers_yes_only_for_listed_skills():
    import re
    entries = deterministic_entries(RESUME, skills=["Python"])
    pats = [e["match"] for e in entries if e["answer"] == "Yes"]
    assert any(re.search(p, "Do you have experience with Python?", re.I)
               for p in pats)
    assert not any(re.search(p, "Do you have experience with Fortran?", re.I)
                   for p in pats)


def test_a_long_skill_name_cannot_hijack_a_how_many_years_question():
    # Regression: the "Yes" skill pattern used to also match "How many years of
    # experience with X?", and _bank_lookup's longest-pattern tie-break handed it
    # the win for any skill name over ~39 chars -- 16 of the user's 66 real skills.
    # A free-text years field would have received "Yes", unreviewed, on a real
    # application.
    import re
    long_skill = "Distributed systems consensus protocols (Raft / Paxos / PBFT / Zab)"
    entries = deterministic_entries(RESUME, skills=[long_skill], now_year=2026)
    yes_pats = [e["match"] for e in entries if e["answer"] == "Yes"]
    assert yes_pats, "expected a skill entry"
    quantity = f"How many years of experience do you have with {long_skill}?"
    assert not any(re.search(p, quantity, re.I) for p in yes_pats)
    # ...but the plain yes/no phrasing must still work.
    plain = f"Do you have experience with {long_skill}?"
    assert any(re.search(p, plain, re.I) for p in yes_pats)


def test_an_inferred_estimate_is_staged_not_active():
    entries = inferred_entries(RESUME, ["Python"], ask=lambda p: "5",
                               now_year=2026)
    assert entries and all(e["status"] == "pending" for e in entries)


def test_heuristic_extractors_are_staged_not_auto_active():
    # years-of-experience (min of every 4-digit year -- a citation year fools it)
    # and current-role (a comma-split of any "...present" line) are heuristics,
    # not proofs. They must NOT be written active into a verified, employer-
    # submittable source; they are staged for the same human review the LLM
    # estimates get. Degree, graduation and skills are genuinely deterministic
    # and stay active.
    entries = deterministic_entries(RESUME, skills=["Python"], now_year=2026)
    status = {e["answer"]: e.get("status", "active") for e in entries}
    assert status["8"] == "pending"                          # total years
    assert status["Graduate Research Assistant"] == "pending"  # current title
    assert status["Apex Tech"] == "pending"                        # current employer
    assert status["PhD"] == "active"                         # degree
    assert status["2027-01"] == "active"                     # graduation
    assert status["Yes"] == "active"                         # a listed skill


def test_an_estimate_beyond_total_experience_is_dropped():
    # Total experience is 8 years. A model claiming 30 years of Python is
    # arithmetically impossible, and it would be submitted unreviewed.
    assert inferred_entries(RESUME, ["Python"], ask=lambda p: "30",
                            now_year=2026) == []


def test_a_skill_absent_from_the_resume_is_dropped():
    assert inferred_entries(RESUME, ["Fortran"], ask=lambda p: "3",
                            now_year=2026) == []


def test_an_unparseable_reply_is_dropped():
    assert inferred_entries(RESUME, ["Python"], ask=lambda p: "quite a while",
                            now_year=2026) == []
