import pytest

from jobbot.resume_facts import ResumeFacts, extract_from_markdown, save

# This fixture doubles as the parser's format spec, so its STRUCTURE mirrors the
# real resume exactly: "### Title <em dash> org chain, City, ST <middle dot>
# M/YYYY <en dash> M/YYYY", education as bullets with NO spaces around the range
# dash, an experience heading that does not read "## Experience", and a bare-year
# entry. Only the identity strings inside it are fake.
#
# An earlier version of this file mirrored a resume shape that was invented
# rather than observed, and it passed while the parser extracted zero rows from
# the real document. Do not "tidy" the punctuation here.
SAMPLE = """
# Alex Rivera, Ph.D. Candidate
Springfield, ST · alex.rivera@example.com · +1-555-555-0100 · linkedin.com/in/alexrivera

## Summary
Software engineer with doctoral training in distributed systems.

## Education
- **Ph.D. Computer Science** (Expected 2027) — Springfield State University (SYS), Springfield, ST. Doctoral Candidate, 1/2021–Present. Advisor: Dr. A. Example.
- **B.S. Software Engineering, Minor in Mathematics** — Springfield State University (SYS), Springfield, ST, 8/2017–11/2020. Academic Honors.

## Technical Skills
- **Systems & Architecture**: Distributed systems, microservices design

## Research & Professional Experience

### Research Assistant — Dr. A. Example's Lab, Computer Science Department, Springfield State University (SYS), Springfield, ST · 1/2021 – Present
- Developing distributed consensus protocols
- Applied performance modeling to guide system architecture

### Teaching Assistant — Computer Systems Lab, Springfield State University (SYS), Springfield, ST · 2025
- Taught students core systems programming skills.

### Senior Operations Manager — Office of Sustainability, Springfield State University (SYS), Springfield, ST · 8/2018 – 11/2020
- Managed the data-analysis division for campus energy data.

## Publications
1. Rivera, A. et al. *A paper title.* Journal, 2025.
"""


def test_extracts_every_experience_row():
    """Row count is the assertion that would have caught the invented format.

    'Conservative' must never quietly degrade into 'extracts nothing'.
    """
    facts = extract_from_markdown(SAMPLE)
    assert len(facts.employment) == 3
    assert [e.title for e in facts.employment] == [
        "Research Assistant", "Teaching Assistant", "Senior Operations Manager"]


def test_employer_keeps_the_org_chain_and_splits_off_the_location():
    # Which link in the chain belongs in an ATS "Employer" box is a judgment
    # call, so the parser keeps the whole chain rather than silently picking
    # one. The human corrects it in `jobbot facts review`.
    first = extract_from_markdown(SAMPLE).employment[0]
    assert first.employer == (
        "Dr. A. Example's Lab, Computer Science Department, "
        "Springfield State University (SYS)")
    assert first.location == "Springfield, ST"


def test_normalizes_slash_dates_to_yyyy_mm():
    facts = extract_from_markdown(SAMPLE)
    assert facts.employment[0].start == "2021-01"
    assert facts.employment[2].start == "2018-08"
    assert facts.employment[2].end == "2020-11"


def test_present_becomes_current_with_no_end_date():
    first = extract_from_markdown(SAMPLE).employment[0]
    assert first.current is True
    assert first.end == ""


def test_bare_year_keeps_the_row_but_leaves_dates_empty():
    """Never invent a month.

    A fabricated start month is a false statement about the user's history on a
    real job application; a blank one is a gap they fill in during review.
    """
    ta = extract_from_markdown(SAMPLE).employment[1]
    assert ta.title == "Teaching Assistant"
    assert ta.start == ""
    assert ta.end == ""
    assert ta.current is False


def test_extracts_education_bullets_with_no_spaces_around_the_dash():
    facts = extract_from_markdown(SAMPLE)
    assert len(facts.education) == 2
    phd, bs = facts.education
    assert phd.degree == "Ph.D. Computer Science"
    assert phd.school == "Springfield State University (SYS)"
    assert phd.location == "Springfield, ST"
    assert phd.start == "2021-01"
    assert phd.end == ""
    assert bs.start == "2017-08"
    assert bs.end == "2020-11"


def test_degree_containing_a_comma_survives():
    bs = extract_from_markdown(SAMPLE).education[1]
    assert bs.degree == "B.S. Software Engineering, Minor in Mathematics"


def test_identity_is_captured_from_a_middle_dot_contact_line():
    ident = extract_from_markdown(SAMPLE).identity
    assert ident["email"] == "alex.rivera@example.com"
    assert ident["linkedin"] == "linkedin.com/in/alexrivera"
    assert "Alex Rivera" in ident["name"]


def test_bullets_are_captured():
    facts = extract_from_markdown(SAMPLE)
    assert any("distributed consensus" in b for b in facts.employment[0].bullets)
    # A publications entry is not a bullet of the last job.
    assert not any("A paper title" in b for b in facts.employment[2].bullets)


def test_extracted_facts_are_never_pre_marked_reviewed():
    """A parser must not silently decide when the user left a job."""
    assert extract_from_markdown(SAMPLE).reviewed is False


def test_extractor_output_survives_its_own_validator(tmp_path, monkeypatch):
    """The extractor and save() must agree on date shape, or neither works."""
    import jobbot.resume_facts as rf
    monkeypatch.setattr(rf, "_path", lambda: tmp_path / "facts.json")
    facts = extract_from_markdown(SAMPLE)
    save(facts)
    assert len(rf.load().employment) == 3


def test_unmatched_lines_are_dropped_not_guessed():
    facts = extract_from_markdown(
        "## Research & Professional Experience\n\n### Just a title with no dates\n")
    assert facts.employment == []


@pytest.mark.parametrize("md", ["", "   ", "# Name only\n"])
def test_empty_input_is_not_an_error(md):
    facts = extract_from_markdown(md)
    assert facts.employment == []
    assert facts.education == []
    assert facts.reviewed is False


def test_identity_is_empty_when_the_document_carries_none():
    # Asserted against a literal, not against another call to the function
    # under test: an expected value built by the code being tested cannot fail.
    assert extract_from_markdown("## Education\n- nothing\n").identity == {}
