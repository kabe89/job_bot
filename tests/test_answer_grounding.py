import pytest

from jobbot import profile_context, resume_facts as rf
from jobbot.apply_questions import FormQuestion, gate_free_text


@pytest.fixture(autouse=True)
def _facts(tmp_path, monkeypatch):
    monkeypatch.setattr(rf, "_path", lambda: tmp_path / "facts.json")
    rf.save(rf.ResumeFacts(
        employment=[rf.Employment(employer="Acme Research",
                                  title="Research Assistant",
                                  location="Springfield, ST", start="2022-06",
                                  end="", current=True,
                                  bullets=["Optimized buffer conditions"])],
        education=[], identity={}, reviewed=True))
    profile_context.refresh()


def test_context_includes_employment_facts():
    ctx = profile_context.applicant_context()
    assert "Acme Research" in ctx
    assert "Research Assistant" in ctx


def test_unreviewed_facts_are_not_injected(tmp_path, monkeypatch):
    monkeypatch.setattr(rf, "_path", lambda: tmp_path / "unrev.json")
    rf.save(rf.ResumeFacts(
        employment=[rf.Employment(employer="Ghost Corp", title="X",
                                  start="2020-01", current=True)],
        education=[], identity={}, reviewed=False))
    profile_context.refresh()
    assert "Ghost Corp" not in profile_context.applicant_context()


def test_draft_that_invents_a_date_is_rejected():
    source = "Research Assistant, Acme Research, June 2022 - Present"
    draft = "I have worked at Acme Research since 2015."
    text, passed = gate_free_text(source, draft)
    assert passed is False


def test_faithful_draft_passes():
    source = "Research Assistant, Acme Research, June 2022 - Present"
    draft = "As a Research Assistant at Acme Research, I optimized buffers."
    text, passed = gate_free_text(source, draft)
    assert passed is True
    assert text == draft


def test_failed_gate_leaves_the_answer_blank_with_a_guess():
    """A low-confidence hold is blank plus q.guess and needs_user, never a
    populated answer that reads confident and is wrong."""
    from jobbot.apply_questions import apply_free_text_answer

    q = FormQuestion(text="Describe your experience", qtype="textarea")
    apply_free_text_answer(q, source="Acme Research, June 2022 - Present",
                           draft="I founded Acme Research in 1901.")
    assert q.answer == ""
    assert q.guess != ""
    assert q.needs_user is True


def test_gate_does_not_reject_a_forward_looking_date():
    from jobbot.apply_questions import gate_free_text

    source = "Research Assistant, Acme Research, June 2022 - Present"
    draft = "I could start in September 2026."
    text, passed = gate_free_text(source, draft)
    assert passed is True
    assert text == draft


_RESUME_TEXT = (
    "Jane Doe\njane@example.com\n\n"
    "## Experience\n"
    "### Research Assistant — Acme Research, Springfield, ST · 6/2022-Present\n"
    "- Optimized buffer conditions\n"
)


def test_ai_answer_wires_free_text_through_the_gate_on_pass(monkeypatch):
    """The real answering path (_ai_answer's free-text branch), not just the
    gate function in isolation, must apply gate_free_text before a draft can
    reach q.answer. No network/LLM call: the model call is stubbed."""
    from jobbot.apply_questions import _ai_answer
    import jobbot.ai_client as gc

    q = FormQuestion(text="Describe your experience", qtype="textarea")
    faithful = "As a Research Assistant at Acme Research, I optimized buffers."

    def fake(resume, title, company, jd, formatted):
        return {q.text: faithful}

    monkeypatch.setattr(gc, "answer_application_questions", fake)
    _ai_answer([q], _RESUME_TEXT, "Scientist", "Recursion", "jd")

    assert q.answer == faithful
    assert q.answer_source == "ai"
    assert q.needs_user is False


def test_ai_answer_wires_free_text_through_the_gate_on_fail(monkeypatch):
    """Same wiring, but the stubbed model returns a backdated draft: the gate
    must blank it rather than let a fabrication reach q.answer."""
    from jobbot.apply_questions import _ai_answer
    import jobbot.ai_client as gc

    q = FormQuestion(text="Describe your experience", qtype="textarea")
    backdated = "I have worked at Acme Research since 2015."

    def fake(resume, title, company, jd, formatted):
        return {q.text: backdated}

    monkeypatch.setattr(gc, "answer_application_questions", fake)
    _ai_answer([q], _RESUME_TEXT, "Scientist", "Recursion", "jd")

    assert q.answer == ""
    assert q.guess == backdated
    assert q.needs_user is True


# ---------- Fix round 1, Finding 1: an empty source must fail closed ------
#
# gate_free_text("", draft) used to return passed=True for ANY draft, because
# no check in source_fidelity or _fabricated_years has anything to compare
# against when the source is blank. This is reachable in production, not
# just a hypothetical: browser_apply.py fail-opens to base_resume = "" on
# any resume-load error and passes that straight through to _ai_answer.

def test_empty_source_fails_closed():
    text, passed = gate_free_text("", "I received the Engineering Achievement Award.")
    assert passed is False


def test_whitespace_only_source_fails_closed():
    text, passed = gate_free_text("   \n\t  ", "I received the Engineering Achievement Award.")
    assert passed is False


def test_apply_free_text_answer_blanks_when_source_is_empty():
    from jobbot.apply_questions import apply_free_text_answer

    q = FormQuestion(text="Describe your experience", qtype="textarea")
    apply_free_text_answer(q, source="", draft="I founded Acme Research in 1901.")
    assert q.answer == ""
    assert q.guess != ""
    assert q.needs_user is True


def test_ai_answer_blanks_free_text_when_resume_failed_to_load(monkeypatch):
    """Mirrors browser_apply.py's fail-open exactly: base_resume = "" on any
    load error, passed straight through to answer_questions -> _ai_answer.
    An unreadable resume must not let every free-text draft on that
    application sail through ungated."""
    from jobbot.apply_questions import _ai_answer
    import jobbot.ai_client as gc

    q = FormQuestion(text="Describe your experience", qtype="textarea")
    fabricated = "I received the Engineering Achievement Award for research excellence."

    def fake(resume, title, company, jd, formatted):
        return {q.text: fabricated}

    monkeypatch.setattr(gc, "answer_application_questions", fake)
    _ai_answer([q], "", "Scientist", "Recursion", "jd")  # resume="" like the fail-open

    assert q.answer == ""
    assert q.guess == fabricated
    assert q.needs_user is True


def test_empty_source_logs_a_distinct_reason_from_a_fidelity_defect(caplog):
    """The human debugging a stuck needs_user answer must be able to tell
    "couldn't verify, no source" apart from "verified, and it's wrong" --
    they call for different fixes (reload the resume vs. distrust the model).
    """
    import logging
    caplog.set_level(logging.INFO, logger="jobbot.apply_questions")

    caplog.clear()
    gate_free_text("", "I received the Engineering Achievement Award.")
    assert "no source" in caplog.text.lower()

    caplog.clear()
    gate_free_text("Research Assistant, Acme Research, June 2022 - Present",
                   "I have worked at Acme Research since 2015.")
    assert "backdated" in caplog.text.lower()
    assert "no source" not in caplog.text.lower()


# ---------- Fix round 1, Finding 2: the gate must not overstate itself -----

def test_gate_docstring_discloses_known_uncaught_fabrication_classes():
    """gate_free_text only detects omissions/alterations against the source
    (dropped credentials, altered ranges, backdated years, ...); it does NOT
    detect a wholly invented claim (a fabricated employer, title, degree,
    publication, award, or skill) that doesn't collide with one of those
    shapes. A caller reading passed=True must not mistake it for "no
    fabrication present" -- the docstring has to say so plainly, since this
    module feeds a real job application."""
    doc = gate_free_text.__doc__ or ""
    low = doc.lower()
    for phrase in ("employer", "degree", "publication", "award", "skill",
                  "forward-dated", "job title"):
        assert phrase in low, f"docstring silent on {phrase!r}"


# ---------- Fix round 2: gate on fabrications, not omissions --------------
#
# repair_fidelity (and therefore the round-1 gate) merges omissions and
# fabrications into one defect list. Omission checks (dropped DOI, dropped
# named credential) exist to compare a full tailored RESUME against a
# profile -- they are the wrong tool for a single free-text answer, which is
# not supposed to restate the candidate's entire publication list and every
# award just to avoid a false rejection. On the user's real resume (which
# has DOIs and named credentials), this silently blanked essentially every
# ordinary free-text answer. Placeholder DOI/credentials below -- never the
# user's real ones.

_DOI_CREDENTIAL_SOURCE = (
    "Research Assistant, Acme Research, Springfield, ST, June 2022 - Present\n"
    "Publications: 10.1234/example.5678\n"
    "Honors: Engineering Achievement Award, Academic Honors, Honor Roll\n"
)


def test_gate_ignores_omitted_publications_and_credentials():
    """A short, ordinary free-text answer that doesn't repeat the DOI or the
    named credentials must still pass -- omitting them is not a fabrication.
    This is the exact case Fix round 2 exists for."""
    draft = ("I'm drawn to this role because my work on distributed "
            "systems maps directly onto your team's pipeline.")
    text, passed = gate_free_text(_DOI_CREDENTIAL_SOURCE, draft)
    assert passed is True
    assert text == draft


def test_gate_still_rejects_a_genuine_fabrication_against_the_same_source():
    """Omitting the awards is fine; claiming there are none is not -- that is
    an asserted-absence FABRICATION (not an omission) and must still fail,
    even against the same DOI/credential-bearing source as the test above.

    The draft must be the EXACT line shape source_fidelity._check_asserted_
    absence matches (a whole line that is just "None."/"None listed."/etc.)
    -- an earlier version of this test used "Awards and honors: None.",
    which does not match that pattern and only "passed" under the
    round-1 gate because the OMITTED-credentials check (now deliberately
    ignored) happened to catch it too, for the wrong reason. Verified
    directly against jobbot.source_fidelity.check_fidelity before fixing:
    "Awards and honors: None." -> fabrications == [] (asserted_absence does
    not fire on it), while "None." -> one asserted-absence fabrication."""
    draft = "None."
    text, passed = gate_free_text(_DOI_CREDENTIAL_SOURCE, draft)
    assert passed is False
