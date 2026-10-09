"""Personal-info sheet -> auto-answers.

The user maintains a plain-text "My Information.txt" (label / value lines, an
exported filled application) to populate answers the .env identity fields don't
cover: address lines, EEO self-ID, "how did you hear", work authorization. These
tests pin the parser and its integration into the answer pipeline so a Workday
(or any ATS) standard question set fills from the sheet instead of falling to
needs_user.
"""
from jobbot import personal_info as pi
from jobbot.apply_questions import FormQuestion, answer_questions, _WD_RACE_OPTIONS

SHEET = """My Information
How Did You Hear About Us?
University Job Board
Legal Name
Jane Doe
Address
123 Main Street
Springfield, IL 62701
United States of America
Email Address
jane.doe@example.com
Phone
+1 (555) 555-0199 (Cell)
My Experience
Work Experience 1
Job Title
Research Assistant
Company
Example State University
Education
Education 1
School or University
Example State University
Degree
Doctor of Philosophy (PhD)
Field of Study
Biochemistry
Resume/CV/Cover Letter
cover_letter.txt
Application Questions
1. Are you legally entitled to work?
Yes
Voluntary Disclosures
Gender
Male
Race/Ethnicity
White (United States of America)
Veteran Status
I am NOT a protected veteran
Please check one of the boxes below:
Yes, I have a disability, or have had one in the past
"""


def _write(tmp_path):
    p = tmp_path / "My Information.txt"
    p.write_text(SHEET, encoding="utf-8")
    return str(p)


def test_load_parses_identity_and_address(tmp_path):
    info = pi.load_personal_info(_write(tmp_path))
    assert info["first_name"] == "Jane"
    assert info["last_name"] == "Doe"
    assert info["email"] == "jane.doe@example.com"
    assert info["phone"].startswith("+1")
    assert info["address_line1"] == "123 Main Street"
    assert info["city"] == "Springfield"
    assert info["state"] == "IL"
    assert info["postal_code"] == "62701"
    assert info["country"] == "United States of America"


def test_load_parses_eeo_and_how_heard(tmp_path):
    info = pi.load_personal_info(_write(tmp_path))
    assert info["gender"] == "Male"
    assert "white" in info["race"].lower()
    assert "not a protected veteran" in info["veteran_status"].lower()
    assert "disability" in info["disability_status"].lower()
    assert info["how_heard"] == "University Job Board"


def test_missing_file_is_empty(tmp_path):
    assert pi.load_personal_info(str(tmp_path / "nope.txt")) == {}
    assert pi.resume_context(str(tmp_path / "nope.txt")) == ""


def test_resume_context_keeps_experience_strips_form_noise(tmp_path):
    ctx = pi.resume_context(_write(tmp_path))
    # Resume-relevant facts are kept.
    assert "Research Assistant" in ctx
    assert "Example State University" in ctx
    assert "Doctor of Philosophy (PhD)" in ctx
    assert "Biochemistry" in ctx
    assert "jane.doe@example.com" in ctx
    # Form / EEO / screening noise is stripped.
    assert "Voluntary Disclosures" not in ctx
    assert "Veteran" not in ctx
    assert "disability" not in ctx.lower()
    assert "Application Questions" not in ctx
    assert "University Job Board" not in ctx          # the "how did you hear" answer
    assert "Cover Letter" not in ctx          # uploaded-file listing section


def test_bank_entries_fill_workday_question_set(tmp_path, monkeypatch):
    """The parsed sheet flows through answer_questions and fills the address +
    EEO fields that .env/config don't cover — no AI, no needs_user."""
    monkeypatch.setattr("jobbot.config.settings.personal_info_path", _write(tmp_path))
    qs = [
        FormQuestion(text="Address Line 1", qtype="text", kind="identity"),
        FormQuestion(text="City", qtype="text", kind="identity"),
        FormQuestion(text="State", qtype="text", kind="identity"),
        FormQuestion(text="Postal Code", qtype="text", kind="identity"),
        FormQuestion(text="Gender", qtype="select",
                     options=["Male", "Female", "I do not wish to answer"], kind="eeo"),
        FormQuestion(text="Race/Ethnicity", qtype="select",
                     options=_WD_RACE_OPTIONS, kind="eeo"),
        FormQuestion(text="Veteran Status", qtype="select",
                     options=["I am not a protected veteran",
                              "I identify as one or more of the classifications of a protected veteran",
                              "I do not wish to answer"], kind="eeo"),
    ]
    answer_questions(qs, resume="", job_title="Scientist", company="Acme",
                     job_description="", use_ai=False)
    by = {q.text: q for q in qs}
    assert by["Address Line 1"].answer == "123 Main Street"
    assert by["City"].answer == "Springfield"
    assert by["State"].answer == "IL"
    assert by["Postal Code"].answer == "62701"
    assert by["Gender"].answer == "Male"
    assert "White" in by["Race/Ethnicity"].answer
    assert by["Veteran Status"].answer == "I am not a protected veteran"
    for q in qs:
        assert not q.needs_user, f"{q.text} should be filled from the sheet"
