from jobbot.answer_memory import canonical_key, classify_kind
from jobbot.apply_questions import FormQuestion


def test_canonical_key_collapses_company_specific_phrasing():
    a = canonical_key("Why do you want to work at Acme Corp?")
    b = canonical_key("why do you want to work at  Genentech!")
    assert a == b  # same slot regardless of company / punctuation / case


def test_canonical_key_distinguishes_different_questions():
    assert canonical_key("Salary expectations?") != canonical_key("Earliest start date?")


def test_identity_and_screening_are_static():
    assert classify_kind(FormQuestion(text="First Name", kind="identity")) == "static"
    assert classify_kind(FormQuestion(
        text="Are you authorized to work in the US?", kind="screening")) == "static"
    assert classify_kind(FormQuestion(text="Gender", kind="eeo")) == "static"


def test_identity_fields_do_not_collide_with_ordinal_questions():
    assert canonical_key("First Name") != canonical_key("First Language")
    assert canonical_key("Last Name") != canonical_key("Last Employer")
    # company name still collapses
    assert canonical_key("Why do you want to work at Acme Corp?") == \
        canonical_key("why do you want to work at Genentech!")


def test_title_case_after_plain_stopword_is_kept():
    # "a"/"the" before a Title-Case word must NOT strip it (only company prepositions do)
    assert "masters" in canonical_key("Do you have a Masters Degree")
    assert "degree" in canonical_key("Do you have a Masters Degree")


def test_essays_are_tailored():
    assert classify_kind(FormQuestion(
        text="Why are you a strong fit for this role?", qtype="textarea")) == "tailored"
