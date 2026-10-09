# tests/test_reconcile.py
from jobbot.apply_questions import FormQuestion
from jobbot.reconcile import reconcile


def _by_status(recs, status):
    return [r for r in recs if r.status == status]


def test_matched_when_label_present():
    planned = [FormQuestion(text="First Name", answer="Jane")]
    recs = reconcile(planned, ["First Name", "Last Name"])
    matched = _by_status(recs, "matched")
    assert len(matched) == 1
    assert matched[0].text == "First Name"
    assert matched[0].value == "Jane"
    assert matched[0].live_label == "First Name"


def test_planned_field_missing_on_page():
    planned = [FormQuestion(text="Portfolio URL", answer="http://x")]
    recs = reconcile(planned, ["First Name"])
    assert len(_by_status(recs, "missing_on_page")) == 1


def test_extra_field_on_page_not_planned():
    planned = [FormQuestion(text="First Name", answer="Jane")]
    recs = reconcile(planned, ["First Name", "Are you over 18?"])
    extra = _by_status(recs, "extra_on_page")
    assert len(extra) == 1
    assert extra[0].live_label == "Are you over 18?"


def test_matching_ignores_case_punctuation_and_required_star():
    planned = [FormQuestion(text="Email Address", answer="a@b.com")]
    recs = reconcile(planned, ["email address *"])
    assert _by_status(recs, "matched")[0].value == "a@b.com"


def test_live_label_matched_to_only_one_planned_question():
    from jobbot.apply_questions import FormQuestion
    planned = [FormQuestion(text="Email", answer="a@b"),
               FormQuestion(text="Email Address", answer="c@d")]
    recs = reconcile(planned, ["Email"])
    assert len([r for r in recs if r.status == "matched"]) == 1
    assert len([r for r in recs if r.status == "missing_on_page"]) == 1
