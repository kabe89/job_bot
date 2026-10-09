"""Workday's CXS posting JSON returns `country` as a `{descriptor, id}` object,
not a bare string. Regression: that dict was stuffed straight into the Country
question's `answer`, then crashed downstream at `q.answer.strip()` (auto-apply
died at status=preparing with no recorded error). The extractor must coerce the
object to its display string.
"""
from jobbot import apply_questions as aq
from jobbot.apply_questions import _fetch_workday, answer_questions


_WD_URL = "https://innotech.wd1.myworkdayjobs.com/en-US/Careers/job/City/x_R48428"


def _posting(country):
    return {"jobPostingInfo": {"canApply": True, "title": "Associate Scientist",
                               "country": country}}


def test_workday_country_object_is_coerced_to_string(monkeypatch):
    monkeypatch.setattr(aq, "_workday_cxs_fetch",
                        lambda url: _posting({"descriptor": "United States of America",
                                              "id": "bc33aa3152ec42d4995f4791a106ed09"}))
    qs = _fetch_workday(_WD_URL)
    country_q = next(q for q in qs if q.text == "Country")
    assert isinstance(country_q.answer, str)
    assert country_q.answer == "United States of America"


def test_workday_country_plain_string_still_works(monkeypatch):
    monkeypatch.setattr(aq, "_workday_cxs_fetch",
                        lambda url: _posting("Canada"))
    qs = _fetch_workday(_WD_URL)
    country_q = next(q for q in qs if q.text == "Country")
    assert country_q.answer == "Canada"


def test_answer_questions_survives_workday_question_set(monkeypatch):
    """The end-to-end crash guard: answering a Workday-extracted set must not
    raise even when the raw posting carried a dict country."""
    monkeypatch.setattr(aq, "_workday_cxs_fetch",
                        lambda url: _posting({"descriptor": "United States of America",
                                              "id": "abc"}))
    qs = _fetch_workday(_WD_URL)
    # use_ai=False keeps it offline/deterministic; must not raise.
    answer_questions(qs, "resume", "Associate Scientist", "Innotech", "desc",
                     use_ai=False)
