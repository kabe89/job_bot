"""A Workday posting that has been pulled returns 403 with errorCode S22
("permission denied") from the CXS detail endpoint -- NOT 404. Verified against
the live site: the public page for such a URL renders "The page you are looking
for doesn't exist."

That is a liveness signal, not a fetch failure. Callers must be able to tell
"this job is dead" (expire it) from "the fetch broke" (retry later), so a
transient network blip never expires a live job.
"""
import pytest

from jobbot.scrapers import workday

_URL = ("https://gilead.wd1.myworkdayjobs.com/en-US/gileadcareers/job/"
        "United-States---California---Foster-City/Counsel--IP_R0052365-1")


class _Resp:
    def __init__(self, payload, status=200):
        self._payload = payload
        self.status_code = status

    def json(self):
        return self._payload


_GONE = {"errorCode": "S22", "errorCaseId": "C3D4E4MRO7KNN0",
         "httpStatus": 403, "message": "permission denied", "messageParams": {}}
_LIVE = {"jobPostingInfo": {"title": "Scientist, Biology", "jobReqId": "R1",
                            "location": "Foster City",
                            "jobDescription": "<p>Do science</p>"}}


def test_pulled_posting_reports_gone(monkeypatch):
    monkeypatch.setattr(workday.requests, "get", lambda u, **k: _Resp(_GONE, 403))
    body, state = workday.fetch_posting(_URL)
    assert state == "gone"
    assert body == ""


def test_live_posting_reports_ok_with_a_body(monkeypatch):
    monkeypatch.setattr(workday.requests, "get", lambda u, **k: _Resp(_LIVE, 200))
    body, state = workday.fetch_posting(_URL)
    assert state == "ok"
    assert "Do science" in body


def test_404_is_also_gone(monkeypatch):
    monkeypatch.setattr(workday.requests, "get", lambda u, **k: _Resp({}, 404))
    assert workday.fetch_posting(_URL)[1] == "gone"


@pytest.mark.parametrize("status", [500, 502, 503, 429])
def test_server_trouble_is_transient_not_gone(status, monkeypatch):
    """A 5xx/429 must NEVER expire a job -- it is the site having a bad day."""
    monkeypatch.setattr(workday.requests, "get", lambda u, **k: _Resp({}, status))
    assert workday.fetch_posting(_URL)[1] == "error"


def test_network_exception_is_transient_not_gone(monkeypatch):
    def boom(u, **k):
        raise RuntimeError("connection reset")
    monkeypatch.setattr(workday.requests, "get", boom)
    assert workday.fetch_posting(_URL)[1] == "error"


def test_a_200_with_an_empty_body_is_an_error_not_gone(monkeypatch):
    """Present but bodyless: don't kill the job over it."""
    monkeypatch.setattr(workday.requests, "get", lambda u, **k: _Resp(
        {"jobPostingInfo": {"jobDescription": ""}}, 200))
    body, state = workday.fetch_posting(_URL)
    assert state == "error" and body == ""


def test_a_403_without_the_s22_marker_is_treated_as_transient(monkeypatch):
    """A bot-block/WAF 403 looks different from a pulled posting. Only S22 /
    'permission denied' means gone; anything else gets the benefit of the doubt
    so a WAF hiccup cannot mass-expire a live board."""
    monkeypatch.setattr(workday.requests, "get", lambda u, **k: _Resp(
        {"error": "Forbidden by WAF"}, 403))
    assert workday.fetch_posting(_URL)[1] == "error"


def test_non_workday_url_is_an_error(monkeypatch):
    def boom(u, **k):
        raise AssertionError("must not fetch")
    monkeypatch.setattr(workday.requests, "get", boom)
    assert workday.fetch_posting("https://boards.greenhouse.io/x/jobs/1") == ("", "error")


def test_fetch_description_still_returns_a_bare_string(monkeypatch):
    """The scraper path keeps its simple str contract."""
    monkeypatch.setattr(workday.requests, "get", lambda u, **k: _Resp(_LIVE, 200))
    assert "Do science" in workday.fetch_description(_URL)
    monkeypatch.setattr(workday.requests, "get", lambda u, **k: _Resp(_GONE, 403))
    assert workday.fetch_description(_URL) == ""
