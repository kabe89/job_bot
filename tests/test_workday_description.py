"""Workday's search endpoint returns listing CARDS only (title/location/postedOn)
with no description, so `_scrape_one_tenant` was storing a ~200-char stub of
scraper metadata as the job description. Every Workday job therefore had no real
JD, and its match_score was computed against that noise. `fetch_description`
pulls the real body from the per-job CXS endpoint.
"""
import json

from jobbot.scrapers import workday


_URL = ("https://innotech.wd1.myworkdayjobs.com/en-US/InnoCareers/job/"
        "Cambridge-Massachusetts/Scientist--Platform-and-TA-R19096-1")


class _Resp:
    def __init__(self, payload, status=200):
        self._payload = payload
        self.status_code = status

    def json(self):
        return self._payload


def _posting(**kw):
    info = {"title": "Scientist, Platform Software Engineering",
            "jobReqId": "R19096",
            "location": "Cambridge, Massachusetts",
            "jobDescription": "<p>The Role</p><ul><li>Design software systems</li></ul>"}
    info.update(kw)
    return {"jobPostingInfo": info}


def test_fetch_description_builds_the_cxs_url_from_a_job_url(monkeypatch):
    seen = {}

    def fake_get(url, **kw):
        seen["url"] = url
        return _Resp(_posting())

    monkeypatch.setattr(workday.requests, "get", fake_get)
    workday.fetch_description(_URL)
    assert seen["url"] == (
        "https://innotech.wd1.myworkdayjobs.com/wday/cxs/innotech/InnoCareers/job/"
        "Cambridge-Massachusetts/Scientist--Platform-and-TA-R19096-1")


def test_fetch_description_returns_real_body_not_a_stub(monkeypatch):
    monkeypatch.setattr(workday.requests, "get", lambda url, **kw: _Resp(_posting()))
    out = workday.fetch_description(_URL)
    assert "The Role" in out and "Design software systems" in out
    assert "R19096" in out
    assert "<p>" not in out and "<li>" not in out      # HTML flattened
    assert not out.startswith("Job ID:")               # not the old card stub


def test_fetch_description_unescapes_entities_and_strips_nbsp(monkeypatch):
    monkeypatch.setattr(workday.requests, "get", lambda url, **kw: _Resp(
        _posting(jobDescription="<p>Identity &amp; Ratio</p><p>8&#43; years</p>")))
    out = workday.fetch_description(_URL)
    assert "Identity & Ratio" in out
    assert "8+" in out
    # narrow-NBSP would blow up a cp1252 Windows console downstream
    assert " " not in out and " " not in out


def test_fetch_description_fail_open_on_http_error(monkeypatch):
    monkeypatch.setattr(workday.requests, "get", lambda url, **kw: _Resp({}, status=503))
    assert workday.fetch_description(_URL) == ""


def test_fetch_description_fail_open_when_the_request_raises(monkeypatch):
    def boom(url, **kw):
        raise RuntimeError("network down")

    monkeypatch.setattr(workday.requests, "get", boom)
    assert workday.fetch_description(_URL) == ""


def test_fetch_description_fail_open_on_empty_body(monkeypatch):
    monkeypatch.setattr(workday.requests, "get", lambda url, **kw: _Resp(
        _posting(jobDescription="")))
    assert workday.fetch_description(_URL) == ""


def test_fetch_description_rejects_a_non_workday_url(monkeypatch):
    def boom(url, **kw):  # must never be called
        raise AssertionError("should not fetch a non-Workday URL")

    monkeypatch.setattr(workday.requests, "get", boom)
    assert workday.fetch_description("https://boards.greenhouse.io/x/jobs/1") == ""
    assert workday.fetch_description("") == ""


def test_fetch_description_handles_a_locale_free_job_url(monkeypatch):
    """Workday serves the same posting with and without the /en-US/ segment."""
    seen = {}

    def fake_get(url, **kw):
        seen["url"] = url
        return _Resp(_posting())

    monkeypatch.setattr(workday.requests, "get", fake_get)
    out = workday.fetch_description(
        "https://innotech.wd1.myworkdayjobs.com/InnoCareers/job/Cambridge-Massachusetts/X_R1")
    assert out
    assert seen["url"] == ("https://innotech.wd1.myworkdayjobs.com/wday/cxs/"
                           "innotech/InnoCareers/job/Cambridge-Massachusetts/X_R1")
