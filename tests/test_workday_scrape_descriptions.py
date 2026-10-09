"""_scrape_one_tenant must attach the REAL JD body, not the listing-card stub.

The stub bug meant every scraped Workday job carried ~200 chars of scraper
metadata as its description, so its match_score was computed against noise.
Fetching the body costs one extra HTTP call per NEW job, so it is bounded by
settings.workday_fetch_descriptions and skipped for jobs already in the DB.
"""
import pytest

from jobbot.scrapers import workday


def _card(path, title="Software Engineer"):
    return {"externalPath": path, "title": title,
            "locationsText": "Cambridge, Massachusetts",
            "postedOn": "Posted 9 Days Ago", "bulletFields": ["R19096"]}


@pytest.fixture(autouse=True)
def _no_network(monkeypatch):
    """Nothing in this file may touch the network."""
    def boom(*a, **k):
        raise AssertionError("unexpected real HTTP call")
    monkeypatch.setattr(workday.requests, "post", boom)
    monkeypatch.setattr(workday.requests, "get", boom)


def _fake_query(monkeypatch, cards):
    monkeypatch.setattr(workday, "_query", lambda t, b, h, kw: list(cards))


def test_scrape_attaches_the_real_description(monkeypatch):
    _fake_query(monkeypatch, [_card("/job/Cambridge/Engineer_R19096")])
    monkeypatch.setattr(workday, "fetch_description",
                        lambda url, **kw: "The Role\nDesign scalable backend services.")
    monkeypatch.setattr(workday.settings, "workday_fetch_descriptions", True)

    out = workday._scrape_one_tenant("innotech", "InnoCareers", "Innotech", "wd1", ["python"])

    assert len(out) == 1
    assert out[0].description == "The Role\nDesign scalable backend services."
    assert not out[0].description.startswith("Job ID:")


def test_scrape_falls_back_to_the_card_stub_when_the_body_is_unavailable(monkeypatch):
    """A dead detail endpoint must not lose the job -- keep the card metadata."""
    _fake_query(monkeypatch, [_card("/job/Cambridge/Engineer_R19096")])
    monkeypatch.setattr(workday, "fetch_description", lambda url, **kw: "")
    monkeypatch.setattr(workday.settings, "workday_fetch_descriptions", True)

    out = workday._scrape_one_tenant("innotech", "InnoCareers", "Innotech", "wd1", ["python"])

    assert len(out) == 1                       # job survives
    assert "Job ID:" in out[0].description     # degraded to the old stub


def test_scrape_skips_the_fetch_when_disabled(monkeypatch):
    _fake_query(monkeypatch, [_card("/job/Cambridge/Engineer_R19096")])
    calls = []
    monkeypatch.setattr(workday, "fetch_description",
                        lambda url, **kw: calls.append(url) or "body")
    monkeypatch.setattr(workday.settings, "workday_fetch_descriptions", False)

    out = workday._scrape_one_tenant("innotech", "InnoCareers", "Innotech", "wd1", ["python"])

    assert calls == []                         # no extra HTTP
    assert "Job ID:" in out[0].description


def test_scrape_fetches_once_per_unique_job_not_once_per_keyword(monkeypatch):
    """The same posting surfaces under several keywords; dedupe happens before
    the fetch, or we'd pay N HTTP calls for one job."""
    card = _card("/job/Cambridge/Engineer_R19096")
    monkeypatch.setattr(workday, "_query", lambda t, b, h, kw: [card])
    calls = []
    monkeypatch.setattr(workday, "fetch_description",
                        lambda url, **kw: calls.append(url) or "body")
    monkeypatch.setattr(workday.settings, "workday_fetch_descriptions", True)

    out = workday._scrape_one_tenant("innotech", "InnoCareers", "Innotech", "wd1",
                                     ["python", "software", "backend"])

    assert len(out) == 1
    assert len(calls) == 1


def test_scrape_is_fail_open_when_the_fetch_raises(monkeypatch):
    _fake_query(monkeypatch, [_card("/job/Cambridge/Engineer_R19096")])

    def boom(url, **kw):
        raise RuntimeError("detail endpoint exploded")

    monkeypatch.setattr(workday, "fetch_description", boom)
    monkeypatch.setattr(workday.settings, "workday_fetch_descriptions", True)

    out = workday._scrape_one_tenant("innotech", "InnoCareers", "Innotech", "wd1", ["python"])

    assert len(out) == 1                       # never lose the job
    assert "Job ID:" in out[0].description


def test_scrape_still_sets_company_from_the_display_name(monkeypatch):
    _fake_query(monkeypatch, [_card("/job/Cambridge/Engineer_R19096")])
    monkeypatch.setattr(workday, "fetch_description", lambda url, **kw: "body")
    monkeypatch.setattr(workday.settings, "workday_fetch_descriptions", True)

    out = workday._scrape_one_tenant("innotech", "InnoCareers", "Innotech", "wd1", ["python"])

    assert out[0].company == "Innotech"
    assert "/" not in out[0].company
