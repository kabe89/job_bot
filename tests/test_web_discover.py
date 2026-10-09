import pytest

from jobbot import web
from jobbot.discovery.model import DiscoveredTarget


@pytest.fixture
def client(monkeypatch):
    monkeypatch.setattr(web, "discovery_service", web.discovery_service)  # ensure attr exists
    app = web.create_app()
    app.config.update(TESTING=True)
    return app.test_client()


def test_discover_page_lists(monkeypatch, client):
    monkeypatch.setattr(web.discovery_service, "review",
                        lambda min_fit=None: [DiscoveredTarget(provider="ashby", key="xaira",
                                              display_name="Xaira", source="llm", valid=True, fit_score=0.9)])
    r = client.get("/discover")
    assert r.status_code == 200 and b"xaira" in r.data


def test_discover_action_approve(monkeypatch, client):
    seen = {}
    monkeypatch.setattr(web.discovery_service, "parse_coord", lambda s: tuple(s.split(":", 1)))
    monkeypatch.setattr(web.discovery_service, "approve",
                        lambda coords, watchlist=False: seen.update(c=coords, w=watchlist) or {"approved": 1})
    r = client.post("/discover/approve_watchlist", data={"coord": "ashby:xaira"})
    assert r.status_code in (302, 303) and seen["c"] == [("ashby", "xaira")] and seen["w"] is True


def test_discover_run_starts_background(monkeypatch, client):
    called = {"n": 0}
    monkeypatch.setattr(web.discovery_service, "run_harvest",
                        lambda: called.__setitem__("n", called["n"] + 1) or {"added": 0})
    r = client.post("/discover/run", data={"kind": "harvest"})
    assert r.status_code in (302, 303)
    # status endpoint responds
    assert client.get("/discover/run/status").status_code == 200
