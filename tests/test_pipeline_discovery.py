import pytest

from jobbot import pipeline
from jobbot.models import Job, init_db, session


@pytest.fixture(autouse=True)
def _clean(monkeypatch):
    init_db()
    with session() as db:
        db.query(Job).delete(); db.commit()
    monkeypatch.setattr(pipeline, "load_resume", lambda p: "r")
    monkeypatch.setattr(pipeline.registry, "run_all", lambda kw: [])
    monkeypatch.setattr(pipeline, "load_companies", lambda: [])
    monkeypatch.setattr(pipeline, "backup_db", lambda: None)
    monkeypatch.setattr(pipeline, "load_profile", lambda: type("P", (), {"embedding": None})())
    monkeypatch.setattr(pipeline, "expand_queries", lambda p: [])
    monkeypatch.setattr(pipeline.settings, "digest_email_to", "")
    monkeypatch.setattr(pipeline.settings, "auto_prepare_kits", 0)
    monkeypatch.setattr(pipeline.settings, "auto_prefetch_contacts", 0)
    monkeypatch.setattr(pipeline.settings, "rerank_enabled", False)
    yield


def test_auto_harvest_runs_when_enabled(monkeypatch):
    calls = {"n": 0}
    monkeypatch.setattr(pipeline.settings, "auto_discover_harvest", True)
    monkeypatch.setattr(pipeline, "run_harvest", lambda: calls.__setitem__("n", calls["n"] + 1) or {"added": 2})
    stats = pipeline.run_scrape_cycle(send_digest=False)
    assert calls["n"] == 1 and stats.get("discovered") == 2


def test_auto_harvest_skipped_when_disabled(monkeypatch):
    calls = {"n": 0}
    monkeypatch.setattr(pipeline.settings, "auto_discover_harvest", False)
    monkeypatch.setattr(pipeline, "run_harvest", lambda: calls.__setitem__("n", calls["n"] + 1) or {})
    pipeline.run_scrape_cycle(send_digest=False)
    assert calls["n"] == 0
