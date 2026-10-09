import pytest

from jobbot import pipeline
from jobbot.models import init_db


@pytest.fixture(autouse=True)
def _ensure_tables():
    # Real init_db (table creation) must run once before the test stubs
    # pipeline.init_db to a no-op — mirrors tests/test_pipeline_discovery.py.
    init_db()


def test_run_scrape_cycle_calls_liveness(monkeypatch):
    calls = {}

    monkeypatch.setattr(pipeline.settings, "liveness_check_per_cycle", 7)
    # Stub everything heavy so the cycle is a no-op except the liveness hook.
    monkeypatch.setattr(pipeline, "init_db", lambda: None)
    monkeypatch.setattr(pipeline, "backup_db", lambda: None)
    monkeypatch.setattr(pipeline.registry, "run_all", lambda terms: [])
    monkeypatch.setattr(pipeline, "load_profile", lambda: type("P", (), {
        "embedding": None})())
    monkeypatch.setattr(pipeline, "load_tags", lambda: [])
    monkeypatch.setattr(pipeline, "load_locations", lambda: ["remote"])
    monkeypatch.setattr(pipeline, "scrape_locations", lambda: ["remote"])
    monkeypatch.setattr(pipeline, "includes_remote", lambda: True)
    monkeypatch.setattr(pipeline, "load_companies", lambda: [])
    monkeypatch.setattr(pipeline.settings, "semantic_ranking_enabled", False)
    monkeypatch.setattr(pipeline.settings, "auto_prepare_kits", 0)
    monkeypatch.setattr(pipeline.settings, "auto_prefetch_contacts", 0)
    monkeypatch.setattr(pipeline.settings, "auto_discover_harvest", False)
    monkeypatch.setattr(pipeline.settings, "rerank_enabled", False)

    def fake_sweep(limit, **k):
        calls["limit"] = limit
        return {"checked": 0, "expired": 3, "alive": 0, "errors": 0}
    monkeypatch.setattr(pipeline, "sweep_expired", fake_sweep)

    stats = pipeline.run_scrape_cycle(send_digest=False)
    assert calls["limit"] == 7
    assert stats["expired"] == 3
