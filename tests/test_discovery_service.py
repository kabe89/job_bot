import types

import pytest

from jobbot.discovery import service as SVC
from jobbot.discovery.model import DiscoveredTarget


@pytest.fixture(autouse=True)
def _tmp(tmp_path, monkeypatch):
    monkeypatch.setattr(SVC.store.settings, "discovery_pending_path", str(tmp_path / "p.json"))
    monkeypatch.setattr(SVC, "load_profile", lambda: types.SimpleNamespace(embedding=[1.0], summary="s"))
    monkeypatch.setattr(SVC.targets, "existing_coords", lambda: {("greenhouse", "already")})
    yield


def test_run_harvest_validates_scores_and_stores(monkeypatch):
    cand = DiscoveredTarget(provider="greenhouse", key="new", display_name="New", source="harvest")
    dup = DiscoveredTarget(provider="greenhouse", key="already", display_name="Dup", source="harvest")
    monkeypatch.setattr(SVC, "harvest_from_jobs", lambda: [cand, dup])
    monkeypatch.setattr(SVC, "validate", lambda t: (setattr(t, "valid", True), setattr(t, "sample_titles", ["Sci"]), t)[-1])
    monkeypatch.setattr(SVC, "fit_score", lambda t, p: (setattr(t, "fit_score", 0.7), t)[-1])
    res = SVC.run_harvest()
    assert res["added"] == 1                      # dup skipped via existing_coords
    pend = SVC.review()
    assert [t.key for t in pend] == ["new"] and pend[0].fit_score == 0.7


def test_approve_appends_and_marks(monkeypatch):
    t = DiscoveredTarget(provider="ashby", key="techflow", display_name="TechFlow", source="llm", valid=True, fit_score=0.9)
    SVC.store.upsert(t)
    appended = {}
    monkeypatch.setattr(SVC.targets, "append", lambda x: appended.setdefault("t", x) or True)
    monkeypatch.setattr(SVC.targets, "append_watchlist", lambda x: appended.setdefault("w", x) or True)
    res = SVC.approve([("ashby", "techflow")], watchlist=True)
    assert res["approved"] == 1 and appended["t"].key == "techflow" and "w" in appended
    assert SVC.store.get(("ashby", "techflow")).status == "approved"


def test_parse_coord():
    assert SVC.parse_coord("greenhouse:nexustech") == ("greenhouse", "nexustech")
    assert SVC.parse_coord("workday:pinnacle/Pinnacle-Careers") == ("workday", "pinnacle/Pinnacle-Careers")
