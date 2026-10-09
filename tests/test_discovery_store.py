import pytest

from jobbot.discovery import store as S
from jobbot.discovery.model import DiscoveredTarget


@pytest.fixture(autouse=True)
def _tmp(tmp_path, monkeypatch):
    monkeypatch.setattr(S.settings, "discovery_pending_path", str(tmp_path / "pending.json"))
    yield


def _t(key, fit=0.0, valid=True, status="pending"):
    return DiscoveredTarget(provider="greenhouse", key=key, display_name=key,
                            source="harvest", valid=valid, fit_score=fit, status=status)


def test_upsert_dedups_and_preserves_status():
    assert S.upsert(_t("a")) is True
    assert S.upsert(_t("a", status="pending")) is False     # already present → not re-added
    S.set_status(("greenhouse", "a"), "rejected")
    assert S.upsert(_t("a")) is False                        # rejected entry not overwritten
    assert S.get(("greenhouse", "a")).status == "rejected"


def test_list_pending_sorted_and_filtered():
    S.upsert(_t("low", fit=0.2))
    S.upsert(_t("high", fit=0.9))
    S.upsert(_t("invalid", fit=0.95, valid=False))
    keys = [t.key for t in S.list_pending()]
    assert keys == ["high", "low"]                           # sorted desc, invalid excluded
    assert [t.key for t in S.list_pending(min_fit=0.5)] == ["high"]


def test_rejected_excluded_from_pending_list():
    S.upsert(_t("x", fit=0.8))
    S.set_status(("greenhouse", "x"), "rejected")
    assert S.list_pending() == []
