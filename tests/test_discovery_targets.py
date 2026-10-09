from pathlib import Path

import pytest

from jobbot.discovery import targets as T
from jobbot.discovery.model import DiscoveredTarget


@pytest.fixture(autouse=True)
def _isolate(tmp_path, monkeypatch):
    # Point all four target files at temp paths; clear DEFAULT_TARGETS seeds.
    monkeypatch.setattr(T, "ATS_TARGETS_FILE", tmp_path / "ats_targets.txt")
    monkeypatch.setattr(T, "ASHBY_TARGETS_FILE", tmp_path / "ashby_targets.txt")
    monkeypatch.setattr(T, "WORKDAY_TARGETS_FILE", tmp_path / "workday_targets.txt")
    monkeypatch.setattr(T, "COMPANIES_FILE", tmp_path / "companies.md")
    monkeypatch.setattr(T, "_seed_coords", lambda: set())
    monkeypatch.setattr(T, "_watchlist_names", lambda: set())
    yield


def test_append_greenhouse_and_dedup():
    t = DiscoveredTarget(provider="greenhouse", key="nexustech", display_name="Nexus Tech", source="harvest")
    assert T.append(t) is True
    assert (T.ATS_TARGETS_FILE).read_text().strip() == "greenhouse,nexustech"
    assert ("greenhouse", "nexustech") in T.existing_coords()
    assert T.append(t) is False           # idempotent


def test_append_ashby_and_workday_formats():
    T.append(DiscoveredTarget(provider="ashby", key="techflow", display_name="Techflow", source="llm"))
    assert T.ASHBY_TARGETS_FILE.read_text().strip() == "techflow,Techflow"
    T.append(DiscoveredTarget(provider="workday", key="pinnacle/Pinnacle-Careers",
                              display_name="Pinnacle Systems", source="harvest", wd_host="wd5"))
    assert T.WORKDAY_TARGETS_FILE.read_text().strip() == "pinnacle,Pinnacle-Careers,Pinnacle Systems,wd5"


def test_append_watchlist():
    t = DiscoveredTarget(provider="greenhouse", key="nexustech", display_name="Nexus Tech", source="harvest")
    assert T.append_watchlist(t) is True
    assert "- Nexus Tech" in T.COMPANIES_FILE.read_text()
    assert T.append_watchlist(t) is False   # idempotent (name already present)
