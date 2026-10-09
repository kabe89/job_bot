from click.testing import CliRunner

from jobbot import cli as cli_mod
from jobbot.discovery.model import DiscoveredTarget


def test_discover_review_lists_pending(monkeypatch):
    monkeypatch.setattr(cli_mod.discovery_service, "review",
                        lambda min_fit=None: [DiscoveredTarget(provider="greenhouse", key="beamtx",
                                              display_name="Beam", source="harvest", valid=True, fit_score=0.8)])
    res = CliRunner().invoke(cli_mod.cli, ["discover", "review"])
    assert res.exit_code == 0 and "greenhouse:beamtx" in res.output and "Beam" in res.output


def test_discover_approve_calls_service(monkeypatch):
    seen = {}
    monkeypatch.setattr(cli_mod.discovery_service, "parse_coord", lambda s: tuple(s.split(":", 1)))
    monkeypatch.setattr(cli_mod.discovery_service, "approve",
                        lambda coords, watchlist=False: seen.update(coords=coords, watchlist=watchlist) or {"approved": 1})
    res = CliRunner().invoke(cli_mod.cli, ["discover", "approve", "ashby:xaira", "--watchlist"])
    assert res.exit_code == 0 and seen["coords"] == [("ashby", "xaira")] and seen["watchlist"] is True
