# tests/test_warm_path_cli.py
from click.testing import CliRunner

from jobbot import cli as cli_mod


def test_warm_path_daily_runs_all_stages(monkeypatch):
    calls = []
    monkeypatch.setattr("jobbot.pipeline.run_scrape_cycle",
                        lambda send_digest=False: calls.append("scrape") or {"new": 2})
    monkeypatch.setattr("jobbot.pipeline.load_profile", lambda: object())
    monkeypatch.setattr("jobbot.ranking.rescore_jobs",
                        lambda profile: calls.append("rescore") or {"rescored": 3})
    monkeypatch.setattr("jobbot.outreach.scan",
                        lambda: calls.append("scan") or {"warm_intro": 1, "cold": 2})

    result = CliRunner().invoke(cli_mod.cli, ["warm-path", "daily"])
    assert result.exit_code == 0, result.output
    assert calls == ["scrape", "rescore", "scan"]
    assert "warm_intro" in result.output


def test_warm_path_daily_failopen_on_scrape(monkeypatch):
    def boom(send_digest=False):
        raise RuntimeError("scrape down")
    monkeypatch.setattr("jobbot.pipeline.run_scrape_cycle", boom)
    monkeypatch.setattr("jobbot.pipeline.load_profile", lambda: object())
    monkeypatch.setattr("jobbot.ranking.rescore_jobs", lambda p: {"rescored": 0})
    monkeypatch.setattr("jobbot.outreach.scan", lambda: {"cold": 0})
    result = CliRunner().invoke(cli_mod.cli, ["warm-path", "daily"])
    assert result.exit_code == 0, result.output  # fail-open, still exits clean
