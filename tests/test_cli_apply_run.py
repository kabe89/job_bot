from click.testing import CliRunner

from jobbot import cli as cli_mod


def test_apply_run_command_invokes_runner(monkeypatch):
    called = {}
    def fake_run_apply(job_id, **kwargs):
        called["job_id"] = job_id
        called["headed"] = kwargs.get("headed")
        return 42
    monkeypatch.setattr(cli_mod.apply_runner, "run_apply", fake_run_apply)

    result = CliRunner().invoke(cli_mod.cli, ["apply-run", "7", "--no-headed"])
    assert result.exit_code == 0
    assert called["job_id"] == 7
    assert called["headed"] is False
    assert "42" in result.output
