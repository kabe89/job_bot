from click.testing import CliRunner

from jobbot import cli as cli_mod
from jobbot.profile import CandidateProfile


def _prof():
    return CandidateProfile(
        skills=["molecular dynamics"], role_titles=["Computational Chemist"],
        role_archetypes=[], domains=["biochemistry"], seniority="mid",
        dealbreakers=[], summary="A computational biochemist.",
        source_resume_hash="h", built_at="t", embedding=[0.1])


def test_profile_show(monkeypatch):
    monkeypatch.setattr(cli_mod.profile, "load_profile", lambda force=False: _prof())
    res = CliRunner().invoke(cli_mod.cli, ["profile", "show"])
    assert res.exit_code == 0
    assert "Computational Chemist" in res.output
    assert "molecular dynamics" in res.output


def test_profile_rebuild_forces(monkeypatch):
    seen = {}
    monkeypatch.setattr(cli_mod.profile, "load_profile",
                        lambda force=False: seen.update(force=force) or _prof())
    res = CliRunner().invoke(cli_mod.cli, ["profile", "rebuild"])
    assert res.exit_code == 0
    assert seen.get("force") is True
