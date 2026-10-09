from click.testing import CliRunner
from jobbot import cli as cli_mod


def test_paper_podcast_invokes_make(monkeypatch, tmp_path):
    seen = {}

    def fake_make(sources, out_path=None, mode="explainer", style="two_host",
                  contact="", title="", audio=True, voices=None):
        seen.update(sources=sources, mode=mode, audio=audio, contact=contact)
        return {"mode": mode, "style": style, "title": "T", "n_papers": 1,
                "n_lines": 12, "script_path": "s.txt", "audio_path": None}

    monkeypatch.setattr(cli_mod.paper_podcast, "make_paper_podcast", fake_make)
    res = CliRunner().invoke(
        cli_mod.cli,
        ["paper-podcast", "a.pdf", "--mode", "networking",
         "--contact", "Jane Researcher", "--script-only"])
    assert res.exit_code == 0, res.output
    assert seen["mode"] == "networking"
    assert seen["audio"] is False
    assert seen["contact"] == "Jane Researcher"
    assert "a.pdf" in list(seen["sources"])
