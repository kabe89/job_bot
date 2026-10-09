from click.testing import CliRunner
from jobbot import cli as cli_mod


def test_model_list_shows_rows(monkeypatch):
    rows = [
        {"group": "Ollama — local", "provider": "ollama", "model": "qwen3.5:latest",
         "available": True, "reason": "", "current": True},
        {"group": "Gemini", "provider": "gemini", "model": "gemini-2.5-flash",
         "available": False, "reason": "set GEMINI_API_KEY", "current": False},
    ]
    monkeypatch.setattr(cli_mod.model_registry, "list_selectable", lambda: rows)
    res = CliRunner().invoke(cli_mod.cli, ["model", "list"])
    assert res.exit_code == 0
    assert "qwen3.5:latest" in res.output
    assert "gemini-2.5-flash" in res.output


def test_model_set_calls_registry(monkeypatch):
    seen = {}
    monkeypatch.setattr(cli_mod.model_registry, "set_selection",
                        lambda p, m: seen.update(provider=p, model=m))
    res = CliRunner().invoke(cli_mod.cli, ["model", "set", "ollama", "gpt-oss:120b-cloud"])
    assert res.exit_code == 0
    assert seen == {"provider": "ollama", "model": "gpt-oss:120b-cloud"}


def test_model_pull_reports_result(monkeypatch):
    monkeypatch.setattr(cli_mod.model_registry, "pull_ollama_model",
                        lambda name, progress=None: (progress and progress("pulling")) or True)
    res = CliRunner().invoke(cli_mod.cli, ["model", "pull", "llama3.3:70b"])
    assert res.exit_code == 0
    assert "llama3.3:70b" in res.output
