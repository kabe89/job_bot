"""Tests for the AI model selector (registry + cloud routing)."""
import jobbot.model_registry as mr
from jobbot.config import settings


def test_ollama_cloud_base_default():
    assert settings.ollama_cloud_base.rstrip("/") == "https://ollama.com"


def test_list_selectable_groups_and_availability(monkeypatch):
    monkeypatch.setattr(mr, "list_ollama_models", lambda: ["qwen3.5:latest"])
    monkeypatch.setattr(mr.ollama_client, "_server_reachable", lambda *a, **k: True)
    monkeypatch.setattr(mr.ollama_client, "_can_autostart", lambda: False)
    monkeypatch.setattr(mr.settings, "ollama_api_key", "")
    monkeypatch.setattr(mr.settings, "gemini_api_key", "")
    monkeypatch.setattr(mr.settings, "anthropic_api_key", "")

    rows = mr.list_selectable()
    groups = {r["group"] for r in rows}
    # Local present; cloud absent (no OLLAMA_API_KEY); cloud providers shown-but-disabled.
    assert "Ollama — local" in groups
    assert "Ollama — cloud" not in groups
    gemini = [r for r in rows if r["group"] == "Gemini"]
    assert gemini and all(r["available"] is False for r in gemini)
    assert any("GEMINI_API_KEY" in r["reason"] for r in gemini)


def test_ollama_cloud_group_appears_with_key(monkeypatch):
    monkeypatch.setattr(mr, "list_ollama_models", lambda: [])
    monkeypatch.setattr(mr.ollama_client, "_server_reachable", lambda *a, **k: False)
    monkeypatch.setattr(mr.ollama_client, "_can_autostart", lambda: False)
    monkeypatch.setattr(mr.settings, "ollama_api_key", "key123")
    monkeypatch.setattr(mr.settings, "gemini_api_key", "")
    monkeypatch.setattr(mr.settings, "anthropic_api_key", "")
    rows = mr.list_selectable()
    cloud = [r for r in rows if r["group"] == "Ollama — cloud"]
    assert cloud and all(r["available"] for r in cloud)
    assert all(r["provider"] == "ollama" for r in cloud)


def test_current_selection_reads_settings(monkeypatch):
    monkeypatch.setattr(mr.settings, "ai_provider", "ollama")
    monkeypatch.setattr(mr.settings, "ollama_model", "qwen3.5:latest")
    assert mr.current_selection() == ("ollama", "qwen3.5:latest")


def test_set_selection_writes_env_and_settings(tmp_path, monkeypatch):
    env = tmp_path / ".env"
    env.write_text("AI_PROVIDER=auto\nOLLAMA_MODEL=old:latest\nFOO=bar\n", encoding="utf-8")
    monkeypatch.setattr(mr.ollama_client, "reset_fallback_state", lambda: None)
    # set_selection mutates the LIVE global settings; snapshot the touched attrs
    # so monkeypatch restores them at teardown and we don't poison other tests'
    # view of the active model (e.g. tests/test_ollama_local.py).
    monkeypatch.setattr(mr.settings, "ai_provider", mr.settings.ai_provider)
    monkeypatch.setattr(mr.settings, "ollama_model", mr.settings.ollama_model)

    mr.set_selection("ollama", "gpt-oss:120b-cloud", env_path=env)

    text = env.read_text(encoding="utf-8")
    assert "AI_PROVIDER=ollama" in text
    assert "OLLAMA_MODEL=gpt-oss:120b-cloud" in text
    assert "FOO=bar" in text                      # untouched key preserved
    assert mr.settings.ai_provider == "ollama"
    assert mr.settings.ollama_model == "gpt-oss:120b-cloud"


def test_set_selection_appends_missing_key(tmp_path, monkeypatch):
    env = tmp_path / ".env"
    env.write_text("AI_PROVIDER=auto\n", encoding="utf-8")
    monkeypatch.setattr(mr.ollama_client, "reset_fallback_state", lambda: None)
    # Snapshot the live settings attrs this mutates so they restore at teardown.
    monkeypatch.setattr(mr.settings, "ai_provider", mr.settings.ai_provider)
    monkeypatch.setattr(mr.settings, "gemini_model", mr.settings.gemini_model)
    mr.set_selection("gemini", "gemini-2.5-flash", env_path=env)
    text = env.read_text(encoding="utf-8")
    assert "GEMINI_MODEL=gemini-2.5-flash" in text


def test_set_selection_rejects_bad_input(tmp_path):
    import pytest
    with pytest.raises(ValueError):
        mr.set_selection("bogus", "x", env_path=tmp_path / ".env")
    with pytest.raises(ValueError):
        mr.set_selection("ollama", "", env_path=tmp_path / ".env")


def test_pull_without_binary_returns_false(monkeypatch):
    monkeypatch.setattr(mr.ollama_client, "_ollama_bin", lambda: None)
    seen = []
    ok = mr.pull_ollama_model("llama3.3:70b", progress=seen.append)
    assert ok is False
    assert any("not found" in s for s in seen)
