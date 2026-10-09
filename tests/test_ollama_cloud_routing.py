import types
import jobbot.ollama_client as oc


def test_is_cloud_model_detects_suffix():
    assert oc._is_cloud_model("gpt-oss:120b-cloud") is True
    assert oc._is_cloud_model("qwen3.5:latest") is False


def test_chat_endpoint_local_vs_cloud(monkeypatch):
    # Local model -> localhost host, no auth header.
    monkeypatch.setattr(oc.settings, "ollama_model", "qwen3.5:latest")
    url, headers = oc._chat_endpoint()
    assert "localhost" in url or "127.0.0.1" in url
    assert "Authorization" not in headers

    # Cloud model -> cloud base + bearer header.
    monkeypatch.setattr(oc.settings, "ollama_model", "gpt-oss:120b-cloud")
    monkeypatch.setattr(oc.settings, "ollama_api_key", "secret-key")
    monkeypatch.setattr(oc.settings, "ollama_cloud_base", "https://ollama.com")
    url, headers = oc._chat_endpoint()
    assert url == "https://ollama.com/api/chat"
    assert headers["Authorization"] == "Bearer secret-key"


def test_generate_cloud_requires_key(monkeypatch):
    monkeypatch.setattr(oc.settings, "ollama_model", "gpt-oss:120b-cloud")
    monkeypatch.setattr(oc.settings, "ollama_api_key", "")
    import pytest
    with pytest.raises(RuntimeError, match="OLLAMA_API_KEY"):
        oc._generate("hi", include_profile=False)
