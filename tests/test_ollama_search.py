"""Tests for the Ollama hosted web-search client (HTTP mocked)."""
from jobbot import ollama_search
from jobbot.config import settings


class _Resp:
    def __init__(self, status, payload):
        self.status_code = status
        self._payload = payload
        self.text = str(payload)

    def json(self):
        return self._payload


def test_not_available_without_key(monkeypatch):
    monkeypatch.setattr(settings, "ollama_api_key", "", raising=False)
    assert ollama_search.is_available() is False
    # No key -> empty results, no exception.
    assert ollama_search.web_search("anything") == []
    assert ollama_search.web_fetch("http://x.com") == ""


def test_web_search_normalizes_content_to_snippet(monkeypatch):
    monkeypatch.setattr(settings, "ollama_api_key", "k", raising=False)
    payload = {"results": [
        {"title": "Paper A", "url": "http://a.com", "content": "about A"},
        {"title": "No URL", "content": "skip me"},  # dropped (no url)
        {"title": "Paper B", "url": "http://b.com", "content": "about B"},
    ]}
    monkeypatch.setattr(ollama_search.requests, "post",
                        lambda *a, **k: _Resp(200, payload))
    rows = ollama_search.web_search("q", max_results=5)
    assert [r["url"] for r in rows] == ["http://a.com", "http://b.com"]
    assert rows[0]["snippet"] == "about A"   # content -> snippet


def test_web_search_respects_max_results(monkeypatch):
    monkeypatch.setattr(settings, "ollama_api_key", "k", raising=False)
    payload = {"results": [{"title": f"t{i}", "url": f"http://x/{i}",
                            "content": "c"} for i in range(10)]}
    monkeypatch.setattr(ollama_search.requests, "post",
                        lambda *a, **k: _Resp(200, payload))
    assert len(ollama_search.web_search("q", max_results=3)) == 3


def test_web_search_http_error_returns_empty(monkeypatch):
    monkeypatch.setattr(settings, "ollama_api_key", "k", raising=False)
    monkeypatch.setattr(ollama_search.requests, "post",
                        lambda *a, **k: _Resp(401, {"error": "bad key"}))
    assert ollama_search.web_search("q") == []


def test_web_fetch_returns_content(monkeypatch):
    monkeypatch.setattr(settings, "ollama_api_key", "k", raising=False)
    monkeypatch.setattr(ollama_search.requests, "post",
                        lambda *a, **k: _Resp(200, {"content": "page text here"}))
    assert ollama_search.web_fetch("http://x.com") == "page text here"


def test_selftest_reports_unconfigured(monkeypatch):
    monkeypatch.setattr(settings, "ollama_api_key", "", raising=False)
    res = ollama_search.selftest()
    assert res["configured"] is False and res["ok"] is False
    assert "OLLAMA_API_KEY" in res["error"]


def test_research_enricher_prefers_ollama(monkeypatch):
    """When Ollama search is available, research_enricher._web_search uses it."""
    from jobbot import research_enricher
    monkeypatch.setattr(ollama_search, "is_available", lambda: True)
    monkeypatch.setattr(ollama_search, "web_search",
                        lambda q, n=6: [{"title": "T", "url": "http://o", "snippet": "s"}])
    rows = research_enricher._web_search("q", 6)
    assert rows and rows[0]["url"] == "http://o"
