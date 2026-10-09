import json
import types

from jobbot.discovery import discoverers as D


def _prof():
    return types.SimpleNamespace(domains=["biochemistry"], role_titles=["Computational Chemist"],
                                 skills=["MD"], summary="s")


def test_websearch_extracts_coords(monkeypatch):
    monkeypatch.setattr(D, "_web_search", lambda q, max_results=8: [
        {"url": "https://boards.greenhouse.io/beamtx/jobs/1"},
        {"url": "https://example.com/none"},
        {"url": "https://jobs.ashbyhq.com/xaira/0d6f1b2c-1111-2222-3333-444455556666"},
    ])
    monkeypatch.setattr(D.settings, "discovery_websearch_queries", 1)
    out = D.websearch_companies(_prof())
    assert {t.coord() for t in out} == {("greenhouse", "beamtx"), ("ashby", "xaira")}
    assert all(t.source == "websearch" for t in out)


def test_websearch_failopen(monkeypatch):
    def boom(*a, **k): raise RuntimeError("quota")
    monkeypatch.setattr(D, "_web_search", boom)
    assert D.websearch_companies(_prof()) == []


def test_llm_resolves_names_via_search(monkeypatch):
    monkeypatch.setattr(D.oc, "_generate", lambda *a, **k: json.dumps({"companies": ["Beam Therapeutics"]}))
    monkeypatch.setattr(D, "_web_search", lambda q, max_results=8: [
        {"url": "https://boards.greenhouse.io/beamtx/jobs/9"}])
    monkeypatch.setattr(D.settings, "discovery_llm_max_companies", 5)
    out = D.llm_suggest_companies(_prof())
    assert out and out[0].coord() == ("greenhouse", "beamtx") and out[0].source == "llm"


def test_llm_failopen(monkeypatch):
    def boom(*a, **k): raise RuntimeError("ollama down")
    monkeypatch.setattr(D.oc, "_generate", boom)
    assert D.llm_suggest_companies(_prof()) == []
