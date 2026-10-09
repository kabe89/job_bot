import numpy as np
import pytest

from jobbot import embeddings


def test_cosine_basic():
    assert embeddings.cosine([1.0, 0.0], [1.0, 0.0]) == pytest.approx(1.0)
    assert embeddings.cosine([1.0, 0.0], [0.0, 1.0]) == pytest.approx(0.0)


def test_cosine_zero_vector_is_zero():
    assert embeddings.cosine([0.0, 0.0], [1.0, 2.0]) == 0.0
    assert embeddings.cosine([], []) == 0.0


def test_embed_posts_to_ollama_and_returns_vector(monkeypatch):
    captured = {}

    class _Resp:
        status_code = 200
        text = ""
        def raise_for_status(self): pass
        def json(self): return {"embedding": [0.1, 0.2, 0.3]}

    def fake_post(url, json=None, timeout=None):
        captured["url"] = url
        captured["model"] = json["model"]
        captured["prompt"] = json["prompt"]
        return _Resp()

    monkeypatch.setattr(embeddings.oc, "ensure_ready", lambda **k: True)
    monkeypatch.setattr(embeddings.oc, "_host", lambda: "http://localhost:11434")
    monkeypatch.setattr(embeddings.oc, "_timeout", lambda: 30)
    monkeypatch.setattr(embeddings.requests, "post", fake_post)

    vec = embeddings.embed("computational biochemist")
    assert vec == [0.1, 0.2, 0.3]
    assert captured["url"].endswith("/api/embeddings")
    assert captured["model"] == embeddings.settings.ollama_embed_model


def test_embed_raises_on_empty(monkeypatch):
    class _Resp:
        status_code = 200
        text = ""
        def raise_for_status(self): pass
        def json(self): return {"embedding": []}
    monkeypatch.setattr(embeddings.oc, "ensure_ready", lambda **k: True)
    monkeypatch.setattr(embeddings.oc, "_host", lambda: "http://localhost:11434")
    monkeypatch.setattr(embeddings.oc, "_timeout", lambda: 30)
    monkeypatch.setattr(embeddings.requests, "post", lambda *a, **k: _Resp())
    with pytest.raises(RuntimeError):
        embeddings.embed("x")


def test_embed_raises_actionable_when_model_not_pulled(monkeypatch):
    # A 404 "model not found" must become an actionable 'ollama pull' message,
    # not a bare HTTPError, so the semantic-ranking degradation is diagnosable.
    class _Resp:
        status_code = 404
        text = 'model "nomic-embed-text" not found, try pulling it first'
        def raise_for_status(self): raise AssertionError("should not reach raise_for_status")
        def json(self): return {}
    monkeypatch.setattr(embeddings.oc, "ensure_ready", lambda **k: True)
    monkeypatch.setattr(embeddings.oc, "_host", lambda: "http://localhost:11434")
    monkeypatch.setattr(embeddings.oc, "_timeout", lambda: 30)
    monkeypatch.setattr(embeddings.requests, "post", lambda *a, **k: _Resp())
    with pytest.raises(RuntimeError, match="ollama pull"):
        embeddings.embed("x")
