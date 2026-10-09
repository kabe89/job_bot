import pytest

from jobbot import ranking


def test_semantic_score_returns_cosine_and_embedding(monkeypatch):
    monkeypatch.setattr(ranking.embeddings, "embed", lambda t: [1.0, 0.0])
    score, emb = ranking.semantic_score("a job", [1.0, 0.0])
    assert score == pytest.approx(1.0)
    assert emb == [1.0, 0.0]


def test_semantic_score_clamps_to_unit_interval(monkeypatch):
    monkeypatch.setattr(ranking.embeddings, "embed", lambda t: [-1.0, 0.0])
    score, _ = ranking.semantic_score("a job", [1.0, 0.0])  # cosine = -1
    assert score == 0.0


def test_semantic_score_propagates_embed_failure(monkeypatch):
    def boom(t): raise RuntimeError("down")
    monkeypatch.setattr(ranking.embeddings, "embed", boom)
    with pytest.raises(RuntimeError):
        ranking.semantic_score("a job", [1.0, 0.0])
