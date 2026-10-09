import types

import pytest

from jobbot.discovery import validate as V
from jobbot.discovery.model import DiscoveredTarget


class _Resp:
    def __init__(self, payload): self._p = payload
    def raise_for_status(self): pass
    def json(self): return self._p


def test_validate_greenhouse_titles(monkeypatch):
    monkeypatch.setattr(V.requests, "get",
                        lambda *a, **k: _Resp({"jobs": [{"title": "Scientist"}, {"title": "Engineer"}]}))
    t = V.validate(DiscoveredTarget(provider="greenhouse", key="beamtx", display_name="Beam", source="harvest"))
    assert t.valid is True and t.job_count == 2 and "Scientist" in t.sample_titles


def test_validate_failure_marks_invalid(monkeypatch):
    def boom(*a, **k): raise RuntimeError("404")
    monkeypatch.setattr(V.requests, "get", boom)
    t = V.validate(DiscoveredTarget(provider="lever", key="dead", display_name="Dead", source="llm"))
    assert t.valid is False and t.job_count == 0


def test_fit_score_uses_embeddings(monkeypatch):
    monkeypatch.setattr(V.embeddings, "embed", lambda text: [1.0, 0.0])
    monkeypatch.setattr(V.oc, "_generate", lambda *a, **k: "Strong computational-bio fit.")
    prof = types.SimpleNamespace(embedding=[1.0, 0.0], role_titles=[], skills=[], domains=[], summary="s")
    t = DiscoveredTarget(provider="greenhouse", key="x", display_name="X", source="harvest",
                         valid=True, sample_titles=["Computational Scientist"])
    t = V.fit_score(t, prof)
    assert t.fit_score == pytest.approx(1.0) and t.fit_reason


def test_fit_score_failopen_when_embed_down(monkeypatch):
    def boom(text): raise RuntimeError("down")
    monkeypatch.setattr(V.embeddings, "embed", boom)
    prof = types.SimpleNamespace(embedding=[1.0, 0.0])
    t = DiscoveredTarget(provider="greenhouse", key="x", display_name="X", source="harvest",
                         valid=True, sample_titles=["Sci"])
    t = V.fit_score(t, prof)
    assert t.fit_score == 0.0   # no crash
