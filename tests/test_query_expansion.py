import json

from jobbot import query_expansion as Q
from jobbot.profile import CandidateProfile


def _prof():
    return CandidateProfile(
        skills=["molecular dynamics"], role_titles=["Computational Chemist"],
        role_archetypes=["molecular modeler"], domains=["biochemistry"],
        seniority="mid", dealbreakers=[], summary="x",
        source_resume_hash="h", built_at="t")


def test_expand_queries_dedupes_and_caps(monkeypatch):
    payload = {"queries": ["Computational Chemist", "computational chemist",
                           "Molecular Modeler", "MD Scientist", "Protein Engineer"]}
    monkeypatch.setattr(Q.oc, "_generate", lambda *a, **k: json.dumps(payload))
    monkeypatch.setattr(Q.settings, "query_expansion_max", 3)
    out = Q.expand_queries(_prof())
    assert len(out) == 3                       # capped
    lowered = [q.lower() for q in out]
    assert len(lowered) == len(set(lowered))   # deduped case-insensitively


def test_expand_queries_returns_empty_on_failure(monkeypatch):
    def boom(*a, **k): raise RuntimeError("ollama down")
    monkeypatch.setattr(Q.oc, "_generate", boom)
    assert Q.expand_queries(_prof()) == []
