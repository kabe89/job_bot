import json
import types

from jobbot import ranking
from jobbot.profile import CandidateProfile


def _prof():
    return CandidateProfile(
        skills=["x"], role_titles=["Scientist"], role_archetypes=[], domains=[],
        seniority="mid", dealbreakers=[], summary="s",
        source_resume_hash="h", built_at="t")


def _job(jid, score):
    return types.SimpleNamespace(id=jid, title="T", company="C",
                                 description="D", match_score=score)


def test_rerank_topk_judges_only_k_highest(monkeypatch):
    calls = {"n": 0}
    def fake_gen(*a, **k):
        calls["n"] += 1
        return json.dumps({"score": 0.9, "rationale": "great fit"})
    monkeypatch.setattr(ranking.oc, "_generate", fake_gen)
    jobs = [_job(1, 0.2), _job(2, 0.8), _job(3, 0.5)]
    out = ranking.rerank_topk(_prof(), jobs, k=2)
    assert calls["n"] == 2                       # only top-2 judged
    assert {r.job_id for r in out} == {2, 3}     # the two highest match_score
    assert all(0.0 <= r.score <= 1.0 for r in out)
    assert all(r.rationale for r in out)


def test_rerank_topk_skips_failed_jobs(monkeypatch):
    def boom(*a, **k): raise RuntimeError("down")
    monkeypatch.setattr(ranking.oc, "_generate", boom)
    out = ranking.rerank_topk(_prof(), [_job(1, 0.9)], k=5)
    assert out == []                              # failure skipped, not raised
