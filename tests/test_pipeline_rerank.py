import pytest

from jobbot import pipeline
from jobbot.models import Job, init_db, session
from jobbot.ranking import RerankResult


@pytest.fixture(autouse=True)
def _clean():
    init_db()
    with session() as db:
        db.query(Job).delete(); db.commit()
    yield


def test_apply_rerank_persists_scores_and_rationale(monkeypatch):
    with session() as db:
        j = Job(hash="h1", source="t", title="T", company="C", url="u",
                description="d", match_score=0.7)
        db.add(j); db.commit()
        jid = j.id

    monkeypatch.setattr(pipeline.settings, "rerank_enabled", True)
    monkeypatch.setattr(pipeline.settings, "rerank_top_k", 5)
    monkeypatch.setattr(pipeline, "rerank_topk",
                        lambda prof, jobs, k: [RerankResult(jid, 0.95, "excellent fit")])

    pipeline._apply_rerank(profile=object(), job_ids=[jid])

    with session() as db:
        job = db.get(Job, jid)
        assert job.rerank_score == pytest.approx(0.95)
        assert job.rerank_rationale == "excellent fit"


def test_apply_rerank_noop_when_disabled(monkeypatch):
    monkeypatch.setattr(pipeline.settings, "rerank_enabled", False)
    called = {"n": 0}
    monkeypatch.setattr(pipeline, "rerank_topk",
                        lambda *a, **k: called.__setitem__("n", called["n"] + 1) or [])
    pipeline._apply_rerank(profile=object(), job_ids=[1, 2])
    assert called["n"] == 0
