import json

import pytest

from jobbot import calibrate
from jobbot.models import Job, init_db, session
from jobbot.profile import CandidateProfile


@pytest.fixture(autouse=True)
def _clean():
    init_db()
    with session() as db:
        db.query(Job).delete(); db.commit()
    yield


def _prof():
    return CandidateProfile(skills=["x"], role_titles=[], role_archetypes=[],
                            domains=[], seniority="mid", dealbreakers=[], summary="s",
                            source_resume_hash="h", built_at="t", embedding=[1.0, 0.0])


def test_cosine_distribution_uses_stored_embeddings():
    with session() as db:
        for i, emb in enumerate([[1.0, 0.0], [0.0, 1.0], [0.7, 0.7]]):
            db.add(Job(hash=f"h{i}", source="t", title="T", company="C",
                       url=f"u{i}", description="d", embedding=json.dumps(emb)))
        db.commit()
    dist = calibrate.cosine_distribution(_prof())
    assert dist["n"] == 3
    assert 0.0 <= dist["p50"] <= 1.0


def test_recommend_thresholds_are_ordered():
    dist = {"n": 100, "p10": 0.2, "p25": 0.35, "p50": 0.5, "p75": 0.65, "p90": 0.8}
    rec = calibrate.recommend_thresholds(dist)
    assert 0.0 <= rec["semantic_recall_threshold"] <= 1.0
    assert rec["min_match_score"] >= rec["semantic_recall_threshold"]
