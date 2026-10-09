from datetime import datetime

import pytest

from jobbot import pipeline
from jobbot.models import Job, init_db, session
from jobbot.profile import CandidateProfile
from jobbot.scrapers import RawJob


@pytest.fixture(autouse=True)
def _clean(monkeypatch):
    init_db()
    with session() as db:
        db.query(Job).delete(); db.commit()
    # No real scrapers / resume / digest.
    monkeypatch.setattr(pipeline, "load_resume", lambda p: "resume text")
    monkeypatch.setattr(pipeline.registry, "run_all", lambda kw: _RAW)
    monkeypatch.setattr(pipeline, "load_companies", lambda: [])
    monkeypatch.setattr(pipeline, "backup_db", lambda: None)
    monkeypatch.setattr(pipeline.settings, "digest_email_to", "")
    monkeypatch.setattr(pipeline.settings, "auto_prepare_kits", 0)
    monkeypatch.setattr(pipeline.settings, "auto_prefetch_contacts", 0)
    # Profile with a known embedding; expansion off; semantic on.
    prof = CandidateProfile(skills=["x"], role_titles=[], role_archetypes=[],
                            domains=[], seniority="mid", dealbreakers=[], summary="s",
                            source_resume_hash="h", built_at="t", embedding=[1.0, 0.0])
    monkeypatch.setattr(pipeline, "load_profile", lambda: prof)
    monkeypatch.setattr(pipeline, "expand_queries", lambda p: [])
    monkeypatch.setattr(pipeline.settings, "semantic_recall_threshold", 0.5)
    monkeypatch.setattr(pipeline.settings, "min_match_score", 0.5)
    yield


_RAW = [
    RawJob(source="t", title="Great Fit", company="Acme",
           url="http://x/1", location="Mars", description="great fit role"),
    RawJob(source="t", title="Bad Fit", company="Beta",
           url="http://x/2", location="Chicago, IL", description="unrelated role"),
]


def test_semantic_score_and_recall_filter_ignore_location(monkeypatch):
    # Embedding [1,0] -> score 1.0 for "great fit", [0,1] -> 0.0 for "bad fit".
    def fake_semantic(text, prof_emb):
        emb = [1.0, 0.0] if "great fit" in text.lower() else [0.0, 1.0]
        from jobbot import embeddings
        return round(embeddings.cosine(emb, prof_emb), 4), emb
    monkeypatch.setattr(pipeline, "semantic_score", fake_semantic)

    stats = pipeline.run_scrape_cycle(send_digest=False)

    with session() as db:
        great = db.query(Job).filter(Job.title == "Great Fit").one()
        bad = db.query(Job).filter(Job.title == "Bad Fit").one()
    # Far location ("Mars") but strong fit -> kept in range; embedding stored.
    assert great.match_score == pytest.approx(1.0)
    assert great.embedding and "1.0" in great.embedding
    # Local but weak fit -> below threshold -> out of range (stored, not in-range).
    assert bad.match_score == pytest.approx(0.0)
    assert stats["in_range_stored"] == 1


def test_falls_back_to_bag_of_words_when_embed_down(monkeypatch):
    def boom(text, prof_emb): raise RuntimeError("ollama down")
    monkeypatch.setattr(pipeline, "semantic_score", boom)
    # Should not raise; jobs still stored with legacy scores.
    stats = pipeline.run_scrape_cycle(send_digest=False)
    with session() as db:
        assert db.query(Job).count() == 2
