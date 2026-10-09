"""Re-score backfill: recompute match_score for active jobs against the current
profile embedding. Root-cause fix for stale scores — jobs scraped before semantic
ranking was live kept legacy bag-of-words scores and no embedding, so genuinely
relevant roles were underrated (e.g. a comp-bio Research Scientist stuck at 0.30).
"""
import json

from jobbot import ranking
from jobbot.models import Job, init_db, session
from jobbot.profile import CandidateProfile


def _prof(emb):
    return CandidateProfile(
        skills=["x"], role_titles=["Scientist"], role_archetypes=[], domains=[],
        seniority="mid", dealbreakers=[], summary="s",
        source_resume_hash="h", built_at="t", embedding=emb)


def _job(db, **kw):
    d = dict(hash=kw.pop("hash", "j1"), source="greenhouse:test", title="Research Scientist",
             company="Acme", url="https://x", description="comp bio", match_score=0.30,
             status="new", embedding=None)
    d.update(kw)
    j = Job(**d); db.add(j); db.commit(); db.refresh(j); return j


def test_rescore_updates_stale_legacy_score_and_persists_embedding(monkeypatch):
    """A job with no embedding + a low legacy score gets a fresh semantic score,
    and its embedding is stored for cheap future re-scores."""
    monkeypatch.setattr(ranking.embeddings, "embed", lambda t: [1.0, 0.0])
    init_db()
    with session() as db:
        j = _job(db, hash="stale", match_score=0.30, embedding=None)
        jid = j.id
    stats = ranking.rescore_jobs(_prof([1.0, 0.0]))  # cosine == 1.0
    with session() as db:
        got = db.get(Job, jid)
        assert got.match_score == 1.0            # was 0.30 (legacy) -> real cosine
        assert got.embedding                     # embedding persisted
        assert json.loads(got.embedding) == [1.0, 0.0]
    assert stats["embedded"] >= 1 and stats["rescored"] >= 1


def test_rescore_reuses_stored_embedding(monkeypatch):
    """A job that already has an embedding is NOT re-embedded (cheap re-score)."""
    calls = {"n": 0}
    def fake_embed(t):
        calls["n"] += 1
        return [0.0, 1.0]
    monkeypatch.setattr(ranking.embeddings, "embed", fake_embed)
    init_db()
    with session() as db:
        _job(db, hash="hasemb", match_score=0.9, embedding=json.dumps([1.0, 0.0]))
    ranking.rescore_jobs(_prof([1.0, 0.0]))
    assert calls["n"] == 0                        # reused stored embedding, no embed call
    with session() as db:
        got = db.query(Job).filter_by(hash="hasemb").one()
        assert got.match_score == 1.0             # cosine([1,0],[1,0]) == 1.0


def test_rescore_is_fail_open_and_keeps_prior_score(monkeypatch):
    """If embedding a job raises, its previous score is preserved (not zeroed)."""
    def boom(t): raise RuntimeError("ollama down")
    monkeypatch.setattr(ranking.embeddings, "embed", boom)
    init_db()
    with session() as db:
        _job(db, hash="fail", match_score=0.42, embedding=None)
    stats = ranking.rescore_jobs(_prof([1.0, 0.0]))
    with session() as db:
        got = db.query(Job).filter_by(hash="fail").one()
        assert got.match_score == 0.42            # unchanged
    assert stats["failed"] >= 1


def test_rescore_skips_expired_jobs(monkeypatch):
    monkeypatch.setattr(ranking.embeddings, "embed", lambda t: [1.0, 0.0])
    init_db()
    with session() as db:
        _job(db, hash="exp", match_score=0.10, embedding=None, status="expired")
    ranking.rescore_jobs(_prof([1.0, 0.0]))
    with session() as db:
        got = db.query(Job).filter_by(hash="exp").one()
        assert got.match_score == 0.10            # expired left alone


def test_rescore_noop_without_profile_embedding(monkeypatch):
    monkeypatch.setattr(ranking.embeddings, "embed", lambda t: [1.0, 0.0])
    init_db()
    with session() as db:
        _job(db, hash="np", match_score=0.30, embedding=None)
    stats = ranking.rescore_jobs(_prof(None))
    assert stats["rescored"] == 0
    with session() as db:
        got = db.query(Job).filter_by(hash="np").one()
        assert got.match_score == 0.30
