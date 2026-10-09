"""Interview-readiness signal: bounded, never a penalty, monotonic in score."""
from datetime import datetime

import pytest

from jobbot import interview_coach as ic
from jobbot import predict
from jobbot.models import InterviewSession, Job, init_db, session


def _job(db, **kw):
    d = dict(hash=kw.pop("hash", "p1"), source="greenhouse:test",
             title="Research Scientist", company="Acme Bio", url="https://x/1",
             description="d", match_score=0.7, status="new")
    d.update(kw)
    j = Job(**d); db.add(j); db.commit(); db.refresh(j); return j


def _sessions(db, job_id, scores):
    for s in scores:
        db.add(InterviewSession(job_id=job_id, mode="general", overall_score=s,
                                ended_at=datetime.utcnow()))
    db.commit()


def test_zero_sessions_is_neutral_never_a_penalty():
    init_db()
    with session() as db:
        jid = _job(db, hash="p_none").id
    bonus, why = ic.readiness_bonus(jid)
    assert bonus == 0.0 and why == ""


def test_open_sessions_do_not_count():
    init_db()
    with session() as db:
        jid = _job(db, hash="p_open").id
        db.add(InterviewSession(job_id=jid, mode="general", overall_score=5.0))
        db.commit()
    assert ic.readiness_bonus(jid)[0] == 0.0


def test_a_bad_score_is_never_negative():
    init_db()
    with session() as db:
        jid = _job(db, hash="p_bad").id
        _sessions(db, jid, [0.0, 1.0])
    assert ic.readiness_bonus(jid)[0] == 0.0


def test_bonus_is_monotonic_in_the_best_score():
    init_db()
    out = []
    for n, best in enumerate([3.0, 4.0, 5.0]):
        with session() as db:
            jid = _job(db, hash=f"p_mono{n}").id
            _sessions(db, jid, [best, best, best])
        out.append(ic.readiness_bonus(jid)[0])
    assert out == sorted(out) and out[0] < out[-1]


def test_bonus_never_exceeds_the_user_set_cap(monkeypatch):
    monkeypatch.setattr(ic.settings, "interview_practice_weight", 0.05)
    init_db()
    with session() as db:
        jid = _job(db, hash="p_cap").id
        _sessions(db, jid, [5.0] * 10)
    bonus, why = ic.readiness_bonus(jid)
    assert bonus == pytest.approx(0.05)
    assert "practice" in why.lower()


def test_the_cap_is_user_settable(monkeypatch):
    monkeypatch.setattr(ic.settings, "interview_practice_weight", 0.20)
    init_db()
    with session() as db:
        jid = _job(db, hash="p_cap2").id
        _sessions(db, jid, [5.0] * 10)
    assert ic.readiness_bonus(jid)[0] == pytest.approx(0.20)


def test_a_single_session_earns_only_partial_credit(monkeypatch):
    monkeypatch.setattr(ic.settings, "interview_practice_weight", 0.05)
    init_db()
    with session() as db:
        one = _job(db, hash="p_one").id
        _sessions(db, one, [5.0])
        many = _job(db, hash="p_many").id
        _sessions(db, many, [5.0, 5.0, 5.0])
    assert ic.readiness_bonus(one)[0] < ic.readiness_bonus(many)[0]


def test_readiness_bonus_is_fail_open(monkeypatch):
    def boom(*a, **k):
        raise RuntimeError("db gone")
    monkeypatch.setattr(ic, "session", boom)
    assert ic.readiness_bonus(1) == (0.0, "")


def test_predict_job_folds_in_the_readiness_bonus(monkeypatch):
    monkeypatch.setattr(ic.settings, "interview_practice_weight", 0.05)
    init_db()
    with session() as db:
        jid = _job(db, hash="p_predict").id
        j_before = db.get(Job, jid)
        baseline = predict.predict_job(j_before).callback_probability
        _sessions(db, jid, [5.0, 5.0, 5.0])
        j_after = db.get(Job, jid)
        after = predict.predict_job(j_after)
    assert after.callback_probability > baseline
    assert after.logit_contributions["interview_practice"] == pytest.approx(0.05)


def test_predict_job_unchanged_without_practice():
    init_db()
    with session() as db:
        jid = _job(db, hash="p_nopractice").id
        p = predict.predict_job(db.get(Job, jid))
    assert "interview_practice" not in p.logit_contributions


def test_predict_job_stays_within_the_probability_clamp(monkeypatch):
    monkeypatch.setattr(ic.settings, "interview_practice_weight", 0.5)
    init_db()
    with session() as db:
        jid = _job(db, hash="p_clamp", match_score=0.99).id
        _sessions(db, jid, [5.0] * 5)
        p = predict.predict_job(db.get(Job, jid))
    assert p.callback_probability <= 0.85
