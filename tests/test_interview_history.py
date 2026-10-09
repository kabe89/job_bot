"""Session history: the analytics read model + its two front-ends."""
import json

import pytest
from click.testing import CliRunner

from jobbot import cli as cli_mod
from jobbot import interview_coach as ic
from jobbot import web as web_mod
from jobbot.models import InterviewSession, InterviewTurn, Job, init_db, session


def _job(db, **kw):
    d = dict(hash=kw.pop("hash", "h1"), source="greenhouse:test",
             title="Research Scientist", company="Acme Bio", url="https://x/1",
             description="d", match_score=0.7, status="new",
             interview_question_bank=json.dumps(
                 {"behavioral": [{"q": "B1", "grounded_in": "jd"}],
                  "technical": [], "research": [], "contact": []}))
    d.update(kw)
    j = Job(**d); db.add(j); db.commit(); db.refresh(j); return j


def _ended(db, job_id, mode, score, n_turns=1):
    from datetime import datetime
    s = InterviewSession(job_id=job_id, mode=mode, overall_score=score,
                         ended_at=datetime.utcnow(), summary="s")
    db.add(s); db.commit(); db.refresh(s)
    for i in range(n_turns):
        db.add(InterviewTurn(session_id=s.id, idx=i, question=f"Q{i}",
                             question_kind=mode, answer="a", score=score,
                             rubric_json=json.dumps({"relevance": score})))
    db.commit()
    return s


def test_history_lists_sessions_newest_first_with_per_mode_means():
    init_db()
    with session() as db:
        jid = _job(db, hash="h_list").id
        _ended(db, jid, "behavioral", 3.0)
        _ended(db, jid, "behavioral", 5.0)
        _ended(db, jid, "research", 4.0)
    out = ic.history(jid)
    assert len(out["sessions"]) == 3
    created = [s["created_at"] for s in out["sessions"]]
    assert created == sorted(created, reverse=True)
    assert out["by_mode"]["behavioral"] == 4.0     # mean(3, 5)
    assert out["by_mode"]["research"] == 4.0
    assert out["mean"] == 4.0                       # mean(3, 5, 4)
    assert out["trend"] == [3.0, 5.0, 4.0]          # oldest -> newest


def test_history_excludes_open_sessions_from_the_stats():
    init_db()
    with session() as db:
        jid = _job(db, hash="h_open").id
        _ended(db, jid, "behavioral", 4.0)
        db.add(InterviewSession(job_id=jid, mode="behavioral"))  # still open
        db.commit()
    out = ic.history(jid)
    assert len(out["sessions"]) == 2               # both listed
    assert out["mean"] == 4.0                      # only the ended one scores
    assert out["trend"] == [4.0]


def test_history_with_no_job_id_spans_every_job():
    init_db()
    with session() as db:
        a = _job(db, hash="h_all_a", company="Acme Bio").id
        b = _job(db, hash="h_all_b", company="Other Bio").id
        _ended(db, a, "behavioral", 4.0)
        _ended(db, b, "technical", 2.0)
    out = ic.history()
    # Isolation-robust: the session-scoped test DB may carry sessions from
    # earlier tests, so assert both jobs' sessions are spanned rather than an
    # exact count over the whole DB.
    assert {"Acme Bio", "Other Bio"} <= {s["company"] for s in out["sessions"]}


def test_history_is_empty_and_neutral_with_no_sessions():
    init_db()
    out = ic.history(job_id=987654)
    assert out == {"sessions": [], "by_mode": {}, "trend": [], "mean": None}


def test_interviews_page_renders():
    init_db()
    with session() as db:
        jid = _job(db, hash="h_page").id
        _ended(db, jid, "behavioral", 4.0)
    app = web_mod.create_app()
    app.config["TESTING"] = True
    with app.test_client() as c:
        r = c.get("/interviews")
    assert r.status_code == 200
    assert b"Acme Bio" in r.data


def test_history_cli_lists_sessions():
    init_db()
    with session() as db:
        jid = _job(db, hash="h_cli").id
        _ended(db, jid, "behavioral", 4.0)
    res = CliRunner().invoke(cli_mod.cli, ["interview", "history", str(jid)])
    assert res.exit_code == 0, res.output
    assert "behavioral" in res.output and "4.0" in res.output
    assert res.output.isascii()


def test_history_cli_prints_a_read_only_transcript():
    init_db()
    with session() as db:
        jid = _job(db, hash="h_cli_t").id
        sid = _ended(db, jid, "behavioral", 4.0, n_turns=2).id
    res = CliRunner().invoke(cli_mod.cli, ["interview", "history", "--session", str(sid)])
    assert res.exit_code == 0, res.output
    assert "Q0" in res.output and "Q1" in res.output
    assert res.output.isascii()
