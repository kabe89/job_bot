from datetime import datetime

import pytest

from jobbot.browser_apply import mark_applied
from jobbot.models import Application, Job, OutcomeEvent, init_db, session


@pytest.fixture(autouse=True)
def _clean_tables():
    # The test DB is a single session-scoped temp SQLite file (see
    # tests/conftest.py), not reset per-test. _app() always seeds a Job with
    # the same deterministic hash, so without this cleanup the second test in
    # this file would hit a UNIQUE constraint on jobs.hash before it ever
    # reached mark_applied. Matches the pattern in tests/test_apply_runner.py.
    init_db()
    with session() as db:
        db.query(OutcomeEvent).delete()
        db.query(Application).delete()
        db.query(Job).delete()
        db.commit()
    yield


def _app() -> int:
    """A job with one draft application. Returns the application id."""
    init_db()
    with session() as db:
        job = Job(hash=Job.make_hash("t", "http://x", "T"), source="t",
                  title="T", company="C", url="http://x")
        db.add(job)
        db.flush()
        app = Application(job_id=job.id)
        db.add(app)
        db.commit()
        return app.id


def test_mark_applied_logs_one_applied_event():
    aid = _app()
    mark_applied(aid)
    with session() as db:
        events = db.query(OutcomeEvent).filter_by(application_id=aid).all()
        assert [e.stage for e in events] == ["applied"]


def test_mark_applied_is_idempotent():
    # record_outcome appends unconditionally, so a second call would log a
    # duplicate applied event and skew the Bayesian priors' denominator.
    aid = _app()
    mark_applied(aid)
    mark_applied(aid)
    with session() as db:
        events = db.query(OutcomeEvent).filter_by(application_id=aid).all()
        assert len(events) == 1


def test_mark_applied_leaves_status_sent():
    # Ordering regression: status must be set to "sent" BEFORE record_outcome
    # runs. If it ran first, status would still be "draft" (rank 0) and
    # record_outcome would overwrite it with "applied" -- and followup.py, which
    # filters on status == "sent", would stop seeing the application.
    aid = _app()
    mark_applied(aid)
    with session() as db:
        app = db.get(Application, aid)
        assert app.status == "sent"
        assert app.job.status == "applied"


def test_mark_applied_honours_a_back_dated_when():
    aid = _app()
    when = datetime(2026, 7, 17, 14, 30)
    mark_applied(aid, when=when)
    with session() as db:
        assert db.get(Application, aid).sent_at == when


def test_pipeline_send_also_logs_the_funnel_event(monkeypatch):
    # pipeline.py had its own `app.status = "sent"` writer that skipped the
    # funnel entirely -- a second source of applications missing from the priors.
    from jobbot import pipeline
    aid = _app()
    pipeline._record_sent(aid)
    with session() as db:
        events = db.query(OutcomeEvent).filter_by(application_id=aid).all()
        assert [e.stage for e in events] == ["applied"]
        assert db.get(Application, aid).status == "sent"


def test_evening_submit_renders_as_the_same_local_day():
    # 8:30pm EDT on 2026-07-18 is 00:30 UTC on the 19th. Rendering the stored
    # UTC value raw showed tomorrow's date for any evening application -- on a
    # feature whose whole point is the date submitted.
    from jobbot.web import _localdt
    stored_utc = datetime(2026, 7, 19, 0, 30)
    assert _localdt(stored_utc, "%Y-%m-%d") == "2026-07-18"


def test_localdt_passes_none_through():
    from jobbot.web import _localdt
    assert _localdt(None, "%Y-%m-%d") == ""


def test_mark_applied_does_not_rebump_a_back_dated_send():
    # A redundant bare re-mark must not overwrite a previously back-dated
    # sent_at with "now" -- the funnel event is idempotent, the date write was
    # not. A correction with an explicit `when` still updates it (below).
    aid = _app()
    when = datetime(2026, 7, 17, 14, 30)
    mark_applied(aid, when=when)
    mark_applied(aid)
    with session() as db:
        assert db.get(Application, aid).sent_at == when


def test_mark_applied_honours_a_later_explicit_correction():
    aid = _app()
    mark_applied(aid, when=datetime(2026, 7, 17, 14, 30))
    corrected = datetime(2026, 7, 10, 9, 0)
    mark_applied(aid, when=corrected)
    with session() as db:
        assert db.get(Application, aid).sent_at == corrected


def test_fill_plan_pauses_on_a_needs_review_answer():
    # A required question answered only by an unverified guess is flagged
    # needs_review with a POPULATED answer (needs_user stays False). fill_plan
    # must still emit a pause-user step for it -- otherwise, with autosubmit on,
    # a model guess reaches the employer unreviewed. The planner used to gate
    # only on needs_user / blank, so this answer sailed through as a fill step.
    from jobbot.browser_apply import fill_plan
    from jobbot.apply_questions import FormQuestion, questions_to_json
    q = FormQuestion(text="Why do you want this job?", qtype="textarea",
                     required=True, answer="Because I love it.",
                     answer_source="ai", needs_review=True)
    init_db()
    with session() as db:
        job = Job(hash=Job.make_hash("t", "http://x", "T"), source="t",
                  title="T", company="C", url="http://x")
        db.add(job)
        db.flush()
        db.add(Application(job_id=job.id,
                           questions_json=questions_to_json([q], "greenhouse")))
        db.commit()
        job_id = job.id
    plan = fill_plan(job_id)
    step = next(s for s in plan["steps"] if s.get("label") == q.text)
    assert step["action"] == "pause-user"
    assert q.text in plan["needs_user"]


def test_cli_mark_applied_marks_the_jobs_application():
    from click.testing import CliRunner

    from jobbot.cli import cli
    aid = _app()
    with session() as db:
        job_id = db.get(Application, aid).job_id
    result = CliRunner().invoke(cli, ["mark-applied", str(job_id),
                                      "--date", "2026-07-17"])
    assert result.exit_code == 0, result.output
    with session() as db:
        app = db.get(Application, aid)
        assert app.status == "sent"
        assert app.sent_at.date().isoformat() == "2026-07-17"
