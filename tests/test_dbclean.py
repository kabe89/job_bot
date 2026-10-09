"""Tests for `jobbot clean` — liveness sweep_all, embedding reclaim, duplicate
application merge."""
from datetime import datetime, timedelta

import pytest

from jobbot import liveness, models


class _Resp:
    def __init__(self, status_code=200, text="", url="http://x"):
        self.status_code = status_code
        self.text = text
        self.url = url


def _alive_body():
    return _Resp(200, "Apply now. Responsibilities. Requirements.")


@pytest.fixture
def db(tmp_path, monkeypatch):
    """A throwaway DB bound into models AND liveness (which imported by name)."""
    from sqlalchemy import create_engine
    from sqlalchemy.orm import sessionmaker

    monkeypatch.setattr(models.settings, "db_path", str(tmp_path / "clean.db"))
    eng = create_engine(f"sqlite:///{models.settings.db_path}", future=True)
    monkeypatch.setattr(models, "_engine", eng)
    monkeypatch.setattr(models, "SessionLocal", sessionmaker(bind=eng, future=True))
    models.init_db()
    monkeypatch.setattr(liveness, "session", models.session)
    monkeypatch.setattr(liveness, "Job", models.Job)
    return models


def _job(db, **kw):
    defaults = dict(source="s", title="t", company="c", status="new",
                    url="http://live/job")
    defaults.update(kw)
    with db.session() as s:
        j = db.Job(**defaults)
        s.add(j)
        s.commit()
        return j.id


# ---------------------------------------------------------------- sweep_all

def test_sweep_all_skips_a_recently_checked_job(db):
    fresh = _job(db, hash="fresh", last_checked_at=datetime.utcnow())
    stale = _job(db, hash="stale", last_checked_at=None)

    probed = []

    def fetch(url, **k):
        probed.append(url)
        return _alive_body()

    stats = liveness.sweep_all(fetch=fetch, recheck_after_hours=24)
    assert stats["checked"] == 1, "only the never-checked job should be probed"
    assert stats["skipped"] == 1
    with db.session() as s:
        assert s.get(db.Job, stale).last_checked_at is not None
        assert s.get(db.Job, fresh).last_checked_at is not None


def test_sweep_all_stamps_last_checked_even_when_the_probe_errors(db):
    jid = _job(db, hash="boom")

    def fetch(url, **k):
        raise RuntimeError("network down")

    stats = liveness.sweep_all(fetch=fetch)
    assert stats["errors"] == 1
    assert stats["expired"] == 0, "an error must never expire a job"
    with db.session() as s:
        job = s.get(db.Job, jid)
        assert job.last_checked_at is not None, "an errored probe must still advance the cursor"
        assert job.status == "new"


def test_sweep_all_aborts_on_an_error_burst_without_expiring(db):
    for i in range(6):
        _job(db, hash=f"j{i}")

    def fetch(url, **k):
        raise RuntimeError("WAF")

    stats = liveness.sweep_all(fetch=fetch, workers=1,
                               error_abort_min=3, error_abort_ratio=0.5)
    assert stats["aborted"] is True
    assert stats["expired"] == 0
    assert stats["checked"] < 6, "the breaker should stop before the whole pool"


def test_sweep_all_expires_only_the_dead(db):
    dead = _job(db, hash="d", url="http://dead/job")
    live = _job(db, hash="l", url="http://live/job")

    def fetch(url, **k):
        return _Resp(404) if "dead" in url else _alive_body()

    stats = liveness.sweep_all(fetch=fetch)
    assert stats["expired"] == 1 and stats["alive"] == 1
    with db.session() as s:
        assert s.get(db.Job, dead).status == "expired"
        assert s.get(db.Job, live).status == "new"


# ------------------------------------------------------------- embeddings

def test_clear_expired_embeddings_spares_active_jobs(db):
    from jobbot import dbclean

    active = _job(db, hash="a", embedding="[1.0, 2.0]")
    gone = _job(db, hash="b", embedding="[3.0]", status="expired",
                expired_at=datetime.utcnow())

    dbclean.clear_expired_embeddings(apply=True)
    with db.session() as s:
        assert s.get(db.Job, active).embedding == "[1.0, 2.0]"
        assert not s.get(db.Job, gone).embedding


def test_clear_expired_embeddings_dry_run_changes_nothing(db):
    from jobbot import dbclean

    gone = _job(db, hash="b", embedding="[3.0]", status="expired",
                expired_at=datetime.utcnow())
    report = dbclean.clear_expired_embeddings(apply=False)
    assert report["jobs"] == 1
    with db.session() as s:
        assert s.get(db.Job, gone).embedding == "[3.0]"


# --------------------------------------------------------- duplicate apps

def _two_apps(db):
    """A job with two application rows: the older one carries the outcome
    event, the newer one carries the real kit (questions_json)."""
    jid = _job(db, hash="dup")
    with db.session() as s:
        old = db.Application(job_id=jid, status="sent", questions_json="",
                             sent_at=datetime.utcnow() - timedelta(days=1))
        new = db.Application(job_id=jid, status="draft",
                             questions_json='{"source": "ashby", "questions": []}')
        s.add_all([old, new])
        s.commit()
        s.add(db.OutcomeEvent(application_id=old.id, stage="applied"))
        s.commit()
        return jid, old.id, new.id


def test_duplicate_plan_picks_the_kit_row_as_survivor(db):
    from jobbot import dbclean

    jid, old_id, new_id = _two_apps(db)
    plan = dbclean.duplicate_app_plan()
    assert len(plan) == 1
    assert plan[0]["survivor"] == new_id, "the row holding questions_json wins"
    assert plan[0]["losers"] == [old_id]


def test_merge_repoints_outcome_events_and_never_deletes_them(db):
    from jobbot import dbclean

    jid, old_id, new_id = _two_apps(db)
    dbclean.merge_duplicate_apps(apply=True)
    with db.session() as s:
        events = s.query(db.OutcomeEvent).all()
        assert len(events) == 1, "the funnel event must survive the merge"
        assert events[0].application_id == new_id
        assert s.get(db.Application, old_id) is None
        survivor = s.get(db.Application, new_id)
        assert survivor.status == "sent", "furthest-along status carries forward"
        assert survivor.sent_at is not None


def test_merge_dry_run_changes_nothing(db):
    from jobbot import dbclean

    jid, old_id, new_id = _two_apps(db)
    dbclean.merge_duplicate_apps(apply=False)
    with db.session() as s:
        assert s.get(db.Application, old_id) is not None
        assert s.query(db.OutcomeEvent).first().application_id == old_id


def test_merge_leaves_a_single_application_job_untouched(db):
    from jobbot import dbclean

    jid = _job(db, hash="solo")
    with db.session() as s:
        s.add(db.Application(job_id=jid, status="draft"))
        s.commit()
    assert dbclean.duplicate_app_plan() == []
    report = dbclean.merge_duplicate_apps(apply=True)
    assert report["jobs"] == 0
    with db.session() as s:
        assert s.query(db.Application).count() == 1


# ---------------------------------------------------------------- CLI wiring

def test_clean_duplicate_apps_cli_is_dry_run_by_default(db, monkeypatch):
    """The CLI must not mutate without --apply: the default has to be safe."""
    from click.testing import CliRunner
    from jobbot.cli import cli

    jid, old_id, new_id = _two_apps(db)
    result = CliRunner().invoke(cli, ["clean", "duplicate-apps"])
    assert result.exit_code == 0, result.output
    with db.session() as s:
        assert s.get(db.Application, old_id) is not None, "dry run must not delete"


def test_clean_embeddings_cli_reports_without_writing(db):
    from click.testing import CliRunner
    from jobbot.cli import cli

    gone = _job(db, hash="z", embedding="[9.0]", status="expired")
    result = CliRunner().invoke(cli, ["clean", "embeddings"])
    assert result.exit_code == 0, result.output
    with db.session() as s:
        assert s.get(db.Job, gone).embedding == "[9.0]"


def test_clean_group_is_registered_when_cli_runs_as_a_module():
    """Guards a real bug: a command defined BELOW cli.py's
    `if __name__ == "__main__"` guard registers fine on import (so CliRunner
    sees it) but not when the module is exec'd as __main__, because cli() has
    already run. Only a subprocess exercises that path."""
    import subprocess
    import sys

    out = subprocess.run([sys.executable, "-m", "jobbot.cli", "clean", "--help"],
                         capture_output=True, text=True, timeout=120)
    assert out.returncode == 0, out.stderr or out.stdout
    assert "duplicate-apps" in out.stdout
