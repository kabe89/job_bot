from datetime import datetime

import pytest

from jobbot.config import settings
from jobbot.models import Application, Job, OutcomeEvent, init_db, session


@pytest.fixture(autouse=True)
def _clean_db():
    """Wipe outcome/app/job rows before AND after each test so this file never
    leaks fixtures into other test files (a leaked Application whose Job a later
    test deletes would 500 the /applications route)."""
    def _wipe():
        init_db()
        with session() as db:
            db.query(OutcomeEvent).delete()
            db.query(Application).delete()
            db.query(Job).delete()
            db.commit()
    _wipe()
    yield
    _wipe()


def _job(db, **kw):
    d = dict(hash=kw.pop("hash", "j1"), source="greenhouse:test", title="Scientist",
             company="Acme", url="https://x", description="d", match_score=0.6)
    d.update(kw)
    j = Job(**d); db.add(j); db.commit(); db.refresh(j); return j


def test_config_defaults_present():
    assert settings.ghost_after_days == 30
    assert settings.outcome_prior_strength == 8
    assert settings.outcome_adjustment_cap == 0.08
    assert settings.outcome_interview_weight == 1.5
    assert settings.outcome_min_trials == 3


def test_outcome_event_table_roundtrips():
    init_db()
    with session() as db:
        j = _job(db)
        a = Application(job_id=j.id, status="applied", sent_at=datetime.utcnow())
        db.add(a); db.commit(); db.refresh(a)
        db.add(OutcomeEvent(application_id=a.id, stage="responded", note="email"))
        db.commit()
        a2 = db.query(Application).get(a.id)
        assert len(a2.outcome_events) == 1
        assert a2.outcome_events[0].stage == "responded"


from datetime import timedelta
import pytest
from jobbot import outcomes


def _app(db, status="applied", sent_days_ago=0):
    from jobbot.models import Application
    j = _job(db, hash=f"j{status}{sent_days_ago}{db.query(Application).count()}")
    sent = datetime.utcnow() - timedelta(days=sent_days_ago)
    a = Application(job_id=j.id, status=status, sent_at=sent)
    db.add(a); db.commit(); db.refresh(a); return a


def test_stage_rank_orders_funnel():
    assert outcomes.stage_rank("applied") < outcomes.stage_rank("responded")
    assert outcomes.stage_rank("interview") < outcomes.stage_rank("offer")
    assert outcomes.stage_rank("rejected") == -1


def test_record_outcome_advances_and_never_regresses():
    init_db()
    with session() as db:
        a = _app(db, status="applied")
        aid = a.id
    outcomes.record_outcome(aid, "interview")
    with session() as db:
        assert db.query(Application).get(aid).status == "interview"
    outcomes.record_outcome(aid, "responded")  # earlier stage
    with session() as db:
        a = db.query(Application).get(aid)
        assert a.status == "interview"           # not regressed
        assert len(a.outcome_events) == 2        # but event still logged


def test_record_outcome_terminal_is_sticky_and_validates():
    init_db()
    with session() as db:
        aid = _app(db, status="applied").id
    outcomes.record_outcome(aid, "rejected")
    with session() as db:
        assert db.query(Application).get(aid).status == "rejected"
    outcomes.record_outcome(aid, "screen")       # positive after terminal
    with session() as db:
        assert db.query(Application).get(aid).status == "rejected"  # sticky
    with pytest.raises(ValueError):
        outcomes.record_outcome(aid, "not_a_stage")


def test_sweep_ghosted_marks_only_silent_past_window():
    init_db()
    with session() as db:
        old = _app(db, status="applied", sent_days_ago=40).id
        recent = _app(db, status="applied", sent_days_ago=5).id
        responded = _app(db, status="applied", sent_days_ago=40).id
    outcomes.record_outcome(responded, "responded")
    n = outcomes.sweep_ghosted()
    assert n == 1
    with session() as db:
        assert db.query(Application).get(old).status == "ghosted"
        assert db.query(Application).get(recent).status == "applied"
        assert db.query(Application).get(responded).status == "responded"
    assert outcomes.sweep_ghosted() == 0          # idempotent


class _FakeJob:
    def __init__(self, source="greenhouse:x", title="Research Scientist",
                 company="Acme", is_remote=False, match_score=0.6, rerank_score=None):
        self.source = source; self.title = title; self.company = company
        self.is_remote = is_remote; self.match_score = match_score
        self.rerank_score = rerank_score


def _mk_app(source, is_remote, title, company, reached):
    """A stand-in application object exposing what compute_priors reads."""
    class A: pass
    a = A()
    a.status = reached
    a.job = _FakeJob(source=source, is_remote=is_remote, title=title, company=company)
    a.outcome_events = []
    return a


def test_beta_smoothing_sparse_is_near_base_not_zero():
    # 0/2 for a feature must stay near the global base rate, not collapse to 0.
    apps = ([_mk_app("greenhouse:x", True, "Scientist", "A", "responded")] * 5 +
            [_mk_app("greenhouse:x", True, "Scientist", "A", "applied")] * 5 +
            [_mk_app("lever:y", False, "Scientist", "B", "applied")] * 2)
    p = outcomes.compute_priors(apps=apps)
    base = p["base_response"]
    # lever has 0/2 responses; smoothed rate stays within a small band of base.
    lever = p["dims"]["source"]["lever"]
    rate = outcomes._smoothed_rate(lever["responses"], lever["trials"], base)
    assert 0.0 < rate < base + 0.01
    assert abs(rate - base) < base  # not collapsed to ~0


def test_adjustment_bounded_and_zero_below_min_trials():
    # A matching group that always converts, plus a differently-featured control
    # group that never does -- so the base rate sits below the matching group's
    # rate and there's an actual signal to detect (an all-identical fixture is
    # perfectly confounded with the global base and yields adj == 0 by
    # construction, which isn't what this test is checking).
    apps = ([_mk_app("greenhouse:x", True, "Scientist", "A", "interview")] * 10 +
            [_mk_app("other:y", False, "Junior Analyst", "Zeta", "applied")] * 10)
    p = outcomes.compute_priors(apps=apps)
    adj, why = outcomes.adjustment_for_job(_FakeJob(source="greenhouse:x", is_remote=True), p)
    assert 0 < adj <= settings.outcome_adjustment_cap + 1e-9
    assert why  # non-empty rationale
    # A source seen once (< min_trials) contributes nothing.
    p2 = outcomes.compute_priors(apps=[_mk_app("rare:z", False, "X", "Q", "applied")])
    adj2, _ = outcomes.adjustment_for_job(_FakeJob(source="rare:z"), p2)
    assert adj2 == 0.0


def test_blended_clamps_and_prefers_rerank():
    p = outcomes.compute_priors(apps=[])           # cold start -> neutral
    j = _FakeJob(match_score=0.5, rerank_score=0.9)
    assert outcomes.blended(j, p) == 0.9           # rerank preferred, no nudge
    j2 = _FakeJob(match_score=0.99, rerank_score=None)
    assert 0.0 <= outcomes.blended(j2, p) <= 1.0   # clamped


def test_adjustment_fail_open_on_bad_priors():
    adj, why = outcomes.adjustment_for_job(_FakeJob(), {"garbage": True})
    assert adj == 0.0 and why == ""


def test_outcome_route_records_and_redirects():
    from jobbot.web import create_app
    init_db()
    with session() as db:
        j = _job(db, hash="route1")
        a = Application(job_id=j.id, status="applied", sent_at=datetime.utcnow())
        db.add(a); db.commit(); aid = a.id
    c = create_app().test_client()
    r = c.post(f"/applications/{aid}/outcome", data={"stage": "responded"})
    assert r.status_code in (302, 303)
    with session() as db:
        assert db.query(Application).get(aid).status == "responded"


def test_outcome_route_unknown_stage_flashes_no_500():
    from jobbot.web import create_app
    init_db()
    with session() as db:
        j = _job(db, hash="route2")
        a = Application(job_id=j.id, status="applied"); db.add(a); db.commit(); aid = a.id
    c = create_app().test_client()
    r = c.post(f"/applications/{aid}/outcome", data={"stage": "bogus"},
               follow_redirects=True)
    assert r.status_code == 200  # flashed, not crashed


def test_applications_page_renders_with_insights():
    from jobbot.web import create_app
    init_db()
    assert create_app().test_client().get("/applications").status_code == 200


def test_applications_page_survives_orphaned_job():
    """An Application whose Job was deleted (leftover FK) must not 500 the page."""
    from jobbot.web import create_app
    init_db()
    with session() as db:
        j = _job(db, hash="orphan1")
        a = Application(job_id=j.id, status="applied", sent_at=datetime.utcnow())
        db.add(a); db.commit()
        db.query(Job).filter(Job.id == j.id).delete()
        db.commit()
    r = create_app().test_client().get("/applications")
    assert r.status_code == 200


def test_cli_outcome_records():
    from click.testing import CliRunner
    from jobbot.cli import cli
    init_db()
    with session() as db:
        j = _job(db, hash="cli1")
        a = Application(job_id=j.id, status="applied"); db.add(a); db.commit(); aid = a.id
    res = CliRunner().invoke(cli, ["outcome", str(aid), "responded"])
    assert res.exit_code == 0
    with session() as db:
        assert db.query(Application).get(aid).status == "responded"


def test_cli_outcomes_insights_runs():
    from click.testing import CliRunner
    from jobbot.cli import cli
    init_db()
    res = CliRunner().invoke(cli, ["outcomes-insights"])
    assert res.exit_code == 0


def test_sent_ranks_equal_to_applied():
    # "sent" is the storage spelling of the applied stage. Ranking it -1 (the
    # unknown-stage value) let record_outcome overwrite a sent application's
    # status, which followup.py and predict.py both filter on.
    assert outcomes.stage_rank("sent") == outcomes.stage_rank("applied") == 1

