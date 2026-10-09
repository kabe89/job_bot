import threading
import time
from jobbot import web_apply


def test_runs_registry_is_bounded_and_keeps_active(monkeypatch):
    # A long-running process must not accumulate finished runs forever, but an
    # active (still-polled) run must never be evicted.
    monkeypatch.setattr(web_apply, "_runs", {}, raising=True)
    monkeypatch.setattr(web_apply, "_MAX_RUNS", 5, raising=True)
    web_apply._runs[1] = {"token": 1, "phase": "awaiting_review",
                          "event": threading.Event()}
    for t in range(2, 30):
        web_apply._runs[t] = {"token": t, "phase": "done",
                              "event": threading.Event()}
    with web_apply._lock:
        web_apply._prune_runs_locked()
    assert len(web_apply._runs) <= 5
    assert 1 in web_apply._runs  # active run preserved despite being oldest


def test_get_state_unknown_token_has_full_shape():
    # An unknown token must still return every key the review template reads
    # (results/screenshot/...) so /apply-run/view/<token> can't 500.
    st = web_apply.get_state(999999)
    assert st["phase"] == "error"
    for key in ("token", "job_id", "phase", "run_id", "results",
                "screenshot", "error", "confirmation"):
        assert key in st
    assert st["results"] == []


def test_checkpoint_status_is_not_reported_as_done(monkeypatch):
    # run_apply can return early on a CAPTCHA/login wall ("checkpoint") without
    # calling confirm(); the worker must surface that, not a blind "done".
    monkeypatch.setattr(web_apply, "_lookup_run_status", lambda run_id: "checkpoint")

    def fake_run_apply(job_id, *, confirm, edit, pause_for_login, headed):
        return 555  # returns immediately, never calls confirm (CAPTCHA path)

    token = web_apply.start_web_apply(9, runner=fake_run_apply)
    for _ in range(100):
        ph = web_apply.get_state(token)["phase"]
        if ph in ("checkpoint", "done", "error"):
            break
        time.sleep(0.02)
    st = web_apply.get_state(token)
    assert st["phase"] == "checkpoint"
    assert "CAPTCHA" in st["confirmation"] or "manual" in st["confirmation"].lower()


def _fake_run_apply_factory(recorder):
    """Return a run_apply stand-in that calls the web confirm callback once and
    respects its decision, mimicking apply_runner.run_apply's contract."""
    def fake_run_apply(job_id, *, confirm, edit, pause_for_login, headed):
        results = [{"label": "Resume/CV", "actual": "/tmp/r.pdf", "ok": True,
                    "status": "matched"}]
        decision = confirm(results, "/tmp/shot.png")
        recorder["decision"] = decision
        return 4242
    return fake_run_apply


def test_submit_flow_publishes_results_then_completes():
    rec = {}
    token = web_apply.start_web_apply(
        1, runner=_fake_run_apply_factory(rec))
    # Wait for the worker to reach awaiting_review.
    for _ in range(100):
        if web_apply.get_state(token)["phase"] == "awaiting_review":
            break
        time.sleep(0.02)
    state = web_apply.get_state(token)
    assert state["phase"] == "awaiting_review"
    assert state["results"][0]["label"] == "Resume/CV"

    assert web_apply.submit_decision(token, "submit") is True
    for _ in range(100):
        if web_apply.get_state(token)["phase"] == "done":
            break
        time.sleep(0.02)
    assert web_apply.get_state(token)["phase"] == "done"
    assert rec["decision"] == "submit"


def test_skip_decision_is_relayed():
    rec = {}
    token = web_apply.start_web_apply(2, runner=_fake_run_apply_factory(rec))
    for _ in range(100):
        if web_apply.get_state(token)["phase"] == "awaiting_review":
            break
        time.sleep(0.02)
    web_apply.submit_decision(token, "skip")
    for _ in range(100):
        if web_apply.get_state(token)["phase"] == "done":
            break
        time.sleep(0.02)
    assert rec["decision"] == "skip"
