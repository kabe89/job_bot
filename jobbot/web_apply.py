"""Adapt the synchronous apply_runner.run_apply into a web-reviewable flow.

run_apply fills the real form in a headed Chromium, then calls a `confirm`
callback to decide submit/skip. On the CLI that's a blocking input(); here we
supply a callback that PUBLISHES the read-back results into a run registry and
BLOCKS on a threading.Event until the web UI posts a decision. A /status route
polls get_state(); a /decision route calls submit_decision()."""
from __future__ import annotations

import logging
import threading
from itertools import count
from typing import Callable, Dict, List, Optional

from .config import settings

log = logging.getLogger("jobbot.web_apply")

_runs: Dict[int, dict] = {}
_lock = threading.Lock()
_token_seq = count(1)

# Cap the in-memory run registry so a long-running Flask process doesn't
# accumulate finished runs forever. Only terminal runs are evicted.
_MAX_RUNS = 200
_TERMINAL_PHASES = ("done", "error", "checkpoint")


def _new_token() -> int:
    with _lock:
        return next(_token_seq)


def _prune_runs_locked() -> None:
    """Evict the oldest TERMINAL runs so `_runs` stays bounded. Caller must hold
    `_lock`. Active runs (starting/awaiting_review/finishing) are never evicted —
    the UI is still polling them. Tokens increase monotonically, so dict
    insertion order is oldest-first."""
    if len(_runs) <= _MAX_RUNS:
        return
    for token in list(_runs):
        if len(_runs) <= _MAX_RUNS:
            break
        if _runs[token]["phase"] in _TERMINAL_PHASES:
            del _runs[token]


def get_state(token: int) -> dict:
    keys = ("token", "job_id", "phase", "run_id", "results",
            "screenshot", "error", "confirmation")
    with _lock:
        run = _runs.get(token)
        if not run:
            # Full shape (not just token/phase/error) so the review template can
            # iterate results/screenshot without raising on an unknown token.
            return {"token": token, "job_id": None, "phase": "error",
                    "run_id": None, "results": [], "screenshot": "",
                    "error": "unknown token", "confirmation": ""}
        return {k: run[k] for k in keys}


def submit_decision(token: int, choice: str) -> bool:
    """Record the user's submit/skip decision and release the worker."""
    if choice not in ("submit", "skip"):
        return False
    with _lock:
        run = _runs.get(token)
        if not run or run["phase"] != "awaiting_review":
            return False
        run["decision"] = choice
        run["phase"] = "finishing"
        run["event"].set()
    return True


def _lookup_run_status(run_id) -> str:
    """Best-effort read of the ApplyRun.status for a finished run. Fail-open:
    returns '' if the id isn't a real ApplyRun (e.g. a test's fake runner) or on
    any DB error, so the worker degrades to reporting a plain 'done'."""
    try:
        from .models import ApplyRun, session
        with session() as db:
            run = db.get(ApplyRun, run_id)
            return run.status if run else ""
    except Exception as e:  # noqa: BLE001
        log.info("web-apply: could not read ApplyRun %r status: %s", run_id, e)
        return ""


def start_web_apply(job_id: int, *, runner: Optional[Callable] = None) -> int:
    """Launch run_apply in a background thread and return a polling token."""
    from .apply_runner import run_apply
    runner = runner or run_apply
    token = _new_token()
    ev = threading.Event()
    with _lock:
        _runs[token] = {"token": token, "job_id": job_id, "phase": "starting",
                        "run_id": None, "results": [], "screenshot": "",
                        "error": "", "confirmation": "", "decision": None,
                        "event": ev}
        _prune_runs_locked()  # keep the registry bounded (evicts old terminal runs)

    def _web_confirm(results: List[dict], screenshot_path: str) -> str:
        with _lock:
            run = _runs[token]
            run["results"] = results
            run["screenshot"] = screenshot_path
            run["phase"] = "awaiting_review"
        # Block until the web UI posts a decision (or the safety timeout).
        got = ev.wait(timeout=settings.web_apply_decision_timeout)
        with _lock:
            run = _runs[token]
            decision = run["decision"]
            if not got or decision is None:
                # Close the review gate under the lock BEFORE returning skip, so a
                # late submit_decision() (fired after the timeout but before the
                # worker finishes) is rejected instead of falsely reporting
                # "submitted" for a run that already took the skip path.
                run["phase"] = "finishing"
                run["decision"] = "skip"
        if not got or decision is None:
            log.warning("web-apply %d: decision timed out — skipping.", token)
            return "skip"
        return decision

    def _web_edit(results: List[dict]):
        return []  # web review is submit/skip only; edits happen in the Apply Kit

    def _web_pause_for_login(reason: str) -> None:
        # Headed browser is visible to the user; just note it and continue.
        with _lock:
            _runs[token]["confirmation"] = f"Paused: {reason}"

    def _worker():
        try:
            run_id = runner(job_id, confirm=_web_confirm, edit=_web_edit,
                            pause_for_login=_web_pause_for_login,
                            headed=settings.playwright_headed)
            # Map the real ApplyRun outcome to a web phase instead of always
            # reporting "done": run_apply can return early on a CAPTCHA/login
            # wall ("checkpoint") WITHOUT ever calling confirm(), so a blind
            # "done" would tell the user the application finished while a real
            # Chromium window is stuck waiting for a human.
            ats_status = _lookup_run_status(run_id)
            with _lock:
                run = _runs[token]
                run["run_id"] = run_id
                if ats_status == "checkpoint":
                    run["phase"] = "checkpoint"
                    run["confirmation"] = ("Needs a manual step (CAPTCHA / login) "
                                           "— finish it in the browser window.")
                elif ats_status in ("failed", "error"):
                    run["phase"] = "error"
                    run["error"] = run["error"] or f"apply run status: {ats_status}"
                else:
                    # submitted | skipped | anything terminal-benign
                    run["phase"] = "done"
                    if ats_status:
                        run["confirmation"] = run["confirmation"] or ats_status
        except Exception as e:  # noqa: BLE001
            log.exception("web-apply %d failed", token)
            with _lock:
                run = _runs[token]
                run["phase"] = "error"
                run["error"] = repr(e)

    threading.Thread(target=_worker, daemon=True,
                     name=f"web-apply-{token}").start()
    return token
