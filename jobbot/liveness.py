"""Probe job-posting URLs and soft-expire ones that no longer exist.

A posting is considered DEAD when the URL 404s/410s, redirects to a bare
careers root, or the page contains an explicit "no longer accepting" phrase.
Dead jobs are flipped to status='expired' (restorable) — never hard-deleted.
Fail-open: network/parse errors count as 'error', never as dead."""
from __future__ import annotations

import logging
from datetime import datetime, timedelta
from typing import Callable, Optional, Tuple

from .config import settings
from .models import Job, init_db, session

try:
    import truststore
    truststore.inject_into_ssl()
except Exception:
    pass

log = logging.getLogger("jobbot.liveness")

DEAD_PHRASES = (
    # Direct "no longer" statements
    "no longer accepting applications",
    "no longer accepting new applications",
    "no longer accepting",
    "not currently accepting applications",
    "not accepting applications",
    "no longer open",
    "no longer active",
    "is no longer active",
    "no longer available",
    "The vacancy you are looking for does not exist",
    "is no longer available",
    "this position is no longer available",
    "this position is no longer",
    "this job is no longer available",
    "this job is no longer active",
    "this job is no longer",
    "this role is no longer available",
    "this role is no longer",
    "this posting is no longer available",
    "this posting is no longer active",
    "this posting is no longer",
    "this listing is no longer available",
    "this listing is no longer",
    "this vacancy is no longer",
    "job posting is no longer active",
    "posting is no longer active",
    "position is no longer active",
    # Filled positions
    "position has been filled",
    "position is filled",
    "job has been filled",
    "job is filled",
    "role has been filled",
    "role is filled",
    "opening has been filled",
    "has been filled",
    # Expired listings
    "this listing has expired",
    "this listing is expired",
    "listing has expired",
    "listing is expired",
    "this job has expired",
    "this job is expired",
    "job has expired",
    "job is expired",
    "this position has expired",
    "this role has expired",
    "posting has expired",
    "posting is expired",
    "this job post has expired",
    # Closed positions
    "posting has closed",
    "posting is closed",
    "position has closed",
    "position is closed",
    "this position has closed",
    "this position is closed",
    "job has closed",
    "job is closed",
    "this job has closed",
    "this job is closed",
    "this role has closed",
    "this role is closed",
    "applications are closed",
    "applications have closed",
    "applications are now closed",
    "application is closed",
    "application has closed",
    "requisition is closed",
    "requisition has been closed",
    "requisition is no longer",
    "this vacancy is closed",
    "vacancy has closed",
    "vacancy is closed",
    # Removals and 404 / generic copies
    "job has been removed",
    "job was removed",
    "this job has been removed",
    "this job was removed",
    "this listing has been removed",
    "job no longer exists",
    "page no longer exists",
    "the job you are looking for",
    "the page you are looking for",
    "the position you are looking for",
    "page you are looking for doesn't exist",
    "page you are looking for does not exist",
    "could not be found",
    "cannot be found",
    "job not found",
    "posting not found",
    "position not found",
    "could not find that job",
    "couldn't find that job",
    "can't find that page",
    "cannot find that page",
    # Polite rejection headers
    "sorry, this job is no longer",
    "sorry, this position is no longer",
    "sorry, this role is no longer",
    "sorry! this job is no longer",
    "we're sorry, this job is no longer",
    "we are sorry, this job is no longer",
    "thank you for your interest, but this position",
    "thank you for your interest, but this job",
    "thank you for your interest, this position",
    "thank you for your interest, this job",
    "thanks for your interest, but this position",
    "thanks for your interest, but this job",
    # ATS-specific artifacts
    "closed-job",
    "job-alert-banner",
    "trk=expired_jd_redirect",
    "trk=expired_link",
)

_ACTIVE_STATUSES = ("new", "tailored", "starred")


def _is_workday(url: str) -> bool:
    from .apply_questions import WORKDAY_RE
    return bool(url and WORKDAY_RE.match(url))


def _default_fetch(url: str, *, timeout: int):
    import requests
    return requests.get(
        url,
        timeout=timeout,
        allow_redirects=True,
        headers={
            "User-Agent": (
                "Mozilla/5.0 (Windows NT 10.0; Win64; x64) "
                "AppleWebKit/537.36 (KHTML, like Gecko) "
                "Chrome/126.0.0.0 Safari/537.36"
            ),
            "Accept": "text/html,application/xhtml+xml,application/xml;q=0.9,*/*;q=0.8",
            "Accept-Language": "en-US,en;q=0.9",
        },
    )


def _workday_state(url: str) -> str:
    """'ok' | 'gone' | 'error' for a Workday posting, via CXS endpoints."""
    from .apply_questions import workday_job_status
    from .scrapers.workday import fetch_posting

    # First check via structured CXS parser in apply_questions
    try:
        ws = workday_job_status(url)
        if ws.get("can_apply") is False or ws.get("alive") is False:
            return "gone"
    except Exception:
        pass

    # Then verify via scraper fetch_posting
    return fetch_posting(url)[1]


def check_job_alive(url: str, *, fetch: Optional[Callable] = None) -> Tuple[bool, str]:
    """Return (alive, reason). alive=True means keep; False means expire.
    On any fetch error return (True, 'error:...') so we never expire on a fluke."""
    if not url:
        return True, "no-url"

    # Workday serves a JS app shell: a PULLED posting still answers 200 with no
    # "no longer available" text (that message is rendered client-side), so the
    # generic probe below reports every dead Workday job as ALIVE. Ask the CXS
    # endpoint, which distinguishes pulled (403/S22 or canApply=False) from transient.
    if _is_workday(url):
        try:
            state = _workday_state(url)
        except Exception as e:  # noqa: BLE001
            return True, f"error:{e}"[:200]
        if state == "gone":
            return False, "workday: posting pulled or closed (canApply=False / CXS 403)"
        if state == "error":
            return True, "error:workday detail unavailable"
        return True, "alive"

    fetch = fetch or _default_fetch
    try:
        resp = fetch(url, timeout=settings.liveness_timeout)
    except Exception as e:  # noqa: BLE001
        return True, f"error:{e}"[:200]
    code = getattr(resp, "status_code", 200)
    if code in (404, 410):
        return False, f"HTTP {code}"
    if code >= 500:
        return True, f"error:HTTP {code}"   # server hiccup — don't expire

    # Check for redirect to generic careers root or jobs search page
    final_url = getattr(resp, "url", "")
    if final_url and final_url.rstrip("/") != url.rstrip("/"):
        from urllib.parse import urlparse
        orig_parts = [x for x in (urlparse(url).path or "").split("/") if x]
        final_parts = [x for x in (urlparse(final_url).path or "").split("/") if x]
        p_final = (urlparse(final_url).path or "").rstrip("/").lower()
        if len(orig_parts) >= 2 and len(final_parts) < len(orig_parts):
            return False, f"redirected to listing root: {final_url}"
        if any(p_final.endswith(sfx) for sfx in ("/careers", "/jobs", "/search", "/vacancies")):
            return False, f"redirected to careers root: {final_url}"
        final_lower = final_url.lower()
        if any(tok in final_lower for tok in ("trk=expired", "expired_jd_redirect", "expired_link", "job_not_found", "job-not-found", "/expired")):
            return False, f"redirected to expired target: {final_url}"

    text = (getattr(resp, "text", "") or "").lower()
    for phrase in DEAD_PHRASES:
        if phrase in text:
            return False, f"page says '{phrase}'"
    return True, "alive"


def expire_job(job_id: int, reason: str) -> None:
    init_db()
    with session() as db:
        job = db.get(Job, job_id)
        if job:
            job.status = "expired"
            job.expired_at = datetime.utcnow()
            job.expiry_reason = reason[:256]
            db.commit()


def restore_job(job_id: int) -> None:
    init_db()
    with session() as db:
        job = db.get(Job, job_id)
        if job:
            job.status = "new"
            job.expired_at = None
            job.expiry_reason = ""
            db.commit()


def sweep_expired(limit: int = 50, *, fetch: Optional[Callable] = None) -> dict:
    """Probe up to `limit` active, not-yet-applied jobs; expire the dead ones."""
    init_db()
    stats = {"checked": 0, "expired": 0, "alive": 0, "errors": 0}
    with session() as db:
        candidates = (db.query(Job)
                      .filter(Job.status.in_(_ACTIVE_STATUSES))
                      .order_by(Job.discovered_at.asc())
                      .limit(limit).all())
        rows = [(j.id, j.url) for j in candidates]
    for job_id, url in rows:
        stats["checked"] += 1
        alive, reason = check_job_alive(url, fetch=fetch)
        if not alive:
            expire_job(job_id, reason)
            stats["expired"] += 1
        elif reason.startswith("error:"):
            stats["errors"] += 1
        else:
            stats["alive"] += 1
    log.info("Liveness sweep: %s", stats)
    return stats


# --- On-demand full sweep ---------------------------------------------------
#
# `sweep_expired` above is the per-scrape-cycle trickle (40 jobs, ordered by
# discovery). It cannot keep up: at 40 a cycle a 4,500-job pool takes ~115
# cycles to cover once, so most rows go months without a probe and the
# dashboard ranks postings nobody can apply to. `sweep_all` is the on-demand
# version behind `jobbot clean liveness`.
#
# Two properties `sweep_expired` lacks:
#   RESUMABLE  ordered by last_checked_at (NULLs first) and stamped on EVERY
#              outcome, so a re-run continues rather than restarting.
#   BOUNDED    concurrent, but capped per host. 2,469 of these URLs are
#              Workday and 635 are one tenant; an unbounded pool is a
#              self-inflicted denial of service against a company we are
#              actively applying to.

_DEFAULT_WORKERS = 8
_PER_HOST_WORKERS = 2


def _host_of(url: str) -> str:
    from urllib.parse import urlparse
    try:
        return (urlparse(url).hostname or "").lower()
    except Exception:  # noqa: BLE001
        return ""


def _stamp_checked(job_id: int) -> None:
    with session() as db:
        job = db.get(Job, job_id)
        if job:
            job.last_checked_at = datetime.utcnow()
            db.commit()


import threading


class _LivenessProgress:
    """Thread-safe liveness progress tracker singleton."""

    def __init__(self):
        self._lock = threading.RLock()
        self.reset()

    def reset(self):
        with self._lock:
            self.state: str = "idle"  # idle | running | done | error
            self.started_at: Optional[datetime] = None
            self.finished_at: Optional[datetime] = None
            self.total: int = 0
            self.checked: int = 0
            self.expired: int = 0
            self.alive: int = 0
            self.errors: int = 0
            self.skipped: int = 0
            self.aborted: bool = False
            self.error: str = ""

    def start(self, total: int, skipped: int = 0):
        with self._lock:
            self.reset()
            self.state = "running"
            self.started_at = datetime.utcnow()
            self.total = total
            self.skipped = skipped

    def update(self, stats: dict):
        with self._lock:
            self.checked = stats.get("checked", 0)
            self.expired = stats.get("expired", 0)
            self.alive = stats.get("alive", 0)
            self.errors = stats.get("errors", 0)
            self.skipped = stats.get("skipped", self.skipped)
            self.aborted = stats.get("aborted", False)

    def finish(self, stats: Optional[dict] = None):
        with self._lock:
            if stats:
                self.update(stats)
            self.state = "done"
            self.finished_at = datetime.utcnow()

    def fail(self, err: str):
        with self._lock:
            self.state = "error"
            self.finished_at = datetime.utcnow()
            self.error = err

    def snapshot(self) -> dict:
        with self._lock:
            elapsed = None
            if self.started_at:
                end = self.finished_at or datetime.utcnow()
                elapsed = round((end - self.started_at).total_seconds(), 1)
            pct = int(100 * self.checked / self.total) if self.total else (100 if self.state == "done" else 0)
            return {
                "state": self.state,
                "checked": self.checked,
                "total": self.total,
                "expired": self.expired,
                "alive": self.alive,
                "errors": self.errors,
                "skipped": self.skipped,
                "aborted": self.aborted,
                "percent": pct,
                "error": self.error,
                "elapsed_seconds": elapsed,
            }


liveness_progress = _LivenessProgress()
progress = liveness_progress


def sweep_all(*, limit: Optional[int] = None, workers: int = _DEFAULT_WORKERS,
              fetch: Optional[Callable] = None, recheck_after_hours: int = 24,
              error_abort_min: int = 250, error_abort_ratio: float = 0.9,
              progress: Optional[Callable[[dict], None]] = None) -> dict:
    """Probe every active job whose last check is older than `recheck_after_hours`.

    Returns {checked, expired, alive, errors, skipped, aborted}.

    The breaker (`error_abort_*`) exists because `check_job_alive` fails OPEN:
    a WAF block or a dropped uplink makes every probe report "alive", which is
    safe but silent — the sweep would burn an hour and change nothing while
    looking like a success. Aborting turns that into a visible result. It can
    never cause an expiry; it only stops work.
    """
    from concurrent.futures import ThreadPoolExecutor
    from threading import Lock, Semaphore

    init_db()
    cutoff = datetime.utcnow() - timedelta(hours=recheck_after_hours)
    with session() as db:
        q = (db.query(Job)
             .filter(Job.status.in_(_ACTIVE_STATUSES))
             .order_by(Job.last_checked_at.is_(None).desc(),
                       Job.last_checked_at.asc()))
        rows = [(j.id, j.url, j.last_checked_at) for j in q.all()]

    due, skipped = [], 0
    for job_id, url, checked in rows:
        if checked is not None and checked > cutoff:
            skipped += 1
        else:
            due.append((job_id, url))
    if limit is not None:
        due = due[:limit]

    cb = progress
    liveness_progress.start(len(due), skipped=skipped)

    stats = {"checked": 0, "expired": 0, "alive": 0, "errors": 0,
             "skipped": skipped, "aborted": False}
    lock = Lock()
    gates: dict = {}
    gates_lock = Lock()

    def _gate(url: str) -> Semaphore:
        host = _host_of(url)
        with gates_lock:
            if host not in gates:
                gates[host] = Semaphore(_PER_HOST_WORKERS)
            return gates[host]

    def _tripped() -> bool:
        return (stats["checked"] >= error_abort_min
                and stats["errors"] >= stats["checked"] * error_abort_ratio)

    def probe(item) -> None:
        job_id, url = item
        with lock:
            if stats["aborted"]:
                return
        gate = _gate(url)
        with gate:
            try:
                alive, reason = check_job_alive(url, fetch=fetch)
            except Exception as e:  # noqa: BLE001
                # check_job_alive is already fail-open, but a caller-supplied
                # fetch can raise from anywhere. Treat it as an error, never
                # as death.
                alive, reason = True, f"error:{e}"
        _stamp_checked(job_id)
        with lock:
            stats["checked"] += 1
            if not alive:
                expire_job(job_id, reason)
                stats["expired"] += 1
            elif reason.startswith("error:"):
                stats["errors"] += 1
            else:
                stats["alive"] += 1
            if _tripped():
                stats["aborted"] = True
            liveness_progress.update(dict(stats))
            if cb:
                cb(dict(stats))

    try:
        with ThreadPoolExecutor(max_workers=max(1, workers)) as pool:
            list(pool.map(probe, due))
    except Exception as e:
        liveness_progress.fail(str(e))
        raise

    liveness_progress.finish(dict(stats))

    if stats["aborted"]:
        log.warning("Liveness sweep ABORTED after %d checks (%d errors): the "
                    "probes are failing, not the postings. %s",
                    stats["checked"], stats["errors"], stats)
    else:
        log.info("Liveness sweep_all: %s", stats)
    return stats
