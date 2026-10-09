"""Structural DB hygiene behind `jobbot clean`.

Two jobs live here, both DRY-RUN BY DEFAULT — every entry point takes
`apply=False` and reports what it *would* do. `apply=True` is the only thing
that writes.

  clear_expired_embeddings  reclaim disk from rows nobody will rank again
  merge_duplicate_apps      collapse the >1 Application rows a job accumulates

The liveness sweep is deliberately NOT here; it lives in `liveness.py` beside
the probe it drives.
"""
from __future__ import annotations

import logging
import os
from typing import List, Optional

from sqlalchemy import func

from . import models
from .outcomes import TERMINAL, stage_rank

log = logging.getLogger("jobbot.dbclean")

# Statuses whose embedding will never be needed again. An embedding is
# regenerable from the job description at any time, so dropping it costs a
# recompute, never data.
_DONE_STATUSES = ("expired", "applied", "sent", "rejected")


def _db_bytes() -> int:
    try:
        return os.path.getsize(models.settings.db_path)
    except OSError:
        return 0


def clear_expired_embeddings(*, apply: bool = False) -> dict:
    """Blank `Job.embedding` on expired/applied/rejected rows, then VACUUM.

    Reports {jobs, bytes_before, bytes_after}. With apply=False nothing is
    written and `jobs` is the count that WOULD be cleared.
    """
    models.init_db()
    before = _db_bytes()
    with models.session() as db:
        q = (db.query(models.Job)
             .filter(models.Job.embedding.isnot(None),
                     models.Job.embedding != "")
             .filter((models.Job.status.in_(_DONE_STATUSES))
                     | (models.Job.expired_at.isnot(None))))
        rows = q.all()
        n = len(rows)
        if apply:
            for job in rows:
                job.embedding = ""
            db.commit()

    if apply and n:
        # VACUUM cannot run inside a transaction, so it needs its own
        # connection outside the session above.
        from sqlalchemy import text
        with models._engine.connect() as conn:
            conn.execute(text("VACUUM"))

    report = {"jobs": n, "bytes_before": before,
              "bytes_after": _db_bytes() if apply else before}
    log.info("clear_expired_embeddings(apply=%s): %s", apply, report)
    return report


# --- freelance / gig backfill ----------------------------------------------
#
# The scrape-time title filter only ever sees NEW postings: run_scrape_cycle
# dedupes on the row hash and `continue`s before it reaches the exclude checks,
# so rows already in the DB are never re-judged. Hiding them is a separate job,
# and it is the half the user actually notices.
#
# The channel is `status='skipped'`, NOT `expired_at`:
#   * liveness._ACTIVE_STATUSES is ("new", "tailored", "starred"), so a skipped
#     row is never re-probed and can never be resurrected by `restore_job`,
#     which clears status/expired_at/expiry_reason back to 'new'.
#   * 'skipped' already means "the user dismissed this" (web.py's skip button),
#     and auto_apply / apply_fill already exclude it.
# Statuses that record something real are never rewritten -- an applied gig job
# stays applied, because that is history.
_HIDEABLE_STATUSES = ("new", "tailored", "starred")

# Stamped into Job.tags so the undo restores ONLY what this tool hid. Without a
# marker, undo would also resurrect the 316 gig postings that were already
# status='skipped' before the filter existed -- rows a human dismissed, which
# an "undo the freelance hide" must not touch. `tags` is the machine-managed
# comma list (pipeline already appends "watchlist" there); every reader does a
# substring test, and this tag collides with none of them.
_GIG_TAG = "gig-hidden"


def _freelance_title_rows(db):
    """Active rows whose TITLE reads as freelance work."""
    from .matcher import excluded

    terms = models.settings.excludes_title
    rows = (db.query(models.Job)
            .filter(models.Job.status.in_(_HIDEABLE_STATUSES)).all())
    return [j for j in rows if excluded(j.title or "", terms)]


def hide_freelance_jobs(*, apply: bool = False) -> dict:
    """Set `status='skipped'` on active jobs whose title reads as gig work.

    Dry run by default. Reports {jobs, titles, applied} where `jobs` is the
    count that would be hidden. Reversible with `unhide_freelance_jobs`.
    """
    models.init_db()
    with models.session() as db:
        rows = _freelance_title_rows(db)
        titles = [j.title for j in rows[:20]]
        if apply:
            for job in rows:
                job.status = "skipped"
                tags = job.tags or ""
                if _GIG_TAG not in tags:
                    job.tags = f"{tags},{_GIG_TAG}" if tags else _GIG_TAG
            db.commit()
    report = {"jobs": len(rows), "titles": titles, "applied": apply}
    log.info("hide_freelance_jobs(apply=%s): %d row(s)", apply, len(rows))
    return report


def unhide_freelance_jobs(*, apply: bool = False) -> dict:
    """Undo `hide_freelance_jobs` -- put its hidden rows back to 'new'.

    Scoped to rows carrying `_GIG_TAG`, so a posting the user skipped by hand
    stays skipped even when its title matches the freelance terms.

    Restores to 'new' unconditionally: the prior status is not stored, so a
    'tailored' or 'starred' gig posting comes back as 'new'. `Job.starred` is a
    separate boolean column and survives; only the status word is flattened.
    """
    models.init_db()
    with models.session() as db:
        rows = (db.query(models.Job)
                .filter(models.Job.status == "skipped",
                        models.Job.tags.contains(_GIG_TAG)).all())
        titles = [j.title for j in rows[:20]]
        if apply:
            for job in rows:
                job.status = "new"
                job.tags = ",".join(t for t in (job.tags or "").split(",")
                                    if t.strip() and t.strip() != _GIG_TAG)
            db.commit()
    report = {"jobs": len(rows), "titles": titles, "applied": apply}
    log.info("unhide_freelance_jobs(apply=%s): %d row(s)", apply, len(rows))
    return report


def _survivor_of(apps: List["models.Application"]) -> "models.Application":
    """The row to keep: the one carrying the real form kit.

    `questions_json` is what makes an Application row useful — it is the
    scraped form plus every planned answer. A row without it is a stub the
    pipeline created and abandoned. Ties (both or neither carry a kit) go to
    the newest row, which is the one the most recent kit build wrote.
    """
    with_kit = [a for a in apps if (a.questions_json or "").strip()]
    pool = with_kit or apps
    return max(pool, key=lambda a: a.id)


def _merged_status(apps: List["models.Application"]) -> str:
    """The furthest-along status across the rows.

    Terminal statuses (rejected, etc.) are sticky and win outright, matching
    `outcomes.record_outcome`; otherwise the highest positive stage rank wins.
    """
    for a in apps:
        if a.status in TERMINAL:
            return a.status
    return max((a.status for a in apps), key=stage_rank)


def duplicate_app_plan() -> List[dict]:
    """One entry per job holding more than one Application row.

    Pure read. `[]` means nothing to do.
    """
    models.init_db()
    plan: List[dict] = []
    with models.session() as db:
        job_ids = [r[0] for r in
                   db.query(models.Application.job_id)
                   .group_by(models.Application.job_id)
                   .having(func.count(models.Application.id) > 1).all()]
        for job_id in job_ids:
            apps = (db.query(models.Application)
                    .filter_by(job_id=job_id).order_by(models.Application.id).all())
            survivor = _survivor_of(apps)
            losers = [a.id for a in apps if a.id != survivor.id]
            events = (db.query(models.OutcomeEvent)
                      .filter(models.OutcomeEvent.application_id.in_(losers))
                      .count()) if losers else 0
            plan.append({
                "job_id": job_id,
                "survivor": survivor.id,
                "losers": losers,
                "events_to_repoint": events,
                "merged_status": _merged_status(apps),
            })
    return plan


def merge_duplicate_apps(*, apply: bool = False) -> dict:
    """Collapse each job's Application rows onto the one holding the kit.

    OutcomeEvents and ApplyRuns are RE-POINTED at the survivor, never deleted:
    an outcome event is funnel history that `outcomes.priors` counts, and
    dropping one silently biases every future ranking. Only the now-unreferenced
    loser Application rows are removed.
    """
    plan = duplicate_app_plan()
    report = {"jobs": len(plan), "rows_removed": 0, "events_repointed": 0,
              "runs_repointed": 0, "applied": apply}
    if not apply or not plan:
        report["rows_removed"] = sum(len(p["losers"]) for p in plan)
        report["events_repointed"] = sum(p["events_to_repoint"] for p in plan)
        if not apply:
            return report
    if not plan:
        return report

    report.update(rows_removed=0, events_repointed=0)
    with models.session() as db:
        for entry in plan:
            survivor = db.get(models.Application, entry["survivor"])
            losers = entry["losers"]
            if survivor is None or not losers:
                continue
            for ev in (db.query(models.OutcomeEvent)
                       .filter(models.OutcomeEvent.application_id.in_(losers)).all()):
                ev.application_id = survivor.id
                report["events_repointed"] += 1
            for run in (db.query(models.ApplyRun)
                        .filter(models.ApplyRun.application_id.in_(losers)).all()):
                run.application_id = survivor.id
                report["runs_repointed"] += 1

            loser_rows = (db.query(models.Application)
                          .filter(models.Application.id.in_(losers)).all())
            # Carry the record forward BEFORE the rows go: the earliest real
            # send is when this application actually left, and a survivor stuck
            # at 'draft' would drop out of followup.py's `status == "sent"`.
            sents = [a.sent_at for a in loser_rows + [survivor] if a.sent_at]
            if sents:
                survivor.sent_at = min(sents)
            survivor.status = entry["merged_status"]
            for a in loser_rows:
                db.delete(a)
                report["rows_removed"] += 1
        db.commit()

    log.info("merge_duplicate_apps(apply=True): %s", report)
    return report
