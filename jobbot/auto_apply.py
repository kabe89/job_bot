"""Auto-apply engine.

Selects candidate jobs based on **user-defined thresholds** AND the
**predictor's recommendation**, then runs the full tailor → send pipeline.

Safety model (all defaults conservative):
  - Headless form-filling bots are decommissioned in favor of In-Browser Copilot
  - Dry-run by default — must pass confirm=True to actually send email applications
  - Rate-limited per day (APPLY_RATE_LIMIT_PER_DAY)
  - Will not apply to a job already in 'applied' / 'sent' / 'skipped' status
  - Honors watchlist-only mode
  - Honors a per-company blocklist (data/auto_apply_blocklist.txt)
"""
from __future__ import annotations

import logging
import re
from dataclasses import dataclass, field
from datetime import datetime, timedelta
from pathlib import Path
from typing import List, Optional

from .config import settings
from .matcher import excluded
from .models import Application, Job, init_db, session
from .pipeline import daily_apply_quota_remaining, send_application, tailor_for_job
from .predict import predict_job

log = logging.getLogger("jobbot.auto_apply")

EMAIL_RE = re.compile(r"\b[A-Za-z0-9._%+\-]+@[A-Za-z0-9.\-]+\.[A-Za-z]{2,}\b")
BLOCKLIST_FILE = Path("data/auto_apply_blocklist.txt")


@dataclass
class AutoApplySettings:
    """User-configurable inputs. Defaults are the bot's *recommended* values."""
    # USER INPUT — explicit per-run knobs
    confirm: bool = False                       # if False → dry-run, prints plan only
    max_applies: Optional[int] = None           # absolute cap for this run (else daily quota)
    default_recipient: str = ""                 # fallback email if posting has none
    watchlist_only: bool = False                # only auto-apply to watchlist companies
    dry_run_only_high_confidence: bool = True   # skip when prediction confidence is "low"

    # RECOMMENDED INPUT — model-driven defaults (overridable per-run)
    min_match_score: float = 0.55               # heuristic match floor
    min_callback_probability: float = 0.10      # model-predicted callback floor
    require_watchlist_or_score: float = 0.65    # auto-apply outside watchlist only if score >= this
    skip_stale_days: int = 30                   # don't apply to postings older than this
    skip_companies: List[str] = field(default_factory=list)


def _blocklist() -> List[str]:
    if not BLOCKLIST_FILE.exists():
        return []
    return [ln.strip().lower() for ln in BLOCKLIST_FILE.read_text(encoding="utf-8").splitlines()
            if ln.strip() and not ln.strip().startswith("#")]


def extract_recruiter_email(text: str) -> Optional[str]:
    """Pull the first plausible recruiter email from a job posting body.
    Avoids obvious junk addresses (postmaster, noreply, do-not-reply, support, sales)."""
    if not text:
        return None
    JUNK = ("noreply", "no-reply", "donotreply", "do-not-reply", "postmaster",
            "support", "sales", "marketing", "newsletter", "webmaster", "info@indeed",
            "abuse", "privacy", "press")
    seen = set()
    for m in EMAIL_RE.finditer(text):
        addr = m.group(0).lower()
        if addr in seen:
            continue
        seen.add(addr)
        if any(j in addr for j in JUNK):
            continue
        return addr
    return None


@dataclass
class Plan:
    job: Job
    callback_probability: float
    chosen_recipient: str
    reasons: List[str]


def select_candidates(cfg: AutoApplySettings, require_recipient: bool = True) -> List[Plan]:
    """Pick jobs that meet thresholds + model recommendation.

    require_recipient=True  -> email mode: skip jobs with no recruiter email
                               and no default recipient (legacy behavior).
    require_recipient=False -> kit mode: recipient is informational only;
                               the application happens on the ATS form.
    """
    init_db()
    blocklist = set(_blocklist()) | {c.lower() for c in cfg.skip_companies}
    stale_cutoff = (datetime.utcnow() - timedelta(days=cfg.skip_stale_days)
                    if cfg.skip_stale_days and cfg.skip_stale_days > 0 else None)
    plans: List[Plan] = []
    with session() as db:
        candidates = (
            db.query(Job)
            .filter(Job.status.in_(["new", "tailored"]))
            .order_by(Job.match_score.desc())
            .all()
        )
        excludes = [e.lower() for e in settings.excludes]
        title_excludes = settings.excludes_title
        for job in candidates:
            reasons = []
            if job.company.lower() in blocklist:
                continue
            # Never spend apply quota on excluded roles (intern, sales, ...)
            # even if they slipped past the scrape-time filter.
            title_l = job.title.lower()
            if any(e in title_l for e in excludes):
                continue
            # Freelance/gig work never spends apply quota either, even for rows
            # stored before the title filter existed. Word-boundary matched, so
            # "Contract Strategy" is not mistaken for contract employment.
            if excluded(job.title, title_excludes):
                continue
            if job.match_score < cfg.min_match_score:
                continue
            # Skip postings older than the stale cutoff (use posted_at when
            # known, else fall back to when we first discovered the job).
            if stale_cutoff is not None:
                ref = job.posted_at or job.discovered_at
                if ref is not None and ref < stale_cutoff:
                    continue
            in_watch = "watchlist" in (job.tags or "")
            if cfg.watchlist_only and not in_watch:
                continue
            if not in_watch and job.match_score < cfg.require_watchlist_or_score:
                continue

            # Recipient resolution (mandatory only in email mode)
            recipient = extract_recruiter_email(job.description) or cfg.default_recipient
            if require_recipient and not recipient:
                continue
            if recipient:
                reasons.append(f"recipient={recipient}")

            # Prediction gate
            app_row = job.applications[0] if job.applications else None
            pred = predict_job(job, app_row)
            if cfg.dry_run_only_high_confidence and pred.confidence == "low" and pred.callback_probability < cfg.min_callback_probability:
                continue
            if pred.callback_probability < cfg.min_callback_probability:
                continue
            reasons.append(f"P(callback)={pred.callback_probability:.1%}")
            reasons.append(f"score={job.match_score:.2f}{' [watchlist]' if in_watch else ''}")
            plans.append(Plan(job=job, callback_probability=pred.callback_probability,
                              chosen_recipient=recipient, reasons=reasons))
    # Best prospects first — rank by predicted callback, then match score.
    plans.sort(key=lambda p: (p.callback_probability, p.job.match_score), reverse=True)
    # Respect daily quota
    quota = daily_apply_quota_remaining()
    cap = cfg.max_applies if cfg.max_applies is not None else quota
    cap = min(cap, quota)
    return plans[:cap]


def run_auto_apply(cfg: AutoApplySettings) -> dict:
    """Execute (or dry-run) the plan. Returns a structured report."""
    plans = select_candidates(cfg)
    report = {
        "selected": len(plans),
        "confirmed_send": bool(cfg.confirm),
        "quota_remaining_before": daily_apply_quota_remaining(),
        "details": [],
        "sent_ids": [],
        "errors": [],
    }
    if not plans:
        return report

    for plan in plans:
        entry = {
            "job_id": plan.job.id,
            "title": plan.job.title,
            "company": plan.job.company,
            "recipient": plan.chosen_recipient,
            "p_callback": plan.callback_probability,
            "reasons": plan.reasons,
            "action": "would-send (dry-run)",
        }
        if not cfg.confirm:
            report["details"].append(entry)
            continue
        try:
            # Ensure tailored materials exist
            with session() as db:
                fresh = db.get(Job, plan.job.id)
                app_row = fresh.applications[0] if fresh and fresh.applications else None
            if not app_row:
                result = tailor_for_job(plan.job.id)
                app_id = result["application_id"]
            else:
                app_id = app_row.id
            send_application(app_id, plan.chosen_recipient,
                             custom_note="(Sent via JobBot auto-apply — please review and reply if interested.)")
            entry["action"] = "sent"
            report["sent_ids"].append(plan.job.id)
        except Exception as e:  # noqa: BLE001
            entry["action"] = f"error: {e}"
            report["errors"].append({"job_id": plan.job.id, "error": str(e)})
        report["details"].append(entry)
    report["quota_remaining_after"] = daily_apply_quota_remaining()
    return report


# =====================  APPLY QUEUE (kit mode)  =============================
# Email auto-apply only works for the rare posting with a recruiter address.
# Kit mode is the main path to actually landing applications: pick the best
# candidates, build a ready-to-submit Apply Kit for each (real ATS form
# questions, auto-answered), and queue them so submitting is a 2-minute
# review instead of a 20-minute form slog.

def _select_adaptive(cfg: AutoApplySettings, want: int) -> tuple[List[Plan], str]:
    """Kit-mode selection with graceful relaxation.

    The heuristic match scores rarely exceed ~0.5, so the conservative email
    gates (0.55/0.65) often select nothing. Rather than silently doing
    nothing, relax in tiers until `want` candidates are found:
      1. configured  — the user's auto-apply gates as-is
      2. relaxed     — score floor drops to the scrape-keep floor
                       (MIN_MATCH_SCORE), watchlist score gate equalized,
                       callback floor halved
      3. best-available — top jobs by score, blocklist/staleness still honored
    Returns (plans, tier_name).
    """
    from dataclasses import replace

    plans = select_candidates(cfg, require_recipient=False)
    if len(plans) >= want:
        return plans, "configured"

    floor = min(cfg.min_match_score, settings.min_match_score)
    relaxed = replace(cfg,
                      min_match_score=floor,
                      require_watchlist_or_score=floor,
                      min_callback_probability=cfg.min_callback_probability / 2,
                      dry_run_only_high_confidence=False)
    plans = select_candidates(relaxed, require_recipient=False)
    if plans:
        return plans, "relaxed"

    last_resort = replace(relaxed, min_match_score=0.0,
                          require_watchlist_or_score=0.0,
                          min_callback_probability=0.0)
    return select_candidates(last_resort, require_recipient=False), "best-available"


def prepare_apply_queue(cfg: AutoApplySettings, build_limit: int = 5,
                        rebuild: bool = False) -> dict:
    """Build Apply Kits for the top candidate jobs (kit-mode auto-apply).

    Selection uses the same threshold + prediction gates as email auto-apply
    but does NOT require a recruiter email. Each build fires AI calls
    (tailor + question answering), so `build_limit` caps the run.
    Jobs that already have a kit are skipped unless rebuild=True.
    """
    from .browser_apply import build_package

    plans, tier = _select_adaptive(cfg, build_limit)
    report = {"selected": len(plans), "built": [], "skipped_existing": [],
              "errors": [], "build_limit": build_limit, "selection_tier": tier}
    built = 0
    for plan in plans:
        if built >= build_limit:
            break
        job_id = plan.job.id
        with session() as db:
            fresh = db.get(Job, job_id)
            app_row = fresh.applications[0] if fresh and fresh.applications else None
            has_kit = bool(app_row and app_row.questions_json)
        if has_kit and not rebuild:
            report["skipped_existing"].append({"job_id": job_id, "title": plan.job.title,
                                               "company": plan.job.company})
            continue
        try:
            pkg = build_package(job_id)
            blanks = len(pkg.blanks)
            report["built"].append({
                "job_id": job_id, "title": pkg.title, "company": pkg.company,
                "p_callback": plan.callback_probability,
                "source": pkg.questions_source or "common",
                "questions": len(pkg.form_questions),
                "blanks": blanks,
                "ready": blanks == 0,
                "kit": pkg.kit_markdown_path,
                "url": pkg.url,
            })
            built += 1
        except Exception as e:  # noqa: BLE001
            log.warning("Kit build failed for job %d: %s", job_id, e)
            report["errors"].append({"job_id": job_id, "error": str(e)})
    return report


def apply_queue_status() -> List[dict]:
    """Everything prepared but not yet submitted, best prospects first.

    Returns one dict per queued application: job/title/company/url, score,
    P(callback), question source, blanks count and ready flag.
    """
    from .apply_questions import needs_attention, questions_from_json
    init_db()
    out: List[dict] = []
    with session() as db:
        rows = (
            db.query(Application)
            .join(Job)
            .filter(Application.questions_json != "",
                    Application.status != "sent",
                    Job.status.notin_(["applied", "rejected", "skipped"]))
            .all()
        )
        for app_row in rows:
            job = app_row.job
            try:
                questions, source = questions_from_json(app_row.questions_json)
            except Exception:  # noqa: BLE001
                questions, source = [], ""
            # "Needs you" spans outright blanks AND answers held for
            # confirmation - a kit is only READY when neither remains.
            pending = needs_attention(questions)
            blanks = len(pending)
            pred = predict_job(job, app_row)
            out.append({
                "job_id": job.id,
                "application_id": app_row.id,
                "title": job.title,
                "company": job.company,
                "url": job.url,
                "score": job.match_score,
                "p_callback": pred.callback_probability,
                "source": source or "common",
                "questions": len(questions),
                "blanks": blanks,
                "unanswered": sum(1 for q in pending if q.needs_user),
                "unconfirmed": sum(1 for q in pending if q.needs_review),
                "ready": bool(questions) and blanks == 0,
            })
    out.sort(key=lambda r: (r["ready"], r["p_callback"], r["score"]), reverse=True)
    return out
