"""Follow-up engine — surface due follow-ups, draft emails, record sends."""
from __future__ import annotations

import json
import logging
from datetime import datetime, timedelta
from typing import List, Optional

from .config import settings
from .models import Application, Job, init_db, session

log = logging.getLogger("jobbot.followup")


# ---------------------------------------------------------------------------
# Internal helpers
# ---------------------------------------------------------------------------

def _first_contact_email(job: Job) -> tuple[str, str]:
    """Return (contact_name, contact_email) from contacts_json, or ("","")."""
    raw = (job.contacts_json or "").strip()
    if raw:
        try:
            contacts = json.loads(raw)
            for c in contacts:
                if isinstance(c, dict) and c.get("email"):
                    return c.get("name", ""), c["email"]
        except (json.JSONDecodeError, TypeError):
            pass
    return "", ""


def _fallback_email(job: Job) -> str:
    """Try extract_recruiter_email on the job description."""
    from .auto_apply import extract_recruiter_email
    return extract_recruiter_email(job.description or "") or ""


def _days_since(dt: datetime) -> int:
    """Whole days elapsed since a UTC datetime."""
    return (datetime.utcnow() - dt).days


def _touch_dt(app: Application) -> datetime:
    """Latest touch point: last_followup_at if set, else sent_at."""
    return app.last_followup_at if app.last_followup_at else app.sent_at  # type: ignore[return-value]


# ---------------------------------------------------------------------------
# 1. followups_due
# ---------------------------------------------------------------------------

def followups_due() -> List[dict]:
    """Return applications that are due for a follow-up, most-overdue first.

    Criteria:
    - status == "sent"
    - sent_at is not None
    - job.status not in ("rejected", "closed")
    - followup_count < settings.followup_max
    - days since max(sent_at, last_followup_at) >= settings.followup_after_days
    """
    rows: List[dict] = []

    with session() as db:
        apps = (
            db.query(Application)
            .filter(Application.status == "sent", Application.sent_at.isnot(None))
            .all()
        )

        for app in apps:
            if app.followup_count >= settings.followup_max:
                continue

            job = app.job
            if job is None:
                continue
            if job.status in ("rejected", "closed"):
                continue

            touch = _touch_dt(app)
            days_since_touch = _days_since(touch)
            if days_since_touch < settings.followup_after_days:
                continue

            days_since_sent = _days_since(app.sent_at)

            contact_name, contact_email = _first_contact_email(job)
            if not contact_email:
                contact_email = _fallback_email(job)

            rows.append({
                "application_id": app.id,
                "job_id": job.id,
                "title": job.title,
                "company": job.company,
                "url": job.url,
                "sent_at": app.sent_at.isoformat(),
                "days_since_sent": days_since_sent,
                "days_since_touch": days_since_touch,
                "followup_count": app.followup_count,
                "contact_name": contact_name,
                "contact_email": contact_email,
            })

    rows.sort(key=lambda r: r["days_since_touch"], reverse=True)
    return rows


# ---------------------------------------------------------------------------
# 2. draft_followup
# ---------------------------------------------------------------------------

def draft_followup(application_id: int) -> dict:
    """Return {to, subject, body} for a deterministic follow-up email.

    First follow-up (followup_count == 0): full 4-6 sentence version.
    Second follow-up (followup_count == 1): shorter "final check-in" variant.
    """
    with session() as db:
        app = db.get(Application, application_id)
        if app is None:
            raise ValueError(f"Application {application_id} not found")

        job = app.job
        sent_date = app.sent_at.strftime("%B %d").replace(" 0", " ").strip() if app.sent_at else "recently"
        followup_count = app.followup_count

        contact_name, contact_email = _first_contact_email(job)
        if not contact_email:
            contact_email = _fallback_email(job)

        name = settings.applicant_name
        title = settings.applicant_title
        email = settings.applicant_email or settings.smtp_user
        phone = settings.applicant_phone

        greeting = f"Dear {contact_name}," if contact_name else "Dear Hiring Team,"
        job_title = job.title
        job_company = job.company

    subject = f"Following up: {job_title} application"

    if followup_count == 0:
        # First follow-up: full version
        body = (
            f"{greeting}\n\n"
            f"I wanted to follow up on my application for the {job_title} position at "
            f"{job_company}, which I submitted on {sent_date}. "
            f"I remain genuinely enthusiastic about this opportunity and believe my background "
            f"as a {title} aligns well with the needs of your team. "
            f"I would welcome any update you can share on the hiring timeline or next steps. "
            f"Please do not hesitate to reach out if you need any additional materials from me.\n\n"
            f"Thank you for your time and consideration.\n\n"
            f"Best regards,\n"
            f"{name}"
        )
        if email:
            body += f"\n{email}"
        if phone:
            body += f"\n{phone}"
    else:
        # Second (final) follow-up: shorter check-in
        body = (
            f"{greeting}\n\n"
            f"I am writing for a brief final check-in regarding the {job_title} role at "
            f"{job_company}. I applied on {sent_date} and remain very interested in joining "
            f"your team. I understand hiring timelines can shift, and I appreciate any update "
            f"you are able to share.\n\n"
            f"Thank you again for your consideration.\n\n"
            f"Best regards,\n"
            f"{name}"
        )
        if email:
            body += f"\n{email}"
        if phone:
            body += f"\n{phone}"

    return {
        "to": contact_email,
        "subject": subject,
        "body": body,
    }


# ---------------------------------------------------------------------------
# 3. record_followup
# ---------------------------------------------------------------------------

def record_followup(application_id: int) -> None:
    """Stamp last_followup_at = utcnow and increment followup_count."""
    with session() as db:
        app = db.get(Application, application_id)
        if app is None:
            raise ValueError(f"Application {application_id} not found")
        # Read job info before commit (avoid DetachedInstanceError after flush)
        job_title = app.job.title if app.job else "?"
        job_company = app.job.company if app.job else "?"
        app.last_followup_at = datetime.utcnow()
        app.followup_count = (app.followup_count or 0) + 1
        db.commit()
        log.info(
            "Recorded follow-up #%d for application %d (%s @ %s)",
            app.followup_count, application_id, job_title, job_company,
        )


# ---------------------------------------------------------------------------
# 4. send_followup
# ---------------------------------------------------------------------------

def send_followup(application_id: int, to_email: str) -> dict:
    """Draft, send (via SMTP), and record a follow-up for the given application.

    Returns the draft dict that was sent.
    Raises ValueError if SMTP is not configured or to_email is empty.
    """
    if not settings.smtp_user:
        raise ValueError("SMTP is not configured (SMTP_USER is empty in .env).")
    if not to_email or not to_email.strip():
        raise ValueError("to_email must not be empty.")

    draft = draft_followup(application_id)
    # Override the to address with the caller-supplied one (may differ from contacts_json)
    draft["to"] = to_email.strip()

    # Convert plain-text body to simple HTML paragraphs
    paragraphs = draft["body"].split("\n\n")
    body_html = "\n".join(
        f"<p>{p.replace(chr(10), '<br>')}</p>" for p in paragraphs if p.strip()
    )

    from .emailer import send as email_send
    email_send(
        to=draft["to"],
        subject=draft["subject"],
        body_html=body_html,
        body_text=draft["body"],
    )

    record_followup(application_id)
    log.info("Follow-up sent to %s for application %d", to_email, application_id)
    return draft
