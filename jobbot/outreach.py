"""Approve-then-send outreach queue.

Auto-drafts warm-intro / follow-up / cold networking messages (and paste-in
replies) into OutreachDraft rows. Nothing is emailed until a draft is explicitly
approved via approve() and then send(). Every source is fail-open: any failure
contributes no drafts rather than raising.
"""
from __future__ import annotations

import logging
import re
from datetime import datetime
from email.utils import parseaddr
from typing import Optional, Tuple

from . import ollama_client
from .config import settings
from .models import (Contact, Job, OutreachDraft, init_db,
                     session)

log = logging.getLogger(__name__)


def _live_key_exists(db, dedupe_key: str) -> bool:
    """True if a non-dismissed draft with this key already exists."""
    if not dedupe_key:
        return False
    return (db.query(OutreachDraft)
              .filter(OutreachDraft.dedupe_key == dedupe_key,
                      OutreachDraft.status != "dismissed")
              .count() > 0)


def _queue(*, kind: str, subject: str, body: str, recipient_email: str = "",
           job_id=None, contact_id=None, application_id=None, rationale: str = "",
           priority: float = 0.0, inbound: str = "",
           dedupe_key: str = "", thread_id=None) -> Optional[OutreachDraft]:
    """Insert a pending draft unless a live one with dedupe_key exists."""
    init_db()
    with session() as db:
        if _live_key_exists(db, dedupe_key):
            return None
        d = OutreachDraft(
            kind=kind, status="pending", recipient_email=(recipient_email or "").strip(),
            subject=subject or "", body=body or "", inbound=inbound or "",
            rationale=rationale or "", priority=float(priority or 0.0),
            job_id=job_id, contact_id=contact_id, application_id=application_id,
            dedupe_key=dedupe_key or "", thread_id=thread_id,
        )
        db.add(d); db.commit(); db.refresh(d)
        return d


MAX_HISTORY_MSGS = 6
MAX_HISTORY_CHARS = 4000

_DASHES = ("—", "–")  # em dash, en dash


def _scrub(text: str) -> str:
    """Strip em/en dashes from drafted message text.

    Enforced in code rather than by prompt instruction: local models reach for
    em dashes constantly, and a line in the prompt is not a guarantee.

    Only a token-internal "--" survives: that is the URL/DOI/slug shape, and
    rewriting it would silently corrupt a link the user is about to send. Every
    other run of hyphens is being used AS a dash, whatever follows it, so
    "Talk soon --" gets rewritten the same as "soon -- but".
    """
    if not text:
        return ""
    out = text
    for d in _DASHES:
        out = out.replace(f" {d} ", ", ").replace(d, ", ")
    out = re.sub(r"(?<=\S)-{2,}(?=\S)",                # park token-internal runs
                 lambda m: "\x00" * len(m.group(0)), out)
    out = re.sub(r"\s*-{2,}\s*", ", ", out)
    out = out.replace("\x00", "-")
    while ", ," in out:
        out = out.replace(", ,", ",")
    return out.replace(" ,", ",")


def _trim_history(history) -> list:
    """Newest-last window over history, bounded by message count and characters."""
    rows = list(history or [])[-MAX_HISTORY_MSGS:]
    total = 0
    kept = []
    for row in reversed(rows):          # newest first while budgeting
        size = len(str(row.get("text", "")))
        if kept and total + size > MAX_HISTORY_CHARS:
            break
        total += size
        kept.append(row)
    kept.reverse()                      # back to oldest-first for the prompt
    return kept


def _history_block(history) -> str:
    """The trimmed thread transcript, formatted for a prompt. "" when empty."""
    trimmed = _trim_history(history)
    if not trimmed:
        return ""
    lines = []
    for h in trimmed:
        tag = "SENT" if h.get("direction") == "sent" else "RECEIVED"
        at = h.get("at")
        stamp = at.strftime("%Y-%m-%d") if hasattr(at, "strftime") else str(at or "")
        lines.append(f"{tag} ({stamp}): {str(h.get('text', '')).strip()}")
    return "--- Conversation so far ---\n" + "\n".join(lines)


def _fallback_reply_body() -> str:
    """The deterministic reply used whenever generation does not happen.

    Exposed so callers can tell a generated draft from boilerplate by comparing
    against it. Probing ollama_client.is_available() separately does not answer
    that question: _reply_text catches its own generation failures and falls
    back here even when the server is reachable.
    """
    name = getattr(settings, "applicant_name", "") or "Applicant"
    return (
        "Hello,\n\n"
        "Thank you for reaching out and for the update. I am very interested and "
        "happy to continue the conversation. I am available at your convenience, "
        "so please let me know a few times that work and I will make it happen.\n\n"
        "Best regards,\n"
        f"{name}"
    )


def _reply_text(inbound: str, sender: str = "", role: str = "",
                scenario: str = "", history=(), subject: str = "") -> Tuple[str, str]:
    """Draft a reply. Ollama when available, else a deterministic template.

    scenario/history/subject are the thread's standing context; all default to
    empty so the threadless caller draft_reply() is unaffected.
    """
    if subject:
        subj = subject if subject.lower().startswith("re:") else f"Re: {subject}"
    elif role:
        subj = f"Re: {role}"
    else:
        subj = "Re: your message"

    try:
        if ollama_client.is_available():
            blocks = []
            if (scenario or "").strip():
                blocks.append(f"--- Situation ---\n{scenario.strip()}")
            hist = _history_block(history)
            if hist:
                blocks.append(hist)
            blocks.append(f"--- Their latest email ---\n{inbound}")
            prompt = (
                "You are drafting a concise, warm, professional reply to the last "
                "email below. Keep it under 150 words, propose concrete next steps, "
                "and do not invent facts. Never use em dashes.\n\n"
                + "\n\n".join(blocks)
                + "\n\nWrite only the reply body."
            )
            out = ollama_client._generate(prompt, system="You write job-search emails.")
            if out and out.strip():
                return subj, _scrub(out.strip())
    except Exception as exc:  # noqa: BLE001
        log.warning("Reply generation failed, using template: %s", exc)

    return subj, _scrub(_fallback_reply_body())


def _fallback_nudge_body() -> str:
    """Deterministic follow-up used whenever generation does not happen.

    Deliberately gives them an out ("no pressure at all") and offers to do the
    writing. A nudge that only repeats the original ask reads as pressure, and
    pressure is what costs you a warm contact.
    """
    name = getattr(settings, "applicant_name", "") or "Applicant"
    return (
        "Hi,\n\n"
        "Just floating this back up in case it slipped, no pressure at all. I am "
        "still very interested and still glad to draft the intro note myself so "
        "all you have to do is forward it.\n\n"
        "Either way, thank you again for the time.\n\n"
        "Best,\n"
        f"{name}"
    )


def _nudge_text(days: int, scenario: str = "", history=(),
                subject: str = "") -> Tuple[str, str]:
    """Draft a follow-up to a message of ours that got no reply.

    Separate from _reply_text because the writing problem is the opposite one:
    there is no inbound email to answer, and the risk is sounding impatient
    rather than sounding vague.
    """
    subj = (subject if subject.lower().startswith("re:") else f"Re: {subject}") \
        if subject else "Re: following up"

    try:
        if ollama_client.is_available():
            blocks = []
            if (scenario or "").strip():
                blocks.append(f"--- Situation ---\n{scenario.strip()}")
            hist = _history_block(history)
            if hist:
                blocks.append(hist)
            blocks.append(f"--- Timing ---\nIt has been {days} days since we last "
                          "wrote and there has been no reply.")
            prompt = (
                "You are drafting a short, low-pressure follow-up to a message "
                "that got no reply. Under 120 words. Give them an easy out, do "
                "not repeat the original request verbatim, do not guilt them, "
                "and do not invent facts. Never use em dashes.\n\n"
                + "\n\n".join(blocks)
                + "\n\nWrite only the email body."
            )
            out = ollama_client._generate(prompt, system="You write job-search emails.")
            if out and out.strip():
                return subj, _scrub(out.strip())
    except Exception as exc:  # noqa: BLE001
        log.warning("Nudge generation failed, using template: %s", exc)

    return subj, _scrub(_fallback_nudge_body())


def draft_reply(inbound_text: str, sender: str = "", job_id=None) -> OutreachDraft:
    """Queue a reply draft for a pasted inbound email. Seam for future read_inbox()."""
    _name, addr = parseaddr(sender or "")
    role = ""
    if job_id:
        try:
            init_db()
            with session() as db:
                j = db.get(Job, job_id)
                if j:
                    role = f"{j.title} at {j.company}"
        except Exception:  # noqa: BLE001
            pass
    subject, body = _reply_text(inbound_text or "", sender=sender, role=role)
    d = _queue(kind="reply", subject=subject, body=body,
               recipient_email=addr, job_id=job_id, inbound=inbound_text or "",
               rationale=f"Reply to {addr or sender or 'inbound email'}".strip(),
               priority=0.9,
               dedupe_key=f"reply:{datetime.utcnow().timestamp()}")
    return d


from . import contacts as _contacts_mod  # noqa: E402  (module-level for monkeypatch)
from . import followup  # noqa: E402
from . import referrals  # noqa: E402
from . import people_finder as _people_finder  # noqa: E402


def _contact_to_dict(c: Contact) -> dict:
    return {"name": c.name or "", "email": c.email or "", "title": c.title or "",
            "company": c.company or ""}


def _scan_warm_intros(added: dict) -> None:
    """Queue a warm-intro draft per job that has a matching warm contact."""
    try:
        index = referrals.build_index()
    except Exception as exc:  # noqa: BLE001
        log.warning("warm-intro scan: build_index failed: %s", exc)
        return
    if not index:
        return
    with session() as db:
        jobs = (db.query(Job).filter(Job.status != "expired")
                  .order_by(Job.match_score.desc()).limit(300).all())
        for job in jobs:
            if added["_total"] >= settings.outreach_max_per_scan:
                return
            try:
                matches = referrals.contacts_for_job(job, index)
                if not matches:
                    continue
                # Field gate: don't draft a warm intro for a job the candidate is
                # a poor fit for, even if a contact works there.
                if (job.match_score or 0.0) < settings.warm_intro_min_score:
                    continue
                c = matches[0]  # highest warmth
                if not getattr(c, "id", None):
                    continue
                _bonus, why = referrals.referral_bonus(job, index)
                mail = _contacts_mod.networking_email(job.id, _contact_to_dict(c))
                d = _queue(kind="warm_intro", subject=mail.get("subject", ""),
                           body=mail.get("body", ""),
                           recipient_email=getattr(c, "email", "") or "",
                           job_id=job.id, contact_id=c.id,
                           rationale=why or f"Warm intro via {c.name}",
                           priority=float(getattr(c, "warmth_score", 0.0) or 0.5),
                           dedupe_key=f"warm_intro:{job.id}:{c.id}")
                if d is not None:
                    added["warm_intro"] += 1
                    added["_total"] += 1
            except Exception as exc:  # noqa: BLE001
                log.warning("warm-intro scan: job %s failed: %s", job.id, exc)
                continue


def _scan_followups(added: dict) -> None:
    """Queue a follow-up draft per due application."""
    try:
        due = followup.followups_due()
    except Exception as exc:  # noqa: BLE001
        log.warning("follow-up scan failed: %s", exc)
        return
    for row in due:
        if added["_total"] >= settings.outreach_max_per_scan:
            return
        try:
            app_id = row.get("application_id")
            draft = followup.draft_followup(app_id)
            d = _queue(kind="follow_up", subject=draft.get("subject", ""),
                       body=draft.get("body", ""),
                       recipient_email=draft.get("to", "") or row.get("contact_email", ""),
                       job_id=row.get("job_id"), application_id=app_id,
                       rationale=f"Follow-up ({row.get('days_since_touch', '?')}d since touch)",
                       priority=0.7,
                       dedupe_key=f"follow_up:{app_id}:{row.get('followup_count', 0)}")
            if d is not None:
                added["follow_up"] += 1
                added["_total"] += 1
        except Exception as exc:  # noqa: BLE001
            log.warning("follow-up scan: app %s failed: %s", row.get("application_id"), exc)
            continue


def _scan_cold(added: dict) -> None:
    """Cold outreach for top matched jobs with NO warm contact."""
    try:
        if not _contacts_mod.search_configured():
            return
    except Exception as exc:  # noqa: BLE001
        log.warning("cold scan: search_configured failed: %s", exc)
        return
    try:
        index = referrals.build_index()
    except Exception:  # noqa: BLE001
        index = []
    with session() as db:
        jobs = (db.query(Job).filter(Job.status != "expired")
                  .order_by(Job.match_score.desc())
                  .limit(settings.outreach_cold_max).all())
        job_ids = [(j.id, j) for j in jobs]
    for job_id, job in job_ids:
        if added["_total"] >= settings.outreach_max_per_scan:
            return
        try:
            # Skip jobs that already have a warm contact (those are warm_intro).
            if index and referrals.contacts_for_job(job, index):
                continue
            # Prefer a shared-thread contact (coauthor/institution/field) with an
            # email; fall back to the generic public-web contact finder.
            threaded = _people_finder.find_shared_thread_contacts(job_id) or []
            contact = next((c for c in threaded if (c.get("email") or "").strip()), None)
            rationale = ""
            if contact is not None:
                rationale = f"Cold outreach (thread: {contact.get('thread', '')})"
            else:
                found = _contacts_mod.find_contacts(job_id) or []
                contact = next((c for c in found if (c.get("email") or "").strip()), None)
                if contact is None:
                    continue
                rationale = f"Cold outreach to {contact.get('name', 'hiring team')}"
            mail = _contacts_mod.networking_email(job_id, contact)
            d = _queue(kind="cold", subject=mail.get("subject", ""),
                       body=mail.get("body", ""),
                       recipient_email=contact.get("email", ""),
                       job_id=job_id,
                       rationale=rationale,
                       priority=float(getattr(job, "match_score", 0.0) or 0.0),
                       dedupe_key=f"cold:{job_id}")
            if d is not None:
                added["cold"] += 1
                added["_total"] += 1
        except Exception as exc:  # noqa: BLE001
            log.warning("cold scan: job %s failed: %s", job_id, exc)
            continue


def scan() -> dict:
    """Run all source scanners, fail-open, bounded and idempotent."""
    added = {"warm_intro": 0, "follow_up": 0, "cold": 0, "reply": 0, "_total": 0}
    if not settings.outreach_enabled:
        added.pop("_total")
        return added
    for fn in (_scan_warm_intros, _scan_followups, _scan_cold):
        try:
            fn(added)
        except Exception as exc:  # noqa: BLE001
            log.warning("scan source %s failed: %s", fn.__name__, exc)
    added.pop("_total")
    return added


from . import emailer  # noqa: E402


def _get(db, draft_id: int) -> OutreachDraft:
    d = db.get(OutreachDraft, draft_id)
    if d is None:
        raise ValueError(f"OutreachDraft {draft_id} not found")
    return d


def approve(draft_id: int) -> OutreachDraft:
    with session() as db:
        d = _get(db, draft_id)
        # kind="sent" rows are history, not outbox: conversations.record_sent()
        # files email that already left from a real mail client. send() only
        # accepts approved drafts, so refusing here closes the only door.
        if d.kind == "sent":
            raise ValueError(
                f"Draft {draft_id} is a record of an email you already sent. "
                "It cannot be approved or re-sent.")
        d.status = "approved"
        db.commit(); db.refresh(d)
        return d


def dismiss(draft_id: int) -> OutreachDraft:
    with session() as db:
        d = _get(db, draft_id)
        d.status = "dismissed"
        db.commit(); db.refresh(d)
        return d


def edit(draft_id: int, subject: str, body: str) -> OutreachDraft:
    with session() as db:
        d = _get(db, draft_id)
        d.subject = subject or ""
        d.body = body or ""
        db.commit(); db.refresh(d)
        return d


def send(draft_id: int) -> dict:
    """Email an APPROVED draft. Refuses otherwise. Blank recipient => refuse."""
    with session() as db:
        d = _get(db, draft_id)
        if d.status != "approved":
            raise ValueError("Draft must be approved before sending.")
        recipient = (d.recipient_email or "").strip()
        if not recipient:
            raise ValueError("Draft has no recipient email (copy-paste only).")
        subject, body = d.subject, d.body
        kind, application_id = d.kind, d.application_id
    paragraphs = (body or "").split("\n\n")
    body_html = "\n".join(
        f"<p>{p.replace(chr(10), '<br>')}</p>" for p in paragraphs if p.strip()
    )
    message_id = emailer.send(to=recipient, subject=subject,
                              body_html=body_html, body_text=body)
    with session() as db:
        d = _get(db, draft_id)
        d.status = "sent"
        d.sent_at = datetime.utcnow()
        if message_id:            # tests stub emailer.send as `lambda **kw: None`
            d.message_id = message_id
        db.commit()
    # Let the outcome-tracking loop learn from the touch (follow-ups only).
    if kind == "follow_up" and application_id:
        try:
            followup.record_followup(application_id)
        except Exception as exc:  # noqa: BLE001
            log.warning("record_followup failed for app %s: %s", application_id, exc)
    return {"to": recipient, "subject": subject, "status": "sent"}


def queue_list(kind: Optional[str] = None) -> list:
    """Pending + approved drafts, highest priority first."""
    with session() as db:
        q = db.query(OutreachDraft).filter(OutreachDraft.status.in_(["pending", "approved"]))
        if kind:
            q = q.filter(OutreachDraft.kind == kind)
        return q.order_by(OutreachDraft.priority.desc(),
                          OutreachDraft.created_at.desc()).all()
