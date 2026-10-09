"""Email conversation threads.

A thread holds the standing situation for one ongoing email exchange; the
messages themselves are OutreachDraft rows pointing back via thread_id.

This module decides what context a reply draft should see and files it through
outreach._queue(). It never sends email: outreach.send() remains the only path
to SMTP, so the approve-then-send gate is inherited rather than reimplemented.
"""
from __future__ import annotations

import hashlib
import logging
from datetime import datetime
from email.utils import parseaddr
from typing import List, Optional

from . import outcomes, outreach
from .models import (Application, Contact, Job, OutreachDraft, OutreachThread,
                     init_db, session)

log = logging.getLogger(__name__)


def _get(db, thread_id: int) -> OutreachThread:
    t = db.get(OutreachThread, thread_id)
    if t is None:
        raise ValueError(f"OutreachThread {thread_id} not found")
    return t


def start(contact_id: Optional[int] = None, job_id: Optional[int] = None,
          application_id: Optional[int] = None, scenario: str = "",
          subject: str = "") -> OutreachThread:
    """Create an open thread. All links are optional."""
    init_db()
    with session() as db:
        name = email = ""
        if contact_id:
            c = db.get(Contact, contact_id)
            if c is not None:
                name, email = c.name or "", c.email or ""
        subj = subject or ""
        if not subj and job_id:
            j = db.get(Job, job_id)
            if j is not None:
                subj = f"{j.title} at {j.company}"
        last_outbound = None
        if application_id:
            app = db.get(Application, application_id)
            if app is not None:
                last_outbound = app.sent_at
        t = OutreachThread(
            contact_id=contact_id, job_id=job_id, application_id=application_id,
            subject=subj, scenario=scenario or "", status="open",
            counterpart_name=name, counterpart_email=email,
            last_outbound_at=last_outbound,
        )
        db.add(t); db.commit(); db.refresh(t)
        db.expunge(t)
        return t


def history(thread_id: int) -> List[dict]:
    """Ordered message log, oldest first.

    Entries are flattened from every draft on the thread and THEN sorted by
    their own timestamp: a draft created Monday and sent Friday must fall after
    a reply received Wednesday, which sorting by draft.created_at would get
    wrong.
    """
    with session() as db:
        t = _get(db, thread_id)
        rows: List[dict] = []

        # Synthetic opening entry: the original application. Its body was never
        # stored (pipeline.send_application emails directly), so we record only
        # that it happened and when.
        if t.application_id:
            app = db.get(Application, t.application_id)
            if app is not None and app.sent_at:
                job = db.get(Job, app.job_id)
                where = f"{job.title} at {job.company}" if job else "the role"
                rows.append({"direction": "sent", "draft_id": None,
                             "at": app.sent_at,
                             "text": f"Applied to {where}."})

        drafts = (db.query(OutreachDraft)
                    .filter(OutreachDraft.thread_id == thread_id)
                    .all())
        for d in drafts:
            if (d.inbound or "").strip():
                rows.append({"direction": "received", "draft_id": d.id,
                             "at": d.created_at, "text": d.inbound.strip()})
            # A pending draft has not happened yet, so it is not history.
            if (d.body or "").strip() and d.sent_at:
                rows.append({"direction": "sent", "draft_id": d.id,
                             "at": d.sent_at, "text": d.body.strip()})

        rows.sort(key=lambda r: r["at"])
        return rows


def set_scenario(thread_id: int, text: str) -> OutreachThread:
    with session() as db:
        t = _get(db, thread_id)
        t.scenario = text or ""
        db.commit(); db.refresh(t); db.expunge(t)
        return t


def close(thread_id: int) -> OutreachThread:
    with session() as db:
        t = _get(db, thread_id)
        t.status = "closed"
        db.commit(); db.refresh(t); db.expunge(t)
        return t


def list_open() -> List[OutreachThread]:
    """Open threads, most recent inbound first (never-replied last)."""
    with session() as db:
        threads = (db.query(OutreachThread)
                     .filter(OutreachThread.status == "open")
                     .all())
        threads.sort(key=lambda t: t.last_inbound_at or datetime.min, reverse=True)
        for t in threads:
            db.expunge(t)
        return threads


def _normalize(text: str) -> str:
    """Canonical form of a pasted email, for content-based dedupe.

    Lowercases, collapses whitespace, and drops quoted-reply lines so the same
    email pasted twice (with different trailing spaces or quote depth) hashes
    identically.
    """
    lines = []
    for raw in (text or "").splitlines():
        s = raw.strip()
        if s.startswith(">"):
            continue
        s = " ".join(s.split())
        if s:
            lines.append(s)
    return "\n".join(lines).lower()


def _dedupe_key(thread_id: int, inbound: str) -> str:
    """Content-derived, NOT positional.

    A positional key such as f"reply:{thread_id}:{n}" (n = existing draft
    count) would advance on every call, so the second paste of the same email
    produces a different key and _live_key_exists() never fires. That is the
    same dead-check defect as the old timestamp key.
    """
    digest = hashlib.sha1(_normalize(inbound).encode("utf-8")).hexdigest()[:16]
    return f"reply:{thread_id}:{digest}"


def _touch_outbound(thread_id: int, when: datetime) -> None:
    """Advance last_outbound_at, never rewind it.

    Sent messages get recorded out of order (you remember Tuesday's email after
    filing Friday's), and rewinding the clock would make the nudge guard think
    the thread is older than it is.
    """
    with session() as db:
        t = _get(db, thread_id)
        if t.last_outbound_at is None or when > t.last_outbound_at:
            t.last_outbound_at = when
            db.commit()


def record_sent(thread_id: int, body: str, subject: str = "",
                sent_at: Optional[datetime] = None,
                recipient: str = "") -> OutreachDraft:
    """File an email you sent by hand, outside the bot, onto a thread.

    Networking email leaves from a real mail client, so without this the thread
    has no outbound history and every later draft is written blind. The row is
    written already-sent: it must never become approvable, because approving it
    would send the same email twice.
    """
    if not (body or "").strip():
        raise ValueError("body must not be empty.")
    init_db()
    when = sent_at or datetime.utcnow()

    with session() as db:
        t = _get(db, thread_id)
        subj = subject or t.subject or ""
        to = recipient or t.counterpart_email or ""
        job_id, contact_id, application_id = t.job_id, t.contact_id, t.application_id

    key = f"sent:{thread_id}:{hashlib.sha1(_normalize(body).encode('utf-8')).hexdigest()[:16]}"
    with session() as db:
        existing = (db.query(OutreachDraft)
                      .filter(OutreachDraft.dedupe_key == key,
                              OutreachDraft.status != "dismissed")
                      .first())
        if existing is not None:
            db.expunge(existing)
            existing.is_new = False
            _touch_outbound(thread_id, when)
            return existing

        d = OutreachDraft(
            kind="sent", status="sent", thread_id=thread_id, subject=subj,
            body=body, recipient_email=to, dedupe_key=key, sent_at=when,
            job_id=job_id, contact_id=contact_id, application_id=application_id,
            rationale="Sent by hand, recorded after the fact.", priority=0.0,
        )
        db.add(d); db.commit(); db.refresh(d); db.expunge(d)
    d.is_new = True
    _touch_outbound(thread_id, when)
    return d


# Warm contacts get two weeks, not the 8 days an application follow-up gets.
# Someone doing you a favour on their own time is not a hiring pipeline.
NUDGE_QUIET_DAYS = 14


def draft_nudge(thread_id: int, force: bool = False) -> OutreachDraft:
    """Queue a low-pressure follow-up for an outbound message that got no reply."""
    init_db()
    with session() as db:
        t = _get(db, thread_id)
        if t.status != "open":
            raise ValueError(f"Thread {thread_id} is closed.")
        last_out, last_in = t.last_outbound_at, t.last_inbound_at
        scenario, subject = t.scenario or "", t.subject or ""
        recipient = t.counterpart_email or ""
        job_id, contact_id, application_id = t.job_id, t.contact_id, t.application_id

    if last_out is None:
        raise ValueError(
            "Nothing has been sent on this thread yet. Record the email you sent "
            "first: jobbot convo sent <thread> --file <path>")
    if last_in is not None and last_in >= last_out:
        raise ValueError(
            "They already replied, so this needs a reply and not a nudge. "
            "Use: jobbot convo paste <thread> --file <path>")

    days = (datetime.utcnow() - last_out).days
    if days < NUDGE_QUIET_DAYS and not force:
        raise ValueError(
            f"Only {days} days since your last message on thread {thread_id}. "
            f"Warm contacts get {NUDGE_QUIET_DAYS}. Pass --force to draft anyway.")

    prior = history(thread_id)
    try:
        subj, body = outreach._nudge_text(days=days, scenario=scenario,
                                          history=prior, subject=subject)
    except Exception as exc:  # noqa: BLE001
        log.warning("nudge drafting failed for thread %s: %s", thread_id, exc)
        subj, body = (f"Re: {subject}" if subject else "Re: following up", "")

    # Keyed on the message being chased, so a second call is a no-op but a nudge
    # for a LATER unanswered email is still allowed through.
    key = f"nudge:{thread_id}:{last_out.date().isoformat()}"
    d = outreach._queue(
        kind="follow_up", subject=outreach._scrub(subj), body=body,
        recipient_email=recipient, thread_id=thread_id, job_id=job_id,
        contact_id=contact_id, application_id=application_id, priority=0.85,
        rationale=f"Nudge on thread {thread_id}, {days} days with no reply.",
        dedupe_key=key)

    if d is None:
        with session() as db:
            d = (db.query(OutreachDraft)
                   .filter(OutreachDraft.dedupe_key == key,
                           OutreachDraft.status != "dismissed")
                   .first())
            db.expunge(d)
        d.is_new = False
        return d
    d.is_new = True
    return d


def ingest(thread_id: int, inbound_text: str, sender: str = "") -> OutreachDraft:
    """Record an inbound email on a thread and queue a context-aware reply.

    Always returns a draft. On a dedupe hit it returns the existing one, with
    `.is_new = False` so the caller can say so rather than implying a fresh
    draft was written.
    """
    if not (inbound_text or "").strip():
        raise ValueError("inbound_text must not be empty.")

    with session() as db:
        t = _get(db, thread_id)
        if t.status != "open":
            raise ValueError(f"Thread {thread_id} is closed.")
        scenario, subject = t.scenario or "", t.subject or ""
        application_id = t.application_id
        job_id = t.job_id
        contact_id = t.contact_id
        counterpart = t.counterpart_email or ""

    def _record_arrival() -> None:
        """Side effects owed to any inbound message, deduped draft or not.

        A repeated paste is still evidence a human wrote back, so it must move
        the clock and suppress the follow-up nudge even when no new draft is
        written. Two genuinely different emails can also share a dedupe key
        (identical visible text, e.g. "Thanks!"), and skipping these would
        silently drop the suppression the whole feature rests on.
        """
        with session() as db:
            t_ = _get(db, thread_id)
            t_.last_inbound_at = datetime.utcnow()
            db.commit()
        # Fail-open: a bookkeeping error must never cost the user their email.
        if application_id:
            try:
                outcomes.record_outcome(application_id, "responded",
                                        note="inbound reply")
            except Exception as exc:  # noqa: BLE001
                log.warning("record_outcome failed for app %s: %s",
                            application_id, exc)

    key = _dedupe_key(thread_id, inbound_text)
    with session() as db:
        existing = (db.query(OutreachDraft)
                      .filter(OutreachDraft.dedupe_key == key,
                              OutreachDraft.status != "dismissed")
                      .first())
        if existing is not None:
            db.expunge(existing)
            existing.is_new = False
            _record_arrival()
            return existing

    prior = history(thread_id)
    _name, addr = parseaddr(sender or "")
    recipient = addr or counterpart

    try:
        subj, body = outreach._reply_text(
            inbound_text, sender=sender, scenario=scenario,
            history=prior, subject=subject)
    except Exception as exc:  # noqa: BLE001
        log.warning("reply drafting failed for thread %s: %s", thread_id, exc)
        subj, body = (f"Re: {subject}" if subject else "Re: your message", "")

    # Compare against the boilerplate rather than probing is_available(): a
    # reachable server still yields the template when _generate raises or comes
    # back empty, and mislabelling that draft as generated is exactly backwards
    # from what the reader needs to know.
    generated = bool(body.strip()) and body.strip() != outreach._scrub(
        outreach._fallback_reply_body()).strip()

    rationale = (f"Reply on thread {thread_id}"
                 if generated else f"Reply on thread {thread_id} (ungenerated template)")

    d = outreach._queue(kind="reply", subject=outreach._scrub(subj), body=body,
                        recipient_email=recipient, inbound=inbound_text,
                        thread_id=thread_id, job_id=job_id,
                        contact_id=contact_id, application_id=application_id,
                        rationale=rationale, priority=0.9, dedupe_key=key)

    if d is not None:
        d.is_new = True
    _record_arrival()
    return d
