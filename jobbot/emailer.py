"""SMTP email sender — digests + applications."""
from __future__ import annotations

import logging
import smtplib
from email.message import EmailMessage
from email.utils import make_msgid
from pathlib import Path
from typing import Iterable

from .config import settings

log = logging.getLogger("jobbot.email")


def _smtp():
    if not settings.smtp_user or not settings.smtp_password:
        raise RuntimeError("SMTP_USER / SMTP_PASSWORD not configured in .env")
    s = smtplib.SMTP(settings.smtp_host, settings.smtp_port, timeout=30)
    s.starttls()
    s.login(settings.smtp_user, settings.smtp_password)
    return s


def send(to: str, subject: str, body_html: str, body_text: str = "", attachments: Iterable[str | Path] = ()) -> str:
    """Send one email. Returns the Message-ID it stamped on the message.

    The returned id is what WE set. A submission service may overwrite it in
    transit, in which case the stored value is advisory: it still helps the
    recipient's client thread the message, but it will not match the
    In-Reply-To that comes back. Recovering the true sent id would require
    reading the Sent folder over IMAP, which this feature deliberately does not
    do.
    """
    msg = EmailMessage()
    msg["From"] = f"{settings.smtp_from_name} <{settings.smtp_user}>"
    msg["To"] = to
    msg["Subject"] = subject
    # Set explicitly so the id can be stored and later matched against a
    # reply's In-Reply-To. Domain comes from smtp_user: a non-resolving domain
    # inside a Message-ID is a spam signal.
    domain = (settings.smtp_user.split("@", 1)[-1] or "localhost").strip()
    msg["Message-ID"] = make_msgid(domain=domain)
    msg.set_content(body_text or "This email requires an HTML-capable client.")
    msg.add_alternative(body_html, subtype="html")
    for att in attachments:
        p = Path(att)
        if not p.exists():
            log.warning("Attachment missing: %s", p)
            continue
        data = p.read_bytes()
        sub = "pdf" if p.suffix.lower() == ".pdf" else "octet-stream"
        msg.add_attachment(data, maintype="application", subtype=sub, filename=p.name)
    with _smtp() as s:
        s.send_message(msg)
    log.info("Email sent to %s: %s", to, subject)
    return msg["Message-ID"]


def digest_html(jobs) -> str:
    rows = []
    for j in jobs:
        rows.append(f"""
        <tr>
          <td><a href="{j.url}">{j.title}</a></td>
          <td>{j.company}</td>
          <td>{j.location}</td>
          <td>{j.source}</td>
          <td>{j.match_score:.2f}</td>
        </tr>""")
    return f"""
    <html><body style="font-family:system-ui,Arial,sans-serif">
    <h2>JobBot Digest — {len(jobs)} new matches</h2>
    <table border="1" cellpadding="6" cellspacing="0" style="border-collapse:collapse;">
      <thead><tr><th>Title</th><th>Company</th><th>Location</th><th>Source</th><th>Score</th></tr></thead>
      <tbody>{''.join(rows)}</tbody>
    </table>
    <p style="color:#666">Open the dashboard to tailor and apply.</p>
    </body></html>"""
