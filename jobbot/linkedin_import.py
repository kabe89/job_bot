# jobbot/linkedin_import.py
"""Import a LinkedIn 'Connections.csv' export into the Contact store.

Official data-export CSV only (no scraping / no API). Fail-open: any error
returns the zeroed stats dict rather than raising.
"""
from __future__ import annotations

import csv
import logging
import os
from pathlib import Path

from .config import settings
from .contacts import _company_tokens
from .models import Contact, Job, init_db, session

log = logging.getLogger("jobbot.linkedin_import")

_HEADER_COL = "First Name"


def _affiliation_terms() -> list:
    return [a.strip().lower()
            for a in (settings.affiliation_terms or "").split(",") if a.strip()]


def _norm_linkedin(url: str) -> str:
    return (url or "").strip().rstrip("/").lower()


def _find_header_index(lines: list) -> int:
    """Index of the LinkedIn header row (skips the export's Notes preamble)."""
    for i, ln in enumerate(lines):
        s = ln.lstrip().strip('"')
        if s.startswith(_HEADER_COL + ",") or s.startswith(_HEADER_COL + '"'):
            return i
    return -1


def import_connections(csv_path: "str | os.PathLike") -> dict:
    """Import connections from *csv_path*. Returns counts; fail-open."""
    stats = {"imported": 0, "updated": 0, "skipped": 0, "at_target": 0}
    p = Path(csv_path)
    if not p.exists():
        log.warning("LinkedIn CSV not found: %s", csv_path)
        return stats
    try:
        text = p.read_text(encoding="utf-8-sig", errors="replace")
    except Exception as exc:  # noqa: BLE001
        log.warning("Could not read %s: %s", csv_path, exc)
        return stats

    lines = text.splitlines()
    hi = _find_header_index(lines)
    if hi < 0:
        log.warning("No LinkedIn header row in %s", csv_path)
        return stats

    affils = _affiliation_terms()
    init_db()
    try:
        with session() as db:
            # Pre-build the set of distinctive tokens across all job companies,
            # so we can flag imported connections that sit at a target company.
            job_tokens: set = set()
            for (comp,) in db.query(Job.company).distinct():
                job_tokens |= set(_company_tokens(comp or ""))

            for row in csv.DictReader(lines[hi:]):
                first = (row.get("First Name") or "").strip()
                last = (row.get("Last Name") or "").strip()
                name = (first + " " + last).strip()
                if not name:
                    stats["skipped"] += 1
                    continue
                company = (row.get("Company") or "").strip()
                title = (row.get("Position") or "").strip()
                email = (row.get("Email Address") or "").strip()
                url = _norm_linkedin(row.get("URL") or "")
                connected = (row.get("Connected On") or "").strip()
                notes = f"LinkedIn 1st-degree; connected {connected}".strip()
                rel = "alum" if (affils and any(
                    a in f"{company} {title}".lower() for a in affils)) else "unknown"

                existing = None
                if url:
                    existing = db.query(Contact).filter(Contact.linkedin == url).first()
                if existing is None:
                    existing = (db.query(Contact)
                                .filter(Contact.name == name, Contact.company == company)
                                .first())

                if existing is None:
                    db.add(Contact(name=name, company=company, title=title,
                                   email=email, linkedin=url, source="import",
                                   relationship=rel, notes=notes))
                    stats["imported"] += 1
                else:
                    # Enrich empties; only rewrite curated fields on import rows.
                    existing.company = existing.company or company
                    existing.title = existing.title or title
                    existing.email = existing.email or email
                    if not existing.linkedin:
                        existing.linkedin = url
                    if existing.source == "import":
                        existing.notes = notes
                        if existing.relationship in ("", "unknown"):
                            existing.relationship = rel
                    stats["updated"] += 1

                if company and (set(_company_tokens(company)) & job_tokens):
                    stats["at_target"] += 1
            db.commit()
    except Exception as exc:  # noqa: BLE001
        log.warning("import_connections failed: %s", exc)
        return {"imported": 0, "updated": 0, "skipped": 0, "at_target": 0}
    return stats
