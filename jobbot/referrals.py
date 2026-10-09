# jobbot/referrals.py
"""Referral-first routing: the user's warm-network Contact store + a bounded,
explainable ranking boost for jobs at companies where a contact exists.

Deterministic and fail-open: no contacts / no search provider / any error ->
neutral (empty list, 0.0 bonus) and ranking is unchanged. Reuses contacts.py's
web search and networking_package for outreach; adds referral_bonus on top of
outcomes.blended (sort-only).
"""
from __future__ import annotations

import json
import logging
import re
from pathlib import Path
from typing import List, Set, Tuple

from .config import settings
from .contacts import _company_tokens
from .models import Contact, session

log = logging.getLogger("jobbot.referrals")

_PEER_TITLE = re.compile(r"scientist|professor|principal|\bpi\b|faculty|"
                         r"director of research|research lead|investigator", re.I)
_RECRUITER_TITLE = re.compile(r"recruit|talent|sourcer", re.I)


def profile_terms() -> Set[str]:
    """Lowercased tokens from the candidate profile's skills+domains, for field
    matching. Reads the JSON directly (no Ollama). Fail-open -> empty set."""
    try:
        p = Path(settings.candidate_profile_path)
        if not p.exists():
            return set()
        data = json.loads(p.read_text(encoding="utf-8"))
        terms: Set[str] = set()
        for key in ("skills", "domains"):
            for item in data.get(key) or []:
                terms.add(str(item).strip().lower())
        return {t for t in terms if t}
    except Exception:  # noqa: BLE001
        return set()


def warmth_score(contact, terms: Set[str]) -> Tuple[float, str]:
    """Deterministic warmth in [0,1] + rationale. Fail-open -> (0.0, '')."""
    try:
        score = 0.0
        reasons: List[str] = []
        title = getattr(contact, "title", "") or ""
        text = f"{getattr(contact, 'company', '')} {title} {getattr(contact, 'notes', '')}".lower()
        if getattr(contact, "pinned", False):
            score += 0.6; reasons.append("pinned")
        rel = (getattr(contact, "relationship", "") or "").lower()
        if rel in ("colleague", "alum", "pi"):
            score += 0.2; reasons.append(rel)
        if terms and any(t in text for t in terms):
            score += 0.2; reasons.append("shared field")
        affils = [a.strip().lower() for a in (settings.affiliation_terms or "").split(",") if a.strip()]
        if affils and any(a in text for a in affils):
            score += 0.3; reasons.append("shared institution")
        if _PEER_TITLE.search(title):
            score += 0.15; reasons.append("peer/PI")
        elif _RECRUITER_TITLE.search(title):
            score += 0.02
        if (getattr(contact, "source", "") or "").lower() == "import":
            score += 0.25; reasons.append("1st-degree connection")
        score = max(0.0, min(1.0, score))
        return score, ", ".join(reasons)
    except Exception:  # noqa: BLE001
        return 0.0, ""


def _alias_patterns(raw: str) -> list:
    """Lowercased substring patterns from a contact's company_aliases field
    (semicolon / newline separated — NOT comma, since commas occur inside org
    names like 'Health Research, Inc.'). Matched against a job's RAW company
    string — precise where token overlap is not (e.g. 'department of health'
    catches 'Health, Department of' without matching 'Department of Taxation')."""
    parts = re.split(r"[;\n]+", (raw or "").lower())
    return [p.strip() for p in parts if p.strip()]


class _CardSnap:
    __slots__ = ("name", "company", "title", "email", "linkedin", "relationship",
                 "pinned", "notes", "id", "tokens", "aliases", "warmth")


def build_index() -> list:
    """Snapshot all contacts with precomputed company tokens + warmth so the
    caller can score jobs without re-querying (recomputed per request, no
    module-global cache -> no staleness)."""
    terms = profile_terms()
    out: list = []
    try:
        with session() as db:
            rows = db.query(Contact).all()
            for c in rows:
                s = _CardSnap()
                s.id = c.id; s.name = c.name; s.company = c.company; s.title = c.title
                s.email = c.email; s.linkedin = c.linkedin
                s.relationship = c.relationship; s.pinned = c.pinned; s.notes = c.notes
                s.tokens = set(_company_tokens(c.company or ""))
                s.aliases = _alias_patterns(getattr(c, "company_aliases", "") or "")
                s.warmth = warmth_score(c, terms)[0]
                out.append(s)
    except Exception as e:  # noqa: BLE001
        log.warning("build_index failed (%s) — no referrals.", e)
        return []
    return out


def contacts_for_job(job, index: list) -> list:
    """Contacts linked to the job's company, warmth desc. A contact matches when
    its distinctive company tokens overlap the job's, OR one of its alias
    substring patterns appears in the job's raw company string."""
    try:
        raw = (getattr(job, "company", "") or "").lower()
        jt = set(_company_tokens(raw))

        def _match(c):
            if jt and (c.tokens & jt):
                return True
            aliases = getattr(c, "aliases", None) or []
            return any(p in raw for p in aliases)

        hits = [c for c in index if _match(c)]
        hits.sort(key=lambda c: c.warmth, reverse=True)
        return hits
    except Exception:  # noqa: BLE001
        return []


def referral_bonus(job, index: list) -> Tuple[float, str]:
    """Bounded ranking boost for a job with a matching contact. (0.0,'') if none.

    Field gate: an org match on a job the candidate is a poor fit for (match_score
    below settings.warm_intro_min_score) yields no boost/badge, so a well-connected
    contact doesn't drag in off-field roles at their institution."""
    try:
        if (getattr(job, "match_score", 0.0) or 0.0) < settings.warm_intro_min_score:
            return 0.0, ""
        hits = contacts_for_job(job, index)
        if not hits:
            return 0.0, ""
        best = hits[0]
        bonus = max(0.0, min(settings.referral_bonus_cap, settings.referral_bonus_cap * best.warmth))
        why = f"+{bonus:.3f}: warm intro via {best.name}" + (f" @ {best.company}" if best.company else "")
        return bonus, why
    except Exception:  # noqa: BLE001
        return 0.0, ""


# ---- discovery --------------------------------------------------------------

class _CompanyJob:
    """Minimal job-like object the contacts search sources read (.company/.title)."""
    __slots__ = ("company", "title", "description", "url", "id")

    def __init__(self, company: str, title: str):
        self.company = company; self.title = title
        self.description = ""; self.url = ""; self.id = 0


def _rep_title() -> str:
    """A representative target title for the team-search query."""
    try:
        data = json.loads(Path(settings.candidate_profile_path).read_text(encoding="utf-8"))
        titles = data.get("role_titles") or data.get("role_archetypes") or []
        if titles:
            return str(titles[0])
    except Exception:  # noqa: BLE001
        pass
    return "Scientist"


def discover_company(company: str, max_contacts=None) -> list:
    """Search for contacts at `company`, store new (deduped) Contact rows.
    Returns the list of newly-added Contact ids. Fail-open."""
    from . import contacts as gc
    company = (company or "").strip()
    if not company or not gc.search_configured():
        return []
    cap = max_contacts or settings.contacts_max_per_company
    added: list = []
    try:
        shim = _CompanyJob(company, _rep_title())
        found: list = []
        for src in (gc._contacts_from_recruiter_search, gc._contacts_from_team_search):
            try:
                found.extend(src(shim) or [])
            except Exception:  # noqa: BLE001
                pass
        with session() as db:
            existing = {(c.name or "").strip().lower()
                        for c in db.query(Contact).filter_by(company=company).all()}
            for d in found:
                name = (d.get("name") or "").strip()
                if not name or name.lower() in existing:
                    continue
                existing.add(name.lower())
                c = Contact(name=name, company=company, title=d.get("role", "") or "",
                            email=d.get("email", "") or "", linkedin=d.get("linkedin", "") or "",
                            source="discovered", relationship="unknown")
                db.add(c); db.flush()
                added.append(c.id)
                if len(added) >= cap:
                    break
            db.commit()
    except Exception as e:  # noqa: BLE001
        log.warning("discover_company(%s) failed (%s).", company, e)
        return []
    return added


def _discovered_path() -> Path:
    return Path(settings.output_dir) / "referral_discovered.json"


def _load_discovered() -> set:
    try:
        return set(json.loads(_discovered_path().read_text(encoding="utf-8")))
    except Exception:  # noqa: BLE001
        return set()


def _save_discovered(names: set) -> None:
    try:
        p = _discovered_path(); p.parent.mkdir(parents=True, exist_ok=True)
        p.write_text(json.dumps(sorted(names)), encoding="utf-8")
    except Exception:  # noqa: BLE001
        pass


def discover_top_companies(max_companies=None, force=False) -> dict:
    """Discover contacts for the top companies by count of in-range non-expired
    jobs. Skips companies already searched (unless force). Returns {company: n}."""
    from .models import Job
    from sqlalchemy import func
    limit = max_companies or settings.referral_discover_max
    done = set() if force else _load_discovered()
    result: dict = {}
    try:
        with session() as db:
            rows = (db.query(Job.company, func.count(Job.id))
                    .filter(Job.status != "expired")
                    .group_by(Job.company)
                    .order_by(func.count(Job.id).desc()).all())
        ranked = [(comp, n) for comp, n in rows if comp and comp.strip()]
        for comp, _n in ranked:
            if len(result) >= limit:
                break
            if comp in done:
                continue
            added = discover_company(comp)
            done.add(comp)
            result[comp] = len(added)
        _save_discovered(done)
    except Exception as e:  # noqa: BLE001
        log.warning("discover_top_companies failed (%s).", e)
    return result


def seed_sample_lead():
    """Insert a sample pinned contact lead once (idempotent). Returns the new
    Contact, or None if already present."""
    with session() as db:
        if db.query(Contact).filter(Contact.name.like("%Taylor%")).first():
            return None
        c = Contact(name="Dr. Alex Taylor", company="Acme Research Institute",
                    company_aliases="acme research; acme institute; department of health; health research",
                    title="Principal Investigator", relationship="pi", source="manual",
                    pinned=True, notes="Sample insider referral contact.")
        db.add(c); db.commit(); db.refresh(c)
        return c

