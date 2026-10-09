"""Seed the answer bank from the resume.

Facts a resume can settle -- highest degree, graduation, current role, years of
experience -- are otherwise re-answered by hand or invented by the AI on every
application.

These become *bank* entries rather than memory entries because the bank is the
regex layer. Memory matches on an exact `canonical_key`, and a seeded key would
never equal the key a real form produces from its own phrasing, so a seeded
memory entry would sit there matching nothing.

Every deterministic value is verified in code before it is written: the bank is a
VERIFIED source, so anything active here is submitted to a real employer without
review.
"""
from __future__ import annotations

import logging
import re
from datetime import datetime
from typing import List, Optional

log = logging.getLogger("jobbot.answer_seed")

# Ordered most- to least-advanced; the first hit wins.
_DEGREES = [
    ("PhD", r"\bph\.?\s?d\b|\bdoctorate\b|\bdoctoral\b"),
    ("Master's", r"\bm\.?s\.?c?\b|\bmaster'?s\b|\bm\.?eng\b|\bmba\b"),
    ("Bachelor's", r"\bb\.?s\.?c?\b|\bb\.?a\b|\bbachelor'?s\b"),
]

_YEAR_RE = re.compile(r"\b(19|20)\d{2}\b")

# The longest plausible career. A parse failure that produced 224 years must
# never reach a form.
_MAX_YEARS = 50


def highest_degree(resume_text: str) -> str:
    """The most advanced degree named in the resume, or ''."""
    for label, pattern in _DEGREES:
        if re.search(pattern, resume_text, re.I):
            return label
    return ""


def total_years_experience(resume_text: str, now_year: Optional[int] = None) -> int:
    """Years from the earliest four-digit year in the resume to now.

    Returns 0 rather than a wrong number when the span is implausible -- a
    citation year or a typo would otherwise become a claim to an employer.
    """
    now_year = now_year or datetime.now().year
    years = [int(m.group(0)) for m in _YEAR_RE.finditer(resume_text)]
    if not years:
        return 0
    span = now_year - min(years)
    return span if 0 <= span <= _MAX_YEARS else 0


def _degree_entry(resume_text: str) -> Optional[dict]:
    degree = highest_degree(resume_text)
    if not degree:
        return None
    return {"match": r"highest (degree|level of education)|education level"
                     r"|degree (earned|obtained|held)",
            "answer": degree}


def _years_entry(resume_text: str, now_year: Optional[int] = None) -> Optional[dict]:
    years = total_years_experience(resume_text, now_year)
    if years <= 0:
        return None
    # Staged, not active: the span is the min of EVERY four-digit year in the
    # resume, so a citation or reference year silently inflates it. The 0-50 band
    # catches only absurd values, not a plausible-but-wrong number, so a human
    # confirms this before it can be submitted to an employer.
    return {"match": r"(total )?years of (professional |relevant |work )?experience"
                     r"|how many years.*experience",
            "answer": str(years),
            "status": "pending"}


_DATED_DEGREE_RE = re.compile(
    r"(?P<year>(?:19|20)\d{2})-(?P<month>\d{2})")


def expected_graduation(resume_text: str, now_year: Optional[int] = None) -> str:
    """The latest future-dated degree completion as 'YYYY-MM', or ''.

    Only a date still ahead of us is a graduation *expectation*; a past one is
    a completed degree and answering with it would misstate availability.
    """
    now_year = now_year or datetime.now().year
    best = ""
    for line in resume_text.splitlines():
        if not re.search(r"\b(ph\.?\s?d|m\.?s|b\.?s|bachelor|master|doctor)",
                         line, re.I):
            continue
        m = _DATED_DEGREE_RE.search(line)
        if not m:
            continue
        year, month = int(m.group("year")), int(m.group("month"))
        if year < now_year or not 1 <= month <= 12:
            continue
        stamp = f"{year:04d}-{month:02d}"
        if stamp > best:
            best = stamp
    return best


def current_role(resume_text: str) -> tuple:
    """(title, employer) of the role marked current, or ('', '').

    A role is current when its line ends in 'present' or 'current'. Returns
    empties rather than guessing at the most recent dated role -- a wrong
    current employer on an application is worse than a blank one.
    """
    for line in resume_text.splitlines():
        if not re.search(r"\b(present|current)\s*$", line.strip(), re.I):
            continue
        parts = [p.strip() for p in line.split(",")]
        if len(parts) >= 2:
            return parts[0], parts[1]
    return "", ""


def _graduation_entry(resume_text: str,
                      now_year: Optional[int] = None) -> Optional[dict]:
    stamp = expected_graduation(resume_text, now_year)
    if not stamp:
        return None
    return {"match": r"expected graduation|graduation date|anticipated "
                     r"(completion|graduation)|when do you (expect to )?graduate",
            "answer": stamp}


def _current_role_entries(resume_text: str) -> List[dict]:
    # Staged, not active: (title, employer) is a comma-split of the first line
    # ending in "present"/"current", which a non-job line ("At present, ...") can
    # fool. A wrong current employer on an application is worse than a blank, so a
    # human confirms these before they can be submitted.
    title, employer = current_role(resume_text)
    out = []
    if title:
        out.append({"match": r"current (job )?title|present title"
                             r"|most recent (job )?title",
                    "answer": title,
                    "status": "pending"})
    if employer:
        out.append({"match": r"current employer|present employer"
                             r"|most recent employer|currently employed by",
                    "answer": employer,
                    "status": "pending"})
    return out


def _skill_entries(skills: List[str]) -> List[dict]:
    """One yes-entry per skill actually listed on the profile.

    Deterministic: the skill is on the profile, so the answer is Yes. No
    inference and nothing for a model to get wrong.
    """
    out = []
    for skill in skills:
        s = (skill or "").strip()
        if not s:
            continue
        out.append({
            # The leading lookahead is load-bearing, not defensive styling. Without
            # it this pattern also matches "How many years of experience do you have
            # with <skill>?" -- a question the years entry owns -- and _bank_lookup's
            # longest-pattern tie-break hands the win to whichever is longer, which
            # for any skill name over ~39 chars is this one. It would then answer
            # "Yes" to a how-many-years question, unreviewed, on a real application.
            "match": r"^(?!.*(?:how many|\byears?\b))"
                     r".*(experience|familiar|proficien\w+|worked).{0,30}"
                     + re.escape(s),
            "answer": "Yes",
        })
    return out


def deterministic_entries(resume_text: str, skills: List[str],
                          now_year: Optional[int] = None) -> List[dict]:
    """Bank-shaped entries extracted from the resume.

    Truly-provable facts (degree, future graduation, listed skills) are `active`
    and auto-submittable. Heuristic extractions (years-of-experience, current
    role) carry `status: "pending"` -- they parse plausibly-wrong values often
    enough that a human must confirm them before they reach an employer. Callers
    should partition by status rather than assume everything here is active.
    """
    entries: List[dict] = []
    for builder in (_degree_entry,
                    lambda t: _years_entry(t, now_year),
                    lambda t: _graduation_entry(t, now_year)):
        entry = builder(resume_text)
        if entry:
            entries.append(entry)
    entries.extend(_current_role_entries(resume_text))
    entries.extend(_skill_entries(skills))
    return entries


_PROMPT = ("Read this resume and answer with a single integer: how many years of "
           "hands-on experience does the candidate have with {skill}? "
           "Reply with the number only.\n\nRESUME:\n{resume}")


def inferred_entries(resume_text: str, skills: List[str],
                     ask, now_year: Optional[int] = None) -> List[dict]:
    """Per-skill years-of-experience estimates, staged for review.

    `ask` takes a prompt and returns the model's reply, so this is testable
    without Ollama.

    The model proposes; this function verifies. An estimate survives only if it
    parses as an integer, does not exceed total experience, and names a skill
    that actually appears in the resume. Anything else is dropped rather than
    staged -- a staged answer is one approval away from being submitted.
    """
    total = total_years_experience(resume_text, now_year)
    if total <= 0:
        return []

    out: List[dict] = []
    for skill in skills:
        s = (skill or "").strip()
        if not s or s.lower() not in resume_text.lower():
            continue
        try:
            reply = ask(_PROMPT.format(skill=s, resume=resume_text))
        except Exception as e:  # noqa: BLE001
            log.info("skill-years estimate failed for %s: %s", s, e)
            continue
        m = re.search(r"\d+", str(reply or ""))
        if not m:
            continue
        years = int(m.group(0))
        if years <= 0 or years > total:
            log.info("dropped %s: %d years exceeds total experience of %d",
                     s, years, total)
            continue
        out.append({
            "match": r"years.{0,30}" + re.escape(s) + r"|" + re.escape(s)
                     + r".{0,30}years",
            "answer": str(years),
            "status": "pending",
        })
    return out


def seed_from_resume(dry_run: bool = False) -> dict:
    """Build bank entries from the resume + candidate profile.

    Deterministic facts go in active; LLM estimates go in pending. Returns
    {"active": [...], "pending": [...]}.
    """
    import json as _json
    from pathlib import Path

    from .apply_questions import write_bank_entries
    from .config import settings

    resume_text = ""
    resume_path = Path(settings.base_resume_path)
    if resume_path.exists():
        resume_text = resume_path.read_text(encoding="utf-8", errors="ignore")

    skills: List[str] = []
    profile_path = Path(settings.candidate_profile_path)
    if profile_path.exists():
        try:
            skills = _json.loads(
                profile_path.read_text(encoding="utf-8")).get("skills", [])
        except Exception as e:  # noqa: BLE001
            log.warning("Could not read candidate profile: %s", e)

    built = deterministic_entries(resume_text, skills)
    active = [e for e in built if e.get("status", "active") == "active"]
    staged = [e for e in built if e.get("status") == "pending"]

    def _ask(prompt: str) -> str:
        from .ollama_client import _generate
        return _generate(prompt, temperature=0.0)

    inferred = inferred_entries(resume_text, skills, ask=_ask)

    if not dry_run:
        # All resume-parsed entries share source="resume" so a re-seed replaces
        # them as one group; write_bank_entries preserves each entry's own status,
        # so the staged heuristics stay pending and load_answer_bank hides them.
        write_bank_entries(built, source="resume")
        write_bank_entries(inferred, source="resume-ai")

    return {"active": active, "pending": staged + inferred}
