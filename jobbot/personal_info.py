"""Personal-info sheet -> auto-answers.

The user maintains a plain-text sheet (``My Information.txt`` by default — an
exported filled application, label line then value line) with the details the
``.env`` identity fields don't cover: full address, EEO self-identification,
"how did you hear about us", etc. This module parses that sheet into canonical
fields and, from those, into answer-bank entries so the existing answer pipeline
(``apply_questions.answer_questions``) fills them on every ATS — Workday included
— instead of dropping them to ``needs_user``.

The curated ``data/answer_bank.json`` is still consulted first, so an explicit
entry there always overrides the sheet.
"""
from __future__ import annotations

import logging
import re
from pathlib import Path
from typing import Dict, List

from .config import settings

log = logging.getLogger("jobbot.personal_info")

# "Springfield, IL 62701"  /  "Springfield, IL 62701-1234"
_CITY_STATE_ZIP = re.compile(
    r"^(.+?),\s*([A-Za-z]{2})\.?\s+(\d{5}(?:-\d{4})?)\s*$")
_COUNTRY_RE = re.compile(r"united states|u\.?s\.?a?\.?$|america|canada", re.I)


def _read_lines(path: str) -> List[str]:
    return [ln.strip() for ln in
            Path(path).read_text(encoding="utf-8", errors="ignore").splitlines()]


def _value_after(lines: List[str], *labels: str) -> str:
    """First non-empty line following the first line that equals any of `labels`
    (case-insensitive)."""
    wanted = {l.lower() for l in labels}
    for i, ln in enumerate(lines):
        if ln.lower() in wanted:
            for nxt in lines[i + 1:]:
                if nxt.strip():
                    return nxt.strip()
    return ""


def _address_block(lines: List[str]) -> Dict[str, str]:
    """Parse the identity address block: a label ('Address') followed by a
    street line, a 'City, ST ZIP' line, and an optional country line."""
    for i, ln in enumerate(lines):
        if ln.lower() in ("address", "mailing address", "home address"):
            vals = [v for v in lines[i + 1:i + 6] if v.strip()]
            if not vals:
                continue
            out = {"address_line1": vals[0], "city": "", "state": "",
                   "postal_code": "", "country": ""}
            for v in vals[1:]:
                m = _CITY_STATE_ZIP.match(v)
                if m:
                    out["city"], out["state"], out["postal_code"] = (
                        m.group(1).strip(), m.group(2).upper(), m.group(3))
                elif _COUNTRY_RE.search(v):
                    out["country"] = v
            return out
    return {}


def load_personal_info(path: str | None = None) -> Dict[str, str]:
    """Parse the personal-info sheet into canonical fields. Returns {} if the
    file is missing/unreadable (fail-open — auto-apply just falls back to
    .env/config/AI as before)."""
    p = path or getattr(settings, "personal_info_path", "My Information.txt")
    try:
        if not Path(p).exists():
            return {}
        lines = _read_lines(p)
    except Exception as e:  # noqa: BLE001
        log.info("Could not read personal-info sheet %s: %s", p, e)
        return {}

    info: Dict[str, str] = {}

    full = _value_after(lines, "Legal Name", "Full Name", "Name")
    if full:
        info["full_name"] = full
        first, _, last = full.partition(" ")
        info["first_name"] = first
        info["last_name"] = last.strip() or first

    email = _value_after(lines, "Email Address", "Email")
    if email:
        info["email"] = email

    phone = _value_after(lines, "Phone", "Phone Number", "Mobile")
    if phone:
        info["phone"] = phone

    info.update(_address_block(lines))

    how = _value_after(lines, "How Did You Hear About Us?", "How did you hear about us")
    if how:
        info["how_heard"] = how

    gender = _value_after(lines, "Gender")
    if gender:
        info["gender"] = gender

    race = _value_after(lines, "Race/Ethnicity", "Race", "Ethnicity")
    if race:
        # Drop a trailing parenthetical ("White (United States of America)") so
        # the value snaps onto the form's real option ("White (Not Hispanic...)").
        info["race"] = re.sub(r"\s*\(.*$", "", race).strip() or race

    vet = _value_after(lines, "Veteran Status")
    if vet:
        info["veteran_status"] = vet

    disability = next(
        (ln for ln in lines
         if re.match(r"^(yes, i have a disability|no, i (do not|don'?t) have a disability)",
                     ln, re.I)),
        "")
    if disability:
        info["disability_status"] = disability

    return info


# Canonical field -> regex that matches the corresponding form question.
_FIELD_QUESTION_RE = {
    "address_line1": r"address line ?1|street address|^address$|^address line$",
    "city": r"^city$|current city|city of residence",
    "state": r"^state$|state ?/ ?province|state/province|^province$",
    "postal_code": r"postal code|zip code|^zip$|post ?code|postcode",
    "country": r"^country$|country of residence",
    "how_heard": r"how did you (hear|find|learn)|hear about (us|this|the)|referral source",
    # NOT "gender" alone: that also matches "I identify as transgender", a
    # separate Yes/No question that was inheriting the user's gender answer.
    "gender": r"(?<!trans)\bgender\b",
    "race": r"\brace\b|ethnic",
    "veteran_status": r"veteran",
    "disability_status": r"disab",
}


# Form-application sections that carry no resume value (and would only add
# noise / bias to tailoring): drop the sheet from the first of these onward.
_RESUME_STOP_HEADERS = {
    "resume/cv/cover letter", "websites", "application questions",
    "voluntary disclosures", "us eeo and veteran self-identification",
    "self identify", "terms and conditions",
}


def resume_context(path: str | None = None) -> str:
    """A clean, resume-relevant slice of the personal-info sheet — contact, work
    experience, education, languages — with application-form noise (EEO, self-ID,
    screening questions, uploaded-file listings) stripped out.

    Returned as authoritative candidate facts to ground resume/cover-letter
    tailoring so it keeps names, dates, employers and degrees accurate and
    doesn't omit or invent detail. Returns '' if the sheet is unavailable."""
    p = path or getattr(settings, "personal_info_path", "My Information.txt")
    try:
        if not Path(p).exists():
            return ""
        lines = _read_lines(p)
    except Exception as e:  # noqa: BLE001
        log.info("Could not read personal-info sheet %s: %s", p, e)
        return ""

    # Cut everything from the first form-noise section header onward.
    end = len(lines)
    for i, ln in enumerate(lines):
        if ln.lower() in _RESUME_STOP_HEADERS:
            end = i
            break

    # Drop stray screening/consent questions (and their following answer line)
    # that sit in the contact block — tailoring shouldn't see "how did you hear".
    out: List[str] = []
    skip_value = False
    for ln in lines[:end]:
        if skip_value and ln.strip():
            skip_value = False
            continue
        if ln.rstrip().endswith("?") or re.search(r"\b(previous|former)\b.*\bemploy", ln, re.I):
            skip_value = True
            continue
        out.append(ln)

    text = re.sub(r"\n{3,}", "\n\n", "\n".join(out)).strip()
    return text


def personal_info_bank_entries(path: str | None = None) -> List[dict]:
    """Answer-bank-shaped entries ({'match': regex, 'answer': value}) built from
    the sheet, for the fields actually present. Consumed by
    ``apply_questions.answer_questions`` after the curated answer bank, so
    ``data/answer_bank.json`` still wins on overlapping questions."""
    info = load_personal_info(path)
    entries: List[dict] = []
    for field, pattern in _FIELD_QUESTION_RE.items():
        val = (info.get(field) or "").strip()
        if val:
            entries.append({"match": pattern, "answer": val})
    return entries
