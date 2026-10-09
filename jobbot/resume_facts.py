# jobbot/resume_facts.py
"""Structured, human-verified resume facts.

data/candidate_profile.json holds skills and an embedding: it is built for
matching, not for filling forms. Workday's Work Experience section needs
employer, title and dates as discrete values, and the bot previously had none,
so it created a repeating block it could not fill and the form failed
validation.

Employment dates and employers are attestable claims on a legal document.
Nothing here is ever LLM-authored: it is extracted, then confirmed by a human.
"""
from __future__ import annotations

import json
import logging
import re
from dataclasses import asdict, dataclass, field
from pathlib import Path
from typing import Dict, List, Optional, Tuple

from .config import settings

log = logging.getLogger("jobbot.resume_facts")

_YYYY_MM = re.compile(r"^\d{4}-(0[1-9]|1[0-2])\Z")


@dataclass
class Employment:
    employer: str
    title: str
    location: str = ""
    start: str = ""          # YYYY-MM
    end: str = ""            # YYYY-MM, or "" when current
    current: bool = False
    bullets: List[str] = field(default_factory=list)


@dataclass
class Education:
    school: str
    degree: str
    field: str = ""
    location: str = ""
    start: str = ""
    end: str = ""


@dataclass
class ResumeFacts:
    employment: List[Employment] = field(default_factory=list)
    education: List[Education] = field(default_factory=list)
    identity: Dict[str, str] = field(default_factory=dict)
    reviewed: bool = False

    def is_reviewed(self) -> bool:
        return bool(self.reviewed)


def _path() -> Path:
    return Path(getattr(settings, "resume_facts_path", "data/resume_facts.json"))


def _validate(facts: ResumeFacts) -> None:
    for e in facts.employment:
        for label, val in (("start", e.start), ("end", e.end)):
            if val and not _YYYY_MM.match(val):
                raise ValueError(
                    f"{e.employer}: {label} must be YYYY-MM, got {val!r}")
        if e.current and e.end:
            raise ValueError(
                f"{e.employer}: a current role must not have an end date")
    for ed in facts.education:
        for label, val in (("start", ed.start), ("end", ed.end)):
            if val and not _YYYY_MM.match(val):
                raise ValueError(
                    f"{ed.school}: {label} must be YYYY-MM, got {val!r}")


def save(facts: ResumeFacts) -> None:
    _validate(facts)
    p = _path()
    p.parent.mkdir(parents=True, exist_ok=True)
    p.write_text(json.dumps(asdict(facts), indent=2), encoding="utf-8")


def load() -> ResumeFacts:
    p = _path()
    if not p.exists():
        ex = p.parent / (p.stem + ".example" + p.suffix)
        if ex.exists():
            p = ex
        else:
            # No file at all is a legitimate first-run state, not an error.
            return ResumeFacts()
    try:
        raw = json.loads(p.read_text(encoding="utf-8"))
    except (ValueError, OSError) as e:
        # The file EXISTS but didn't parse: that's corruption, not "this
        # person has no employment history." A later task fills Workday's
        # repeating employment section from this store and cannot tell those
        # two states apart from an empty ResumeFacts() alone, so raise
        # instead of silently returning empty.
        log.error("resume_facts at %s exists but is unreadable (%s); refusing "
                  "to treat a broken store as empty facts", p, e)
        raise

    if not isinstance(raw, dict):
        # Valid JSON that isn't an object (null, [], "x") parses fine and would
        # otherwise reach raw.get() and die with a bare AttributeError that
        # names neither this module nor the file. Route it through the same
        # deliberate, logged corruption path as a parse failure -- a caller
        # catching (ValueError, OSError) around load(), which is this module's
        # own convention, should not be blindsided by a third exception type.
        log.error("resume_facts at %s parsed to %s, not an object; refusing to "
                  "treat a broken store as empty facts", p, type(raw).__name__)
        raise ValueError(
            f"resume_facts at {p} is not a JSON object (got {type(raw).__name__})")

    reviewed_raw = raw.get("reviewed", False)
    if isinstance(reviewed_raw, bool):
        reviewed = reviewed_raw
    else:
        # bool("false") is True in Python. A hand-edited file (currently the
        # only way a human marks facts reviewed) with the JSON string "false"
        # must not flip this trust gate to the opposite of what was intended.
        log.warning("resume_facts 'reviewed' field is not a JSON boolean "
                    "(%r); treating as False", reviewed_raw)
        reviewed = False

    return ResumeFacts(
        employment=[Employment(**e) for e in raw.get("employment", [])],
        education=[Education(**e) for e in raw.get("education", [])],
        identity=raw.get("identity", {}),
        reviewed=reviewed)


def is_reviewed() -> bool:
    return load().is_reviewed()


# The real resume uses typographic punctuation, not ASCII stand-ins: an em dash
# between title and organisation, a middle dot before the date range, and an en
# dash inside the range (sometimes unspaced). Parsing is written against the
# actual document -- an earlier version targeted an invented shape and extracted
# zero rows from every real file.
_EM_DASH = "—"
_MIDDLE_DOT = "·"

# "1/2021 - Present", "8/2018 - 11/2020", "1/2021-Present" (any dash, optional
# spaces). A BARE year deliberately does not match; see _parse_range.
_RANGE_RE = re.compile(
    r"(?P<m1>\d{1,2})/(?P<y1>\d{4})\s*[-–—]\s*"
    r"(?:(?P<present>Present)|(?P<m2>\d{1,2})/(?P<y2>\d{4}))", re.I)

_STATE_RE = re.compile(r"^[A-Z]{2}$")


def _to_yyyy_mm(month: str, year: str) -> str:
    try:
        m = int(month)
    except (TypeError, ValueError):
        return ""
    return f"{year}-{m:02d}" if 1 <= m <= 12 else ""


def _parse_range(text: str) -> Tuple[str, str, bool]:
    """Return (start, end, current) for a date range found anywhere in `text`.

    A bare year with no month yields empty dates rather than a guessed month.
    `save()` would reject a bare year anyway, but the real reason is that an
    invented start month is a false statement about the user's history on a job
    application, while a blank one is a gap they fill in during review.
    """
    m = _RANGE_RE.search(text or "")
    if not m:
        return "", "", False
    start = _to_yyyy_mm(m.group("m1"), m.group("y1"))
    if m.group("present"):
        return start, "", True
    return start, _to_yyyy_mm(m.group("m2"), m.group("y2")), False


def _split_location(chain: str) -> Tuple[str, str]:
    """Split a comma chain into (everything else, trailing "City, ST").

    Which link of "Some Lab, Some Department, Some University" belongs in an ATS
    "Employer" box is a judgment call, so the whole chain is kept rather than
    silently picking one. `jobbot facts review` is where the human decides.
    """
    parts = [p.strip() for p in (chain or "").split(",") if p.strip()]
    if len(parts) >= 2 and _STATE_RE.match(parts[-1]):
        return ", ".join(parts[:-2]), f"{parts[-2]}, {parts[-1]}"
    return ", ".join(parts), ""


def _section_lines(lines: List[str], keyword: str) -> List[str]:
    """Lines under the first `## ` heading containing `keyword`, up to the next."""
    out: List[str] = []
    inside = False
    for line in lines:
        if line.startswith("## "):
            if inside:
                break
            inside = keyword.lower() in line[3:].lower()
            continue
        if inside:
            out.append(line)
    return out


def extract_from_markdown(md: str) -> ResumeFacts:
    """Best-effort structured extraction from a markdown resume.

    Deliberately conservative and never LLM-backed: if a line does not match
    the expected shape, it is dropped rather than guessed at, because a wrong
    employment date on a real application is worse than an absent one.
    `reviewed` is always False -- a human confirms every date before these
    values reach a form.
    """
    employment: List[Employment] = []
    education: List[Education] = []
    identity: Dict[str, str] = {}

    lines = [l.rstrip() for l in (md or "").splitlines()]

    for l in lines[:8]:
        m = re.search(r"[\w.+-]+@[\w-]+\.[\w.]+", l)
        if m and "email" not in identity:
            identity["email"] = m.group(0)
        m = re.search(r"\(?\d{3}\)?[\s.-]?\d{3}[\s.-]?\d{4}", l)
        if m and "phone" not in identity:
            identity["phone"] = m.group(0)
        m = re.search(r"linkedin\.com/in/[\w-]+", l, re.I)
        if m and "linkedin" not in identity:
            identity["linkedin"] = m.group(0)
        if l.startswith("# ") and "name" not in identity:
            identity["name"] = l[2:].strip()

    # Experience. The real heading reads "Research & Professional Experience",
    # so match on the word rather than the exact heading text.
    exp_lines = _section_lines(lines, "experience")
    current_role: Optional[Employment] = None
    for raw in exp_lines:
        line = raw.strip()
        if line.startswith("### "):
            current_role = None
            head = line[4:]
            if _EM_DASH not in head:
                continue  # unmatched shape: drop rather than guess
            title, _, rest = head.partition(_EM_DASH)
            chain, _, dates = rest.partition(_MIDDLE_DOT)
            if not dates.strip():
                continue
            employer, location = _split_location(chain)
            start, end, is_current = _parse_range(dates)
            current_role = Employment(
                employer=employer, title=title.strip(), location=location,
                start=start, end=end, current=is_current, bullets=[])
            employment.append(current_role)
        elif current_role is not None and line.startswith("- "):
            current_role.bullets.append(line[2:].strip())

    # Education entries are bullets, not headings, and their date range is
    # embedded mid-sentence with no spaces around the dash.
    for raw in _section_lines(lines, "education"):
        line = raw.strip()
        if not line.startswith("- **"):
            continue
        body = line[2:]
        degree = ""
        m = re.match(r"\*\*(?P<degree>.+?)\*\*", body)
        if m:
            degree = m.group("degree").strip()
            body = body[m.end():]
        if _EM_DASH not in body:
            continue
        _, _, tail = body.partition(_EM_DASH)
        start, end, _cur = _parse_range(tail)
        # Everything before the date range, first sentence only, is the school
        # and its location; the rest is honours, advisors and GPA prose.
        pre = tail[:_RANGE_RE.search(tail).start()] if _RANGE_RE.search(tail) else tail
        school_part = pre.split(".")[0].strip().rstrip(",").strip()
        school, location = _split_location(school_part)
        education.append(Education(
            school=school, degree=degree, field="", location=location,
            start=start, end=end))

    return ResumeFacts(employment=employment, education=education,
                       identity=identity, reviewed=False)
