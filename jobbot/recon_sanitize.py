# jobbot/recon_sanitize.py
"""The identity list, and redaction of it out of a captured Workday page.

A recon capture is taken from a SIGNED-IN candidate profile, so the raw HTML
carries the applicant's name, emails, phone, address, every prior employer and
every school - in value attributes, aria-labels, placeholders and body text.
These fixtures are COMMITTED, so redaction runs before anything is written and
tests/test_recon_pii.py fails the suite if a token survives.

Redaction is a denylist by nature: it can only replace tokens it knows. The
assertion test is what makes that safe, and it walks THIS SAME list, so the
sanitizer and the check cannot drift apart. Anything added here is
automatically both redacted and asserted on.
"""
from __future__ import annotations

import logging
import re
from typing import List, Optional, Tuple

from . import resume_facts
from .config import settings

log = logging.getLogger("jobbot.recon_sanitize")

# Placeholder identity, per the convention set in commit 8bc095b. Nothing here
# is the owner's real name, employer or city.
_NAME = "Alex Rivera"
_EMAIL = "alex.rivera@example.com"
_PHONE = "555-010-0000"
_LOCATION = "Springfield, ST"


def _phone_variants(raw: str) -> List[str]:
    """Every punctuation form a 10-digit number is written in on a form."""
    digits = re.sub(r"\D", "", raw or "")
    if len(digits) < 10:
        return []
    d = digits[-10:]
    return [
        digits,
        d,
        f"{d[:3]}-{d[3:6]}-{d[6:]}",
        f"({d[:3]}) {d[3:6]}-{d[6:]}",
        f"({d[:3]}){d[3:6]}-{d[6:]}",
        f"{d[:3]}.{d[3:6]}.{d[6:]}",
        f"{d[:3]} {d[3:6]} {d[6:]}",
        f"+1{d}",
    ]


# ONE pattern, used both to harvest emails out of the profile and to sweep up
# any that the identity list never knew about. Two copies would drift, and the
# copy that drifted would be the one guarding the fixtures.
_EMAIL_RE = re.compile(r"[A-Za-z0-9._%+-]+@[A-Za-z0-9.-]+\.[A-Za-z]{2,}")


def _emails_in(text: str) -> List[str]:
    return _EMAIL_RE.findall(text or "")


def _profile_text() -> str:
    """The freeform profile markdown, or "" when it is absent or unreadable.

    Fail-open on purpose: a missing profile must not stop a capture being
    sanitized for the tokens that ARE available.
    """
    try:
        from pathlib import Path
        return Path(settings.profile_path).read_text(encoding="utf-8", errors="ignore")
    except Exception as e:  # noqa: BLE001
        log.info("profile unreadable for identity list: %s", e)
        return ""


# Components too generic to redact on their own. Replacing a bare "University"
# or "Laboratory" would rewrite unrelated text across the whole capture, and
# over-redaction costs us the DOM facts the capture exists to collect.
_GENERIC_COMPONENTS = {
    "university", "college", "school", "institute", "lab", "labs", "laboratory",
    "laboratories", "department", "dept", "center", "centre", "hospital",
    "office", "division", "group", "team", "inc", "llc", "ltd", "corp",
    "corporation", "company", "the", "of", "at",
}


def _components(raw: str) -> List[str]:
    """Split a composite employer/school string into its redactable parts.

    """
    out: List[str] = []
    for part in re.split(r"[,()/|]|\s+-\s+", raw or ""):
        part = part.strip(" \t\r\n.;:")
        if len(part) < 6:
            continue
        if part.lower() in _GENERIC_COMPONENTS:
            continue
        # A component that is nothing but generic words carries no identity:
        # "Department of Chemistry" does, "The Office of" does not.
        if all(w.lower().strip(".") in _GENERIC_COMPONENTS for w in part.split()):
            continue
        out.append(part)
    return out


def identity_tokens() -> List[Tuple[str, str]]:
    """(real_token, placeholder) pairs, LONGEST real token first.

    Order is load-bearing. Replacing "Dana" before "Dana Whitfield" would leave
    "Alex Whitfield" in the page - a surname surviving redaction because a
    shorter token consumed its prefix.
    """
    pairs: List[Tuple[str, str]] = []

    def add(real: str, placeholder: str) -> None:
        real = (real or "").strip()
        # Two characters is the floor: a one-character "token" would rewrite
        # unrelated text across the whole document.
        if len(real) > 2:
            pairs.append((real, placeholder))

    name = (getattr(settings, "applicant_name", "") or "").strip()
    if name and name.lower() != "applicant":
        add(name, _NAME)
        parts = name.split()
        if len(parts) >= 2:
            add(parts[0], _NAME.split()[0])
            add(parts[-1], _NAME.split()[-1])

    profile = _profile_text()
    seen_emails = set()
    for email in [getattr(settings, "applicant_email", "")] + _emails_in(profile):
        email = (email or "").strip()
        if email and email.lower() not in seen_emails:
            seen_emails.add(email.lower())
            add(email, _EMAIL)

    for phone in [getattr(settings, "applicant_phone", "")]:
        for form in _phone_variants(phone):
            add(form, _PHONE)

    add(getattr(settings, "applicant_location", ""), _LOCATION)
    add(getattr(settings, "applicant_linkedin", ""), "linkedin.com/in/alex-rivera")
    add(getattr(settings, "applicant_github", ""), "github.com/alexrivera")

    try:
        facts = resume_facts.load()
    except Exception as e:  # noqa: BLE001
        log.info("resume facts unavailable for identity list: %s", e)
        facts = None

    if facts is not None:
        for i, job in enumerate(getattr(facts, "employment", []) or []):
            add(getattr(job, "employer", ""), f"Employer {i + 1}")
            for part in _components(getattr(job, "employer", "")):
                add(part, f"Employer {i + 1}")
            add(getattr(job, "location", ""), _LOCATION)
        for i, ed in enumerate(getattr(facts, "education", []) or []):
            add(getattr(ed, "school", ""), f"School {i + 1}")
            for part in _components(getattr(ed, "school", "")):
                add(part, f"School {i + 1}")
            add(getattr(ed, "location", ""), _LOCATION)

    # Longest first. Dedupe on the real token, keeping the first placeholder.
    pairs.sort(key=lambda p: len(p[0]), reverse=True)
    out: List[Tuple[str, str]] = []
    claimed = set()
    for real, placeholder in pairs:
        if real.lower() not in claimed:
            claimed.add(real.lower())
            out.append((real, placeholder))
    return out


# A government ID is STRUCTURAL, not a token: it appears in no profile file, so
# identity_tokens() can never learn it and the denylist can never match it.
# Form capture sanitization masks the FIELD it is typed into; this catches the
# other half, an ID that a Review step RENDERS as body text where there is no
# field to key on.
#
# Only the dashed form. A bare nine-digit run would rewrite requisition ids,
# postal+4 and phone digits, and over-redaction costs us the DOM facts the
# capture exists to collect.
_GOV_ID_RE = re.compile(r"\b\d{3}-\d{2}-\d{4}\b")
_GOV_ID_PLACEHOLDER = "000-00-0000"


def redact(html: str, pairs: Optional[List[Tuple[str, str]]] = None) -> str:
    """Replace every real identity token with its placeholder, case-insensitively.

    Structural patterns run unconditionally, including when `pairs` is passed
    explicitly: a caller narrowing the token list is choosing which NAMES to
    replace, never opting out of government-ID masking.
    """
    if not html:
        return html or ""
    for real, placeholder in (pairs if pairs is not None else identity_tokens()):
        html = re.sub(re.escape(real), placeholder, html, flags=re.IGNORECASE)
    # Identity-list redaction is NAME-based, so it only reaches an email that
    # happens to contain a name it already knows. A Workday capture puts the
    # signed-in account's address in accountSettingsButton's text, and a test
    # account's address need not contain the owner's name at all -- measured:
    # "testuser@example.com" survived the token pass completely intact,
    # and the fixture gate could not flag it either, because it is not a
    # token. Sweeping by SHAPE retires that whole class instead of adding one
    # more name to enumerate. Runs after the token pass so a name inside an
    # address is still destroyed first; substituting _EMAIL for itself on the
    # second pass is a no-op.
    html = _EMAIL_RE.sub(_EMAIL, html)
    return _GOV_ID_RE.sub(_GOV_ID_PLACEHOLDER, html)


# The marker DOM field capture's `trunc()` appends when it cuts a
# string, and the shortest prefix worth replacing. Below three characters a
# "fragment" is noise that would rewrite unrelated text.
_TRUNC_MARK = "...[truncated]"
_MIN_FRAGMENT = 3


def redact_truncated_tail(s: str, pairs: List[Tuple[str, str]]) -> str:
    """Replace a token PREFIX left dangling by truncation.

    The capture truncates at 120 / 200 / 6000 / 8000 / 12000 characters. When
    the cut lands mid-token it leaves a prefix - `Acme ` out of a
    27-character employer - that `redact()`, which matches the full literal,
    never touches. tests/test_recon_pii.py greps for full tokens too, so it
    passes vacuously on the fragment.

    Only the TAIL can carry one, because truncation only ever cuts the end.
    `pairs` is longest-first and the search stops at the first hit, so at most
    one token is applied.
    """
    if not s or not s.endswith(_TRUNC_MARK):
        return s
    body = s[: -len(_TRUNC_MARK)]
    low = body.lower()
    for real, placeholder in pairs:
        rl = real.lower()
        for k in range(min(len(real) - 1, len(body)), _MIN_FRAGMENT - 1, -1):
            if low.endswith(rl[:k]):
                return body[: len(body) - k] + placeholder + _TRUNC_MARK
    return body + _TRUNC_MARK


# Fields whose VALUE is identity no matter what it says. Matched against a
# control's id / name / data-automation-id.
_SENSITIVE_FIELD_RE = re.compile(
    r"address|addressline|postalcode|zipcode|\bzip\b|street"
    r"|birth|dateofbirth|nationalid|national-id|ssn|socialsecurity|taxid",
    re.IGNORECASE,
)
_SENSITIVE_PLACEHOLDER = "[REDACTED field value]"


def sensitive_field_values(data) -> List[str]:
    """Harvest identity values OUT OF the capture, to redact them back into it.

    A street address cannot be learned from a denylist: "205 Somewhere Court"
    appears in no profile file, so identity_tokens() will never contain it, and
    the first live Employer My Information capture wrote it in clear along with
    the city. Found the same way an SSN would be.

    Masking the field at capture time would hide the value but leave the
    rendered copy in `page_text`, which is where a Review step shows the whole
    address as prose. So instead the capture TELLS US what the address is, and
    we redact that string out of the entire document - field, attribute and
    body text alike.
    """
    out: List[str] = []

    def consider(ident: str, value) -> None:
        if not ident or not isinstance(value, str):
            return
        v = value.strip()
        # Two characters is the floor for the same reason as identity_tokens:
        # a shorter "value" would rewrite unrelated text.
        if len(v) > 2 and v != _SENSITIVE_PLACEHOLDER and _SENSITIVE_FIELD_RE.search(ident):
            out.append(v)

    for entry in (data or {}).get("automation_ids", []) or []:
        consider(entry.get("aid", ""), entry.get("value"))
    for entry in (data or {}).get("controls", []) or []:
        attrs = entry.get("attrs", {}) or {}
        ident = " ".join(str(attrs.get(k, "")) for k in
                         ("id", "name", "data-automation-id", "autocomplete"))
        consider(ident, entry.get("value"))
        consider(ident, attrs.get("value"))

    # Longest first, so a city inside a full address line is not consumed by
    # the shorter token. Same ordering rule as identity_tokens().
    return sorted(set(out), key=len, reverse=True)


def with_value_strings(values, pairs: List[Tuple[str, str]]) -> List[Tuple[str, str]]:
    """`pairs` plus harvested values, re-sorted longest-first.

    The sort is the load-bearing part. A city is an identity token AND a
    substring of the address line harvested from the form, so applying the
    short token first turns "12 Somewhere Ct, Albany NY" into
    "12 Somewhere Ct, Springfield NY" - after which the full address literal
    matches nothing and the street survives. Longest-first is the same rule
    identity_tokens() ends on, for the same reason.
    """
    combined = list(pairs) + [(v, _SENSITIVE_PLACEHOLDER) for v in values]
    # Stable, so equal-length tokens keep the caller's ordering.
    combined.sort(key=lambda p: len(p[0]), reverse=True)
    return combined


def with_sensitive_values(data, pairs: List[Tuple[str, str]]) -> List[Tuple[str, str]]:
    """`pairs` plus the identity values this particular capture revealed."""
    return with_value_strings(sensitive_field_values(data), pairs)


def redact_tree(obj, pairs: Optional[List[Tuple[str, str]]] = None):
    """Redact every string leaf of a nested dict/list structure.

    Applied BEFORE json.dumps, not after. Redacting the serialized string
    would miss any token JSON had escaped: `ensure_ascii=True` turns a
    non-ASCII character into \\uXXXX, and a quote or backslash inside a token
    becomes \\" or \\\\, none of which match a literal pattern. Walking the
    leaves is immune to every escape form and cannot corrupt the document.

    Measured against the current identity list, 0 of 18 tokens are altered by
    json.dumps, so redacting the serialized string would work TODAY. It breaks
    the first time someone puts an employer name with a curly apostrophe in
    data/profile.md. Leaf redaction has no such failure mode.
    """
    # Resolve ONCE. identity_tokens() reads files; calling it per leaf would
    # re-read them thousands of times per capture.
    if pairs is None:
        pairs = identity_tokens()
    return _redact_tree(obj, pairs)


def _redact_tree(obj, pairs: List[Tuple[str, str]]):
    if isinstance(obj, str):
        return redact_truncated_tail(redact(obj, pairs), pairs)
    if isinstance(obj, dict):
        # Keys are DOM attribute names, not user data - leave them.
        return {k: _redact_tree(v, pairs) for k, v in obj.items()}
    if isinstance(obj, list):
        return [_redact_tree(v, pairs) for v in obj]
    if isinstance(obj, tuple):
        return tuple(_redact_tree(v, pairs) for v in obj)
    # int, float, bool, None and anything else pass through untouched.
    return obj


def identity_list_health(
    pairs: Optional[List[Tuple[str, str]]] = None,
) -> Tuple[bool, str]:
    """(usable, human-readable reason). Names counts, never tokens.

    `redact()` with an empty or near-empty list returns its input UNCHANGED and
    every caller still reports success. On a fresh clone - where `.env`,
    `data/profile.md` and `data/resume_facts.json` are all gitignored and so
    absent - the list collapses from 18 pairs to 1 and redaction silently
    becomes the identity function.

    The three requirements mirror the anti-vacuous check in
    tests/test_recon_pii.py so the writer and the gate cannot drift.
    """
    if pairs is None:
        pairs = identity_tokens()

    placeholders = [p for _, p in pairs]
    missing = []
    if not any(p == _NAME for p in placeholders):
        missing.append("name")
    if not any(p == _EMAIL for p in placeholders):
        missing.append("email")
    if not any(p.startswith("Employer ") or p.startswith("School ")
               for p in placeholders):
        missing.append("employer/school")

    if missing:
        return False, (
            f"identity list is unusable: no {', '.join(missing)} token "
            f"({len(pairs)} token(s) total). Redaction would be a no-op for "
            f"that category. Check .env, data/profile.md and "
            f"data/resume_facts.json are present and readable."
        )
    return True, f"identity list usable: {len(pairs)} token(s)"
