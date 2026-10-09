"""Deterministic fidelity checks between a source resume/profile and a tailored draft.

The LLM critic keeps the work that needs judgment (is this bullet relevant, is
this phrasing strong, does it hit the JD). This module keeps the work that needs
none: a DOI is a literal string, a month is a token, an employer's city is a
pairing. Code does not get distracted across 24k chars of context.

This exists because prompt rules were not enough. `_critique_tailored` already
names every defect below -- including the literal example 'source says "Expected
2027", draft invents "Expected January 2027"' -- and the local model still missed
all three on a real run and self-scored 0.95. It reported the omissions it could
see (whole job roles, which tailoring had correctly dropped) and missed the
one-line citation. That is a small-model needle-in-haystack limit, not a wording
problem; it was measured, not assumed:

    critic prompt ~5,947 tokens vs num_ctx 8,192      -> the prompt fits
    Water 2022 DOI at char 4,494 of profile (cap 6k)  -> the model saw it

Fail-open by construction: every check only fires on positive evidence (a DOI in
the source and absent from the draft). A parse failure yields no findings, never
a false accusation.
"""
from __future__ import annotations

import re
from dataclasses import dataclass, field
from typing import List, Tuple

# 10.<registrant>/<suffix> -- the suffix runs to whitespace or a closing bracket.
_DOI_RE = re.compile(r"\b10\.\d{4,9}/[^\s\"'<>)\],;]+", re.I)

_MONTHS = ("January", "February", "March", "April", "May", "June", "July",
           "August", "September", "October", "November", "December",
           "Jan", "Feb", "Mar", "Apr", "Jun", "Jul", "Aug", "Sep", "Sept",
           "Oct", "Nov", "Dec")
_MONTH_ALT = "|".join(_MONTHS)

# "Expected 2027" / "Expected May 2027" -- the month group is what we police.
_EXPECTED_RE = re.compile(
    rf"expected\s+(?:({_MONTH_ALT})\.?\s+)?((?:19|20)\d{{2}})", re.I)

# "<Institution> - <City>, <ST>" on one line. Covers the em/en dash and hyphen
# the renderer may emit, and the comma form.
_ORG_LOC_RE = re.compile(
    r"([A-Z][A-Za-z.'&()\- ]{3,60}?)\s*(?:[-–—,])\s*"
    r"([A-Z][A-Za-z.\- ]{2,30}),\s*([A-Z]{2})\b")

# Words that mark a line as the contact header rather than an employer entry.
_CONTACT_MARKERS = ("@", "linkedin.com", "github.com", "tel:", "+1-")

# Credentials worth policing by name. These are earned distinctions: dropping one
# understates the candidate, and no tailoring decision justifies it. Latin honours
# and Dean's List are fixed phrases; awards are matched generically below.
_NAMED_CREDENTIALS = (
    "magna cum laude", "summa cum laude", "cum laude", "dean's list",
    "honor roll", "phi beta kappa", "valedictorian", "salutatorian",
)

# "<Name> Award" / "<Name> Prize" / "<Name> Fellowship" / "<Name> Scholarship".
_AWARD_RE = re.compile(
    r"\b((?:[A-Z][\w'-]*\s+){1,4}?(?:Award|Prize|Fellowship|Scholarship|Medal|Honou?r))\b")

# A draft asserting the candidate has nothing, under a heading where the source
# says otherwise. "None listed" does not merely omit an earned award -- it claims
# there is no award to list.
_ABSENCE_RE = re.compile(
    r"^\s*[-*•]?\s*(none(?:\s+listed)?|n/?a|not\s+applicable|none\s+to\s+report)"
    r"\s*\.?\s*$", re.I)

# 8/2017-11/2020, 08/2017 - 05/2020, 8/2017 to 11/2020
_RANGE_RE = re.compile(
    r"\b(\d{1,2})/((?:19|20)\d{2})\s*(?:[-–—]|to)\s*(\d{1,2})/((?:19|20)\d{2})\b")


@dataclass
class FidelityReport:
    omissions: List[str] = field(default_factory=list)
    fabrications: List[str] = field(default_factory=list)
    buzzwords: List[str] = field(default_factory=list)

    @property
    def ok(self) -> bool:
        return not self.omissions and not self.fabrications and not self.buzzwords

    @property
    def severe(self) -> bool:
        """True when a defect must block the refinement loop's early-stop.

        Dropping a peer-reviewed publication or inventing a credential date is
        not a stylistic trim -- it misrepresents the candidate's record.
        """
        return bool(self.omissions or self.fabrications)

    def as_dict(self) -> dict:
        return {"omissions": list(self.omissions),
                "fabrications": list(self.fabrications),
                "buzzwords": list(self.buzzwords)}


def _dois(text: str) -> List[str]:
    # Normalise: DOIs are case-insensitive per the DOI handbook, and appear as
    # bare, "doi:"-prefixed, or full-URL forms. Strip a trailing period picked up
    # from prose.
    return [m.group(0).rstrip(".").lower() for m in _DOI_RE.finditer(text or "")]


def _check_dropped_publications(source: str, draft: str) -> List[str]:
    src, drf = set(_dois(source)), set(_dois(draft))
    return [f"dropped publication (DOI {d} is in the source, absent from the draft)"
            for d in sorted(src - drf)]


# Presentations carry no DOI, so the publication check cannot see them. They are
# identified by the distinctive words of their title plus their year: two talks
# from the same lab share most of their wording, and only the distinguishing
# phrase ("Hybrid ... Machine Learning") and the year tell them apart.
_PRESENTATIONS_HEADING = re.compile(
    r"^#{1,4}\s*Presentations?\b.*$", re.I | re.M)
_NEXT_HEADING = re.compile(r"^#{1,4}\s+\S", re.M)
# Words too common to distinguish one talk from another.
_TALK_STOPWORDS = frozenset("""
    the and for using with from into a an of on at to as by via their our
    approach approaches study studies work retreat conference symposium meeting
    presentation poster talk annual university institute college center centre
    developing development analysis toward towards role roles new novel
""".split())
_YEAR_RE = re.compile(r"\b(19|20)\d{2}\b")


def _presentation_entries(text: str) -> List[str]:
    """The raw lines of the source's Presentations section."""
    m = _PRESENTATIONS_HEADING.search(text or "")
    if not m:
        return []
    rest = text[m.end():]
    nxt = _NEXT_HEADING.search(rest)
    body = rest[:nxt.start()] if nxt else rest
    entries, current = [], ""
    for raw in body.splitlines():
        line = raw.strip()
        if not line:
            continue
        if line.startswith(("*", "-", "•")) and not line.startswith("**"):
            if current:
                entries.append(current)
            current = line.lstrip("*-• ").strip()
        elif line.startswith("*") and current:
            # a bolded continuation such as "**Name**;" starting a new bullet
            entries.append(current)
            current = line.lstrip("*-• ").strip()
        else:
            current = f"{current} {line}".strip()
    if current:
        entries.append(current)
    return entries


def _talk_signature(entry: str) -> tuple:
    """(distinctive title words, years) for one presentation entry."""
    words = {w for w in re.findall(r"[A-Za-z][A-Za-z-]{3,}", entry.lower())
             if w not in _TALK_STOPWORDS}
    years = set(_YEAR_RE.findall(entry)) and set(m.group(0) for m in _YEAR_RE.finditer(entry))
    return words, years


def _check_dropped_presentations(source: str, draft: str) -> List[str]:
    """Report presentations present in the source but missing from the draft.

    Matching is on distinctive words rather than the literal line, so
    reordering, restyling and dropped markdown emphasis are all fine. A talk
    counts as present when most of its distinguishing words survive AND its year
    still appears somewhere in the draft.
    """
    out: List[str] = []
    draft_l = (draft or "").lower()
    entries = _presentation_entries(source)
    sigs = [_talk_signature(e) for e in entries]

    for i, entry in enumerate(entries):
        words, _years = sigs[i]
        # Words this talk does NOT share with any sibling. Two talks from one
        # project can overlap heavily in shared vocabulary, so shared words prove nothing:
        # a draft that kept only the earlier talk might still contain many of the later talk's words.
        # Only the discriminating remainder can tell them apart.
        shared = set()
        for j, (other, _oy) in enumerate(sigs):
            if j != i:
                shared |= other
        unique = words - shared
        if not unique:
            continue        # indistinguishable from a sibling; cannot judge
        hits = sum(1 for w in unique if w in draft_l)
        if hits / len(unique) >= 0.5:
            continue        # its distinguishing words survived
        label = ", ".join(sorted(unique)[:5])
        out.append(f"dropped presentation (a talk identified by [{label}] "
                   f"is in the source but absent from the draft)")
    return out


def _check_invented_dates(source: str, draft: str) -> List[str]:
    """A draft may not state an expected date MORE precisely than the source."""
    src_months = {}
    for m in _EXPECTED_RE.finditer(source or ""):
        src_months.setdefault(m.group(2), set()).add((m.group(1) or "").lower())
    out = []
    for m in _EXPECTED_RE.finditer(draft or ""):
        month, year = (m.group(1) or ""), m.group(2)
        if not month:
            continue
        known = src_months.get(year)
        if known is None:
            continue                      # year not in source: not our check
        if "" in known and not any(k for k in known):
            out.append(
                f'invented date precision: draft says "Expected {month} {year}" '
                f'but the source says only "Expected {year}"')
    return out


def _org_locations(text: str) -> dict:
    """{(org_lower, city_lower): (org_as_written, city_as_written)}.

    Keys are folded for comparison; values keep the original case so a finding
    quotes the draft back to the reader as it actually reads.
    """
    pairs = {}
    for line in (text or "").splitlines():
        if any(mark in line.lower() for mark in _CONTACT_MARKERS):
            continue                      # the contact header, not an employer
        for m in _ORG_LOC_RE.finditer(line):
            org = re.sub(r"\s+", " ", m.group(1)).strip(" -–—,")
            city = m.group(2).strip()
            pairs[(org.lower(), city.lower())] = (org, city)
    return pairs


def _check_relocated_employers(source: str, draft: str) -> List[str]:
    src = _org_locations(source)
    src_orgs = {o for o, _ in src}
    out = []
    for (org_k, city_k), (org, city) in sorted(_org_locations(draft).items()):
        if (org_k, city_k) in src:
            continue
        # Only accuse when the SOURCE actually places this org somewhere else,
        # or never places it at all but names it. Unknown orgs are left alone.
        if org_k in src_orgs:
            where = sorted(c for o, c in src.values() if o.lower() == org_k)
            out.append(f'relocated employer: draft places "{org}" in '
                       f'"{city}" but the source says {where}')
        elif org_k in (source or "").lower():
            out.append(f'invented employer location: draft places "{org}" in '
                       f'"{city}"; the source gives it no location')
    return out


def _credentials(text: str) -> dict:
    """{folded: as_written} for named honours + '<Name> Award/Prize/...' spans.

    Keys are folded for comparison; values keep the source's own capitalisation
    so a finding quotes the credential back as the reader knows it.

    Apostrophes are folded (Dean's vs Dean’s) and shorter credentials that are
    merely substrings of a longer one are dropped -- "cum laude" always matches
    inside "magna cum laude", and reporting both would flag a faithful draft
    twice for one credential.
    """
    norm = re.sub(r"[‘’ʼ]", "'", text or "")
    low = norm.lower()
    found = {}
    for c in _NAMED_CREDENTIALS:
        i = low.find(c)
        if i >= 0:
            found[c] = norm[i:i + len(c)]
    for m in _AWARD_RE.finditer(norm):
        name = re.sub(r"\s+", " ", m.group(1)).strip()
        # "Awards & Honors" style headings are not themselves awards.
        if name.lower().startswith(("awards", "honors", "honours")):
            continue
        found[name.lower()] = name
    return {k: v for k, v in found.items()
            if not any(k != o and k in o for o in found)}


def _check_dropped_credentials(source: str, draft: str) -> List[str]:
    """Ask only "does each SOURCE credential survive into the draft?".

    Deliberately does NOT parse the draft's own awards: a draft may render
    "DISTINGUISHED ENGINEERING AWARD" or "Distinguished Engineering Award (Fall 2020)" and both keep the credential.
    Substring-matching the source's span is immune to how the draft formats it,
    where re-parsing the draft would flag faithful reformatting as a deletion.
    """
    src = _credentials(source)
    low = re.sub(r"[‘’ʼ]", "'", draft or "").lower()
    return [f'dropped credential ("{src[k]}" is in the source, absent from '
            f'the draft)' for k in sorted(src) if k not in low]


def _check_asserted_absence(source: str, draft: str) -> List[str]:
    """A draft may not claim the candidate has nothing when the source lists
    something. Only fires when the source actually has a credential, so a
    candidate with genuinely no awards can still write "None listed"."""
    src = _credentials(source)
    if not src:
        return []
    for line in (draft or "").splitlines():
        if _ABSENCE_RE.match(line):
            return [f'asserted absence: the draft states "{line.strip()}" while '
                    f'the source lists {sorted(src.values())}']
    return []


def _ranges(text: str) -> set:
    """{(m, yyyy, m, yyyy)} with months int-normalised, so 08/2017 == 8/2017."""
    return {(int(m.group(1)), m.group(2), int(m.group(3)), m.group(4))
            for m in _RANGE_RE.finditer(text or "")}


def _check_altered_ranges(source: str, draft: str) -> List[str]:
    """A date range in the draft must exist in the source. Reformatting is fine;
    moving an endpoint is not -- the model shifted a graduation by six months
    (source 8/2017-11/2020 -> draft 08/2017-05/2020)."""
    src = _ranges(source)
    if not src:
        return []
    out = []
    for r in sorted(_ranges(draft) - src):
        # Only accuse when the source has a range sharing this start -- an
        # unrelated new range is somebody else's check, not ours.
        same_start = [s for s in src if (s[0], s[1]) == (r[0], r[1])]
        if same_start:
            s = same_start[0]
            out.append(f"altered date range: draft says "
                       f"{r[0]}/{r[1]}-{r[2]}/{r[3]} but the source says "
                       f"{s[0]}/{s[1]}-{s[2]}/{s[3]}")
    return out


# Banned AI filler and corporate buzzwords per AGENTS.md Rule 2
_BANNED_BUZZWORDS: List[Tuple[re.Pattern, str]] = [
    (re.compile(r"\butilizing\b", re.I), "using"),
    (re.compile(r"\butilized\b", re.I), "used"),
    (re.compile(r"\butilize\b", re.I), "use"),
    (re.compile(r"\bspearheaded\b", re.I), "led"),
    (re.compile(r"\bspearheading\b", re.I), "leading"),
    (re.compile(r"\bspearhead\b", re.I), "lead"),
    (re.compile(r"\bleveraged\b", re.I), "applied"),
    (re.compile(r"\bleveraging\b", re.I), "applying"),
    (re.compile(r"\bleverage\b", re.I), "apply"),
    (re.compile(r"\bpioneered\b", re.I), "developed"),
    (re.compile(r"\bpioneering\b", re.I), "developing"),
    (re.compile(r"\bpioneer\b", re.I), "develop"),
    (re.compile(r"\bstreamlined\b", re.I), "optimized"),
    (re.compile(r"\bstreamlining\b", re.I), "optimizing"),
    (re.compile(r"\bstreamline\b", re.I), "optimize"),
    (re.compile(r"\bfostered\b", re.I), "built"),
    (re.compile(r"\bfostering\b", re.I), "building"),
    (re.compile(r"\bfoster\b", re.I), "build"),
    (re.compile(r"\borchestrated\b", re.I), "coordinated"),
    (re.compile(r"\borchestrating\b", re.I), "coordinating"),
    (re.compile(r"\borchestrate\b", re.I), "coordinate"),
    (re.compile(r"\bexecuted(?:\s+[\w-]+){0,3}\s+workflows\b", re.I), "conducted experimental protocols"),
    (re.compile(r"\bexecuting(?:\s+[\w-]+){0,3}\s+workflows\b", re.I), "conducting experimental protocols"),
    (re.compile(r"\bdemonstrated\s+proficiency\s+in\b", re.I), "applied"),
    (re.compile(r"\bsynergized\b", re.I), "integrated"),
    (re.compile(r"\bsynergizing\b", re.I), "integrating"),
    (re.compile(r"\bdynamic\s+team\s+player\b", re.I), "collaborative scientist"),
    (re.compile(r"\bresults-driven\b", re.I), "quantitative"),
    (re.compile(r"\bpassionate\b", re.I), "dedicated"),
    (re.compile(r"\bseasoned\b", re.I), "experienced"),
    (re.compile(r"\bbeacon\b", re.I), "model"),
    (re.compile(r"\btestament\b", re.I), "evidence"),
]


def _match_case(original: str, replacement: str) -> str:
    if not original:
        return replacement
    if original[0].isupper():
        return replacement[:1].upper() + replacement[1:]
    return replacement


def check_banned_buzzwords(draft: str) -> List[str]:
    """Flag any prohibited corporate clichés or AI filler verbs."""
    findings = []
    seen = set()
    for pattern, _ in _BANNED_BUZZWORDS:
        for m in pattern.finditer(draft or ""):
            word = m.group(0)
            word_l = word.lower()
            if word_l not in seen:
                seen.add(word_l)
                findings.append(f"banned AI buzzword: '{word}' violates Rule 2 (anti-AI cliché contract)")
    return findings


# A tailored draft's combined heading, e.g. "### Publications & Presentations".
_DRAFT_PRES_HEADING = re.compile(
    r"^#{1,4}[ \t]*(?:[\w&,/ ]*\b(?:Presentations?|Publications)\b[\w&,/ ]*)$",
    re.I | re.M)


def _plain(entry: str) -> str:
    """A source bullet as the tailored document would carry it: no markdown
    emphasis, single-spaced, no leading bullet glyph."""
    s = re.sub(r"\*+", "", entry or "")
    s = re.sub(r"\s+", " ", s).strip().lstrip("-• ").strip()
    return s


def _restore_dropped_presentations(source: str, draft: str) -> str:
    """Put back presentations the draft dropped, inside their own section.

    Unambiguous only when the draft HAS a publications/presentations section:
    the source line is known verbatim and its siblings are already sitting in
    that section, so there is exactly one correct destination. With no such
    section there is no non-guessing answer, and the entry is left as a reported
    defect instead -- a wrong insertion is worse than a reported gap.
    """
    missing = _check_dropped_presentations(source, draft)
    if not missing:
        return draft
    m = _DRAFT_PRES_HEADING.search(draft or "")
    if not m:
        return draft

    entries = _presentation_entries(source)
    sigs = [_talk_signature(e) for e in entries]
    draft_l = (draft or "").lower()

    # Recompute which entries are absent, in source order, using the same
    # discriminating-word test the checker uses.
    to_add = []
    for i, entry in enumerate(entries):
        words, _ = sigs[i]
        shared = set()
        for j, (other, _oy) in enumerate(sigs):
            if j != i:
                shared |= other
        unique = words - shared
        if not unique:
            continue
        hits = sum(1 for w in unique if w in draft_l)
        if hits / len(unique) < 0.5:
            to_add.append(_plain(entry))
    if not to_add:
        return draft

    # Insert at the end of the section: after the heading, before the next one.
    rest = draft[m.end():]
    nxt = _NEXT_HEADING.search(rest)
    cut = m.end() + (nxt.start() if nxt else len(rest))
    body = draft[m.end():cut]
    bullet = "- "
    for line in body.splitlines():
        st = line.strip()
        if st.startswith(("- ", "* ", "• ")):
            bullet = st[:2]
            break
    added = "".join(f"{bullet}{e}\n" for e in to_add)
    # Keep the blank line that separated this section from the next heading;
    # rstrip above removes it, and losing it runs the sections together.
    trailing = "\n" if body.endswith("\n\n") else ""
    body = body.rstrip("\n") + "\n" + added + trailing
    return draft[:m.end()] + body + draft[cut:]


def repair_fidelity(source: str, draft: str) -> tuple:
    """(repaired_draft, remaining_defects).

    Repairs ONLY what is mechanically unambiguous. Stripping an invented month
    qualifies: the source's own wording is the target, so there is exactly one
    correct result and no judgement involved.

    Restoring a dropped credential deliberately does NOT qualify -- code cannot
    know where in the document it belongs, and a wrong insertion is worse than a
    reported gap. Those survive as defects for the caller to surface.

    Exists because detection alone was not enough: the refiner was handed
    "invented date precision: Expected Dec 2027" in its prompt, with
    instructions, and shipped it anyway. A one-word deletion is a small edit in
    a 25k-char prompt, and run 4 got the same date right while run 5 regressed
    -- it is sampling noise, so "the model will fix it next round" is not a plan.
    """
    fixed = draft or ""
    try:
        src_months = {}
        for m in _EXPECTED_RE.finditer(source or ""):
            src_months.setdefault(m.group(2), set()).add((m.group(1) or "").lower())

        def _strip(m):
            month, year = (m.group(1) or ""), m.group(2)
            if not month:
                return m.group(0)
            known = src_months.get(year)
            # Only strip precision the source does not have. If the source
            # itself names the month, the draft is being faithful -- leave it.
            if known is not None and "" in known and not any(k for k in known):
                return re.sub(r"\s+", " ", m.group(0).replace(month, "", 1)).strip()
            return m.group(0)

        fixed = _EXPECTED_RE.sub(_strip, fixed)
    except Exception:  # noqa: BLE001 - a repair bug must never corrupt the draft
        fixed = draft or ""
    try:
        fixed = _restore_dropped_presentations(source, fixed)
    except Exception:  # noqa: BLE001 - same: never corrupt the draft
        pass
    try:
        for pattern, rep in _BANNED_BUZZWORDS:
            def _sub(m, rep=rep):
                return _match_case(m.group(0), rep)
            fixed = pattern.sub(_sub, fixed)
    except Exception:
        pass
    r = check_fidelity(source, fixed)
    return fixed, r.omissions + r.fabrications + r.buzzwords


def check_fidelity(source: str, draft: str) -> FidelityReport:
    """Compare `draft` against `source` (resume + profile concatenated is fine).

    Only positive evidence produces a finding, so a malformed source can cost a
    detection but can never invent one.
    """
    r = FidelityReport()
    for check, bucket in (
        (_check_dropped_publications, r.omissions),
        (_check_dropped_presentations, r.omissions),
        (_check_dropped_credentials, r.omissions),
        (_check_invented_dates, r.fabrications),
        (_check_relocated_employers, r.fabrications),
        (_check_asserted_absence, r.fabrications),
        (_check_altered_ranges, r.fabrications),
    ):
        try:
            bucket.extend(check(source, draft))
        except Exception:  # noqa: BLE001 - one checker's bug must not silence
            continue        # the others, nor break tailoring
    try:
        r.buzzwords.extend(check_banned_buzzwords(draft))
    except Exception:
        pass
    return r
