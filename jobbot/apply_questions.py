"""Application-form question extraction + auto-answering.

Given a job's URL, pull the REAL questions its application form asks:

  * Greenhouse  — public boards API returns the exact form (questions=true)
  * Lever       — the /apply page is server-rendered; parse its form
  * Ashby       — public non-user GraphQL endpoint exposes the application form
  * anything else — best-effort HTML <form> parse of the posting / apply page

Each question is classified and answered in priority order:

  1. answer bank   (data/answer_bank.json — user-curated regex -> answer)
  2. identity      (name/email/phone/links from .env applicant_* fields)
  3. screening     (work auth, sponsorship, salary, start date, ... from .env)
  4. EEO           (defaults to the "decline to self-identify" option if present)
  5. AI            (free-text + choice questions, answered from the resume;
                    choice answers are snapped to a real option or left blank)

Anything still blank is flagged `needs_user` so the UI / CLI can surface it.
Never invents facts: the AI prompt returns "" when the resume lacks the info.
"""
from __future__ import annotations

import json
import logging
import re
from dataclasses import dataclass, field, asdict
from pathlib import Path
from typing import List, Optional, Tuple

import requests

from .config import settings

log = logging.getLogger("jobbot.apply_questions")

UA = {"User-Agent": "Mozilla/5.0 (Windows NT 10.0; Win64; x64) JobBot/2.0"}


@dataclass
class FormQuestion:
    text: str
    qtype: str = "text"          # text|textarea|select|multiselect|boolean|file|date|number
    options: List[str] = field(default_factory=list)
    required: bool = False
    kind: str = ""               # identity|eeo|screening|essay|file
    answer: str = ""
    answer_source: str = ""      # answer_bank|profile|config|eeo-default|ai|file|""
    needs_user: bool = False
    # A value we had but could NOT confidently use — typically because it
    # didn't match any of the form's real options. Kept so the prompt can
    # offer it as a starting point instead of asking from a blank slate.
    guess: str = ""
    # Answered, but from an unverified source (an AI guess on a required
    # question). Submittable only after the user confirms it.
    needs_review: bool = False

    def display(self) -> str:
        opts = f"  [options: {' | '.join(self.options)}]" if self.options else ""
        return f"{self.text}{opts}"


# =====================  EXTRACTION  =========================================

GREENHOUSE_RE = re.compile(
    r"(?:boards|job-boards)\.greenhouse\.io/([A-Za-z0-9_-]+)/jobs/(\d+)")
GREENHOUSE_EMBED_RE = re.compile(
    r"greenhouse\.io/embed/job_app\?[^\"']*?for=([A-Za-z0-9_-]+)[^\"']*?token=(\d+)")
LEVER_RE = re.compile(r"jobs\.(?:eu\.)?lever\.co/([A-Za-z0-9_-]+)/([0-9a-fA-F-]{36})")
ASHBY_RE = re.compile(r"jobs\.ashbyhq\.com/([^/?#]+)/([0-9a-fA-F-]{36})")

# Workday URL pattern.
# Matches: https://{tenant}.{wdN}.myworkdayjobs.com[/{lang}]/{site}/job/{slug}
# Groups: (1) tenant  (2) wd-host-number e.g. "wd1"  (3) optional lang segment
#         e.g. "en-US" (or "")  (4) site  (5) path-after-/job/
WORKDAY_RE = re.compile(
    r"https?://([A-Za-z0-9_-]+)\.(wd\d+)\.myworkdayjobs\.com"
    r"(?:/([A-Za-z]{2}-[A-Za-z]{2}))?"   # optional /en-US style language segment
    r"/([A-Za-z0-9_%-]+)"                 # site
    r"/job/(.+)"                           # everything after /job/
)

_GH_TYPE_MAP = {
    "input_text": "text",
    "input_file": "file",
    "textarea": "textarea",
    "multi_value_single_select": "select",
    "multi_value_multi_select": "multiselect",
}


def _timeout() -> int:
    return getattr(settings, "apply_fetch_timeout", 20)


def fetch_form_questions(url: str) -> Tuple[List[FormQuestion], str]:
    """Return (questions, source). source describes which extractor succeeded;
    "" means nothing could be extracted (caller should fall back to the
    common-question set)."""
    if not url:
        return [], ""
    extractors = (
        ("greenhouse", _fetch_greenhouse),
        ("lever", _fetch_lever),
        ("ashby", _fetch_ashby),
        ("workday", _fetch_workday),
        ("html", _fetch_generic_html),
    )
    for name, fn in extractors:
        try:
            qs = fn(url)
        except Exception as e:  # noqa: BLE001
            log.info("%s extractor failed for %s: %s", name, url, e)
            qs = []
        if qs:
            log.info("Extracted %d real form questions via %s for %s",
                     len(qs), name, url)
            return _dedupe(qs), name
    return [], ""


def _dedupe(qs: List[FormQuestion]) -> List[FormQuestion]:
    seen: set[str] = set()
    out: List[FormQuestion] = []
    for q in qs:
        key = re.sub(r"[^a-z0-9]+", "", q.text.lower())
        if not key or key in seen:
            continue
        seen.add(key)
        out.append(q)
    return out


def _fetch_greenhouse(url: str) -> List[FormQuestion]:
    m = GREENHOUSE_RE.search(url) or GREENHOUSE_EMBED_RE.search(url)
    if not m:
        return []
    board, job_id = m.group(1), m.group(2)
    api = (f"https://boards-api.greenhouse.io/v1/boards/{board}/jobs/{job_id}"
           f"?questions=true")
    r = requests.get(api, headers=UA, timeout=_timeout())
    r.raise_for_status()
    data = r.json()

    raw: List[dict] = list(data.get("questions") or [])
    raw.extend(data.get("location_questions") or [])
    for c in data.get("compliance") or []:
        raw.extend(c.get("questions") or [])
    demo = data.get("demographic_questions") or {}
    if isinstance(demo, dict):
        raw.extend(demo.get("questions") or [])

    out: List[FormQuestion] = []
    for q in raw:
        label = (q.get("label") or "").strip()
        # Skip hidden geo fields the location widget fills automatically.
        if not label or label.lower() in ("latitude", "longitude"):
            continue
        # Two shapes: screening questions nest type/values under "fields";
        # demographic/compliance questions put type/answer_options at top level.
        if q.get("fields"):
            f0 = q["fields"][0]
            qtype = _GH_TYPE_MAP.get(f0.get("type", ""), "text")
            values = f0.get("values") or []
        else:
            qtype = _GH_TYPE_MAP.get(q.get("type", ""), "text")
            values = q.get("answer_options") or []
        options = [v.get("label", "") for v in values if v.get("label")]
        out.append(FormQuestion(text=label, qtype=qtype, options=options,
                                required=bool(q.get("required"))))
    return out


def _fetch_lever(url: str) -> List[FormQuestion]:
    m = LEVER_RE.search(url)
    if not m:
        return []
    company, posting = m.group(1), m.group(2)
    host = "jobs.eu.lever.co" if ".eu.lever.co" in url else "jobs.lever.co"
    apply_url = f"https://{host}/{company}/{posting}/apply"
    r = requests.get(apply_url, headers=UA, timeout=_timeout())
    r.raise_for_status()
    from bs4 import BeautifulSoup
    soup = BeautifulSoup(r.text, "html.parser")

    out: List[FormQuestion] = []
    for li in soup.select(".application-question"):
        label_el = li.select_one(".application-label")
        label = label_el.get_text(" ", strip=True) if label_el else ""
        label = label.replace("✱", "").strip()
        if not label:
            continue
        required = bool(label_el and ("✱" in label_el.get_text()
                                      or label_el.select_one(".required")))
        sel = li.find("select")
        textarea = li.find("textarea")
        file_in = li.find("input", attrs={"type": "file"})
        radios = li.find_all("input", attrs={"type": ["radio", "checkbox"]})
        if sel:
            options = [o.get_text(strip=True) for o in sel.find_all("option")
                       if o.get_text(strip=True) and not o.get("disabled")]
            options = [o for o in options if o.lower() not in ("select one", "select...", "--")]
            out.append(FormQuestion(text=label, qtype="select", options=options,
                                    required=required))
        elif radios:
            options = []
            for inp in radios:
                v = inp.get("value", "").strip()
                if v:
                    options.append(v)
            qtype = "multiselect" if radios[0].get("type") == "checkbox" else "select"
            out.append(FormQuestion(text=label, qtype=qtype, options=options,
                                    required=required))
        elif file_in:
            out.append(FormQuestion(text=label, qtype="file", required=required))
        elif textarea:
            out.append(FormQuestion(text=label, qtype="textarea", required=required))
        else:
            out.append(FormQuestion(text=label, qtype="text", required=required))
    return out


def _fetch_ashby(url: str) -> List[FormQuestion]:
    m = ASHBY_RE.search(url)
    if not m:
        return []
    org, posting = m.group(1), m.group(2)
    gql = {
        "operationName": "ApiJobPosting",
        "variables": {
            "organizationHostedJobsPageName": org,
            "jobPostingId": posting,
        },
        "query": """query ApiJobPosting($organizationHostedJobsPageName: String!, $jobPostingId: String!) {
  jobPosting(organizationHostedJobsPageName: $organizationHostedJobsPageName, jobPostingId: $jobPostingId) {
    id
    applicationForm {
      sections {
        fieldEntries {
          isRequired
          field
        }
      }
    }
  }
}""",
    }
    r = requests.post("https://jobs.ashbyhq.com/api/non-user-graphql?op=ApiJobPosting",
                      json=gql, headers={**UA, "Content-Type": "application/json"},
                      timeout=_timeout())
    r.raise_for_status()
    data = r.json()
    form = (((data.get("data") or {}).get("jobPosting") or {})
            .get("applicationForm") or {})
    out: List[FormQuestion] = []
    for section in form.get("sections") or []:
        for entry in section.get("fieldEntries") or []:
            # `field` is a JSON! scalar in Ashby's schema — arrives as a dict
            # (or a JSON string on some tenants).
            f = entry.get("field") or {}
            if isinstance(f, str):
                try:
                    f = json.loads(f)
                except ValueError:
                    continue
            if not isinstance(f, dict):
                continue
            title = (f.get("title") or "").strip()
            if not title:
                continue
            ftype = (f.get("type") or "").lower()
            options = [v.get("label", "") for v in (f.get("selectableValues") or [])
                       if v.get("label")]
            if "file" in ftype:
                qtype = "file"
            elif "multiselect" in ftype or "multi_select" in ftype:
                qtype = "multiselect"
            elif options or "select" in ftype or ftype == "boolean":
                qtype = "select"
                if ftype == "boolean" and not options:
                    options = ["Yes", "No"]
            elif "longtext" in ftype or "paragraph" in ftype:
                qtype = "textarea"
            else:
                qtype = "text"
            out.append(FormQuestion(text=title, qtype=qtype, options=options,
                                    required=bool(entry.get("isRequired"))))
    return out


def _workday_cxs_fetch(url: str) -> Optional[dict]:
    """GET the Workday CXS JSON for a job posting URL.

    Returns the parsed JSON dict on success, or None on any failure.
    Tries the URL as-is first; if that returns 404 it retries with the
    language segment toggled (added if missing, removed if present).
    """
    m = WORKDAY_RE.match(url)
    if not m:
        return None
    tenant, _wdhost, lang, site, job_path = (
        m.group(1), m.group(2), m.group(3) or "", m.group(4), m.group(5)
    )
    # Strip trailing fragment/query from job_path
    job_path = re.split(r"[?#]", job_path)[0].rstrip("/")

    base = f"https://{tenant}.{_wdhost}.myworkdayjobs.com/wday/cxs/{tenant}/{site}/job/{job_path}"

    hdrs = {**UA, "Accept": "application/json"}

    def _try(endpoint: str) -> Optional[dict]:
        try:
            r = requests.get(endpoint, headers=hdrs, timeout=_timeout())
            if r.status_code == 404:
                return None
            r.raise_for_status()
            return r.json()
        except Exception as exc:  # noqa: BLE001
            log.debug("CXS fetch failed for %s: %s", endpoint, exc)
            return None

    data = _try(base)
    if data is not None:
        return data

    # The Workday CXS path never includes a language segment — the lang
    # portion of the public URL (/en-US/) is not passed to the API path,
    # so there is no alternate URL to try.  Log and return None.
    log.debug("CXS endpoint returned 404 for %s", url)
    return None


# ---- Standard EEO options (EEOC / OFCCP wording used by Workday) ----
_WD_RACE_OPTIONS = [
    "Hispanic or Latino",
    "White (Not Hispanic or Latino)",
    "Black or African American (Not Hispanic or Latino)",
    "Native Hawaiian or Other Pacific Islander (Not Hispanic or Latino)",
    "Asian (Not Hispanic or Latino)",
    "American Indian or Alaska Native (Not Hispanic or Latino)",
    "Two or More Races (Not Hispanic or Latino)",
    "I do not wish to answer",
]


def _fetch_workday(url: str) -> List[FormQuestion]:
    """Extract the STANDARD Workday application question set for a job URL.

    Workday application forms require authentication so individual tenant
    question pages cannot be fetched anonymously.  The public CXS JSON API
    does expose the job posting metadata (title, country, canApply, etc.)
    without auth — we use that to:
      1. Confirm the posting is still open (canApply).
      2. Pull the country / company name to pre-fill a few questions.
      3. Return the highly-standardised "My Information" + "My Experience"
         question set that Workday presents on step 1 of every application
         (the tenant may add custom questionnaire pages behind login that we
         cannot reach here).
    """
    if not WORKDAY_RE.match(url):
        return []

    m = WORKDAY_RE.match(url)
    tenant = m.group(1) if m else "the company"

    data = _workday_cxs_fetch(url)
    if data is None:
        # Network error or endpoint unavailable — surface nothing rather than
        # crashing; the caller will fall back to generic HTML.
        log.debug("_fetch_workday: could not reach CXS for %s", url)
        return []

    posting = data.get("jobPostingInfo") or {}

    # If posting is closed / un-posted return [] so caller doesn't present
    # a stale form to the user.
    can_apply = posting.get("canApply")
    if can_apply is False:
        log.info("Workday posting not open for applications: %s", url)
        return []

    # Company display name: try several common CXS key names.
    company_name = (
        data.get("hiringOrganization", {}).get("name")
        or data.get("organizationName")
        or data.get("company")
        or posting.get("hiringOrganization", {}).get("name")
        or tenant.title()
    )
    if isinstance(company_name, dict):
        company_name = company_name.get("name") or tenant.title()

    # Workday's CXS JSON returns `country` as a {descriptor, id} object on many
    # tenants (like company_name above), not a bare string. Coerce to its display
    # string so it doesn't reach the form as a dict and crash q.answer.strip().
    country = posting.get("country") or "United States of America"
    if isinstance(country, dict):
        country = (country.get("descriptor") or country.get("name")
                   or "United States of America")

    # ------------------------------------------------------------------
    # STANDARD Workday question set — "My Information" + "My Experience"
    # These questions appear on virtually every Workday tenant's step 1.
    # Custom questionnaire pages that appear after login are NOT included.
    # ------------------------------------------------------------------
    qs: List[FormQuestion] = [
        # ---- Contact / Identity ----
        FormQuestion(
            text="First Name",
            qtype="text",
            required=True,
            kind="identity",
        ),
        FormQuestion(
            text="Last Name",
            qtype="text",
            required=True,
            kind="identity",
        ),
        FormQuestion(
            text="Email Address",
            qtype="text",
            required=True,
            kind="identity",
        ),
        FormQuestion(
            text="Phone",
            qtype="text",
            required=True,
            kind="identity",
        ),
        # ---- Address ----
        FormQuestion(
            text="Country",
            qtype="select",
            options=["United States of America"],
            required=True,
            kind="identity",
            answer=country,
            answer_source="profile",
        ),
        FormQuestion(
            text="Address Line 1",
            qtype="text",
            kind="identity",
        ),
        FormQuestion(
            text="City",
            qtype="text",
            kind="identity",
        ),
        FormQuestion(
            text="State",
            qtype="text",
            kind="identity",
        ),
        FormQuestion(
            text="Postal Code",
            qtype="text",
            kind="identity",
        ),
        # ---- Resume / CV ----
        FormQuestion(
            text="Resume/CV",
            qtype="file",
            required=True,
            kind="file",
        ),
        # ---- Referral ----
        FormQuestion(
            text="How Did You Hear About Us?",
            qtype="select",
            options=[
                "Company Careers Site",
                "LinkedIn",
                "Indeed",
                "Employee Referral",
                "Other",
            ],
            kind="screening",
        ),
        # ---- Work history at company ----
        FormQuestion(
            text=f"Have you previously worked for {company_name}?",
            qtype="select",
            options=["Yes", "No"],
            required=True,
            kind="screening",
        ),
        # ---- Work authorisation ----
        FormQuestion(
            text="Are you legally eligible to work in the country where this position is located?",
            qtype="select",
            options=["Yes", "No"],
            required=True,
            kind="screening",
        ),
        FormQuestion(
            text="Will you now or in the future require sponsorship for employment visa status?",
            qtype="select",
            options=["Yes", "No"],
            required=True,
            kind="screening",
        ),
        # ---- Voluntary EEO / Self-identification ----
        FormQuestion(
            text="Gender",
            qtype="select",
            options=["Male", "Female", "I do not wish to answer"],
            kind="eeo",
        ),
        FormQuestion(
            text="Race/Ethnicity",
            qtype="select",
            options=_WD_RACE_OPTIONS,
            kind="eeo",
        ),
        FormQuestion(
            text="Veteran Status",
            qtype="select",
            options=[
                "I am not a protected veteran",
                "I identify as one or more of the classifications of a protected veteran",
                "I do not wish to answer",
            ],
            kind="eeo",
        ),
        FormQuestion(
            text="Disability Status",
            qtype="select",
            options=[
                "Yes, I have a disability, or have had one in the past",
                "No, I do not have a disability and have not had one in the past",
                "I do not want to answer",
            ],
            kind="eeo",
        ),
    ]
    return qs


def workday_job_status(url: str) -> dict:
    """Return liveness / apply-eligibility info for a Workday job URL.

    Used by dead-posting detection and the job pipeline to skip closed roles.

    Return schema::

        {
            "alive":     bool | None,   # None = network error / unknown
            "can_apply": bool | None,   # None = not determinable
            "title":     str,           # empty string if unavailable
        }
    """
    if not WORKDAY_RE.match(url):
        return {"alive": None, "can_apply": None, "title": ""}

    data = _workday_cxs_fetch(url)
    if data is None:
        return {"alive": None, "can_apply": None, "title": ""}

    posting = data.get("jobPostingInfo") or {}
    title = (posting.get("title") or "").strip()
    can_apply = posting.get("canApply")   # bool or None if key absent

    # "alive" = posting exists AND is not explicitly closed.
    # If canApply is explicitly False the job is closed.
    if can_apply is False:
        alive = False
    elif data:           # we got a valid JSON response → posting exists
        alive = True
    else:
        alive = None

    return {
        "alive": alive,
        "can_apply": can_apply,
        "title": title,
    }


_SKIP_INPUT_TYPES = {"hidden", "submit", "button", "image", "reset"}


def _fetch_generic_html(url: str) -> List[FormQuestion]:
    """Best-effort parse of <form> fields on the posting (and an /apply
    variant). JS-rendered ATS pages yield nothing here — that's expected; the
    caller falls back to the common-question set."""
    from bs4 import BeautifulSoup

    candidates = [url]
    if not url.rstrip("/").endswith("apply"):
        candidates.append(url.rstrip("/") + "/apply")

    for target in candidates:
        try:
            r = requests.get(target, headers=UA, timeout=_timeout())
            if r.status_code >= 400:
                continue
        except Exception:  # noqa: BLE001
            continue
        soup = BeautifulSoup(r.text, "html.parser")
        best: List[FormQuestion] = []
        for form in soup.find_all("form"):
            qs: List[FormQuestion] = []
            has_file = False
            for el in form.find_all(["input", "textarea", "select"]):
                itype = (el.get("type") or "text").lower() if el.name == "input" else el.name
                if itype in _SKIP_INPUT_TYPES:
                    continue
                label = _label_for(form, el)
                if not label:
                    continue
                if itype == "file":
                    has_file = True
                    qs.append(FormQuestion(text=label, qtype="file",
                                           required=el.has_attr("required")))
                elif el.name == "select":
                    options = [o.get_text(strip=True) for o in el.find_all("option")
                               if o.get_text(strip=True)]
                    qs.append(FormQuestion(text=label, qtype="select", options=options,
                                           required=el.has_attr("required")))
                elif el.name == "textarea":
                    qs.append(FormQuestion(text=label, qtype="textarea",
                                           required=el.has_attr("required")))
                elif itype in ("checkbox", "radio"):
                    continue  # grouped inputs are noisy without JS context
                else:
                    qs.append(FormQuestion(text=label, qtype="text",
                                           required=el.has_attr("required")))
            # Only trust forms that plausibly ARE the application form.
            if (has_file or len(qs) >= 4) and len(qs) > len(best):
                best = qs
        if best:
            return best
    return []


def _label_for(form, el) -> str:
    el_id = el.get("id")
    if el_id:
        lab = form.find("label", attrs={"for": el_id})
        if lab:
            return lab.get_text(" ", strip=True).replace("*", "").strip()
    for attr in ("aria-label", "placeholder"):
        v = (el.get(attr) or "").strip()
        if v:
            return v
    parent_label = el.find_parent("label")
    if parent_label:
        return parent_label.get_text(" ", strip=True).replace("*", "").strip()
    name = (el.get("name") or "").strip()
    if name and not re.fullmatch(r"[a-f0-9-]{20,}", name):
        return re.sub(r"[_\-\[\]]+", " ", name).strip().title()
    return ""


# =====================  CLASSIFICATION  =====================================

_EEO_PAT = re.compile(
    r"gender|race|ethnic|hispanic|latin|veteran|disab|sexual orientation|"
    r"transgender|lgbt|pronoun|demographic", re.I)
_IDENTITY_PATTERNS = [
    (r"\bfirst\s*name\b", "first_name"),
    (r"\blast\s*name|surname|family name\b", "last_name"),
    (r"\b(full|legal)\s*name\b|^name$", "full_name"),
    (r"e-?mail", "email"),
    (r"phone|mobile", "phone"),
    (r"linkedin", "linkedin"),
    (r"github", "github"),
    (r"portfolio|personal website|website|url", "portfolio"),
    (r"current (city|location)|^location$|city of residence|where (are you|do you) (located|live|reside)", "location"),
    (r"current (title|role|position)", "current_title"),
]
_SCREENING_PATTERNS = [
    # Sponsorship is checked BEFORE work authorization: these are opposite-polarity
    # questions, and a sponsorship question routinely names work authorization too
    # ("require employer sponsorship for work authorization"). Matching the
    # authorization rule first answered such a question from work_authorized ("Yes")
    # and told the employer the candidate needs sponsorship when they do not.
    # A plain work-auth question never says "sponsor", so it still falls through.
    (r"sponsor|visa", "requires_sponsorship"),
    (r"authoriz|legally (able|permitted) to work|right to work|eligib(le|ility) to work|work permit", "work_authorized"),
    (r"salary|compensation|desired pay|pay (expectation|range)|expected (pay|salary)", "salary_expectation"),
    (r"start date|available to start|earliest.*(start|available)|notice period|when (can|could) you start", "earliest_start_date"),
    (r"relocat", "willing_relocate"),
    (r"on-?site|hybrid|in.?office|commut|work (at|from) the (office|listed location)", "willing_onsite"),
    (r"how did you.*(hear|find|learn)|referral source|where did you.*(hear|find)", "how_heard"),
]
_FILE_PAT = re.compile(r"resume|cv\b|curriculum|cover letter|attach", re.I)
_DECLINE_PAT = re.compile(
    r"decline|don'?t wish|do not wish|prefer not|i choose not"
    r"|do(?:n'?t| not) want to (?:answer|disclose|self.?identify)", re.I)


def classify(q: FormQuestion) -> str:
    t = q.text.lower()
    if q.qtype == "file" or (_FILE_PAT.search(t) and len(t) < 60):
        return "file"
    if _EEO_PAT.search(t):
        return "eeo"
    for pat, _key in _IDENTITY_PATTERNS:
        if re.search(pat, t, re.I):
            return "identity"
    for pat, _key in _SCREENING_PATTERNS:
        if re.search(pat, t, re.I):
            # Compound questions ("how did you hear about us AND WHY do you
            # want to work here?") deserve a real answer, not the one-line
            # config default — route them to the AI as essays.
            if re.search(r"\bwhy\b", t) or len(t) > 120:
                return "essay"
            return "screening"
    return "essay" if q.qtype in ("textarea",) or len(t) > 60 else "short"


# =====================  ANSWER BANK  ========================================

def _answer_bank_path() -> Path:
    return Path(getattr(settings, "answer_bank_path", "data/answer_bank.json"))


def load_answer_bank_raw() -> List[dict]:
    """Every bank entry, including ones staged for review."""
    p = _answer_bank_path()
    if not p.exists():
        return []
    try:
        data = json.loads(p.read_text(encoding="utf-8"))
        return [e for e in data if isinstance(e, dict) and e.get("match")]
    except Exception as e:  # noqa: BLE001
        log.warning("Could not read answer bank %s: %s", p, e)
        return []


def load_answer_bank() -> List[dict]:
    """Active bank rules only.

    Entries marked `status: "pending"` are staged for review (see
    `jobbot answers review`) and are invisible to answering until approved --
    the bank is a VERIFIED source, so an unapproved entry would otherwise be
    submitted to a real employer without review.
    """
    return [e for e in load_answer_bank_raw()
            if e.get("status", "active") == "active"]


def write_bank_entries(entries: List[dict], source: str) -> int:
    """Replace every entry tagged `source` with `entries`, leaving all others
    alone. Entries with no `source` key are hand-curated and are never touched.
    Returns how many were written."""
    kept = [e for e in load_answer_bank_raw() if e.get("source") != source]
    tagged = []
    for e in entries:
        e = dict(e)
        e["source"] = source
        e.setdefault("status", "active")
        tagged.append(e)
    p = _answer_bank_path()
    p.parent.mkdir(parents=True, exist_ok=True)
    p.write_text(json.dumps(kept + tagged, indent=2, ensure_ascii=False),
                 encoding="utf-8")
    return len(tagged)


def remember_answer(question_text: str, answer: str) -> None:
    """Remember a user-approved answer for future forms.

    Writes to the answer *memory*, not the bank: the bank is the hand-edited
    rules layer now. Fail-open -- a failed write must not fail an otherwise
    successful gap fill.
    """
    if not question_text.strip() or not answer.strip():
        return
    try:
        from .answer_memory import AnswerMemory
        AnswerMemory(settings.answer_memory_path).remember(
            FormQuestion(text=question_text), answer.strip(), source="user")
    except Exception as e:  # noqa: BLE001
        log.warning("Could not remember %r: %s", question_text, e)


def _personal_info_entries() -> List[dict]:
    """Answer-bank entries derived from the user's personal-info sheet.

    Split out (rather than imported inline) so callers and tests have a single
    seam to stub, and so `personal_info_bank_entries_safe` can wrap it."""
    from .personal_info import personal_info_bank_entries
    return personal_info_bank_entries()


def personal_info_bank_entries_safe() -> List[dict]:
    """`_personal_info_entries` but fail-open: a missing or malformed sheet
    degrades to the curated bank rather than breaking every application."""
    try:
        return _personal_info_entries()
    except Exception as e:  # noqa: BLE001
        log.info("personal-info sheet unavailable (%s) - using curated bank only.", e)
        return []


def _bank_lookup(text: str, bank: List[dict]) -> Optional[str]:
    """The bank answer for `text`, preferring the most specific entry.

    Entries are not mutually exclusive: a broad curated pattern ("how did you
    hear") and a narrower one -- either a hand-added entry in
    data/answer_bank.json or one generated from the personal-info sheet
    (`personal_info_bank_entries_safe`) -- can both match the same question.
    Taking the first hit meant the narrower, more-specific entry was shadowed
    by the broader one and never applied - so an exact whole-question match
    wins, and otherwise the longest (most specific) pattern does.
    """
    exact = re.escape(text.strip())
    best: Optional[str] = None
    best_len = -1
    for entry in bank:
        pat = str(entry.get("match", ""))
        ans = str(entry.get("answer", ""))
        if not pat or not ans:
            continue
        try:
            hit = bool(re.search(pat, text, re.I))
        except re.error:
            hit = pat.lower() in text.lower()
        if not hit:
            continue
        if pat == exact:
            return ans          # question-specific entry: always wins
        if len(pat) > best_len:
            best, best_len = ans, len(pat)
    return best


# =====================  ANSWERING  ==========================================

# Sources we trust enough to submit unreviewed: the user curated them, or they
# came straight off the profile / config / the user's own personal-info sheet.
VERIFIED_SOURCES = frozenset(
    {"answer_bank", "memory", "profile", "config", "eeo-default", "file", "user"})
# The model-guess sources, named for readability at call sites. Anything NOT in
# VERIFIED_SOURCES is treated as unverified, so this list need not be complete.
UNVERIFIED_SOURCES = frozenset({"ai", "ai-default"})


def _set_answer(q: "FormQuestion", value: str, source: str) -> bool:
    """Assign `value` to `q` only if it is actually submittable.

    On a question with a fixed option list, a value that doesn't snap onto one
    of the real options is NOT an answer — writing it produced forms that were
    invalid (or, worse, quietly wrong: the user's gender landing in a Yes/No
    transgender field). Such a value is stashed on `q.guess` so the user gets
    prompted with a starting point instead of a blank. Returns True if an
    answer was set."""
    value = (value or "").strip()
    if not value:
        return False
    if q.options:
        snapped = _snap_to_options(value, q.options)
        if not snapped:
            q.guess = value
            return False
        value = snapped
    q.answer, q.answer_source = value, source
    return True


def needs_attention(questions: List["FormQuestion"]) -> List["FormQuestion"]:
    """Every question the user still has to deal with: outright blanks plus
    answers held for confirmation. This is what the CLI prompts on and what
    the dashboard counts."""
    return [q for q in questions if q.needs_user or q.needs_review]


def _squash(s: str) -> str:
    """Lowercase, keeping only letters and digits, so option text that differs
    from the reply only in punctuation or spacing still compares equal."""
    return re.sub(r"[^a-z0-9]+", "", s.strip().lower())


def _snap_to_options(answer: str, options: List[str]) -> str:
    """Map a free-text answer onto one of the form's real options ('' if no
    confident match) so select answers are always submittable as-is."""
    if not options:
        return answer
    a = answer.strip().lower()
    if not a:
        return ""
    for o in options:
        if o.strip().lower() == a:
            return o
    # Same word, different punctuation ("nonbinary" -> "Non-binary").
    squashed = _squash(a)
    if squashed:
        for o in options:
            if _squash(o) == squashed:
                return o
    # decline-style answers snap onto the form's decline option
    if _DECLINE_PAT.search(a) or "prefer not to say" in a:
        for o in options:
            if _DECLINE_PAT.search(o):
                return o
    yes_no = {"yes": ("yes", "y", "true", "i am", "i do"),
              "no": ("no", "n", "false", "i am not", "i do not", "i don't")}
    for key, variants in yes_no.items():
        # Word boundary: "not sure" opens with the letters of "no" but is not
        # an answer of No, and reading it as one asserts something the user
        # never said.
        if a in variants or re.match(rf"{key}\b", a):
            for o in options:
                if o.strip().lower().startswith(key):
                    return o
    for o in options:
        ol = o.strip().lower()
        # A two-letter option landing inside a longer reply is a coincidence,
        # not a choice — require some substance before trusting containment.
        if (len(ol) >= 3 and ol in a) or a in ol:
            return o
    return ""


def _memory_lookup(q: "FormQuestion") -> Optional[str]:
    """The remembered answer for `q`, or None.

    Tries the exact canonical key first, then a graded fuzzy match. Only `static`
    entries are reusable at either tier: a `tailored` entry was written about one
    specific job, so replaying it into another company's kit would be wrong --
    and since "memory" is a verified source it would be submitted unreviewed.
    (apply_runner applies the same kind guard to its pre-fill; the fuzzy tier
    enforces it too, returning None for non-static kinds before it scores.)

    A confident fuzzy match (>= answer_match_high_threshold with no polarity
    inversion) is returned like an exact hit. A weaker one (down to
    answer_match_review_threshold) is not trusted enough to submit: it is written
    to `q.guess` as a suggestion for the user to confirm, and None is returned so
    the answer stays blank. Blank is the only browser-safe hold -- the autosubmit
    planner (browser_apply) pauses on a blank answer, but does NOT pause on
    needs_review alone, so a held value must never sit in q.answer.

    Kept behind a function (rather than inlined) so the fuzzy tier has one seam to
    extend, and so tests have one thing to stub. Fail-open: an unreadable memory
    file must not break kit building.
    """
    try:
        from .answer_memory import AnswerMemory
        mem = AnswerMemory(settings.answer_memory_path)
        exact = mem.static_answer_for(q)
        if exact is not None:
            return exact
        fuzzy = mem.fuzzy_lookup(
            q,
            high=getattr(settings, "answer_match_high_threshold", 0.93),
            review=getattr(settings, "answer_match_review_threshold", 0.80))
    except Exception as e:  # noqa: BLE001
        log.info("answer memory unavailable (%s) - skipping.", e)
        return None
    if not fuzzy:
        return None
    entry, confident = fuzzy
    if confident:
        return entry.answer
    q.guess = entry.answer
    return None


def answer_questions(questions: List[FormQuestion],
                     resume: str, job_title: str, company: str,
                     job_description: str,
                     identity_fields: Optional[dict] = None,
                     resume_path: str = "", cover_letter_path: str = "",
                     use_ai: bool = True) -> List[FormQuestion]:
    """Fill `answer` / `answer_source` / `needs_user` on every question,
    in-place, and return the list."""
    bank = load_answer_bank()
    # Append answers parsed from the user's personal-info sheet (address, EEO
    # self-ID, "how did you hear", ...). `_bank_lookup` picks the most specific
    # match across both, so a curated entry still beats the sheet's broad
    # patterns on the questions it names. Fail-open if the sheet is unreadable.
    bank = bank + personal_info_bank_entries_safe()
    identity_fields = identity_fields or {}
    ai_batch: List[FormQuestion] = []
    # A posted pay band overrides any stored salary default (see the salary
    # branch below). Computed once: the description does not change per question.
    posted_bands = _posted_salary_bands(job_description)
    posted_pay = _posted_salary_range(job_description)

    for q in questions:
        q.kind = q.kind or classify(q)
        t = q.text

        banked = _bank_lookup(t, bank)
        if banked is not None:
            # A posting that publishes its pay band has already answered the
            # salary question, and that beats a stored default. The bank cannot
            # tell an academic postdoc from an industry Scientist role (the
            # question reads "salary expectations" either way), so job 5934
            # an applicant could ask a university for $135-155K against a posted $65,000 band. Hold it for the user rather than send a number that reads
            # as the wrong pay grade.
            if _SALARY_Q_RE.search(t) and len(posted_bands) > 1:
                # Several bands: the posting spans multiple levels and the right
                # ask depends on which one you are applying for. Not guessable.
                q.guess = "; ".join(f"${lo:,.0f} to ${hi:,.0f}" for lo, hi in posted_bands)
                q.needs_user = True
                log.warning("Salary held for review: posting lists %d pay bands, so the "
                            "level is ambiguous. Stored answer %r not submitted.",
                            len(posted_bands), banked)
                continue
            if _SALARY_Q_RE.search(t) and _salary_answer_conflicts(banked, posted_pay):
                lo, hi = posted_pay
                # The posted band is itself a sensible answer, so offer that as
                # the starting point rather than the stored figure we rejected.
                q.guess = f"${lo:,.0f} to ${hi:,.0f}"
                q.needs_user = True
                log.warning(
                    "Salary held for review: posting states $%s-$%s but the stored "
                    "answer is %r. Not submitting it.", f"{lo:,.0f}", f"{hi:,.0f}", banked)
                continue
            if _set_answer(q, banked, "answer_bank"):
                continue
            # Banked value didn't fit this form's options (an over-broad bank
            # pattern, or a form wording we haven't seen). Ask, don't guess.
            q.needs_user = True
            continue

        remembered = _memory_lookup(q)
        if remembered is not None:
            if _set_answer(q, remembered, "memory"):
                continue
            # Remembered from a form whose options differ from this one's.
            # Ask, don't guess -- same rule as the bank above.
            q.needs_user = True
            continue
        if q.guess:
            # _memory_lookup found only a low-confidence fuzzy match and left the
            # value on q.guess. Surface it as a gap-with-suggestion for the user
            # to confirm; don't let a lower deterministic tier silently overwrite
            # what the user's own memory suggests.
            q.needs_user = True
            continue

        if q.kind == "file":
            low = t.lower()
            if "cover" in low:
                # Cover-letter upload: use the cover letter ONLY. Never fall back
                # to the resume here (that uploaded the resume as the cover
                # letter). Optional + missing -> leave blank; required -> user.
                if cover_letter_path:
                    q.answer, q.answer_source = cover_letter_path, "file"
                elif q.required:
                    q.needs_user = True
            elif resume_path:
                q.answer, q.answer_source = resume_path, "file"
            else:
                q.needs_user = True
            continue

        if q.kind == "identity":
            # Link-type questions are interchangeable on most forms ("Github
            # or website", "Portfolio/LinkedIn") — fall back across whichever
            # links the profile actually has.
            _LINK_FALLBACKS = {"github": ("github", "portfolio", "linkedin"),
                               "portfolio": ("portfolio", "github", "linkedin"),
                               "linkedin": ("linkedin", "portfolio", "github")}
            for pat, key in _IDENTITY_PATTERNS:
                if re.search(pat, t, re.I):
                    keys = _LINK_FALLBACKS.get(key, (key,))
                    val = next((v for k in keys
                                if (v := (identity_fields.get(k) or "").strip())), "")
                    if val:
                        # e.g. profile location "Boston, MA" against a form that
                        # only offers its own office cities — that's a question
                        # for the user, not a value to force in.
                        _set_answer(q, val, "profile")
                    break
            if not q.answer and use_ai:
                ai_batch.append(q)
            elif not q.answer:
                q.needs_user = True
            continue

        if q.kind == "screening":
            for pat, key in _SCREENING_PATTERNS:
                if re.search(pat, t, re.I):
                    val = str(getattr(settings, key, "") or "").strip()
                    if val:
                        _set_answer(q, val, "config")
                    break
            if not q.answer and use_ai:
                ai_batch.append(q)
            elif not q.answer:
                q.needs_user = True
            continue

        if q.kind == "eeo":
            decline = next((o for o in q.options if _DECLINE_PAT.search(o)), "")
            if decline:
                q.answer, q.answer_source = decline, "eeo-default"
            else:
                q.needs_user = True
            continue

        # essay / short free-text / choice questions -> AI
        if use_ai:
            ai_batch.append(q)
        else:
            q.needs_user = True

    if ai_batch and use_ai:
        _ai_answer(ai_batch, resume, job_title, company, job_description)

    for q in questions:
        if not q.answer.strip() and q.answer_source != "file":
            q.needs_user = True
            q.needs_review = False   # a blank is a gap, not a thing to confirm
        elif q.required and q.answer_source not in VERIFIED_SOURCES:
            # Answered only by a model guess, on a question the form requires.
            # Hold it for a human OK rather than submitting it blind. Gating on
            # the allow-list means any source we don't recognise also gets asked
            # about, rather than silently sailing through.
            q.needs_review = True
    return questions


_HOW_HEARD_PAT = re.compile(
    r"how did you (hear|find|learn)|where did you (hear|find)|"
    r"referral source|hear about (us|this|the)|how.*find.*(job|position|role)",
    re.I)


def _norm_key(s: str) -> str:
    """Normalize an AI answer key (or question) for tolerant matching.

    Local models drift the keys we send: they drop the trailing
    ``(You MUST answer ...)`` options hint and prepend enumeration like
    ``4. `` / ``4) `` / ``- ``. Strip all of that, lowercase, and collapse
    whitespace so a drifted key still lines up with its question."""
    s = re.sub(r"\s*\(You MUST answer.*$", "", s or "", flags=re.I | re.S)
    s = re.sub(r"^\s*[-*\d]+[.)]\s*", "", s)
    return re.sub(r"\s+", " ", s).strip().lower()


# A money figure with an optional $ and thousands separators.
_MONEY = r"\$?\s*(\d{1,3}(?:,\d{3})+(?:\.\d{2})?|\d{4,7}(?:\.\d{2})?)"
# "$62,354 - $65,000", "$120,000 to $150,000", "$95,000-$115,000"
_RANGE_RE = re.compile(_MONEY + r"\s*(?:-|–|—|to)\s*" + _MONEY, re.I)
# An hourly band must never be read as an annual one.
_HOURLY_NEAR = re.compile(r"/\s*(?:hr|hour)|per\s+hour|hourly", re.I)

# Below this, a "range" is a placeholder or an hourly figure, not a salary band.
# A posting with "Salary Range: $0.00 - $0.00".
_MIN_PLAUSIBLE_ANNUAL = 10000.0

# Identifies a salary question by its TEXT. Deliberately not keyed on
# FormQuestion.kind: classify() buckets these as "screening", and
# "salary_expectation" is a screening-pattern key, not a kind.
_SALARY_Q_RE = re.compile(
    r"salary|compensation|desired (?:pay|rate)|pay (?:expectation|range)"
    r"|expected (?:pay|salary)|hourly rate", re.I)


def _posted_salary_bands(description: str):
    """Every distinct annual pay band a posting states, as [(low, high), ...].

    More than one band means the posting covers several levels (a talent pool
    running postdoc through senior scientist). Which one applies depends on the
    level being applied for, which is not knowable here - so the caller must
    treat a multi-band posting as a question for the user, not something to
    answer automatically.
    """
    if not description:
        return []
    bands = []
    seen = set()
    for m in _RANGE_RE.finditer(description):
        # Skip a match that is plainly hourly - check the tail of the line it
        # sits on, where "/ hr" lives.
        line_end = description.find("\n", m.end())
        tail = description[m.end():line_end if line_end != -1 else m.end() + 20]
        if _HOURLY_NEAR.search(tail):
            continue
        try:
            lo = float(m.group(1).replace(",", ""))
            hi = float(m.group(2).replace(",", ""))
        except ValueError:
            continue
        if lo < _MIN_PLAUSIBLE_ANNUAL or hi < _MIN_PLAUSIBLE_ANNUAL or hi < lo:
            continue
        if (lo, hi) in seen:
            continue
        seen.add((lo, hi))
        bands.append((lo, hi))
    return bands


def _posted_salary_range(description: str):
    """The single annual band a posting states, as (low, high), or None.

    Returns None when the posting states no band OR states several: with several
    the level is ambiguous, and pretending otherwise is what let a $135K answer
    through against a postdoc line advertised at $62,354. Use
    `_posted_salary_bands` when you need to see them all.
    """
    bands = _posted_salary_bands(description)
    return bands[0] if len(bands) == 1 else None


def _salary_answer_conflicts(answer: str, posted) -> bool:
    """True when `answer` is so far outside `posted` that sending it would hurt.

    Tolerates ordinary negotiation (a bit over the top of the band) and only
    flags answers that read as a different job's pay grade entirely - in either
    direction, since underselling by half is as damaging as asking for double.
    """
    if not posted or not answer:
        return False
    lo, hi = posted
    figures = []
    for raw in re.findall(_MONEY, answer):
        try:
            v = float(raw.replace(",", ""))
        except ValueError:
            continue
        if v >= _MIN_PLAUSIBLE_ANNUAL:
            figures.append(v)
    if not figures:
        return False            # "Negotiable", "Open" - nothing to contradict
    # The ask is the bottom of whatever the candidate stated: that is the
    # number that has to be reachable from the employer's band.
    ask = min(figures)
    return ask > hi * 1.15 or ask < lo * 0.6


def _default_for(q: FormQuestion) -> str:
    """A safe option pick for low-stakes source selects ('how did you hear
    about us') so a large form doesn't stall on one required-but-trivial
    field. Returns '' for anything substantive — those still go to the user."""
    if not q.options or not _HOW_HEARD_PAT.search(q.text):
        return ""
    prefs = ("company website", "company site", "website", "linkedin",
             "job board", "indeed", "search", "other")
    for p in prefs:
        for o in q.options:
            if p in o.strip().lower() and not _DECLINE_PAT.search(o):
                return o
    for o in q.options:  # else the first non-decline option
        if not _DECLINE_PAT.search(o):
            return o
    return ""


# A year the draft states must not predate the source's earliest known year.
# Prose answers rarely use the formatted mm/yyyy ranges source_fidelity's
# checks expect (those are built for tailored-resume-vs-profile comparisons),
# so a backdated employment year like "since 2015" when the source's earliest
# year is 2022 would otherwise sail straight through repair_fidelity
# untouched. This deliberately checks BACKDATING only, not "does this exact
# year appear in the source": a free-text answer legitimately states years
# the resume never mentions (an availability date like "I could start in
# September 2026" is not a claim about the candidate's history, and required
# start-date questions route through this same gate). A forward fabrication
# ("since 2024" when the source's earliest year is 2022) is not caught by
# this check; see the module's known-limitation note.
_YEAR_RE = re.compile(r"\b(?:19|20)\d{2}\b")


def _fabricated_years(source: str, draft: str) -> List[str]:
    src_years = [int(y) for y in _YEAR_RE.findall(source or "")]
    if not src_years:
        return []          # nothing to check backdating against; fail open
    earliest = min(src_years)
    draft_years = sorted(set(int(y) for y in _YEAR_RE.findall(draft or "")))
    return [f'backdated year: draft says "{y}" but the earliest year in the '
            f'source is {earliest}' for y in draft_years if y < earliest]


def gate_free_text(source: str, draft: str) -> Tuple[str, bool]:
    """Check an LLM-authored answer against the candidate source before it can be used.

    Gates on fabrications only (not omissions), ensuring the response does not invent
    credentials, dates, or employers beyond the provided source record. Fails closed
    if the source is empty or if fabrications are detected.

    Known limitations and uncaught fabrication classes:
    This gate checks for omissions/alterations against the source (such as dropped credentials,
    altered date ranges, or backdated years). It does NOT detect wholly invented claims
    that do not collide with known structured shapes, such as a fabricated employer name,
    fabricated job title, fabricated degree or field of study, invented publication,
    invented award name, invented skill, or forward-dated year.
    """
    from .source_fidelity import check_fidelity

    if not (draft or "").strip():
        return "", False
    if not (source or "").strip():
        log.warning("fidelity gate: no source text to verify against; "
                   "refusing to treat an unverifiable draft as clean: %r",
                   draft[:120])
        return draft, False
    try:
        report = check_fidelity(source, draft)
    except Exception as e:  # noqa: BLE001
        log.info("fidelity gate errored (%s); refusing the draft", e)
        return draft, False
    defects = list(report.fabrications) + _fabricated_years(source, draft)
    if defects:
        log.info("fidelity gate rejected draft (%s): %s",
                 "; ".join(defects), draft[:120])
        return draft, False
    return draft, True


def apply_free_text_answer(q: "FormQuestion", source: str, draft: str) -> None:
    """Set a free-text answer only if it survives the fidelity gate.

    A failed gate stores the draft as `guess` and flags needs_user. It must
    never populate `answer`, because a confident-looking wrong answer is worse
    than a blank one."""
    text, passed = gate_free_text(source, draft)
    if passed:
        q.answer = text
        q.answer_source = "ai"
        q.needs_user = False
        return
    q.answer = ""
    q.guess = draft
    q.needs_user = True


def _ai_answer(batch: List[FormQuestion], resume: str, job_title: str,
               company: str, job_description: str) -> None:
    from . import ai_client as gc

    def fmt(q: FormQuestion) -> str:
        if q.options:
            return (f"{q.text} (You MUST answer with exactly one of: "
                    f"{' | '.join(q.options)})")
        return q.text

    formatted = [fmt(q) for q in batch]
    try:
        answers = gc.answer_application_questions(
            resume, job_title, company, job_description, formatted) or {}
    except Exception as e:  # noqa: BLE001
        log.warning("AI question answering failed (%s) — leaving blanks.", e)
        answers = {}

    # Match tolerantly on a normalized key: local models return the answer
    # under a drifted key (numbering added, options-hint dropped), so an exact
    # dict lookup silently loses answers on big forms.
    norm = {}
    for k, v in answers.items():
        norm.setdefault(_norm_key(k), v)

    def _flatten(raw) -> str:
        """Render a model answer as text a human would paste into a form.

        A question like "write 3 bullet points" makes the model reply with a
        JSON list (or, less often, an object). `str()` on those yields a Python
        repr - "['Developed a model...', 'Purified RNA...']" - which reached a
        real apply kit for job 5793 and would have gone to the employer with
        the brackets and quotes intact. Flatten instead of stringifying.
        """
        if raw is None or isinstance(raw, bool):
            return ""
        if isinstance(raw, str):
            return raw.strip()
        if isinstance(raw, dict):
            # Values carry the content; the keys are the model's own labels
            # ("point1", "bullet_2") and are noise to the employer.
            raw = list(raw.values())
        if isinstance(raw, (list, tuple)):
            items = [_flatten(v) for v in raw]
            items = [i for i in items if i]
            if not items:
                return ""
            if len(items) == 1:
                return items[0]
            # Newline-separated so multi-line textareas keep the structure the
            # question asked for. No bullet glyphs: the form usually adds its
            # own, and a stray "- " reads as boilerplate.
            return "\n".join(items)
        return str(raw).strip()

    def _pick(mapping: dict, q: FormQuestion, key: str) -> str:
        nm = {_norm_key(k): v for k, v in mapping.items()}
        raw = (mapping.get(key) or mapping.get(q.text)
               or nm.get(_norm_key(key)) or nm.get(_norm_key(q.text)) or "")
        return _flatten(raw)

    def _retry_one(q: FormQuestion, key: str) -> str:
        # A focused single-question ask recovers option selects that the big
        # batch returned empty (the common local-model failure on long forms).
        try:
            res = gc.answer_application_questions(
                resume, job_title, company, job_description, [key]) or {}
        except Exception:  # noqa: BLE001
            return ""
        return _pick(res, q, key)

    for q, key in zip(batch, formatted):
        raw = _pick(answers, q, key)
        # A focused single-question retry recovers answers the big batch
        # dropped — both option selects and required free-text (the common
        # local-model failure on long forms).
        if not raw and (q.options or q.required):
            raw = _retry_one(q, key)
        if q.options:
            snapped = _snap_to_options(raw, q.options) if raw else ""
            if snapped:
                q.answer, q.answer_source = snapped, "ai"
            else:
                dflt = _default_for(q)
                if dflt:
                    q.answer, q.answer_source = dflt, "ai-default"
        elif raw:
            # Free text has no option list to snap onto, so it is the one
            # answer shape the model can fabricate unconstrained. Gate it
            # against the resume before it can reach `q.answer`.
            apply_free_text_answer(q, source=resume, draft=raw)


# =====================  SERIALIZATION  ======================================

def questions_to_json(questions: List[FormQuestion], source: str) -> str:
    return json.dumps({"source": source,
                       "questions": [asdict(q) for q in questions]},
                      indent=2, ensure_ascii=False)


def questions_from_json(raw: str) -> Tuple[List[FormQuestion], str]:
    if not raw:
        return [], ""
    try:
        data = json.loads(raw)
    except json.JSONDecodeError:
        return [], ""
    qs = []
    for d in data.get("questions", []):
        known = {f for f in FormQuestion.__dataclass_fields__}
        qs.append(FormQuestion(**{k: v for k, v in d.items() if k in known}))
    return qs, data.get("source", "")


def kit_markdown(job_title: str, company: str, url: str,
                 fields: dict, questions: List[FormQuestion],
                 source: str, resume_path: str, cover_letter_path: str) -> str:
    """Human-readable copy/paste sheet written next to the JSON package."""
    lines = [f"# Apply Kit — {job_title} @ {company}", "",
             f"- **Apply URL:** {url}",
             f"- **Resume:** `{resume_path}`",
             f"- **Cover letter:** `{cover_letter_path}`",
             f"- **Questions source:** {source or 'common defaults (form not readable)'}",
             "", "## Your details", ""]
    for k, v in fields.items():
        if v:
            lines.append(f"- **{k.replace('_', ' ').title()}:** {v}")
    lines += ["", "## Form questions", ""]
    for i, q in enumerate(questions, 1):
        req = " *(required)*" if q.required else ""
        lines.append(f"### {i}. {q.text}{req}")
        if q.options:
            lines.append(f"*Options: {' | '.join(q.options)}*")
        if q.answer:
            lines.append("")
            lines.append(q.answer)
            lines.append(f"\n*— source: {q.answer_source}*")
        else:
            lines.append("\n> ⚠️ **NEEDS YOUR ANSWER**")
        lines.append("")
    return "\n".join(lines)
