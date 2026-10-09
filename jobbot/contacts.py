"""Contact finder — locate recruiters / hiring-team members for a job.

Public sources only, no logins.  All network calls use:
  - requests + BeautifulSoup
  - single attempt per URL
  - timeout from settings.apply_fetch_timeout
  - desktop UA header
  - degrade to [] on ANY failure (never raises out of find_contacts)

API
---
find_contacts(job_id, max_contacts=None, save=True) -> List[dict]
load_contacts(job_id)                               -> List[dict]
networking_note(job_id, contact)                    -> str   (<=300 chars)
networking_email(job_id, contact)                   -> {subject, body}
"""
from __future__ import annotations

import json
import logging
import re
from datetime import datetime, timezone
from typing import List, Optional

import requests
from bs4 import BeautifulSoup

from .auto_apply import EMAIL_RE, extract_recruiter_email
from .config import settings
from .models import Job, init_db, session

log = logging.getLogger("jobbot.contacts")

# Quiet the chatty HTTP clients that ddgs uses for each search engine, so bulk
# contact prefetch doesn't flood the console with per-request INFO lines.
for _noisy in ("ddgs", "httpx", "primp", "urllib3"):
    logging.getLogger(_noisy).setLevel(logging.WARNING)

# ---------------------------------------------------------------------------
# Constants
# ---------------------------------------------------------------------------
UA = (
    "Mozilla/5.0 (Windows NT 10.0; Win64; x64) AppleWebKit/537.36 "
    "(KHTML, like Gecko) Chrome/131.0.0.0 Safari/537.36"
)
HEADERS = {
    "User-Agent": UA,
    "Accept-Language": "en-US,en;q=0.9",
    "Accept-Encoding": "gzip, deflate",
}

# Junk email prefixes that should not be treated as recruiter contacts.
JUNK_EMAIL_PREFIXES = (
    "noreply", "no-reply", "donotreply", "do-not-reply", "postmaster",
    "support", "sales", "marketing", "newsletter", "webmaster", "info@indeed",
    "abuse", "privacy", "press", "careers@", "jobs@", "apply@",
)

# LinkedIn title parsing: "First Last - Role Title - Company | LinkedIn"
# Also handles "First Last | Role Title | Company - LinkedIn"
LI_TITLE_RE = re.compile(
    r"^(?P<name>[A-Z][^|–\-]{1,60?}?)\s*[-–|]\s*(?P<role>[^|–\-]{2,80?}?)\s*[-–|]",
    re.UNICODE,
)

# A slightly looser fallback: just capture name up to the first separator
LI_NAME_ONLY_RE = re.compile(
    r"^(?P<name>[A-Z][A-Za-z\s'\-\.]{1,50}?)\s*[-–|]",
    re.UNICODE,
)


# ---------------------------------------------------------------------------
# Web-search helper (DuckDuckGo HTML, fallback to Bing)
# Returns list of {title, url, snippet}
# ---------------------------------------------------------------------------

def serper_configured() -> bool:
    """True if a Serper.dev API key is set in settings/.env."""
    return bool(getattr(settings, "serper_api_key", ""))


def cse_configured() -> bool:
    """True if both Google CSE key and engine ID are set in settings/.env."""
    return bool(getattr(settings, "google_cse_key", "")
                and getattr(settings, "google_cse_cx", ""))


def search_configured() -> bool:
    """True if ANY usable search backend is available: Serper, Google CSE, or
    the keyless `ddgs` library.

    Gates dashboard contact auto-populate so it never falls through to the
    bot-blocked raw-HTML scrapers on a page load.
    """
    return serper_configured() or cse_configured() or _ddgs_available()


def active_search_provider() -> str:
    """Name of the search backend that would be used: serper > cse > ddgs > html."""
    if serper_configured():
        return "serper"
    if cse_configured():
        return "cse"
    if _ddgs_available():
        return "ddgs"
    return "html"


def search_selftest(query: str = "site:linkedin.com/in recruiter") -> dict:
    """Probe the active search backend (Serper preferred, then Google CSE).

    Returns {ok, provider, configured, count, sample, error}. Makes exactly
    one live API call when a backend is configured. Used by `cse-test`.
    """
    provider = active_search_provider()
    out = {"ok": False, "provider": provider, "configured": search_configured(),
           "count": 0, "sample": [], "error": ""}
    if not out["configured"]:
        out["error"] = ("No search backend available. Install the keyless "
                        "'ddgs' library (pip install ddgs), or set "
                        "SERPER_API_KEY / GOOGLE_CSE_KEY + GOOGLE_CSE_CX in .env.")
        return out
    timeout = getattr(settings, "apply_fetch_timeout", 30)
    try:
        if provider == "ddgs":
            rows = _ddgs_search(query, 3)
            out["sample"] = [r["url"] for r in rows]
            out["count"] = len(rows)
            out["ok"] = bool(rows)
            if not rows:
                out["error"] = ("ddgs returned no results (DuckDuckGo may be "
                                "rate-limiting; retry shortly).")
            return out
        if provider == "serper":
            resp = requests.post(
                "https://google.serper.dev/search",
                headers={"X-API-KEY": settings.serper_api_key,
                         "Content-Type": "application/json"},
                json={"q": query, "num": 3},
                timeout=timeout,
            )
            if resp.status_code != 200:
                out["error"] = f"HTTP {resp.status_code}: {resp.text[:200]}"
                return out
            items = resp.json().get("organic", [])
            out["sample"] = [it.get("link", "") for it in items[:3]]
        else:  # cse
            resp = requests.get(
                "https://www.googleapis.com/customsearch/v1",
                params={"key": settings.google_cse_key,
                        "cx": settings.google_cse_cx, "q": query, "num": 3},
                timeout=timeout,
            )
            if resp.status_code != 200:
                try:
                    msg = resp.json().get("error", {}).get("message", "")
                except Exception:  # noqa: BLE001
                    msg = resp.text[:200]
                out["error"] = f"HTTP {resp.status_code}: {msg}"
                return out
            items = resp.json().get("items", [])
            out["sample"] = [it.get("link", "") for it in items[:3]]
        out["count"] = len(out["sample"])
        out["ok"] = True
        return out
    except Exception as exc:  # noqa: BLE001
        out["error"] = str(exc)
        return out


# Back-compat alias (cli/web import this name).
def cse_selftest(query: str = "site:linkedin.com/in recruiter") -> dict:
    return search_selftest(query)


def _serper_search(query: str, max_results: int = 8) -> List[dict]:
    """Serper.dev (Google results via one API key). Preferred backend — no
    Cloud Console / API-enablement dance, just a key. Returns {title,url,snippet}."""
    key = getattr(settings, "serper_api_key", "")
    if not key:
        return []
    try:
        resp = requests.post(
            "https://google.serper.dev/search",
            headers={"X-API-KEY": key, "Content-Type": "application/json"},
            json={"q": query, "num": min(max_results, 10)},
            timeout=getattr(settings, "apply_fetch_timeout", 30),
        )
        resp.raise_for_status()
        return [{"title": it.get("title", ""), "url": it.get("link", ""),
                 "snippet": it.get("snippet", "")}
                for it in resp.json().get("organic", [])][:max_results]
    except Exception as exc:  # noqa: BLE001
        log.debug("Serper search failed for %r: %s", query, exc)
        return []


def _cse_search(query: str, max_results: int = 8) -> List[dict]:
    """Google Programmable Search (official JSON API). Used first when
    GOOGLE_CSE_KEY/GOOGLE_CSE_CX are configured — HTML search engines
    bot-block plain requests (DDG returns a 202 challenge page), so the
    API is the only reliable path for LinkedIn contact discovery."""
    key = getattr(settings, "google_cse_key", "")
    cx = getattr(settings, "google_cse_cx", "")
    if not key or not cx:
        return []
    try:
        resp = requests.get(
            "https://www.googleapis.com/customsearch/v1",
            params={"key": key, "cx": cx, "q": query, "num": min(max_results, 10)},
            timeout=getattr(settings, "apply_fetch_timeout", 30),
        )
        resp.raise_for_status()
        return [{"title": it.get("title", ""), "url": it.get("link", ""),
                 "snippet": it.get("snippet", "")}
                for it in resp.json().get("items", [])][:max_results]
    except Exception as exc:  # noqa: BLE001
        log.debug("CSE search failed for %r: %s", query, exc)
        return []


def _ddgs_available() -> bool:
    """True if the keyless `ddgs` library is importable."""
    import importlib.util
    return importlib.util.find_spec("ddgs") is not None


def _ddgs_search(query: str, max_results: int = 8) -> List[dict]:
    """Keyless DuckDuckGo search via the `ddgs` library (rotates endpoints /
    headers, so it gets through where plain requests hit the 202 challenge).
    Free, no API key. Returns {title,url,snippet}; [] on any failure."""
    try:
        from ddgs import DDGS
    except Exception:  # noqa: BLE001 — not installed
        return []
    try:
        rows = list(DDGS().text(query, max_results=max_results))
        return [{"title": r.get("title", ""), "url": r.get("href", ""),
                 "snippet": r.get("body", "")}
                for r in rows if r.get("href")][:max_results]
    except Exception as exc:  # noqa: BLE001
        log.debug("ddgs search failed for %r: %s", query, exc)
        return []


def _web_search(query: str, max_results: int = 8) -> List[dict]:
    """Minimal web search. Backend order: Serper.dev -> Google CSE -> ddgs
    (keyless) -> DuckDuckGo HTML -> Bing. The API backends are most reliable;
    `ddgs` is the best keyless option; the raw-HTML fallbacks are frequently
    bot-blocked.

    Returns up to *max_results* dicts with keys: title, url, snippet.
    Returns [] on any error — never raises.
    """
    # Ollama hosted web search first when a key is configured (live results).
    try:
        from . import ollama_search
        if ollama_search.is_available():
            results = ollama_search.web_search(query, max_results)
            if results:
                return results
    except Exception as exc:  # noqa: BLE001
        log.debug("Ollama web search failed for %r: %s", query, exc)

    results: List[dict] = _serper_search(query, max_results)
    if results:
        return results
    results = _cse_search(query, max_results)
    if results:
        return results
    results = _ddgs_search(query, max_results)
    if results:
        return results
    timeout = getattr(settings, "apply_fetch_timeout", 30)

    # --- DuckDuckGo HTML -----------------------------------------------
    try:
        resp = requests.get(
            "https://html.duckduckgo.com/html/",
            params={"q": query},
            headers=HEADERS,
            timeout=timeout,
        )
        if resp.status_code == 200:
            soup = BeautifulSoup(resp.text, "html.parser")
            for result_div in soup.select(".result"):
                title_tag = result_div.select_one(".result__title a")
                snippet_tag = result_div.select_one(".result__snippet")
                if not title_tag:
                    continue
                title = title_tag.get_text(strip=True)
                href = title_tag.get("href", "")
                # DDG uses redirect URLs; extract real URL
                real_url = _ddg_extract_url(href)
                snippet = snippet_tag.get_text(strip=True) if snippet_tag else ""
                if real_url and title:
                    results.append({"title": title, "url": real_url, "snippet": snippet})
                if len(results) >= max_results:
                    break
    except Exception as exc:  # noqa: BLE001
        log.debug("DDG search failed for %r: %s", query, exc)

    if results:
        return results[:max_results]

    # --- Bing fallback --------------------------------------------------
    try:
        resp = requests.get(
            "https://www.bing.com/search",
            params={"q": query, "first": 1},
            headers=HEADERS,
            timeout=timeout,
        )
        if resp.status_code == 200:
            soup = BeautifulSoup(resp.text, "html.parser")
            for li in soup.select("li.b_algo"):
                a = li.select_one("h2 a") or li.select_one("a")
                if not a:
                    continue
                title = a.get_text(strip=True)
                href = a.get("href", "")
                snippet_tag = li.select_one(".b_caption p") or li.select_one("p")
                snippet = snippet_tag.get_text(strip=True) if snippet_tag else ""
                if href.startswith("http") and title:
                    results.append({"title": title, "url": href, "snippet": snippet})
                if len(results) >= max_results:
                    break
    except Exception as exc:  # noqa: BLE001
        log.debug("Bing search failed for %r: %s", query, exc)

    return results[:max_results]


def _ddg_extract_url(href: str) -> str:
    """Extract real URL from a DuckDuckGo redirect href."""
    if not href:
        return ""
    # Typical form: //duckduckgo.com/l/?uddg=https%3A%2F%2F...
    match = re.search(r"uddg=([^&]+)", href)
    if match:
        from urllib.parse import unquote
        return unquote(match.group(1))
    if href.startswith("http"):
        return href
    return ""


# ---------------------------------------------------------------------------
# LinkedIn result parser
# ---------------------------------------------------------------------------

# Generic corporate suffixes / filler that don't identify a company.
_COMPANY_STOP = {
    "inc", "llc", "corp", "corporation", "co", "company", "ltd", "plc", "gmbh",
    "pharmaceuticals", "pharmaceutical", "pharma", "therapeutics", "therapeutic",
    "biosciences", "bioscience", "sciences", "science", "technologies",
    "technology", "labs", "laboratories", "laboratory", "bio", "biotech",
    "biotechnology", "group", "holdings", "health", "healthcare", "medical",
    "medicines", "medicine", "systems", "solutions", "the", "and",
    # Generic institutional/academic words — as non-distinctive as the corporate
    # suffixes above. Without these, "Northwest Medical Center" and "Metro Technology
    # Center" collide on the lone shared token "center" and produce false
    # warm-intro matches (referrals.contacts_for_job).
    "center", "centre", "institute", "institutes", "university", "college",
    "hospital", "hospitals", "clinic", "school", "foundation", "department",
    "division", "of",
}


def _company_tokens(name: str) -> List[str]:
    """Distinctive lowercase tokens of a company name (drops corp suffixes)."""
    toks = re.split(r"[^a-z0-9]+", (name or "").lower())
    return [t for t in toks if t and t not in _COMPANY_STOP and len(t) > 1]


def _company_in_text(company: str, text: str) -> bool:
    """True if ALL distinctive token(s) of *company* appear as words in *text*.

    Used to verify a LinkedIn hit's CURRENT employer (from their headline)
    actually matches the target company — instead of accepting any result that
    merely mentions it somewhere in a noisy snippet (past jobs, "people also
    viewed", etc.).
    """
    toks = _company_tokens(company)
    if not toks:
        return False
    tl = (text or "").lower()
    return all(re.search(r"\b" + re.escape(t) + r"\b", tl) for t in toks)


def _parse_linkedin_result(result: dict) -> Optional[dict]:
    """Parse a search result into a contact dict if it looks like a LinkedIn
    profile. Returns name, role, and `company_text` (the headline after the
    name, used to verify the current employer) — plus the raw snippet."""
    url = result.get("url", "")
    title = result.get("title", "")
    snippet = result.get("snippet", "")

    # Must be a personal profile URL.
    if "linkedin.com/in/" not in url.lower():
        return None

    # Strip trailing "... | LinkedIn" / "- LinkedIn" / "| Professional Profile".
    headline = re.sub(r"\s*[|\-–·]\s*linkedin(\.com)?.*$", "", title, flags=re.IGNORECASE).strip()
    headline = re.sub(r"\s*[|\-–·]\s*professional profile.*$", "", headline, flags=re.IGNORECASE).strip()

    # Split "Name - Role - Company" on separators; first chunk is the name.
    parts = [p.strip() for p in re.split(r"\s+[-–|·]\s+", headline) if p.strip()]
    if not parts:
        return None
    name = parts[0]
    # Name sanity: starts with a letter, <=5 words, letters/spaces/.'- only.
    if not re.match(r"^[A-Za-z][A-Za-z.'’\- ]+$", name) or len(name.split()) > 5:
        return None

    tail = " - ".join(parts[1:]).strip()   # role and/or current company
    return {
        "name": name,
        "role": tail or "LinkedIn",
        "company_text": tail,
        "snippet": snippet,
        "email": "",
        "linkedin": url,
        "source_url": url,
        "found_at": datetime.now(tz=timezone.utc).isoformat(),
    }


def _verify_and_clean(c: Optional[dict], company: str) -> Optional[dict]:
    """Keep a parsed contact only if its CURRENT employer — taken from the
    LinkedIn headline (the title segment after the name) — matches *company*.

    The headline reflects the person's present role/company. We deliberately do
    NOT fall back to the snippet: snippets list past employers and "people also
    viewed", which is exactly what produced unrelated recruiters before.
    Strips internal temp fields.
    """
    if not c:
        return None
    if not _company_in_text(company, c.get("company_text", "")):
        return None
    c.pop("company_text", None)
    c.pop("snippet", None)
    return c


# ---------------------------------------------------------------------------
# Source A: emails from job description
# ---------------------------------------------------------------------------

def _contacts_from_description(job: Job) -> List[dict]:
    """Extract any plausible recruiter/contact emails from the job description."""
    text = job.description or ""
    contacts: List[dict] = []
    seen: set = set()
    for m in EMAIL_RE.finditer(text):
        addr = m.group(0).lower()
        if addr in seen:
            continue
        seen.add(addr)
        if any(j in addr for j in JUNK_EMAIL_PREFIXES):
            continue
        contacts.append({
            "name": "",
            "role": "from posting",
            "email": addr,
            "linkedin": "",
            "source_url": job.url or "",
            "found_at": datetime.now(tz=timezone.utc).isoformat(),
        })
    return contacts


# ---------------------------------------------------------------------------
# Source B: LinkedIn recruiter search
# ---------------------------------------------------------------------------

def _contacts_from_recruiter_search(job: Job) -> List[dict]:
    """Search LinkedIn for recruiters / talent-acquisition contacts at the company."""
    company = (job.company or "").strip()
    if not company:
        return []

    query = f'"{company}" ("recruiter" OR "talent acquisition") site:linkedin.com/in'
    results = _web_search(query, max_results=8)
    contacts: List[dict] = []

    for r in results:
        c = _verify_and_clean(_parse_linkedin_result(r), company)
        if c:
            contacts.append(c)

    return contacts


# ---------------------------------------------------------------------------
# Source C: team search (hiring manager / team peer)
# ---------------------------------------------------------------------------

def _contacts_from_team_search(job: Job) -> List[dict]:
    """Team search: search for team peers or hiring managers using the job title keywords."""
    company = (job.company or "").strip()
    title = (job.title or "").strip()
    if not company or not title:
        return []

    # Pull first 2-3 meaningful title words (skip stopwords)
    _STOPWORDS = {"of", "the", "and", "or", "in", "at", "a", "an", "for",
                  "to", "with", "senior", "junior", "staff", "lead", "principal",
                  "associate", "director", "manager"}
    words = [w for w in re.split(r"[\s/,\-]+", title) if w.lower() not in _STOPWORDS and len(w) > 2]
    title_fragment = " ".join(words[:3])
    if not title_fragment:
        return []

    query = f'"{company}" "{title_fragment}" site:linkedin.com/in'
    results = _web_search(query, max_results=8)
    contacts: List[dict] = []

    for r in results:
        c = _verify_and_clean(_parse_linkedin_result(r), company)
        if c:
            contacts.append(c)

    return contacts


# ---------------------------------------------------------------------------
# Deduplication
# ---------------------------------------------------------------------------

def _dedup(contacts: List[dict]) -> List[dict]:
    """Deduplicate contacts by (name.lower() or email)."""
    seen: set = set()
    out: List[dict] = []
    for c in contacts:
        key = (c.get("name") or "").lower().strip() or (c.get("email") or "").lower().strip()
        if not key:
            continue
        if key in seen:
            continue
        seen.add(key)
        out.append(c)
    return out


# ---------------------------------------------------------------------------
# Public API
# ---------------------------------------------------------------------------

def find_contacts(job_id: int, max_contacts: Optional[int] = None, save: bool = True) -> List[dict]:
    """Find recruiter / hiring-team contacts for *job_id*.

    Sources (in order):
      A - emails in the job description
      B - LinkedIn recruiter / talent-acquisition search
      C - LinkedIn team-member / hiring-manager search

    Returns a list of contact dicts.  Degrades to [] on any failure.
    If *save* is True, persists JSON to Job.contacts_json and commits.
    """
    if max_contacts is None:
        max_contacts = getattr(settings, "contacts_max_per_job", 3)

    try:
        init_db()
        with session() as db:
            job = db.get(Job, job_id)
            if not job:
                log.warning("find_contacts: job %d not found", job_id)
                return []
            # Snapshot immutable fields before session closes
            _job_snapshot = _JobSnapshot(job)

        contacts: List[dict] = []

        # Source A
        try:
            contacts.extend(_contacts_from_description(_job_snapshot))
        except Exception as exc:  # noqa: BLE001
            log.debug("Source A failed for job %d: %s", job_id, exc)

        # Source B
        try:
            contacts.extend(_contacts_from_recruiter_search(_job_snapshot))
        except Exception as exc:  # noqa: BLE001
            log.debug("Source B failed for job %d: %s", job_id, exc)

        # Source C
        try:
            contacts.extend(_contacts_from_team_search(_job_snapshot))
        except Exception as exc:  # noqa: BLE001
            log.debug("Source C failed for job %d: %s", job_id, exc)

        contacts = _dedup(contacts)[:max_contacts]

        if save:
            try:
                with session() as db:
                    job = db.get(Job, job_id)
                    if job:
                        job.contacts_json = json.dumps(contacts)
                        db.commit()
            except Exception as exc:  # noqa: BLE001
                log.warning("Could not save contacts for job %d: %s", job_id, exc)

        return contacts

    except Exception as exc:  # noqa: BLE001
        log.error("find_contacts failed for job %d: %s", job_id, exc)
        return []


def _has_contacts(contacts_json: Optional[str]) -> bool:
    """True if a job's contacts_json already holds at least one contact."""
    s = (contacts_json or "").strip()
    return bool(s) and s not in ("[]", "null")


def prefetch_contacts(limit: int = 20, min_score: float = 0.0,
                      refresh: bool = False, delay: float = 2.0,
                      progress=None) -> dict:
    """Bulk-find contacts for the top-scoring jobs that still lack them.

    Selects jobs with a company, ordered by match_score desc, skipping any that
    already have saved contacts (unless *refresh*). Runs find_contacts on each
    with a polite *delay* between jobs (DuckDuckGo rate-limits keyless bursts).

    *progress* — optional callable(done, total, job_title, n_found) for UIs.
    Returns {selected, processed, jobs_with_contacts, total_contacts}.
    """
    import time
    init_db()
    # Pick candidate job ids inside the session (fields don't survive close).
    with session() as db:
        q = db.query(Job).filter(Job.company != "")
        if min_score:
            q = q.filter(Job.match_score >= float(min_score))
        rows = q.order_by(Job.match_score.desc()).all()
        candidates = []
        for job in rows:
            if not refresh and _has_contacts(job.contacts_json):
                continue
            candidates.append((job.id, job.title or ""))
            if len(candidates) >= limit:
                break

    stats = {"selected": len(candidates), "processed": 0,
             "jobs_with_contacts": 0, "total_contacts": 0}
    for i, (jid, title) in enumerate(candidates):
        found = find_contacts(jid, save=True)
        stats["processed"] += 1
        if found:
            stats["jobs_with_contacts"] += 1
            stats["total_contacts"] += len(found)
        if progress:
            try:
                progress(stats["processed"], stats["selected"], title, len(found))
            except Exception:  # noqa: BLE001
                pass
        # Be polite to DuckDuckGo — skip the wait after the last job.
        if delay and i < len(candidates) - 1:
            time.sleep(delay)
    return stats


class _JobSnapshot:
    """Lightweight copy of Job fields so they survive after the session closes."""
    __slots__ = ("id", "title", "company", "url", "description")

    def __init__(self, job: Job) -> None:
        self.id = job.id
        self.title = job.title or ""
        self.company = job.company or ""
        self.url = job.url or ""
        self.description = job.description or ""


def networking_targets(job_id: int) -> List[dict]:
    """Ready-to-click people-search links for a job's company.

    These always work (the user clicks them in their own browser, so no
    bot-blocking) and complement the automated CSE search.  Returns a list of
    {label, url} dicts for finding recruiters, hiring managers, and team
    members on LinkedIn.
    """
    from urllib.parse import quote_plus
    title, company, _url, _desc = _job_fields(job_id)
    company = (company or "").strip()
    if not company:
        return []

    # First 2-3 meaningful words of the title for a "team member" search.
    _STOP = {"of", "the", "and", "or", "in", "at", "a", "an", "for", "to",
             "with", "senior", "junior", "staff", "lead", "principal",
             "associate", "director", "manager", "i", "ii", "iii"}
    words = [w for w in re.split(r"[\s/,\-]+", title or "")
             if w.lower() not in _STOP and len(w) > 2]
    title_frag = " ".join(words[:3])

    def _li(q: str) -> str:
        return "https://www.linkedin.com/search/results/people/?keywords=" + quote_plus(q)

    def _g(q: str) -> str:
        return "https://www.google.com/search?q=" + quote_plus(q)

    targets = [
        {"label": f"Recruiters / talent acquisition at {company}",
         "url": _li(f"{company} recruiter")},
        {"label": f"Hiring managers at {company}",
         "url": _li(f"{company} hiring manager")},
    ]
    if title_frag:
        targets.append({
            "label": f'Team members ("{title_frag}") at {company}',
            "url": _li(f"{company} {title_frag}")})
        targets.append({
            "label": f'Google: {title_frag} at {company} on LinkedIn',
            "url": _g(f'"{company}" "{title_frag}" site:linkedin.com/in')})
    targets.append({
        "label": f"{company} on LinkedIn (company page -> People tab)",
        "url": "https://www.linkedin.com/company/" +
               quote_plus(re.sub(r"[^a-z0-9]+", "-", company.lower()).strip("-"))})
    return targets


def load_contacts(job_id: int) -> List[dict]:
    """Load previously-saved contacts for *job_id*.  Returns [] on empty / bad JSON."""
    try:
        init_db()
        with session() as db:
            job = db.get(Job, job_id)
            if not job or not job.contacts_json:
                return []
            return json.loads(job.contacts_json)
    except Exception as exc:  # noqa: BLE001
        log.debug("load_contacts(%d) failed: %s", job_id, exc)
        return []


# ---------------------------------------------------------------------------
# AI-tailored networking-message generators (Gemini-first, deterministic fallback)
# ---------------------------------------------------------------------------

def _base_resume_text() -> str:
    """Load the candidate base resume as text (md/txt/pdf/docx), '' on failure.

    Uses resume.load_resume so a PDF base resume is extracted to text rather
    than read as raw bytes.  (The AI clients also inject data/profile.md as
    ground truth, so this is supplementary context.)
    """
    try:
        from pathlib import Path
        p = Path(getattr(settings, "base_resume_path", "data/base_resume.md"))
        if not p.exists():
            return ""
        from .resume import load_resume
        return (load_resume(p) or "")[:10000]
    except Exception as exc:  # noqa: BLE001
        log.debug("base resume load failed: %s", exc)
        return ""


def _job_fields(job_id: int) -> tuple[str, str, str, str]:
    """Return (title, company, url, description) for a job; blanks on failure."""
    try:
        init_db()
        with session() as db:
            job = db.get(Job, job_id)
            if job:
                return (job.title or "", job.company or "",
                        job.url or "", job.description or "")
    except Exception:  # noqa: BLE001
        pass
    return ("", "", "", "")


def _ai_networking(job_id: int, contact: dict, channel: str) -> Optional[dict]:
    """Generate a tailored message via AI. Prefers the Gemini backend (per
    project default), falls back to Claude, then to None so callers can use
    the deterministic template. Returns {"subject", "body"} or None."""
    title, company, _url, desc = _job_fields(job_id)
    resume = _base_resume_text()
    from . import gemini_client, claude_client
    # Gemini first by default; Claude as the safety net.
    backends = []
    if gemini_client.is_available():
        backends.append(gemini_client)
    if claude_client.is_available():
        backends.append(claude_client)
    for backend in backends:
        try:
            out = backend.networking_message(
                resume, title, company, desc, contact, channel=channel)
            if out and (out.get("body") or "").strip():
                return out
        except Exception as exc:  # noqa: BLE001
            log.warning("AI networking (%s) failed for job %d: %s",
                        getattr(backend, "__name__", "?"), job_id, exc)
            continue
    return None


def networking_note(job_id: int, contact: dict, use_ai: bool = True) -> str:
    """Generate a short LinkedIn connect-message note (<=300 chars).

    Tailored by AI (Gemini first) to the contact and target role when a key is
    configured; falls back to a deterministic template otherwise.
    """
    if use_ai:
        out = _ai_networking(job_id, contact, channel="linkedin")
        if out and out.get("body"):
            note = out["body"].strip()
            return note[:297] + "..." if len(note) > 300 else note
    return _fallback_note(job_id, contact)


def _fallback_note(job_id: int, contact: dict) -> str:
    """Deterministic LinkedIn note (no AI) — used when AI is unavailable."""
    # Load job for title / company
    role_title = ""
    company = ""
    try:
        init_db()
        with session() as db:
            job = db.get(Job, job_id)
            if job:
                role_title = job.title or ""
                company = job.company or ""
    except Exception:  # noqa: BLE001
        pass

    contact_name = (contact.get("name") or "").strip()
    applicant_title = getattr(settings, "applicant_title", "scientist")
    applicant_name = getattr(settings, "applicant_name", "")

    # Greeting
    greeting = f"Hi {contact_name}," if contact_name else "Hi,"

    role_clause = f" re the {role_title} role at {company}" if role_title and company else ""
    fit_clause = f" I am a {applicant_title} interested in connecting" if applicant_title else " I would love to connect"

    note = f"{greeting}{fit_clause}{role_clause}. Would you be open to connecting? Thank you!"
    # Trim to 300 chars if somehow over
    if len(note) > 300:
        note = note[:297] + "..."
    return note


def networking_email(job_id: int, contact: dict, use_ai: bool = True) -> dict:
    """Generate a longer networking email (subject + body).

    Tailored by AI (Gemini first) to the contact, role, and the candidate's
    real background when a key is configured; falls back to a deterministic
    template otherwise.  Returns {subject: str, body: str}.
    """
    if use_ai:
        out = _ai_networking(job_id, contact, channel="email")
        if out and out.get("body"):
            subject = (out.get("subject") or "").strip()
            if not subject:
                _t, _c, _u, _d = _job_fields(job_id)
                role_at = f"{_t} at {_c}" if _t and _c else (_t or _c or "your team")
                subject = f"Interested in the {role_at} position"
            return {"subject": subject, "body": out["body"].strip()}
    return _fallback_email(job_id, contact)


def _fallback_email(job_id: int, contact: dict) -> dict:
    """Deterministic networking email (no AI) — used when AI is unavailable."""
    role_title = ""
    company = ""
    job_url = ""
    try:
        init_db()
        with session() as db:
            job = db.get(Job, job_id)
            if job:
                role_title = job.title or ""
                company = job.company or ""
                job_url = job.url or ""
    except Exception:  # noqa: BLE001
        pass

    applicant_name = getattr(settings, "applicant_name", "Applicant")
    applicant_title = getattr(settings, "applicant_title", "scientist")
    applicant_email = getattr(settings, "applicant_email", "")
    applicant_linkedin = getattr(settings, "applicant_linkedin", "")
    applicant_location = getattr(settings, "applicant_location", "")

    contact_name = (contact.get("name") or "").strip()
    greeting = f"Hi {contact_name}," if contact_name else "Hello,"
    role_at = f"{role_title} at {company}" if role_title and company else (role_title or company or "your team")

    subject = f"Interested in the {role_at} position"

    url_line = f"\nPosting: {job_url}" if job_url else ""
    linkedin_line = f"\nLinkedIn: {applicant_linkedin}" if applicant_linkedin else ""
    location_line = f" based in {applicant_location}" if applicant_location else ""

    body = (
        f"{greeting}\n\n"
        f"My name is {applicant_name}, a {applicant_title}{location_line}. "
        f"I came across the {role_at} opening and am very interested in the opportunity.\n\n"
        f"I would love to learn more about the team and whether my background might be a good fit. "
        f"Would you have a few minutes to connect or chat briefly?\n\n"
        f"Thank you for your time!{url_line}"
        f"\n\nBest regards,\n{applicant_name}"
        f"\n{applicant_email}{linkedin_line}"
    )

    return {"subject": subject, "body": body}
