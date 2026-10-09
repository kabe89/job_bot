"""Workday CXS public scraper.

Most large biotechs
publish jobs through Workday tenants with a public JSON endpoint:

  POST https://<tenant>.wdN.myworkdayjobs.com/wday/cxs/<tenant>/<board>/jobs

Each entry in `data/workday_targets.txt` is one CSV line:
  tenant,board,display_name[,wd_host]
"""
from __future__ import annotations

import logging
import re
import time
from datetime import datetime
from pathlib import Path
from typing import List

import requests

from ..config import settings
from . import RawJob, registry

log = logging.getLogger("jobbot.scrapers.workday")

TARGETS_FILE = Path("data/workday_targets.txt")

# Default targets (empty by default; configure custom targets in data/workday_targets.txt).
DEFAULT_TARGETS: list[tuple[str, str, str, str]] = []



def _load_targets():
    out = list(DEFAULT_TARGETS)
    if TARGETS_FILE.exists():
        for line in TARGETS_FILE.read_text(encoding="utf-8").splitlines():
            line = line.strip()
            if not line or line.startswith("#"):
                continue
            parts = [p.strip() for p in line.split(",")]
            if len(parts) >= 3:
                tenant, board, name = parts[0], parts[1], parts[2]
                host = parts[3] if len(parts) > 3 else "wd1"
                out.append((tenant, board, name, host))
    return out


HEADERS = {
    "User-Agent": "JobBot/1.0",
    "Accept": "application/json",
    "Content-Type": "application/json",
}


_TAG_BREAK = re.compile(r"</(li|p|div|ul|ol|h[1-6]|tr)>", re.I)
_TAG_ANY = re.compile(r"<[^>]+>")
_BLANKS = re.compile(r"\n{3,}")
_JOB_URL = re.compile(
    r"https?://(?P<tenant>[^.]+)\.(?P<host>wd\d+)\.myworkdayjobs\.com/"
    r"(?:[a-zA-Z-]+/)?(?P<board>[^/]+)(?P<path>/job/[^?#]+)",
    re.IGNORECASE,
)


def _html_to_text(raw: str) -> str:
    """Flatten a Workday jobDescription HTML blob into readable plain text."""
    import html as _html

    text = _TAG_BREAK.sub("\n", raw or "")
    text = _TAG_ANY.sub("", text)
    text = _html.unescape(text)
    # Workday embeds NBSP/narrow-NBSP that break cp1252 consoles downstream.
    text = text.replace(" ", " ").replace(" ", " ").replace("’", "'")
    return _BLANKS.sub("\n\n", text).strip()


def fetch_posting(url: str, timeout: int = 30) -> tuple:
    """Fetch one Workday posting's JD body. Returns (body, state).

    state is one of:
      "ok"    -- body is the real JD text
      "gone"  -- the posting has been PULLED; the job is dead and can be expired
      "error" -- transient (network, 5xx, 429, empty body). Retry later; the
                 caller must NOT treat this as a dead job.

    A pulled Workday posting answers the detail endpoint with 403 + errorCode
    "S22"/"permission denied" -- NOT 404 (verified live: the public page for such
    a URL renders "The page you are looking for doesn't exist"). A 403 WITHOUT
    that marker is treated as transient, so a WAF/bot-block cannot mass-expire a
    live board.
    """
    m = _JOB_URL.match((url or "").strip())
    if not m:
        return "", "error"
    cxs = (f"https://{m['tenant']}.{m['host']}.myworkdayjobs.com/wday/cxs/"
           f"{m['tenant']}/{m['board']}{m['path']}")
    try:
        r = requests.get(cxs, headers={"User-Agent": "Mozilla/5.0 (Windows NT 10.0; Win64; x64) AppleWebKit/537.36 (KHTML, like Gecko) Chrome/122.0.0.0 Safari/537.36 JobBot/1.0",
                                       "Accept": "application/json"}, timeout=timeout)
        try:
            payload = r.json() or {}
        except Exception:  # noqa: BLE001 - non-JSON error page
            payload = {}
        if r.status_code == 404:
            return "", "gone"
        if r.status_code == 403:
            marker = (str(payload.get("errorCode", "")) + " "
                      + str(payload.get("message", ""))).lower()
            if "s22" in marker or "permission denied" in marker:
                return "", "gone"
            return "", "error"          # unrecognised 403 -> assume transient
        if r.status_code != 200:
            return "", "error"
        info = payload.get("jobPostingInfo") or {}
        # Workday explicitly sets canApply=False when a posting is closed/no longer accepting
        if info.get("canApply") is False:
            return "", "gone"
    except Exception as e:  # noqa: BLE001
        log.debug("workday fetch_posting(%s) failed: %s", url, e)
        return "", "error"
    body = _html_to_text(info.get("jobDescription") or "")
    if not body:
        return "", "error"
    head = [f"{info.get('title', '')}".strip(),
            f"Req: {info.get('jobReqId', '')}".strip(),
            f"Location: {info.get('location', '')}".strip()]
    text = "\n".join([h for h in head if h.split(":")[-1].strip()] + ["", body])
    return text[:20000], "ok"


def fetch_description(url: str, timeout: int = 30) -> str:
    """The real JD body for one Workday posting, or "" if unavailable.

    The search endpoint (`_query`) returns listing CARDS only -- title, location,
    postedOn -- with no description. The body lives behind a second per-job CXS
    call. Fail-open by contract: a missing JD must never lose a job. Use
    fetch_posting() when you need to know WHY it was unavailable.
    """
    return fetch_posting(url, timeout=timeout)[0]


def _query(tenant: str, board: str, host: str, keyword: str) -> list:
    url = f"https://{tenant}.{host}.myworkdayjobs.com/wday/cxs/{tenant}/{board}/jobs"
    out: list = []
    max_off = max(20, int(settings.max_workday_offset))
    # Paginate through all available postings (Workday caps per-request at 20)
    for offset in range(0, max_off, 20):
        payload = {
            "appliedFacets": {},
            "limit": 20,
            "offset": offset,
            "searchText": keyword,
        }
        try:
            r = requests.post(url, headers=HEADERS, json=payload, timeout=15)
            if r.status_code != 200:
                break
            batch = r.json().get("jobPostings", []) or []
        except Exception as e:  # noqa: BLE001
            log.debug("workday %s/%s offset=%d failed: %s", tenant, board, offset, e)
            break
        if not batch:
            break
        out.extend(batch)
        if len(batch) < 20:
            break
    return out


def _scrape_one_tenant(
    tenant: str, board: str, display: str, host: str, kw_terms: list[str]
) -> List[RawJob]:
    """Scrape all keywords for a single Workday tenant. Runs in a worker thread."""
    out: List[RawJob] = []
    seen_paths: set[str] = set()
    for kw in kw_terms:
        for j in _query(tenant, board, host, kw):
            path = j.get("externalPath", "")
            # Dedupe BEFORE the detail fetch: one posting surfaces under many
            # keywords, and each fetch is a real HTTP round-trip.
            if not path or path in seen_paths:
                continue
            seen_paths.add(path)
            full_url = f"https://{tenant}.{host}.myworkdayjobs.com/en-US/{board}{path}"
            title = j.get("title", "")
            location = j.get("locationsText", "") or j.get("primaryLocation", "")
            posted = j.get("postedOn", "") or ""
            # The card carries no description. Keep it only as the fallback --
            # scoring a job against this metadata stub is meaningless.
            stub = "\n".join([
                f"Job ID: {j.get('bulletFields', [''])[0] if j.get('bulletFields') else ''}",
                f"Location: {location}",
                f"Posted: {posted}",
                f"Apply: {full_url}",
            ])[:8000]

            description = ""
            if settings.workday_fetch_descriptions:
                try:
                    description = fetch_description(full_url)
                except Exception as e:  # noqa: BLE001 - a bad detail must not lose the job
                    log.debug("workday detail fetch failed for %s: %s", full_url, e)
                if settings.workday_detail_delay:
                    time.sleep(settings.workday_detail_delay)

            out.append(RawJob(
                source=f"workday:{tenant}",
                title=title[:240],
                company=display,
                url=full_url,
                location=_clean_location(location),
                description=description or stub,
                posted_at=None,
            ))
    return out


def scrape_workday(keywords: List[str]) -> List[RawJob]:
    """Scrape all Workday tenants in parallel (one thread per tenant)."""
    from concurrent.futures import ThreadPoolExecutor, as_completed as _as_completed
    targets = _load_targets()
    kw_terms = keywords or [""]
    out: List[RawJob] = []
    # Up to 6 tenants at once — enough to saturate bandwidth without hammering
    # any single domain.
    with ThreadPoolExecutor(max_workers=6, thread_name_prefix="workday") as ex:
        futures = {
            ex.submit(_scrape_one_tenant, tenant, board, display, host, kw_terms): tenant
            for tenant, board, display, host in targets
        }
        for fut in _as_completed(futures, timeout=180):
            name = futures[fut]
            try:
                results = fut.result()
                out.extend(results)
                log.debug("workday:%s -- %d postings", name, len(results))
            except Exception as e:  # noqa: BLE001
                log.info("workday:%s failed: %s", name, e)
    log.info("Workday: %d total postings across %d tenants", len(out), len(targets))
    return out


def _clean_location(loc: str) -> str:
    if not loc:
        return ""
    # Workday joins multiple sites with " | " — keep first 2 for readability
    parts = [p.strip() for p in loc.split("|") if p.strip()]
    return " | ".join(parts[:2])[:200]


registry.register(scrape_workday)
