"""Unified JobSpy scraper — replaces the brittle LinkedIn + Indeed scrapers.

JobSpy (python-jobspy) handles LinkedIn, Indeed, Glassdoor, and ZipRecruiter
behind one call, with built-in rate-limit handling and full description retrieval.

Performance strategy:
  - Build a list of (keyword, location) combos.
  - Run up to JOBSPY_WORKERS concurrent scrape_jobs() calls via ThreadPoolExecutor.
  - Each call internally parallelises its configured site sources.

Expected outcome:
  - LinkedIn re-enabled (it was disabled due to the old manual scraper).
  - Full job descriptions (instead of "View at URL" stubs).
  - Runtime ~60-120 s instead of many minutes of blocked requests.

Tunable via .env:
  JOBSPY_SITES          — comma-separated: linkedin,indeed,glassdoor,zip_recruiter
  JOBSPY_HOURS_OLD      — only return jobs posted within this many hours (default 168 = 1 week)
  JOBSPY_RESULTS_WANTED — results per (keyword, location) call (default 30)
  JOBSPY_WORKERS        — concurrent scrape_jobs() calls (default 3)
  JOBSPY_FETCH_DESCRIPTIONS — True to fetch full LinkedIn descriptions (slower but richer)
"""
from __future__ import annotations

import logging
from concurrent.futures import ThreadPoolExecutor, as_completed
from datetime import datetime
from typing import List

from . import RawJob, registry
from ..config import settings

try:
    import truststore
    truststore.inject_into_ssl()
except Exception:
    pass

log = logging.getLogger("jobbot.scrapers.jobspy")


def _scrape_one(
    site_names: list[str],
    search_term: str,
    location: str,
    results_wanted: int,
    hours_old: int,
    fetch_descriptions: bool,
) -> list[dict]:
    """Single scrape_jobs() call — isolated so it can run in a thread."""
    try:
        from jobspy import scrape_jobs
        df = scrape_jobs(
            site_name=site_names,
            search_term=search_term,
            location=location,
            results_wanted=results_wanted,
            hours_old=hours_old,
            country_indeed="USA",
            linkedin_fetch_description=fetch_descriptions,
            verbose=0,
        )
        if df is None or df.empty:
            return []
        return df.where(df.notna(), other=None).to_dict("records")
    except Exception as e:  # noqa: BLE001
        log.warning("JobSpy batch scrape failed (sites=%s term=%r loc=%r): %s. Attempting fallback per site...", site_names, search_term, location, e)
        if len(site_names) > 1:
            records = []
            for site in site_names:
                try:
                    from jobspy import scrape_jobs
                    df = scrape_jobs(
                        site_name=[site],
                        search_term=search_term,
                        location=location,
                        results_wanted=results_wanted,
                        hours_old=hours_old,
                        country_indeed="USA",
                        linkedin_fetch_description=fetch_descriptions,
                        verbose=0,
                    )
                    if df is not None and not df.empty:
                        records.extend(df.where(df.notna(), other=None).to_dict("records"))
                except Exception as site_err:
                    log.info("JobSpy single-site fallback failed for %s: %s", site, site_err)
            return records
        return []


def _to_raw_job(r: dict) -> RawJob | None:
    """Convert a jobspy record dict to a RawJob. Returns None if unusable."""
    url = str(r.get("job_url") or r.get("job_url_direct") or "")
    title = str(r.get("title") or "")[:240]
    if not url or not title:
        return None

    company = str(r.get("company") or "")[:200]
    location_str = str(r.get("location") or "")[:200]
    description = str(r.get("description") or "")[:20000]
    site = str(r.get("site") or "jobspy")

    # Build a human-readable salary string when min/max amounts are present.
    salary = ""
    mn, mx = r.get("min_amount"), r.get("max_amount")
    if mn is not None or mx is not None:
        interval = r.get("interval") or "year"
        try:
            if mn is not None and mx is not None:
                salary = f"${float(mn):,.0f}–${float(mx):,.0f}/{interval}"
            elif mx is not None:
                salary = f"up to ${float(mx):,.0f}/{interval}"
            elif mn is not None:
                salary = f"from ${float(mn):,.0f}/{interval}"
        except (TypeError, ValueError):
            salary = str(mx or mn)

    posted_at: datetime | None = None
    raw_date = r.get("date_posted")
    if raw_date is not None:
        try:
            # pandas Timestamp
            if hasattr(raw_date, "to_pydatetime"):
                posted_at = raw_date.to_pydatetime().replace(tzinfo=None)
            elif isinstance(raw_date, str):
                posted_at = datetime.fromisoformat(raw_date.replace("Z", "+00:00")).replace(tzinfo=None)
            elif isinstance(raw_date, datetime):
                posted_at = raw_date.replace(tzinfo=None)
        except Exception:
            pass

    return RawJob(
        source=f"jobspy:{site}",
        title=title,
        company=company or f"{site.capitalize()} posting",
        url=url,
        location=location_str,
        description=description,
        salary=salary,
        posted_at=posted_at,
    )


def scrape_jobspy(keywords: List[str]) -> List[RawJob]:
    """Run JobSpy concurrently across keyword × location combos."""
    from .. import user_prefs

    # --- config ---
    raw_sites = getattr(settings, "jobspy_sites", "linkedin,indeed")
    configured_sites = [s.strip().lower() for s in raw_sites.split(",") if s.strip()]
    sites = []
    for s in configured_sites:
        if s in ("glassdoor", "zip_recruiter"):
            log.warning("JobSpy site %r is currently Cloudflare-blocked/unsupported; skipping to prevent failures.", s)
        else:
            sites.append(s)
    if not sites:
        sites = ["linkedin", "indeed"]
    hours_old = int(getattr(settings, "jobspy_hours_old", 200))
    results_wanted = int(getattr(settings, "jobspy_results_wanted", 400))
    workers = int(getattr(settings, "jobspy_workers", 3))
    fetch_desc = bool(getattr(settings, "jobspy_fetch_descriptions", True))

    # Cap the keyword list so we don't blow up the combo count.
    # Prefer the most informative biochem terms.
    preferred = ["biochemistry", "molecular biology", "RNA", "drug discovery", "scientist phd"]
    kw_pool = list(keywords or [])
    terms: list[str] = []
    for p in preferred:
        if p in kw_pool and p not in terms:
            terms.append(p)
    for k in kw_pool:
        if k not in terms:
            terms.append(k)
    terms = terms[:4]

    locs = user_prefs.scrape_locations() or ["Remote"]
    if user_prefs.includes_remote():
        locs = locs + ["Remote"]
    locs = locs[:6]

    tasks = [(term, loc) for term in terms for loc in locs]
    log.info("JobSpy: %d tasks (sites=%s, results_per_call=%d)",
             len(tasks), sites, results_wanted)

    out: List[RawJob] = []
    seen_urls: set[str] = set()

    with ThreadPoolExecutor(max_workers=workers, thread_name_prefix="jobspy") as ex:
        futures = {
            ex.submit(_scrape_one, sites, term, loc, results_wanted, hours_old, fetch_desc): (term, loc)
            for term, loc in tasks
        }
        for fut in as_completed(futures):
            term, loc = futures[fut]
            try:
                records = fut.result()
            except Exception as e:  # noqa: BLE001
                log.info("jobspy future error (term=%r loc=%r): %s", term, loc, e)
                continue
            for r in records:
                rj = _to_raw_job(r)
                if rj is None or rj.url in seen_urls:
                    continue
                seen_urls.add(rj.url)
                out.append(rj)

    log.info("JobSpy: %d unique postings from %d tasks", len(out), len(tasks))
    return out


registry.register(scrape_jobspy)
