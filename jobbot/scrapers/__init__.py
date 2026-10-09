"""Job scrapers — pluggable per-source modules."""
from __future__ import annotations

import logging
import os
import time
from concurrent.futures import ThreadPoolExecutor, as_completed
from concurrent.futures import TimeoutError as FuturesTimeoutError
from dataclasses import dataclass, field
from datetime import datetime
from typing import List, Optional, Callable

log = logging.getLogger("jobbot.scrapers")


@dataclass
class RawJob:
    source: str
    title: str
    company: str
    url: str
    location: str = ""
    description: str = ""
    salary: str = ""
    tags: str = ""
    posted_at: Optional[datetime] = None


@dataclass
class ScraperRegistry:
    scrapers: List[Callable[[List[str]], List[RawJob]]] = field(default_factory=list)

    def register(self, fn: Callable[[List[str]], List[RawJob]]) -> None:
        self.scrapers.append(fn)

    def run_all(self, keywords: List[str]) -> List[RawJob]:
        """Run every registered scraper concurrently with a hard per-scraper
        timeout. Without this, a single hung site (e.g. LinkedIn rate-limit
        backoff, Workday slow tenant) would block the whole cycle.

        Scrapers named in JOBBOT_NO_TIMEOUT_SCRAPERS (default "jobspy") are
        EXEMPT from the deadline — they run to completion even if slow, because
        JobSpy is the primary source and legitimately takes minutes over many
        keyword×location combos. Other scrapers are still abandoned if stuck.

        Tunable via env:
          JOBBOT_SCRAPE_WORKERS      — pool size (default min(8, #scrapers))
          JOBBOT_SCRAPER_TIMEOUT     — seconds before a stuck (non-exempt)
                                       scraper is abandoned (default 90)
          JOBBOT_NO_TIMEOUT_SCRAPERS — comma-separated scraper names exempt
                                       from the deadline (default "jobspy")
        """
        from ..scrape_progress import progress
        out: List[RawJob] = []
        progress.start(total=len(self.scrapers))

        try:
            workers = int(os.environ.get("JOBBOT_SCRAPE_WORKERS", "0") or 0)
        except ValueError:
            workers = 0
        if workers <= 0:
            workers = min(8, max(2, len(self.scrapers)))

        try:
            per_timeout = float(os.environ.get("JOBBOT_SCRAPER_TIMEOUT", "90") or 90)
        except ValueError:
            per_timeout = 90.0

        def _name(fn):
            return getattr(fn, "__name__", "scraper").replace("scrape_", "")

        def _run(fn):
            name = _name(fn)
            progress.scraper_start(name)
            log.info("> %s scraping...", name)
            t0 = time.monotonic()
            try:
                jobs = fn(keywords)
                dt = time.monotonic() - t0
                log.info("+ %s -- %d jobs in %.1fs", name, len(jobs), dt)
                progress.scraper_done(name, len(jobs))
                return jobs
            except Exception as e:  # noqa: BLE001
                dt = time.monotonic() - t0
                log.warning("! %s failed after %.1fs: %s", name, dt, e)
                progress.scraper_error(name, str(e)[:200])
                return []

        if workers == 1 or len(self.scrapers) <= 1:
            for fn in self.scrapers:
                out.extend(_run(fn))
            return out

        # Scrapers exempt from the deadline (JobSpy by default).
        exempt_raw = os.environ.get("JOBBOT_NO_TIMEOUT_SCRAPERS", "jobspy")
        exempt = {n.strip().lower() for n in exempt_raw.split(",") if n.strip()}

        # Cap the whole cycle at per_timeout * 2 — by then every non-exempt
        # scraper that was going to finish has finished, and stragglers get
        # abandoned. Exempt scrapers (JobSpy) are instead waited on to the end.
        ex = ThreadPoolExecutor(max_workers=workers, thread_name_prefix="scraper")
        futures = {ex.submit(_run, fn): _name(fn) for fn in self.scrapers}
        try:
            for fut in as_completed(futures, timeout=per_timeout * 2):
                out.extend(fut.result())
        except FuturesTimeoutError:
            stuck = [(f, n) for f, n in futures.items() if not f.done()]
            abandon = [(f, n) for f, n in stuck if n not in exempt]
            wait_on = [(f, n) for f, n in stuck if n in exempt]
            if abandon:
                log.warning("abandoning %d stuck scraper(s) after %.0fs: %s",
                            len(abandon), per_timeout * 2,
                            ", ".join(n for _, n in abandon))
                for f, n in abandon:
                    progress.scraper_error(n, f"timeout >{int(per_timeout*2)}s")
                    f.cancel()
            # Let exempt scrapers (JobSpy) run to completion — no timeout.
            for f, n in wait_on:
                log.info("waiting for %s to finish (no timeout)...", n)
                try:
                    out.extend(f.result())
                except Exception as e:  # noqa: BLE001
                    log.warning("! %s failed: %s", n, e)
        finally:
            # Don't wait for remaining stragglers — Python threads can't be
            # killed, but the network calls inside requests eventually error out.
            ex.shutdown(wait=False, cancel_futures=True)
        return out


registry = ScraperRegistry()


# Register built-in scrapers on import.
#
# linkedin + indeed are intentionally omitted here — they are superseded by
# jobspy_scraper, which handles both (plus Glassdoor) via the python-jobspy
# library with proper rate-limit handling and full description retrieval.
#
# workday is now re-enabled: the scraper uses ThreadPoolExecutor internally so
# it no longer hangs the cycle.
from . import (  # noqa: E402,F401
    rss, company_pages, nystate, ashby,
    smartrecruiters, oracle_hcm, google_jobs,
    workday, jobspy_scraper,
)
