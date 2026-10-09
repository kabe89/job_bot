"""Generic RSS scraper — biotech/biochem feeds + extras from data/extra_feeds.txt."""
from __future__ import annotations

from datetime import datetime
from pathlib import Path
from typing import List

import feedparser

from . import RawJob, registry

EXTRA_FEEDS_FILE = Path("data/extra_feeds.txt")

# Location-agnostic seed feeds. Many boards expose Indeed-style RSS:
#   https://www.indeed.com/rss?q=<query>&l=<location>
# Location-specific Indeed feeds are added dynamically in `_load_feeds` from the
# user's configured scrape locations (SEARCH_LOCATIONS / user_locations.txt), so
# nothing here hard-codes a personal location.
DEFAULT_FEEDS = [
    # Remote searches (keywords are configurable via SEARCH_KEYWORDS):
    "https://www.indeed.com/rss?q=computational+biology&l=Remote",
    "https://www.indeed.com/rss?q=bioinformatics+scientist&l=Remote",
    # General remote boards (still filtered by your keywords downstream):
    "https://remotive.com/remote-jobs/feed",
    "https://jobicy.com/?feed=job_feed",
]


def _load_feeds() -> List[str]:
    from urllib.parse import quote_plus
    from .. import user_prefs
    feeds = list(DEFAULT_FEEDS)
    # Dynamically add Indeed RSS feeds for the user's scrape locations.
    # Indeed's RSS supports `?q=<term>&l=<location>&radius=<miles>`.
    locs = user_prefs.scrape_locations()
    radius = max(0, int(getattr(__import__("jobbot.config", fromlist=["settings"]).settings,
                                 "scrape_radius_miles", 0)))
    for loc in locs[:8]:
        loc_q = quote_plus(loc)
        for kw in ("biochemistry+phd", "molecular+biology", "protein+chemistry",
                   "research+associate+phd"):
            url = f"https://www.indeed.com/rss?q={kw}&l={loc_q}"
            if radius:
                url += f"&radius={radius}"
            feeds.append(url)
    if EXTRA_FEEDS_FILE.exists():
        for line in EXTRA_FEEDS_FILE.read_text(encoding="utf-8").splitlines():
            line = line.strip()
            if line and not line.startswith("#"):
                feeds.append(line)
    return feeds


def scrape_rss(keywords: List[str]) -> List[RawJob]:
    out: List[RawJob] = []
    kw_lower = [k.lower() for k in keywords] or [""]
    for feed_url in _load_feeds():
        try:
            d = feedparser.parse(feed_url)
        except Exception:
            continue
        source = (d.feed.get("title", feed_url) if hasattr(d, "feed") else feed_url)[:64]
        for entry in d.entries:
            title = entry.get("title", "")
            summary = entry.get("summary", "") or entry.get("description", "")
            haystack = f"{title} {summary}".lower()
            if not any(k in haystack for k in kw_lower):
                continue
            published = entry.get("published_parsed") or entry.get("updated_parsed")
            try:
                posted_dt = datetime(*published[:6]) if published else None
            except Exception:
                posted_dt = None
            company = entry.get("author", "") or ""
            # Indeed embeds location in title: "Title - Company - City, ST"
            location = ""
            if " - " in title and "indeed" in feed_url:
                parts = [p.strip() for p in title.rsplit(" - ", 2)]
                if len(parts) == 3:
                    title, company, location = parts
            out.append(RawJob(
                source=f"rss:{source}",
                title=title,
                company=company,
                url=entry.get("link", ""),
                location=location,
                description=summary[:8000],
                posted_at=posted_dt,
            ))
    return out


registry.register(scrape_rss)
