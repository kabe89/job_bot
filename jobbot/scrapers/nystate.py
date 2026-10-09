"""New York State Civil Service jobs scraper.

Pulls live HTML from the actual public search page (the only thing that works
since the RSS feed was deprecated):

  https://statejobs.ny.gov/employees/vacancyTable.cfm
      ?searchResults=Yes&title=<title>&Keywords=<kw>

Also runs the Research Scientist preset query the user supplied. Covers state health departments, university system openings, and all state agencies.
"""
from __future__ import annotations

import logging
import re
from datetime import datetime
from typing import List
from urllib.parse import urljoin

import requests
from bs4 import BeautifulSoup
from tenacity import retry, stop_after_attempt, wait_exponential

from . import RawJob, registry

log = logging.getLogger("jobbot.scrapers.nystate")

BASE = "https://statejobs.ny.gov"
SEARCH_URL = f"{BASE}/employees/vacancyTable.cfm"
HEADERS = {"User-Agent": "Mozilla/5.0", "Accept": "text/html"}

# Title queries to run. Each runs as a separate HTTP GET and merges results.
TITLE_QUERIES = [
    "Research Scientist",
    "Biochemist",
    "Microbiologist",
    "Laboratory Scientist",
    "Public Health Specialist",
]

# Keyword queries (in addition to title-based search) — catch postings that
# don't use Research Scientist in the title but are still biochem-relevant.
KEYWORD_QUERIES = [
    "biochemistry",
    "molecular biology",
    "protein",
]


@retry(stop=stop_after_attempt(2), wait=wait_exponential(min=1, max=5), reraise=False)
def _fetch(params: dict) -> str:
    r = requests.get(SEARCH_URL, params=params, headers=HEADERS, timeout=20)
    r.raise_for_status()
    return r.text


def _header_columns(soup) -> dict:
    """Map lowercased header label -> column index, from the results table's <th>
    row. Returns {} when the page has no header row."""
    for tr in soup.find_all("tr"):
        ths = tr.find_all("th")
        if len(ths) >= 4:
            return {th.get_text(" ", strip=True).lower().strip(): i
                    for i, th in enumerate(ths)}
    return {}


def _column(cells: list, idx) -> str:
    if idx is None or idx >= len(cells):
        return ""
    return cells[idx]


def _parse_vacancy_table(html: str) -> List[dict]:
    """Parse the HTML <table> the NY State search results page returns.

    Columns are read BY HEADER NAME, not sniffed. The live layout is
        Item # | Title | Grade | Posted | Deadline | Agency | County
    so the old "first cell containing Department/Health/State/University" scan hit
    the TITLE (column 1) long before the Agency (column 5): every title carrying
    one of those words became its own employer. Sniffing also could not find real
    agencies that contain none of the keywords ("Podiatry Board", "Children &
    Family Services, Office").
    """
    soup = BeautifulSoup(html, "html.parser")
    cols = _header_columns(soup)
    agency_i = cols.get("agency")
    county_i = cols.get("county", cols.get("location"))

    rows = []
    for tr in soup.find_all("tr"):
        cells = tr.find_all(["td", "th"])
        if len(cells) < 4:
            continue
        link = tr.find("a", href=re.compile(r"vacancyDetails", re.I))
        if not link:
            continue
        title = link.get_text(strip=True)
        href = urljoin(BASE + "/employees/", link.get("href", ""))
        cell_text = [c.get_text(" ", strip=True) for c in cells]

        agency = _column(cell_text, agency_i).strip()
        # Guard: never let the company become the job title, whatever the layout.
        if not agency or agency == title:
            agency = "NY State"

        location = _column(cell_text, county_i).strip()
        if not location:
            # No County column (unknown layout) -- fall back to a city scan, but
            # never over the title cell.
            for i, c in enumerate(cell_text):
                if c == title:
                    continue
                m = re.search(r"\b(New York|Buffalo|Rochester|Syracuse|Yonkers|"
                              r"White Plains|Binghamton|Ithaca)\b", c)
                if m:
                    location = m.group(1)
                    break

        rows.append({
            "title": title,
            "url": href,
            "agency": agency,
            "location": location,
            "raw": " | ".join(cell_text)[:600],
        })
    return rows


def scrape_nystate(keywords: List[str]) -> List[RawJob]:
    kw_lower = [k.lower() for k in keywords] or [""]
    seen_urls = set()
    out: List[RawJob] = []

    # Run all title-based queries (these tend to be highest yield)
    for title_q in TITLE_QUERIES:
        try:
            html = _fetch({"searchResults": "Yes", "title": title_q})
        except Exception as e:  # noqa: BLE001
            log.info("NY State title query '%s' failed: %s", title_q, e)
            continue
        for row in _parse_vacancy_table(html):
            if row["url"] in seen_urls:
                continue
            seen_urls.add(row["url"])
            haystack = (row["title"] + " " + row["raw"]).lower()
            # Title-query results are already pre-filtered by the server, but
            # still confirm at least one biochem keyword OR health
            if not (any(k in haystack for k in kw_lower) or
                    "health" in haystack):
                continue
            out.append(RawJob(
                source="nystate",
                title=row["title"][:240],
                company=row["agency"][:120],
                url=row["url"],
                location=row["location"],
                description=f"Posted via NY State Civil Service.\n\n{row['raw']}",
                posted_at=None,
            ))

    # Keyword-based queries to catch anything title-search missed.
    # Require a biochem-related word in the row text to suppress generic
    # NY HELPS office staff postings that match agency strings.
    SCI_TOKENS = ("scientist", "biolog", "biochem", "chem", "lab", "research",
                  "virol", "microbi", "pharm", "molecular")
    for kw_q in KEYWORD_QUERIES:
        try:
            html = _fetch({"searchResults": "Yes", "Keywords": kw_q})
        except Exception:
            continue
        for row in _parse_vacancy_table(html):
            if row["url"] in seen_urls:
                continue
            haystack = (row["title"] + " " + row["raw"]).lower()
            if not any(t in haystack for t in SCI_TOKENS):
                continue
            seen_urls.add(row["url"])
            out.append(RawJob(
                source="nystate",
                title=row["title"][:240],
                company=row["agency"][:120],
                url=row["url"],
                location=row["location"],
                description=f"Posted via NY State Civil Service (keyword: {kw_q}).\n\n{row['raw']}",
                posted_at=None,
            ))

    log.info("NY State Civil Service: %d unique postings", len(out))
    return out


registry.register(scrape_nystate)
