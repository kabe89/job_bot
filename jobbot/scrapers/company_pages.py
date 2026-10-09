"""Direct company ATS scrapers (Greenhouse + Lever JSON APIs).

Many biotechs publish open APIs:
  Greenhouse: https://boards-api.greenhouse.io/v1/boards/<slug>/jobs?content=true
  Lever:      https://api.lever.co/v0/postings/<slug>?mode=json
Add (provider, slug) pairs in data/ats_targets.txt.
"""
from __future__ import annotations

from datetime import datetime
from pathlib import Path
from typing import List, Tuple

import requests
from tenacity import retry, stop_after_attempt, wait_exponential

from . import RawJob, registry

ATS_TARGETS_FILE = Path("data/ats_targets.txt")

# Seed list — verified Greenhouse / Lever boards relevant to biochem PhDs.
DEFAULT_TARGETS: List[Tuple[str, str]] = [
    # All slugs verified to return live JSON from boards-api.greenhouse.io.
    ("greenhouse", "recursionpharmaceuticals"),
]

# Map machine slugs -> human readable display names (used for output).
DISPLAY_NAMES = {
    "recursionpharmaceuticals": "Recursion Pharmaceuticals",
    "10xgenomics":              "10x Genomics",
}


def display_name(slug: str) -> str:
    return DISPLAY_NAMES.get(slug.lower(), slug)


def _load_targets() -> List[Tuple[str, str]]:
    targets = list(DEFAULT_TARGETS)
    if ATS_TARGETS_FILE.exists():
        for line in ATS_TARGETS_FILE.read_text(encoding="utf-8").splitlines():
            line = line.strip()
            if not line or line.startswith("#"):
                continue
            if "," in line:
                provider, slug = [x.strip() for x in line.split(",", 1)]
                targets.append((provider, slug))
    return targets


HEADERS = {"User-Agent": "JobBot/1.0"}


@retry(stop=stop_after_attempt(2), wait=wait_exponential(min=1, max=5), reraise=False)
def _get(url: str):
    r = requests.get(url, headers=HEADERS, timeout=15)
    r.raise_for_status()
    return r.json()


def _clean_html(s: str) -> str:
    import html, re
    s = html.unescape(s or "")
    s = re.sub(r"<br\s*/?>", "\n", s)
    s = re.sub(r"<[^>]+>", "", s)
    s = re.sub(r"\n{3,}", "\n\n", s)
    return s.strip()


def _greenhouse(slug: str, kw_lower: List[str]) -> List[RawJob]:
    try:
        data = _get(f"https://boards-api.greenhouse.io/v1/boards/{slug}/jobs?content=true")
    except Exception:
        return []
    out: List[RawJob] = []
    for j in data.get("jobs", []):
        title = j.get("title", "")
        desc = _clean_html(j.get("content") or "")
        haystack = f"{title} {desc}".lower()
        if not any(k in haystack for k in kw_lower):
            continue
        location = (j.get("location") or {}).get("name", "")
        updated = j.get("updated_at")
        try:
            posted_dt = datetime.fromisoformat(updated.replace("Z", "+00:00")) if updated else None
        except Exception:
            posted_dt = None
        out.append(RawJob(
            source="greenhouse",
            title=title[:240],
            company=display_name(slug),
            url=j.get("absolute_url", ""),
            location=location,
            description=desc[:8000],
            posted_at=posted_dt,
        ))
    return out


def _lever(slug: str, kw_lower: List[str]) -> List[RawJob]:
    try:
        data = _get(f"https://api.lever.co/v0/postings/{slug}?mode=json")
    except Exception:
        return []
    out: List[RawJob] = []
    for j in data if isinstance(data, list) else []:
        title = j.get("text", "")
        desc = (j.get("descriptionPlain") or j.get("description") or "")
        haystack = f"{title} {desc}".lower()
        if not any(k in haystack for k in kw_lower):
            continue
        location = (j.get("categories") or {}).get("location", "")
        created = j.get("createdAt")
        try:
            posted_dt = datetime.fromtimestamp(created / 1000) if created else None
        except Exception:
            posted_dt = None
        out.append(RawJob(
            source="lever",
            title=title[:240],
            company=display_name(slug),
            url=j.get("hostedUrl", ""),
            location=location,
            description=desc[:8000],
            posted_at=posted_dt,
        ))
    return out


def scrape_company_pages(keywords: List[str]) -> List[RawJob]:
    kw_lower = [k.lower() for k in keywords] or [""]
    out: List[RawJob] = []
    for provider, slug in _load_targets():
        if provider == "greenhouse":
            out.extend(_greenhouse(slug, kw_lower))
        elif provider == "lever":
            out.extend(_lever(slug, kw_lower))
    return out


registry.register(scrape_company_pages)
