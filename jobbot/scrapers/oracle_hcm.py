"""Oracle Recruiting Cloud (Oracle HCM) public scraper.

Used by Curia Global and many large pharma companies. Their public REST endpoint:

  GET https://<tenant>.fa.<region>.oraclecloud.com/hcmRestApi/resources/latest/recruitingCEJobRequisitions
      ?finder=findReqs;siteNumber=<CX_XXXX>
      &limit=100&onlyData=true&expand=requisitionList.secondaryLocations

Each entry in `data/oracle_hcm_targets.txt` is one line:
  tenant_host,siteNumber,display_name
Example:
  hcug.fa.us2.oraclecloud.com,CX_2001,Curia Global
"""
from __future__ import annotations

import logging
from datetime import datetime
from pathlib import Path
from typing import List

import requests

from . import RawJob, registry

log = logging.getLogger("jobbot.scrapers.oracle_hcm")

TARGETS_FILE = Path("data/oracle_hcm_targets.txt")

# Verified live as of build time.
DEFAULT_TARGETS = [
    # tenant_host, siteNumber, display_name
    ("hcug.fa.us2.oraclecloud.com", "CX_2001", "Curia Global"),
]

HEADERS = {"User-Agent": "Mozilla/5.0", "Accept": "application/json"}


def _load_targets():
    out = list(DEFAULT_TARGETS)
    if TARGETS_FILE.exists():
        for line in TARGETS_FILE.read_text(encoding="utf-8").splitlines():
            line = line.strip()
            if not line or line.startswith("#"):
                continue
            parts = [p.strip() for p in line.split(",")]
            if len(parts) >= 3:
                out.append((parts[0], parts[1], parts[2]))
    return out


def _fetch(tenant_host: str, site_number: str) -> list:
    url = f"https://{tenant_host}/hcmRestApi/resources/latest/recruitingCEJobRequisitions"
    params = {
        "limit": 100,
        "onlyData": "true",
        "finder": f"findReqs;siteNumber={site_number}",
        "expand": "requisitionList.secondaryLocations,requisitionList.workLocation",
    }
    try:
        r = requests.get(url, params=params, headers=HEADERS, timeout=20)
        if r.status_code != 200:
            return []
        items = r.json().get("items", [])
        if not items:
            return []
        # Oracle wraps results in a single "search wrapper" item; jobs live in requisitionList
        return items[0].get("requisitionList", []) or []
    except Exception as e:  # noqa: BLE001
        log.warning("Oracle HCM %s failed: %s", tenant_host, e)
        return []


def _build_apply_url(tenant_host: str, site_number: str, req_id: str) -> str:
    # Public-facing display URL pattern
    return f"https://{tenant_host}/hcmUI/CandidateExperience/en/sites/{site_number}/job/{req_id}"


def scrape_oracle_hcm(keywords: List[str]) -> List[RawJob]:
    kw_lower = [k.lower() for k in keywords] or [""]
    out: List[RawJob] = []
    for tenant_host, site_number, display in _load_targets():
        reqs = _fetch(tenant_host, site_number)
        for req in reqs:
            title = req.get("Title", "") or ""
            location = req.get("PrimaryLocation", "") or ""
            desc_pieces = []
            if req.get("ExternalDescriptionStr"):
                desc_pieces.append(req["ExternalDescriptionStr"])
            if req.get("ExternalQualificationsStr"):
                desc_pieces.append("\n\nQualifications:\n" + req["ExternalQualificationsStr"])
            if req.get("ExternalResponsibilitiesStr"):
                desc_pieces.append("\n\nResponsibilities:\n" + req["ExternalResponsibilitiesStr"])
            desc = "\n".join(desc_pieces)
            # Fallback shallow description if external blob is empty
            if not desc:
                desc = f"Job ID {req.get('Id','')} — {title} — {location}"
            # Strip HTML
            import re
            desc = re.sub(r"<[^>]+>", "", desc)
            desc = re.sub(r"\s+", " ", desc).strip()[:8000]
            haystack = f"{title} {desc}".lower()
            if not any(k in haystack for k in kw_lower):
                continue
            posted = req.get("PostedDate") or req.get("PostingStartDate")
            posted_dt = None
            try:
                if posted:
                    posted_dt = datetime.fromisoformat(posted.replace("Z", "+00:00"))
            except Exception:
                pass
            req_id = str(req.get("Id", ""))
            url = _build_apply_url(tenant_host, site_number, req_id)
            out.append(RawJob(
                source=f"oracle_hcm:{display.lower().replace(' ', '_')}",
                title=title[:240],
                company=display,
                url=url,
                location=location[:200],
                description=desc,
                posted_at=posted_dt,
            ))
    log.info("Oracle HCM: %d postings across %d tenants", len(out), len(_load_targets()))
    return out


registry.register(scrape_oracle_hcm)
