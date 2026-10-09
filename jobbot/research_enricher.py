"""Research enrichment for networking contacts.

For each contact, searches PubMed, ResearchGate, and company/lab pages to gather:
  - Recent publications (with DOI/PMID)
  - Research focus description
  - Company pipeline / lab direction context

Then uses AI (Gemini → Claude → Ollama) to synthesize a "Research Package":
  - Health Impact Summary: what disease/problem they address and what success means
  - Connection to User's Work: how the user's specific skills map to this person's research
  - Key Papers to Read: 3-5 real, findable papers with identifiers
  - Conversation Hook: the sharpest single opening question for outreach

API
---
enrich_contact(contact, profile_text)  -> dict  (contact + research_package key)
build_batch_report(contacts, company_name, profile_text, output_path) -> str  (markdown)
run_enrichment_cli(contacts_json_path, company_name, output_path) -> None
"""
from __future__ import annotations

import json
import logging
import re
import textwrap
from datetime import datetime, timezone
from pathlib import Path
from typing import Optional

import requests
from bs4 import BeautifulSoup

from .config import settings

log = logging.getLogger("jobbot.research_enricher")

# ---------------------------------------------------------------------------
# Constants
# ---------------------------------------------------------------------------
UA = (
    "Mozilla/5.0 (Windows NT 10.0; Win64; x64) AppleWebKit/537.36 "
    "(KHTML, like Gecko) Chrome/131.0.0.0 Safari/537.36"
)
HEADERS = {"User-Agent": UA, "Accept-Language": "en-US,en;q=0.9"}
TIMEOUT = 20

# ---------------------------------------------------------------------------
# Web search (re-uses contacts.py backend logic without importing it)
# ---------------------------------------------------------------------------

def _web_search(query: str, max_results: int = 6) -> list[dict]:
    """Search wrapper: Ollama web search → Serper → Google CSE → ddgs → []."""
    results: list[dict] = []

    # Ollama hosted web search (live results; preferred when a key is set)
    try:
        from . import ollama_search
        if ollama_search.is_available():
            results = ollama_search.web_search(query, max_results)
            if results:
                return results
    except Exception as exc:  # noqa: BLE001
        log.debug("Ollama web search failed for %r: %s", query, exc)

    # Serper
    key = getattr(settings, "serper_api_key", "")
    if key:
        try:
            r = requests.post(
                "https://google.serper.dev/search",
                headers={"X-API-KEY": key, "Content-Type": "application/json"},
                json={"q": query, "num": min(max_results, 10)},
                timeout=TIMEOUT,
            )
            r.raise_for_status()
            results = [{"title": it.get("title", ""), "url": it.get("link", ""),
                        "snippet": it.get("snippet", "")}
                       for it in r.json().get("organic", [])][:max_results]
            if results:
                return results
        except Exception as exc:  # noqa: BLE001
            log.debug("Serper failed for %r: %s", query, exc)

    # Google CSE
    cse_key = getattr(settings, "google_cse_key", "")
    cse_cx = getattr(settings, "google_cse_cx", "")
    if cse_key and cse_cx:
        try:
            r = requests.get(
                "https://www.googleapis.com/customsearch/v1",
                params={"key": cse_key, "cx": cse_cx, "q": query,
                        "num": min(max_results, 10)},
                timeout=TIMEOUT,
            )
            r.raise_for_status()
            results = [{"title": it.get("title", ""), "url": it.get("link", ""),
                        "snippet": it.get("snippet", "")}
                       for it in r.json().get("items", [])][:max_results]
            if results:
                return results
        except Exception as exc:  # noqa: BLE001
            log.debug("CSE failed for %r: %s", query, exc)

    # ddgs (keyless)
    try:
        import importlib.util
        if importlib.util.find_spec("ddgs"):
            from ddgs import DDGS
            rows = list(DDGS().text(query, max_results=max_results))
            results = [{"title": r.get("title", ""), "url": r.get("href", ""),
                        "snippet": r.get("body", "")}
                       for r in rows if r.get("href")][:max_results]
            if results:
                return results
    except Exception as exc:  # noqa: BLE001
        log.debug("ddgs failed for %r: %s", query, exc)

    return []


def _fetch_page_text(url: str, max_chars: int = 4000) -> str:
    """Fetch a URL and return visible text, truncated."""
    try:
        r = requests.get(url, headers=HEADERS, timeout=TIMEOUT)
        if r.status_code != 200:
            return ""
        soup = BeautifulSoup(r.text, "html.parser")
        for tag in soup(["script", "style", "nav", "footer", "header"]):
            tag.decompose()
        return " ".join(soup.get_text(" ", strip=True).split())[:max_chars]
    except Exception as exc:  # noqa: BLE001
        log.debug("fetch_page_text(%s) failed: %s", url, exc)
        return ""


# ---------------------------------------------------------------------------
# Publication / research discovery
# ---------------------------------------------------------------------------

def _pubmed_snippets(name: str, company: str, max_results: int = 5) -> list[dict]:
    """Search for recent papers or technical publications by this person."""
    kw = getattr(settings, "research_enricher_keywords", "")
    terms = " ".join(kw.split(",")[:3]) if kw else ""
    results = _web_search(f"site:pubmed.ncbi.nlm.nih.gov {name} {company} {terms}".strip(), max_results)
    if not results:
        results = _web_search(f"site:pubmed.ncbi.nlm.nih.gov {name} {company}", max_results)
    if not results:
        results = _web_search(f'"{name}" "{company}" research publications', max_results)
    return results


def _lab_page_text(name: str, company: str) -> str:
    """Try to fetch the person's staff or lab page."""
    results = _web_search(
        f'"{name}" {company} research lab publications site:{_company_domain(company)}',
        max_results=3,
    )
    if not results:
        results = _web_search(f'"{name}" {company} research lab publications', max_results=3)

    for r in results:
        url = r.get("url", "")
        if url and ("lab" in url.lower() or "staff" in url.lower()
                    or "people" in url.lower() or "faculty" in url.lower()
                    or name.split()[-1].lower() in url.lower()):
            text = _fetch_page_text(url)
            if len(text) > 200:
                return text
    return ""


def _company_domain(company: str) -> str:
    """Best-guess web domain for a company name."""
    slug = re.sub(r"[^a-z0-9]+", "", company.lower())
    return f"{slug}.com" if slug else "example.com"


def _gather_raw_intelligence(contact: dict) -> dict:
    """Collect raw text evidence for AI synthesis: snippets from PubMed,
    lab pages, and general search. Returns a dict of text blobs."""
    name = contact.get("name", "")
    company = contact.get("company", "")
    title = contact.get("title", "")

    pubmed_results = _pubmed_snippets(name, company)
    pubmed_text = "\n".join(
        f"- {r['title']}: {r['snippet']} [{r['url']}]"
        for r in pubmed_results if r.get("title")
    )

    lab_text = _lab_page_text(name, company)

    general_results = _web_search(f'"{name}" {company} research publications 2023 2024 2025', 4)
    general_text = "\n".join(
        f"- {r['title']}: {r['snippet']}"
        for r in general_results if r.get("title")
    )

    return {
        "name": name,
        "title": title,
        "company": company,
        "pubmed": pubmed_text[:3000],
        "lab_page": lab_text[:3000],
        "general": general_text[:2000],
    }


# ---------------------------------------------------------------------------
# AI synthesis
# ---------------------------------------------------------------------------

_RESEARCH_PACKAGE_PROMPT = """\
You are a scientific intelligence analyst preparing a pre-outreach research package.

## Contact
Name: {name}
Title: {title}
Organization: {company}

## Evidence gathered from public sources
### PubMed / publication search results
{pubmed}

### Lab/staff page text
{lab_page}

### General search snippets
{general}

## User profile (the person reaching out)
{profile}

## Task
Write a structured research package for this contact. Be factual — only reference
papers or programs you can support from the evidence above. Do not invent citations.

Return ONLY the following four sections, with these exact headers:

### Health Impact Summary
(3-5 sentences) What disease or public health problem does this person's research address?
What is their approach? What would success in their work mean for patients or public health?

### Connection to User's Work
(2-3 sentences) How do the user's specific skills and research background connect to this
person's work — overlapping methods, complementary approaches, or adjacent problems?

### Key Papers to Read
List 3-5 papers or pipeline programs grounded in the evidence above.
Format each as: - Title. *Journal*. Year. DOI or PMID or URL if available.
Only include citations you can support from the evidence. If fewer than 3 are verifiable, say so.

### Conversation Hook
(1 sentence) The single sharpest scientific question the user could open with in an email
or first meeting — specific to this person's work and connected to the user's research.
"""


def _ai_generate(prompt: str) -> Optional[str]:
    """Generate text via AI: Gemini → Claude → Ollama → None."""
    try:
        from . import gemini_client
        if gemini_client.is_available():
            result = gemini_client._generate(prompt, include_profile=False)
            if result and result.strip():
                return result.strip()
    except Exception as exc:  # noqa: BLE001
        log.debug("Gemini research enrichment failed: %s", exc)

    try:
        from . import claude_client
        if claude_client.is_available():
            result = claude_client._generate(prompt)
            if result and result.strip():
                return result.strip()
    except Exception as exc:  # noqa: BLE001
        log.debug("Claude research enrichment failed: %s", exc)

    try:
        from . import ollama_client
        if ollama_client.is_available():
            result = ollama_client._generate(prompt)
            if result and result.strip():
                return result.strip()
    except Exception as exc:  # noqa: BLE001
        log.debug("Ollama research enrichment failed: %s", exc)

    return None


def _profile_text() -> str:
    """Load profile.md as text."""
    try:
        p = Path(getattr(settings, "profile_path", "data/profile.md"))
        if p.exists():
            return p.read_text(encoding="utf-8")[:6000]
    except Exception:  # noqa: BLE001
        pass
    return ""


def _synthesize_package(raw: dict, profile: str) -> dict:
    """Call AI to produce the four-section research package.
    Returns a dict with keys: health_impact, skill_connection, key_papers, hook.
    Falls back to a minimal template if AI unavailable.
    """
    prompt = _RESEARCH_PACKAGE_PROMPT.format(
        name=raw["name"],
        title=raw["title"],
        company=raw["company"],
        pubmed=raw["pubmed"] or "(no PubMed results found)",
        lab_page=raw["lab_page"] or "(no lab page text found)",
        general=raw["general"] or "(no general search results)",
        profile=profile[:4000] if profile else "(profile not available)",
    )

    ai_output = _ai_generate(prompt)

    if ai_output:
        return _parse_ai_package(ai_output)

    # Deterministic fallback when AI is unavailable
    return {
        "health_impact": (
            f"{raw['name']} ({raw['title']} at {raw['company']}) — "
            "research details not synthesized (AI unavailable). "
            "Review publication search results above manually."
        ),
        "skill_connection": "See PubMed and lab page snippets above for research context.",
        "key_papers": "AI synthesis unavailable. Run `pip install google-genai` and set GEMINI_API_KEY.",
        "hook": f"I came across your work at {raw['company']} and would love to learn more about your current research direction.",
        "raw_evidence": {
            "pubmed": raw["pubmed"],
            "lab_page": raw["lab_page"][:500] if raw["lab_page"] else "",
        },
    }


def _parse_ai_package(text: str) -> dict:
    """Extract the four sections from the AI output."""
    sections = {
        "health_impact": "",
        "skill_connection": "",
        "key_papers": "",
        "hook": "",
    }
    MARKERS = [
        ("health_impact", r"#{1,3}\s*Health Impact Summary"),
        ("skill_connection", r"#{1,3}\s*Connection to (?:User'?s|Your) Work"),
        ("key_papers", r"#{1,3}\s*Key Papers to Read"),
        ("hook", r"#{1,3}\s*Conversation Hook"),
    ]

    positions: list[tuple[str, int]] = []
    for key, pattern in MARKERS:
        m = re.search(pattern, text, re.IGNORECASE)
        if m:
            positions.append((key, m.end()))

    positions.sort(key=lambda x: x[1])

    for i, (key, start) in enumerate(positions):
        end = positions[i + 1][1] - len(positions[i + 1][0]) - 10 if i + 1 < len(positions) else len(text)
        content = text[start:end].strip()
        # Strip any leaked markdown headers
        content = re.sub(r"^#{1,4}\s.*$", "", content, flags=re.MULTILINE).strip()
        sections[key] = content

    return sections


# ---------------------------------------------------------------------------
# Public API
# ---------------------------------------------------------------------------

def enrich_contact(contact: dict, profile: Optional[str] = None) -> dict:
    """Enrich a single contact with a research package.

    Returns the contact dict with an added 'research_package' key containing
    {health_impact, skill_connection, key_papers, hook}.
    Degrades gracefully — never raises.
    """
    if profile is None:
        profile = _profile_text()

    try:
        raw = _gather_raw_intelligence(contact)
        package = _synthesize_package(raw, profile)
        return {**contact, "research_package": package}
    except Exception as exc:  # noqa: BLE001
        log.warning("enrich_contact failed for %s: %s", contact.get("name"), exc)
        return {**contact, "research_package": None}


def build_batch_report(
    contacts: list[dict],
    company_name: str,
    profile: Optional[str] = None,
    output_path: Optional[str] = None,
) -> str:
    """Enrich all contacts and build a markdown research package report.

    Args:
        contacts: List of contact dicts (name, title, company, email, linkedin).
        company_name: Display name for the organization/batch (used in header).
        profile: User profile text (loads data/profile.md if None).
        output_path: Where to write the markdown. Defaults to
                     output/networking/{slug}_research_packages.md

    Returns the markdown string.
    """
    if profile is None:
        profile = _profile_text()

    slug = re.sub(r"[^a-z0-9]+", "_", company_name.lower()).strip("_")
    if output_path is None:
        out_dir = Path("output/networking")
        out_dir.mkdir(parents=True, exist_ok=True)
        output_path = str(out_dir / f"{slug}_research_packages.md")

    enriched = []
    for i, c in enumerate(contacts):
        log.info("Enriching contact %d/%d: %s", i + 1, len(contacts), c.get("name", "?"))
        enriched.append(enrich_contact(c, profile))

    md = _render_markdown(enriched, company_name)

    try:
        Path(output_path).write_text(md, encoding="utf-8")
        log.info("Research packages written to %s", output_path)
    except Exception as exc:  # noqa: BLE001
        log.warning("Could not write research packages to %s: %s", output_path, exc)

    return md


def _render_markdown(enriched_contacts: list[dict], company_name: str) -> str:
    """Render the full research packages markdown document."""
    now = datetime.now(tz=timezone.utc).strftime("%Y-%m-%d")
    lines: list[str] = [
        f"# {company_name} — Research Intelligence Packages",
        f"**Date:** {now}",
        "**Status:** RESEARCH ONLY — No outreach messages drafted here. "
        "No contact attempted.",
        "",
        "---",
        "",
    ]

    for i, c in enumerate(enriched_contacts, 1):
        name = c.get("name", f"Contact {i}")
        title = c.get("title", "")
        company = c.get("company", company_name)
        email = c.get("email", "")
        linkedin = c.get("linkedin", "")
        pkg = c.get("research_package") or {}

        lines += [
            f"## Contact {i}: {name}",
            f"**Title:** {title}" if title else "",
            f"**Organization:** {company}",
            f"**Email:** {email}" if email else "**Email:** (not confirmed)",
            f"**LinkedIn:** {linkedin}" if linkedin else "**LinkedIn:** (not confirmed)",
            "",
            "### Health Impact Summary",
            pkg.get("health_impact") or "_AI synthesis unavailable — see raw evidence._",
            "",
            "### Connection to User's Work",
            pkg.get("skill_connection") or "_See raw evidence above._",
            "",
            "### Key Papers to Read",
            pkg.get("key_papers") or "_No verified papers synthesized._",
            "",
            "### Conversation Hook",
            pkg.get("hook") or "_No hook generated._",
            "",
            "---",
            "",
        ]

    return "\n".join(l for l in lines if l is not None)


# ---------------------------------------------------------------------------
# CLI entry point (called from cli.py)
# ---------------------------------------------------------------------------

def run_enrichment_cli(
    contacts_json: Optional[str],
    company_name: str,
    output_path: Optional[str] = None,
) -> str:
    """Load contacts from a JSON file (or stdin) and run enrichment.

    contacts_json: path to a JSON file containing a list of contact dicts,
                   or None to read from stdin.
    Returns the output file path.
    """
    import sys

    if contacts_json:
        text = Path(contacts_json).read_text(encoding="utf-8-sig")
    else:
        text = sys.stdin.read().lstrip("﻿")

    contacts = json.loads(text)
    if not isinstance(contacts, list):
        contacts = [contacts]

    slug = re.sub(r"[^a-z0-9]+", "_", company_name.lower()).strip("_")
    if output_path is None:
        out_dir = Path("output/networking")
        out_dir.mkdir(parents=True, exist_ok=True)
        output_path = str(out_dir / f"{slug}_research_packages.md")

    build_batch_report(contacts, company_name, output_path=output_path)
    return output_path
