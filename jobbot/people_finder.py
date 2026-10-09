# jobbot/people_finder.py
"""Shared-thread people finder: locate strangers at a target company who share
a real thread with the candidate (coauthor > institution > field).

Public sources only (company/public web via contacts.find_contacts + the keyless
PubMed E-utilities API). Every helper is fail-open: errors yield empty results,
never raise.
"""
from __future__ import annotations

import logging
import re
from pathlib import Path
from typing import List, Set
from xml.etree import ElementTree as ET

import requests

from . import contacts as _contacts
from . import referrals
from .config import settings

log = logging.getLogger("jobbot.people_finder")

_PUB_HEAD = re.compile(r"^\s{0,3}#{0,4}\s*publications?\b", re.I | re.M)
_NEXT_HEAD = re.compile(r"^\s{0,3}#{1,4}\s+\S", re.M)

# "Taylor NA" / "Smith J" (surname + initials) OR "Grace Hopper" (two names).
_AUTHOR = re.compile(
    r"\b([A-Z][a-z]+(?:[-'][A-Z][a-z]+)?)\s+((?:[A-Z]\.?){1,3}|[A-Z][a-z]+)\b")

# First-token (surname position) words that signal a journal abbreviation, not
# a person. Keyed on the SURNAME slot only, so real single-initial authors like
# "Smith J" (whose second token is "J") are never dropped.
_JOURNAL_STOP = frozenset({
    "j", "mol", "biol", "chem", "phys", "biophys", "sci", "nat", "cell",
    "proc", "natl", "acad", "med", "rev", "res", "am", "eur", "int", "clin",
    "biochem", "genet", "commun", "struct", "func", "bioinform",
})


def _read_one(path) -> str:
    try:
        p = Path(path)
        return p.read_text(encoding="utf-8", errors="replace") if p.exists() else ""
    except Exception:  # noqa: BLE001
        return ""


def _read_sources() -> str:
    """Concatenated text of profile.md + base_resume.md (seam for tests)."""
    prof = _read_one(getattr(settings, "profile_path", "data/profile.md"))
    resume = _read_one(getattr(settings, "base_resume_path", "data/base_resume.md"))
    return prof + "\n\n" + resume


def _publications_block(text: str) -> str:
    m = _PUB_HEAD.search(text or "")
    if not m:
        return ""
    rest = text[m.end():]
    nxt = _NEXT_HEAD.search(rest)
    return rest[:nxt.start()] if nxt else rest


def _self_tokens() -> Set[str]:
    name = (getattr(settings, "applicant_name", "") or "").lower()
    return {t for t in re.split(r"\s+", name) if len(t) > 2}


def _names_from_block(block: str) -> Set[str]:
    out: Set[str] = set()
    for surname, rest in _AUTHOR.findall(block or ""):
        if surname.lower() in _JOURNAL_STOP:
            continue
        out.add(f"{surname} {rest}".strip().lower())
    return out


def _pubmed_coauthors(author: str, max_papers: int = 20) -> Set[str]:
    """Coauthor surnames+initials from the user's PubMed papers. Fail-open."""
    author = (author or "").strip()
    if not author:
        return set()
    timeout = getattr(settings, "apply_fetch_timeout", 30)
    base = "https://eutils.ncbi.nlm.nih.gov/entrez/eutils"
    try:
        r = requests.get(f"{base}/esearch.fcgi", params={
            "db": "pubmed", "term": f"{author}[Author]",
            "retmax": max_papers, "retmode": "json"}, timeout=timeout)
        r.raise_for_status()
        ids = r.json().get("esearchresult", {}).get("idlist", [])
        if not ids:
            return set()
        r2 = requests.get(f"{base}/efetch.fcgi", params={
            "db": "pubmed", "id": ",".join(ids), "retmode": "xml"}, timeout=timeout)
        r2.raise_for_status()
        root = ET.fromstring(r2.text)
        out: Set[str] = set()
        for a in root.iter("Author"):
            ln = (a.findtext("LastName") or "").strip()
            ini = (a.findtext("Initials") or "").strip()
            if ln:
                out.add(f"{ln} {ini}".strip().lower())
        return out
    except Exception as exc:  # noqa: BLE001
        log.debug("PubMed coauthor fetch failed for %r: %s", author, exc)
        return set()


def _seed_coauthors() -> Set[str]:
    """Coauthor names from resume/profile publications + PubMed, minus self."""
    try:
        block = _publications_block(_read_sources())
        seeds = _names_from_block(block)
        seeds |= _pubmed_coauthors(getattr(settings, "applicant_name", "") or "")
        self_toks = _self_tokens()
        return {s for s in seeds if s and not any(t in s for t in self_toks)}
    except Exception as exc:  # noqa: BLE001
        log.debug("_seed_coauthors failed: %s", exc)
        return set()


def _seed_key(seed: str) -> str:
    """A citation-format seed 'surname initials' -> canonical 'surname firstinitial'."""
    toks = [t for t in re.split(r"[^a-z]+", (seed or "").lower()) if t]
    return f"{toks[0]} {toks[1][0]}" if len(toks) >= 2 else ""


def _name_keys(name: str) -> set:
    """Candidate 'surname firstinitial' keys for a person name. We cannot tell
    'First Last' from 'Surname Initials' apart, so emit a key for BOTH readings."""
    toks = [t for t in re.split(r"[^a-z]+", (name or "").lower()) if t]
    if len(toks) < 2:
        return set()
    return {
        f"{toks[-1]} {toks[0][0]}",   # First Last  -> surname=last, initial=first[0]
        f"{toks[0]} {toks[1][0]}",    # Surname Initials -> surname=first, initial=second[0]
    }


def _thread_score(person: dict, seeds: Set[str], terms: Set[str],
                  affils: List[str]) -> "tuple[float, str]":
    """Rank a candidate by shared thread: coauthor > institution > field."""
    name = (person.get("name") or "").strip().lower()
    role = (person.get("role") or "").lower()
    score = 0.0
    why: List[str] = []
    seed_keys = {_seed_key(s) for s in seeds} - {""}
    if seed_keys and (_name_keys(name) & seed_keys):
        score += 0.6; why.append("coauthor")
    if affils and any(a in role for a in affils):
        score += 0.3; why.append("shared institution")
    if terms and any(t in role for t in terms):
        score += 0.15; why.append("shared field")
    return score, ", ".join(why)


def find_shared_thread_contacts(job_id: int, max_contacts=None) -> List[dict]:
    """People at a job's company who share a thread with the candidate.

    Reuses contacts.find_contacts for public-web candidates, scores each by
    shared thread, tags with 'thread'/'thread_score', returns highest first.
    Fail-open -> []."""
    if max_contacts is None:
        max_contacts = getattr(settings, "contacts_max_per_job", 3)
    try:
        seeds = _seed_coauthors()
        terms = referrals.profile_terms()
        affils = [a.strip().lower()
                  for a in (settings.affiliation_terms or "").split(",") if a.strip()]
        found = _contacts.find_contacts(job_id, save=False) or []
        scored: List[dict] = []
        for person in found:
            s, why = _thread_score(person, seeds, terms, affils)
            if s <= 0:
                continue
            person = dict(person)
            person["thread"] = why
            person["thread_score"] = s
            scored.append(person)
        scored.sort(key=lambda p: p["thread_score"], reverse=True)
        return scored[:max_contacts]
    except Exception as exc:  # noqa: BLE001
        log.debug("find_shared_thread_contacts(%s) failed: %s", job_id, exc)
        return []
