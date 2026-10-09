# jobbot/discovery/validate.py
"""Probe a discovered board's public JSON API (valid? + sample titles), then
score profile-fit via Spec-1 embeddings. Both are fail-open (never raise)."""
from __future__ import annotations

import logging
from typing import List

import requests

from .. import embeddings
from .. import ollama_client as oc
from .model import DiscoveredTarget

log = logging.getLogger("jobbot.discovery.validate")

_HEADERS = {"User-Agent": "JobBot/1.0", "Accept": "application/json"}
_TIMEOUT = 15
_MAX_TITLES = 15


def _titles_greenhouse(key: str) -> List[str]:
    r = requests.get(f"https://boards-api.greenhouse.io/v1/boards/{key}/jobs",
                     headers=_HEADERS, timeout=_TIMEOUT)
    r.raise_for_status()
    return [j.get("title", "") for j in (r.json() or {}).get("jobs", []) if j.get("title")]


def _titles_lever(key: str) -> List[str]:
    r = requests.get(f"https://api.lever.co/v0/postings/{key}?mode=json",
                     headers=_HEADERS, timeout=_TIMEOUT)
    r.raise_for_status()
    data = r.json()
    return [j.get("text", "") for j in (data if isinstance(data, list) else []) if j.get("text")]


def _titles_ashby(key: str) -> List[str]:
    r = requests.get(f"https://api.ashbyhq.com/posting-api/job-board/{key}",
                     headers=_HEADERS, timeout=_TIMEOUT)
    r.raise_for_status()
    return [j.get("title", "") for j in (r.json() or {}).get("jobs", []) if j.get("title")]


def _titles_workday(key: str, wd_host: str) -> List[str]:
    tenant, _, board = key.partition("/")
    host = wd_host or "wd1"
    url = f"https://{tenant}.{host}.myworkdayjobs.com/wday/cxs/{tenant}/{board}/jobs"
    r = requests.post(url, json={"limit": 20, "offset": 0, "searchText": ""},
                      headers={**_HEADERS, "Content-Type": "application/json"}, timeout=_TIMEOUT)
    r.raise_for_status()
    return [p.get("title", "") for p in (r.json() or {}).get("jobPostings", []) if p.get("title")]


def validate(t: DiscoveredTarget) -> DiscoveredTarget:
    try:
        if t.provider == "greenhouse":
            titles = _titles_greenhouse(t.key)
        elif t.provider == "lever":
            titles = _titles_lever(t.key)
        elif t.provider == "ashby":
            titles = _titles_ashby(t.key)
        elif t.provider == "workday":
            titles = _titles_workday(t.key, t.wd_host or "wd1")
        else:
            titles = []
        titles = [x for x in titles if x][:_MAX_TITLES]
        t.sample_titles = titles
        t.job_count = len(titles)
        t.valid = len(titles) > 0
    except Exception as e:  # noqa: BLE001
        log.info("validate failed for %s:%s (%s)", t.provider, t.key, e)
        t.valid = False
        t.job_count = 0
        t.sample_titles = []
    return t


def fit_score(t: DiscoveredTarget, profile) -> DiscoveredTarget:
    if not t.valid or not t.sample_titles or not getattr(profile, "embedding", None):
        return t
    try:
        emb = embeddings.embed(" ; ".join(t.sample_titles))
        t.fit_score = round(max(0.0, min(1.0, embeddings.cosine(emb, profile.embedding))), 4)
    except Exception as e:  # noqa: BLE001
        log.info("fit embed failed for %s:%s (%s)", t.provider, t.key, e)
        t.fit_score = 0.0
        return t
    try:
        prompt = (f"In ONE sentence, why might these roles fit the candidate?\n"
                  f"Roles: {', '.join(t.sample_titles[:8])}\n"
                  f"Candidate summary: {getattr(profile, 'summary', '')}")
        t.fit_reason = oc._generate(prompt, temperature=0.2, include_profile=False).strip()[:300]
    except Exception as e:  # noqa: BLE001
        log.info("fit reason failed (%s)", e)
        t.fit_reason = ""
    return t
