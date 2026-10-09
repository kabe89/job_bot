"""Ollama hosted Web Search — live web search + page fetch for the bot.

Ollama offers a hosted web-search API (https://ollama.com) that any client can
call with an API key. This module exposes it to jobbot so research/networking
flows can pull *live* results instead of relying only on Serper/CSE/ddgs, and so
a local model can be driven agentically (it decides what to search).

Set OLLAMA_API_KEY in .env (free tier available at ollama.com -> API keys).

Endpoints (POST, Bearer auth):
  {base}/web_search   {"query": str, "max_results": int} -> {"results":[{title,url,content}]}
  {base}/web_fetch    {"url": str}                        -> {"title","content","links"}

Everything degrades to [] / "" on any failure and never raises.

API
---
is_available()                     -> bool
web_search(query, max_results=5)   -> list[{title, url, snippet}]
web_fetch(url)                     -> str   (page text)
agentic_research(goal, ...)        -> str   (model-driven multi-search summary)
"""
from __future__ import annotations

import logging
from typing import List, Optional

import requests

from .config import settings

log = logging.getLogger("jobbot.ollama_search")


def _key() -> str:
    return (getattr(settings, "ollama_api_key", "") or "").strip()


def _base() -> str:
    return (getattr(settings, "ollama_web_base", "") or "https://ollama.com/api").rstrip("/")


def is_available() -> bool:
    """True if an Ollama web-search API key is configured."""
    return bool(_key())


def _timeout() -> int:
    return int(getattr(settings, "apply_fetch_timeout", 30) or 30)


def _post(path: str, payload: dict) -> Optional[dict]:
    if not _key():
        return None
    try:
        resp = requests.post(
            f"{_base()}/{path}",
            headers={"Authorization": f"Bearer {_key()}",
                     "Content-Type": "application/json"},
            json=payload,
            timeout=_timeout(),
        )
        if resp.status_code != 200:
            log.debug("Ollama %s -> HTTP %s: %s", path, resp.status_code,
                      resp.text[:200])
            return None
        return resp.json()
    except Exception as exc:  # noqa: BLE001
        log.debug("Ollama %s failed: %s", path, exc)
        return None


def web_search(query: str, max_results: int = 5) -> List[dict]:
    """Live web search via Ollama. Returns [{title, url, snippet}]; [] on failure.

    Normalizes the API's `content` field to `snippet` so results are drop-in
    compatible with the rest of jobbot's search helpers.
    """
    data = _post("web_search", {"query": query, "max_results": max_results})
    if not data:
        return []
    rows = data.get("results") or data.get("data") or []
    out: List[dict] = []
    for r in rows:
        url = r.get("url") or r.get("link") or ""
        if not url:
            continue
        out.append({
            "title": r.get("title", ""),
            "url": url,
            "snippet": r.get("content") or r.get("snippet") or r.get("description") or "",
        })
        if len(out) >= max_results:
            break
    return out


def web_fetch(url: str, max_chars: int = 6000) -> str:
    """Fetch a URL's main text via Ollama's web_fetch. '' on failure."""
    data = _post("web_fetch", {"url": url})
    if not data:
        return ""
    text = data.get("content") or data.get("text") or ""
    return text[:max_chars]


def selftest(query: str = "Ollama web search") -> dict:
    """One live call to verify the key works. Returns {ok, configured, count, error}."""
    out = {"ok": False, "configured": is_available(), "count": 0, "error": ""}
    if not out["configured"]:
        out["error"] = ("No OLLAMA_API_KEY set. Get a free key at "
                        "https://ollama.com (Settings -> API keys) and add "
                        "OLLAMA_API_KEY=... to your .env.")
        return out
    rows = web_search(query, 3)
    out["count"] = len(rows)
    out["ok"] = bool(rows)
    if not rows:
        out["error"] = "Key set but search returned nothing (check quota / key)."
    return out


def agentic_research(goal: str, max_rounds: int = 3,
                     max_results: int = 5) -> str:
    """Model-driven research: the local Ollama model proposes search queries,
    we run them via the hosted web-search API, feed results back, and let it
    iterate up to *max_rounds* before writing a grounded summary.

    Falls back to a single direct search + summary if tool-style control fails.
    Returns a markdown summary ('' if nothing could be gathered).
    """
    if not is_available():
        return ""
    from . import ollama_client
    if not ollama_client.is_available():
        # No local model to drive the loop — just dump raw search results.
        rows = web_search(goal, max_results)
        return "\n".join(f"- {r['title']}: {r['snippet']} [{r['url']}]"
                         for r in rows) if rows else ""

    gathered: list[str] = []
    seen_queries: set[str] = set()
    next_query = goal
    for _ in range(max(1, max_rounds)):
        q = (next_query or "").strip()
        if not q or q.lower() in seen_queries:
            break
        seen_queries.add(q.lower())
        rows = web_search(q, max_results)
        if rows:
            gathered.append(f"### Results for: {q}\n" + "\n".join(
                f"- {r['title']}: {r['snippet']} [{r['url']}]" for r in rows))
        # Ask the model what to search next (or to stop).
        try:
            decide = ollama_client._generate(
                "You are researching this goal:\n" + goal +
                "\n\nResults gathered so far:\n" + ("\n\n".join(gathered) or "(none)") +
                "\n\nIf more searching would help, reply with ONLY the next search "
                "query (a single line). If you have enough to write a solid "
                "summary, reply with exactly: DONE",
                temperature=0.3, include_profile=False)
        except Exception:  # noqa: BLE001
            break
        decide = (decide or "").strip().splitlines()[0].strip() if decide else "DONE"
        if not decide or decide.upper().startswith("DONE"):
            break
        next_query = decide

    if not gathered:
        return ""
    try:
        return ollama_client._generate(
            "Using ONLY the web results below, write a concise, factual research "
            "briefing for this goal:\n" + goal + "\n\nWEB RESULTS:\n" +
            "\n\n".join(gathered) + "\n\nDo not invent facts beyond the results. "
            "Cite URLs inline where relevant.",
            temperature=0.3, include_profile=False) or ""
    except Exception:  # noqa: BLE001
        return "\n\n".join(gathered)
