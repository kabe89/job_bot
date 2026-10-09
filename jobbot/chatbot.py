"""Local AI Chatbot with Resume Awareness and Live Online Job Evaluation.

Integrates the user's verified candidate profile and resume (data/base_resume.md,
profile.md, skills) with local AI (Ollama / Qwen / Llama) and live online search
(DuckDuckGo Lite, Wikipedia, and configured search backends) to evaluate jobs,
benchmark compensation, investigate employer reputation, and strategize applications.
"""
from __future__ import annotations

import logging
import re
from pathlib import Path
from typing import Any, Dict, List, Optional

# Ensure native OS SSL certificate store is used on Windows
try:
    import truststore
    truststore.inject_into_ssl()
except Exception:
    pass

import requests
from bs4 import BeautifulSoup

from .config import settings
from .models import Job, session
from .profile_context import applicant_context

log = logging.getLogger("jobbot.chatbot")

_UA = (
    "Mozilla/5.0 (Windows NT 10.0; Win64; x64) AppleWebKit/537.36 "
    "(KHTML, like Gecko) Chrome/131.0.0.0 Safari/537.36"
)


def get_candidate_context() -> str:
    """Return the candidate's complete background: base resume, profile, and verified skills."""
    parts = []

    # 1. Base Resume if available
    base_res_p = Path(settings.base_resume_path)
    if base_res_p.exists():
        try:
            content = base_res_p.read_text(encoding="utf-8").strip()
            if content:
                parts.append("### BASE RESUME\n" + content)
        except Exception as e:
            log.debug("Failed reading base resume: %s", e)

    # 2. Freeform profile & verified skills & employment history
    ctx = applicant_context()
    if ctx:
        parts.append("### APPLICANT PROFILE & VERIFIED SKILLS\n" + ctx)

    return "\n\n".join(parts).strip()


def get_job_context(job_id: Optional[int] = None) -> Optional[Dict[str, Any]]:
    """Retrieve structured job metadata and description by job ID."""
    if not job_id:
        return None
    with session() as db:
        job = db.get(Job, job_id)
        if not job:
            return None
        return {
            "id": job.id,
            "title": job.title or "Position",
            "company": job.company or "Company",
            "location": job.location or "Unspecified",
            "salary": job.salary or "",
            "status": job.status or "new",
            "match_score": job.match_score or 0.0,
            "url": job.url or "",
            "notes": job.notes or "",
            "description": (job.description or "")[:7000],  # Bound prompt length
        }


def search_online_for_company(company: str, role: str = "", query_override: str = "") -> List[Dict[str, str]]:
    """Perform live web search for employer intelligence, culture, news, and salaries."""
    if not company and not query_override:
        return []

    q = query_override
    if not q:
        clean_company = re.sub(r"\b(inc|corp|llc|ltd|co)\b", "", company, flags=re.IGNORECASE).strip()
        role_part = f" {role}" if role else ""
        q = f"{clean_company}{role_part} reviews salary culture"

    results: List[Dict[str, str]] = []

    # 1. DuckDuckGo Lite search via requests (native SSL, keyless, reliable)
    try:
        resp = requests.post(
            "https://lite.duckduckgo.com/lite/",
            data={"q": q},
            headers={"User-Agent": _UA, "Accept-Language": "en-US,en;q=0.9"},
            timeout=15,
        )
        if resp.status_code == 200:
            soup = BeautifulSoup(resp.text, "html.parser")
            links = soup.select("a.result-link")
            snippets = soup.select("td.result-snippet")
            for idx in range(min(len(links), len(snippets), 5)):
                a = links[idx]
                s = snippets[idx]
                href = a.get("href", "")
                title = a.get_text(strip=True)
                snip = s.get_text(strip=True)
                if href and title:
                    results.append({"title": title, "url": href, "snippet": snip})
            if results:
                return results
    except Exception as e:
        log.debug("DDG Lite search failed: %s", e)

    # 2. Wikipedia overview lookup as fallback
    try:
        clean_c = re.sub(r"\b(inc|corp|llc|ltd|co)\b", "", company, flags=re.IGNORECASE).strip()
        w_resp = requests.get(
            "https://en.wikipedia.org/w/api.php",
            params={"action": "query", "list": "search", "srsearch": clean_c, "format": "json"},
            headers={"User-Agent": "JobBot/1.0 (contact@jobbot.local)"},
            timeout=10,
        )
        if w_resp.status_code == 200:
            items = w_resp.json().get("query", {}).get("search", [])
            for item in items[:3]:
                title = item.get("title", "")
                snippet = BeautifulSoup(item.get("snippet", ""), "html.parser").get_text(strip=True)
                results.append({
                    "title": f"Wikipedia: {title}",
                    "url": f"https://en.wikipedia.org/wiki/{title.replace(' ', '_')}",
                    "snippet": snippet,
                })
            if results:
                return results
    except Exception as e:
        log.debug("Wikipedia fallback failed: %s", e)

    # 3. Try Serper or Google CSE if configured in settings
    try:
        from . import contacts
        if getattr(settings, "serper_api_key", ""):
            rows = contacts._serper_search(q, max_results=5)
            if rows:
                return rows
        if getattr(settings, "google_cse_key", "") and getattr(settings, "google_cse_cx", ""):
            rows = contacts._cse_search(q, max_results=5)
            if rows:
                return rows
    except Exception as e:
        log.debug("Contacts search fallback failed: %s", e)

    return results


def build_system_prompt(candidate_ctx: str, job_ctx: Optional[Dict[str, Any]] = None,
                        web_results: Optional[List[Dict[str, str]]] = None) -> str:
    """Construct an authoritative system prompt anchoring the model to truthfulness and context."""
    blocks = [
        "You are JobBot AI Advisor — an elite career coach, scientific recruiter, and job strategist.",
        "You have direct access to the candidate's verified resume and background below.",
        "CRITICAL RULES:",
        "1. GROUND TRUTH: Treat the candidate's background as immutable facts. Do NOT invent degrees, credentials, publications, or skills.",
        "2. CANDID EVALUATION: Provide honest, constructive feedback. Highlight genuine strengths, but also clearly point out qualification gaps and recommend realistic strategies to bridge them.",
        "3. STRATEGIC INSIGHTS: When discussing a job, synthesize the posting requirements, the candidate's profile, and any online employer intelligence.",
        "",
        "=== CANDIDATE PROFILE & RESUME ===",
        candidate_ctx or "(No resume loaded)",
    ]

    if job_ctx:
        desc_preview = job_ctx.get("description", "")
        blocks.extend([
            "",
            "=== TARGET JOB IN FOCUS ===",
            f"Job ID: #{job_ctx.get('id', 'N/A')}",
            f"Title: {job_ctx.get('title')}",
            f"Company: {job_ctx.get('company')}",
            f"Location: {job_ctx.get('location')}",
            f"Salary / Pay: {job_ctx.get('salary') or 'Not listed'}",
            f"Status: {job_ctx.get('status')}",
            f"Posting URL: {job_ctx.get('url')}",
            "Description:",
            desc_preview,
        ])

    if web_results:
        blocks.extend([
            "",
            "=== LIVE ONLINE RESEARCH & EMPLOYER INTELLIGENCE ===",
        ])
        for idx, res in enumerate(web_results, 1):
            blocks.append(f"{idx}. [{res.get('title')}]({res.get('url')}): {res.get('snippet')}")

    return "\n".join(blocks)


def call_llm(messages: List[Dict[str, str]], system_prompt: str, temperature: float = 0.5) -> str:
    """Dispatch chat prompt to Ollama (local) or active AI provider with robust fallback."""
    formatted_convo = []
    for m in messages:
        role = m.get("role", "user").capitalize()
        content = m.get("content", "")
        formatted_convo.append(f"{role}: {content}")
    convo_text = "\n\n".join(formatted_convo)

    full_prompt = f"{system_prompt}\n\n=== CONVERSATION HISTORY ===\n{convo_text}\n\nAssistant:"

    # 1. Primary: Ollama client (local offline AI)
    try:
        from . import ollama_client
        if ollama_client.is_available():
            reply = ollama_client._generate(
                prompt=convo_text,
                system=system_prompt,
                temperature=temperature,
                include_profile=False,  # Already included in system_prompt
            )
            if reply and reply.strip():
                return reply.strip()
    except Exception as e:
        log.warning("Ollama chat failed (%s), attempting fallback...", e)

    # 2. Fallback: Gemini or Claude via ai_client
    try:
        from . import gemini_client, claude_client
        if claude_client.is_available():
            reply = claude_client._generate(full_prompt, temperature=temperature)
            if reply and reply.strip():
                return reply.strip()
        if gemini_client.is_available():
            reply = gemini_client._generate(full_prompt, temperature=temperature)
            if reply and reply.strip():
                return reply.strip()
    except Exception as exc:
        log.error("Cloud AI fallback failed: %s", exc)

    return ("I could not generate a response. Please ensure Ollama is running (`ollama serve`) "
            "or set GEMINI_API_KEY / ANTHROPIC_API_KEY in your .env configuration.")


import json


def load_chat_history(job_id: int) -> List[Dict[str, str]]:
    """Load stored conversation history for a specific job."""
    with session() as db:
        job = db.get(Job, job_id)
        if not job or not job.chat_history_json:
            return []
        try:
            return json.loads(job.chat_history_json)
        except Exception:
            return []


def save_chat_history(job_id: int, history: List[Dict[str, str]]) -> None:
    """Save conversation history for a specific job (bounded to last 50 turns)."""
    with session() as db:
        job = db.get(Job, job_id)
        if job:
            job.chat_history_json = json.dumps(history[-50:])
            db.commit()


def chat_turn(messages: List[Dict[str, str]], job_id: Optional[int] = None,
              enable_web: bool = True) -> Dict[str, Any]:
    """Execute a single conversational turn with candidate context, job anchor, and online search."""
    candidate_ctx = get_candidate_context()
    job_ctx = get_job_context(job_id) if job_id else None

    # Determine if we should perform online search
    web_results = []
    if enable_web:
        last_user_msg = ""
        for m in reversed(messages):
            if m.get("role") == "user":
                last_user_msg = m.get("content", "")
                break

        company = job_ctx.get("company", "") if job_ctx else ""
        role = job_ctx.get("title", "") if job_ctx else ""

        # Trigger search if job is loaded or user asked about news/company/reviews
        search_terms = ["company", "culture", "salary", "news", "reviews", "glassdoor", "reputation"]
        should_search = bool(company) or any(term in last_user_msg.lower() for term in search_terms)

        if should_search:
            web_results = search_online_for_company(company=company, role=role, query_override="")

    system_prompt = build_system_prompt(candidate_ctx, job_ctx, web_results)
    reply = call_llm(messages, system_prompt)

    # Persist chat history if anchored to a job
    if job_id:
        updated_history = list(messages)
        updated_history.append({"role": "assistant", "content": reply})
        save_chat_history(job_id, updated_history)

    return {
        "reply": reply,
        "job": job_ctx,
        "web_results": web_results,
    }


def evaluate_job(job_id: int, enable_web: bool = True) -> Dict[str, Any]:
    """Generate a structured, in-depth evaluation of a specific job posting."""
    job_ctx = get_job_context(job_id)
    if not job_ctx:
        return {"error": f"Job #{job_id} not found."}

    candidate_ctx = get_candidate_context()
    web_results = []
    if enable_web:
        web_results = search_online_for_company(
            company=job_ctx.get("company", ""),
            role=job_ctx.get("title", ""),
        )

    eval_prompt = (
        f"Perform an exhaustive, deep evaluation of Job #{job_id}: {job_ctx.get('title')} at {job_ctx.get('company')}.\n\n"
        "Please structure your evaluation with the following clear headings:\n"
        "1. 🎯 Executive Summary & Strategic Fit Rating (Score 1-100%)\n"
        "2. 💪 Key Candidate Strengths & Direct Match Evidence (from resume/profile)\n"
        "3. ⚠️ Critical Qualification Gaps & Honest Risks\n"
        "4. 🌐 Employer Intelligence & Market Sentiment (incorporating web research findings)\n"
        "5. 💰 Compensation & Salary Analysis (market estimate vs requirements)\n"
        "6. 🚀 Actionable Tactical Recommendations (Resume tailoring angles, networking hooks, interview traps to avoid)"
    )

    messages = [{"role": "user", "content": eval_prompt}]
    system_prompt = build_system_prompt(candidate_ctx, job_ctx, web_results)
    reply = call_llm(messages, system_prompt, temperature=0.3)

    # Persist evaluation into chat history for this job
    history = load_chat_history(job_id)
    history.append({"role": "user", "content": "🎯 Run deep strategic evaluation for this role."})
    history.append({"role": "assistant", "content": reply})
    save_chat_history(job_id, history)

    return {
        "job": job_ctx,
        "evaluation": reply,
        "web_results": web_results,
    }
