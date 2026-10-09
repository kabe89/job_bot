"""Ultimate Networking Package — full, person-centric outreach prep kit.

Given a target person (name + optional company/role/context), this builds the
complete kit the way a sharp human strategist would:

  1. RESEARCH   — web / PubMed / lab-page intelligence on the person
                  (reuses research_enricher).
  2. PREP DOC   — a tailored master brief: who they are, why they matter, the
                  overlap with the user, a call game-plan, smart questions,
                  questions to expect, collaboration hooks, the soft ask,
                  logistics, a one-page cheat sheet, follow-up and landmines.
  3. OUTREACH   — ready-to-send messages (confirm, cold email, LinkedIn note,
                  reschedule) tuned to the chosen format.
  4. ONE-PAGER  — a research summary of the USER, written for this reader, to
                  send as a follow-up.
  5. PODCAST    — optional two-host audio breakdown of the prep (via podcast).

All generation is grounded in the gathered evidence and the user's real
profile/resume — prompts forbid invented facts. Everything degrades gracefully:
if the LLM is down a deterministic template is produced instead, and research
failures just shrink the evidence rather than aborting.

Public API
----------
build_package(name, company="", role="", context="", goals="", fmt="call",
              do_podcast=True, podcast_style="two_host", out_dir=None,
              generate=None, research=None) -> dict
"""
from __future__ import annotations

import logging
import re
from datetime import date
from pathlib import Path
from typing import Callable, Optional

from .config import settings

log = logging.getLogger("jobbot.networking_package")

FORMATS = ("call", "video", "email", "linkedin", "event", "coffee")


# ===========================================================================
# Context gathering
# ===========================================================================

def _slug(name: str, company: str = "") -> str:
    base = f"{name} {company}".strip()
    return re.sub(r"[^a-z0-9]+", "_", base.lower()).strip("_") or "contact"


def _user_context() -> str:
    """User profile + base resume as grounding text for tailoring."""
    parts: list[str] = []
    try:
        from . import research_enricher
        prof = research_enricher._profile_text()
        if prof:
            parts.append("## USER PROFILE\n" + prof)
    except Exception:  # noqa: BLE001
        pass
    try:
        from .resume import load_resume
        rp = Path(getattr(settings, "base_resume_path", "data/base_resume.md"))
        if rp.exists():
            txt = load_resume(rp) or ""
            if txt:
                parts.append("## USER RESUME\n" + txt[:8000])
    except Exception:  # noqa: BLE001
        pass
    return "\n\n".join(parts)


def _gather_research(name: str, company: str, role: str) -> dict:
    """Public-source intelligence on the person. Returns the raw evidence dict
    from research_enricher (pubmed / lab_page / general), or blanks on failure."""
    try:
        from . import research_enricher
        rsch = research_enricher._gather_raw_intelligence(
            {"name": name, "company": company, "title": role})
    except Exception as exc:  # noqa: BLE001
        log.warning("research gathering failed for %s: %s", name, exc)
        rsch = {"name": name, "title": role, "company": company,
                "pubmed": "", "lab_page": "", "general": ""}

    # If Ollama hosted web search is configured, run a model-driven (agentic)
    # research pass for richer, current intel and fold it into the evidence.
    try:
        from . import ollama_search
        if ollama_search.is_available():
            goal = (f"Background, current research, recent work, and notable "
                    f"facts about {name}"
                    + (f", {role}" if role else "")
                    + (f" at {company}" if company else "") + ".")
            summary = ollama_search.agentic_research(goal, max_rounds=3)
            if summary:
                rsch["general"] = (rsch.get("general", "") +
                                   "\n\n[Ollama live web research]\n" + summary)[:4000]
    except Exception as exc:  # noqa: BLE001
        log.debug("agentic research skipped: %s", exc)

    return rsch


def _default_generate(prompt: str) -> str:
    """LLM text generation: Gemini -> Claude -> Ollama (per research_enricher)."""
    from . import research_enricher
    return research_enricher._ai_generate(prompt) or ""


def _evidence_block(rsch: dict) -> str:
    return (
        f"NAME: {rsch.get('name','')}\n"
        f"ROLE/TITLE: {rsch.get('title','') or '(unknown)'}\n"
        f"ORGANIZATION: {rsch.get('company','') or '(unknown)'}\n\n"
        "PUBLICATIONS / PUBMED:\n" + (rsch.get("pubmed") or "(none found)") + "\n\n"
        "LAB / STAFF PAGE:\n" + (rsch.get("lab_page") or "(none found)") + "\n\n"
        "GENERAL WEB SNIPPETS:\n" + (rsch.get("general") or "(none found)")
    )


# ===========================================================================
# Prompts
# ===========================================================================

_NO_PLACEHOLDERS = (
    "IMPORTANT: Use the user's REAL name and contact details from their profile/"
    "resume above. NEVER output bracketed placeholders such as [Your Name], "
    "[Date], [Time], or [Your Phone Number] — substitute the actual value, or "
    "omit that detail entirely if it is genuinely not in the profile. "
    "NEVER guess, infer, or fabricate contact details (email, phone, handles): "
    "copy them verbatim from the profile/resume, and if one is absent simply "
    "leave it out — do not invent an address or add notes like '(inferred)'. "
    "An unknown future meeting time/date in a scheduling message may stay as a "
    "clearly bracketed blank for the user to fill."
)


_FORMAT_HINT = {
    "call": "a scheduled phone informational call",
    "video": "a scheduled video (Zoom) informational call",
    "email": "an email-based outreach and exchange",
    "linkedin": "a LinkedIn outreach and conversation",
    "event": "a run-in at a conference / seminar / campus event",
    "coffee": "an in-person coffee / face-to-face meeting",
}


def _prep_prompt(rsch, user_ctx, role, context, goals, fmt) -> str:
    fmt_desc = _FORMAT_HINT.get(fmt, "a networking conversation")
    return f"""You are an elite career & networking strategist. Write the ULTIMATE
preparation brief for the USER to network with the TARGET PERSON below, over {fmt_desc}.

Ground EVERYTHING in the evidence and the user's real background. Do NOT invent
facts about the person, and never fabricate what they will say or promise. Where
evidence is thin, say so plainly rather than guessing.

=== TARGET PERSON (evidence from public sources) ===
{_evidence_block(rsch)}

=== EXTRA CONTEXT FROM THE USER ===
Relationship/how they connect: {context or '(not specified)'}
User's goals for this contact: {goals or '(general relationship building + advice)'}

=== THE USER (reaching out) ===
{user_ctx or '(profile unavailable)'}

=== OUTPUT: Markdown, using these sections (scale each to what the evidence supports) ===
# Networking Prep — {rsch.get('name','the contact')}
## 1. Why this matters (the quick why)
## 2. Dossier — who they are (role, org, training, current work; cite specifics from evidence)
## 3. The overlap map (a table: their world | the user's matching strength | a talking move)
## 4. The user's 30-second pitch (ready to say aloud, grounded in the real resume)
## 5. Call game-plan (how to sequence the user's goals — relationship first, asks last)
## 6. Smart questions to ask (6-8, specific to their work and the user's goals)
## 7. Questions they may ask the user — with strong, honest answers
## 8. Collaboration hooks (1-3 concrete things the user could offer)
## 9. The one soft ask + close
## 10. Logistics & etiquette for {fmt_desc}
## 11. One-page cheat sheet (condensed, for during the conversation)
## 12. Follow-up (within 24h) + landmines to avoid

Be specific, honest, and genuinely useful. No filler, no clichés.
""" + _NO_PLACEHOLDERS


def _outreach_prompt(rsch, user_ctx, context, goals, fmt) -> str:
    return f"""You are a networking copywriter. Draft short, high-signal outreach
messages from the USER to the TARGET PERSON. Be honest — never claim shared history
or mutual contacts that aren't in the context. Lead with specifics about the
person's real work and the user's real background. No clichés, no "I hope this
finds you well", no AI throat-clearing.

=== TARGET PERSON ===
{_evidence_block(rsch)}

=== CONTEXT ===
How they connect: {context or '(not specified)'}
User goals: {goals or '(relationship building + advice)'}
Primary format: {_FORMAT_HINT.get(fmt, 'conversation')}

=== THE USER ===
{user_ctx or '(profile unavailable)'}

=== OUTPUT: Markdown with these labeled messages ===
# Outreach Messages — {rsch.get('name','the contact')}
## A. Cold/warm first email (subject + 120-170 word body, one concrete hook, one clear ask)
## B. LinkedIn connection note (<= 300 characters, specific, warm)
## C. Confirming a scheduled {fmt} (short)
## D. Polite reschedule (short)
End each message ready to copy-paste.

""" + _NO_PLACEHOLDERS


def _onepager_prompt(rsch, user_ctx) -> str:
    return f"""Write a one-page RESEARCH SUMMARY OF THE USER, tuned for the TARGET
PERSON below to read (e.g. as a post-conversation follow-up). It should make the
user's relevance obvious to this specific reader. Use ONLY real facts from the
user's profile/resume — no fabrication. Lead with the strengths most relevant to
the reader's field.

=== TARGET PERSON (the intended reader) ===
{_evidence_block(rsch)}

=== THE USER ===
{user_ctx or '(profile unavailable)'}

=== OUTPUT: Markdown, one page ===
# <User name> — Research One-Pager
- One-sentence positioning
- Current work / thesis in brief
- What the user brings (computational + experimental, mapped to the reader's world)
- Selected output (real publications/posters only)
- Where a collaboration could click (specific to the reader's systems)
- What the user is looking for + contact line
Keep it crisp and scannable.

""" + _NO_PLACEHOLDERS


# ===========================================================================
# Deterministic fallbacks (no LLM)
# ===========================================================================

def _fallback_prep(rsch, context, goals, fmt) -> str:
    return f"""# Networking Prep — {rsch.get('name','the contact')}
*(LLM unavailable — assembled from raw evidence. Review and flesh out manually.)*

## Target
- **Name:** {rsch.get('name','')}
- **Role:** {rsch.get('title','') or '(unknown)'}
- **Organization:** {rsch.get('company','') or '(unknown)'}
- **Format:** {_FORMAT_HINT.get(fmt, fmt)}
- **How you connect:** {context or '(not specified)'}
- **Your goals:** {goals or '(relationship building + advice)'}

## Publications / PubMed
{rsch.get('pubmed') or '(none found)'}

## Lab / staff page
{rsch.get('lab_page') or '(none found)'}

## General web
{rsch.get('general') or '(none found)'}

## Reminders
- Relationship first; lead with their work, not your ask.
- Have a 30-second honest pitch and 6-8 specific questions ready.
- Close with one soft ask + a request to stay in touch. Follow up within 24h.
"""


# ===========================================================================
# Orchestrator
# ===========================================================================

def build_package(name: str, company: str = "", role: str = "",
                  context: str = "", goals: str = "", fmt: str = "call",
                  do_podcast: bool = True, podcast_style: str = "two_host",
                  out_dir: Optional[str] = None,
                  generate: Optional[Callable[[str], str]] = None,
                  research: Optional[dict] = None) -> dict:
    """Build the full networking package for *name*. Returns a dict of artifacts:
    {name, slug, dir, research, prep_path, outreach_path, onepager_path,
     podcast_path, script_path, used_llm}.

    *generate* and *research* are injectable for testing (defaults hit the
    network/LLM).
    """
    if not name or not name.strip():
        raise ValueError("A contact name is required.")
    if fmt not in FORMATS:
        fmt = "call"
    name = name.strip()
    gen = generate or _default_generate

    slug = _slug(name, company)
    base = Path(out_dir) if out_dir else Path("output/networking") / slug
    base.mkdir(parents=True, exist_ok=True)

    rsch = research if research is not None else _gather_research(name, company, role)
    user_ctx = _user_context()

    def _try(prompt: str) -> str:
        try:
            return (gen(prompt) or "").strip()
        except Exception as exc:  # noqa: BLE001
            log.warning("generation failed: %s", exc)
            return ""

    prep = _try(_prep_prompt(rsch, user_ctx, role, context, goals, fmt))
    used_llm = bool(prep)
    if not prep:
        prep = _fallback_prep(rsch, context, goals, fmt)

    outreach = _try(_outreach_prompt(rsch, user_ctx, context, goals, fmt))
    onepager = _try(_onepager_prompt(rsch, user_ctx))

    today = date.today().isoformat()
    prep_path = base / f"{slug}_prep.md"
    prep_path.write_text(prep + f"\n\n---\n*Generated {today}.*\n", encoding="utf-8")

    out = {"name": name, "slug": slug, "dir": str(base), "research": rsch,
           "prep_path": str(prep_path), "outreach_path": None,
           "onepager_path": None, "podcast_path": None, "script_path": None,
           "used_llm": used_llm}

    if outreach:
        p = base / f"{slug}_outreach.md"
        p.write_text(outreach + f"\n\n---\n*Generated {today}.*\n", encoding="utf-8")
        out["outreach_path"] = str(p)
    if onepager:
        p = base / f"{slug}_onepager.md"
        p.write_text(onepager + f"\n\n---\n*Generated {today}.*\n", encoding="utf-8")
        out["onepager_path"] = str(p)

    if do_podcast:
        try:
            from . import podcast as pod
            res = pod.make_podcast(str(prep_path),
                                   out_path=str(base / f"{slug}_podcast.mp3"),
                                   style=podcast_style,
                                   title=f"Networking with {name}")
            out["podcast_path"] = res.get("audio_path")
            out["script_path"] = res.get("script_path")
        except Exception as exc:  # noqa: BLE001
            log.warning("podcast generation failed: %s", exc)

    return out
