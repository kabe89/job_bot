"""Claude client — Anthropic SDK (official `anthropic` package).

Mirrors the public surface of `gemini_client` so `ai_client` can dispatch to
either backend interchangeably:

    analyze_match, tailor_resume, generate_cover_letter,
    interview_prep, company_intel, advise_resume,
    is_available, probe, refresh_profile, reset_fallback_state

Design notes
------------
* Uses Claude Opus 4.8 by default with **adaptive thinking** + the **effort**
  parameter (Opus 4.8 rejects temperature/top_p/budget_tokens).
* **Prompt caching:** the frozen system instruction, the applicant profile,
  and the base resume are placed in a single cached system prefix. Because the
  base resume is identical for every job and every call in a run, every call
  after the first reads that prefix from cache (~0.1x cost) instead of
  re-billing it. Only the per-job instruction + job text are uncached.
* Streams automatically when max_tokens is large, to avoid SDK HTTP timeouts.
* Walks primary -> fallback model on overload (529) / transient errors.
"""
from __future__ import annotations

import json
import logging
import re
from typing import Optional

from tenacity import retry, stop_after_attempt, wait_exponential

from . import gemini_client as _g
from .config import settings

log = logging.getLogger("jobbot.claude")


def _profile_context() -> str:
    from .profile_context import applicant_context
    return applicant_context()


def refresh_profile() -> None:
    from .profile_context import refresh
    refresh()


# ----- SDK glue (anthropic) -------------------------------------------------
_client = None
_unavailable_models: set[str] = set()  # models that 5xx'd / errored this process


def _get_client():
    global _client
    if _client is not None:
        return _client
    if not settings.anthropic_api_key:
        raise RuntimeError("ANTHROPIC_API_KEY not set in .env")
    import anthropic
    _client = anthropic.Anthropic(api_key=settings.anthropic_api_key)
    return _client


def _is_overload_error(exc: Exception) -> bool:
    s = f"{type(exc).__name__} {exc}".lower()
    return any(tok in s for tok in (
        "overloaded", "529", "rate limit", "ratelimit", "429",
        "too many requests", "internal server", "500", "503",
    ))


def _model_chain() -> list[str]:
    out: list[str] = [settings.claude_model]
    fb = (settings.claude_fallback_model or "").strip()
    if fb and fb not in out:
        out.append(fb)
    return out


def _build_system(cache_text: str, include_profile: bool) -> list[dict]:
    """Build the cached system prefix.

    Block 1: frozen system instruction + (optional) applicant profile.
    Block 2: the base resume, carrying the cache_control breakpoint so the
             whole prefix (blocks 1+2) is cached together.
    """
    head = SYSTEM_RESUME_TAILOR
    if include_profile and _profile_context():
        head += ("\n\nADDITIONAL APPLICANT CONTEXT (treat as ground truth — do "
                 "not invent beyond this):\n" + _profile_context())
    blocks = [{"type": "text", "text": head}]
    if cache_text:
        blocks.append({"type": "text", "text": "CANDIDATE BASE RESUME:\n" + cache_text})
    # Cache the last block -> caches the entire system prefix before it.
    blocks[-1] = {**blocks[-1], "cache_control": {"type": "ephemeral"}}
    return blocks


@retry(stop=stop_after_attempt(2), wait=wait_exponential(min=2, max=20), reraise=True)
def _generate(prompt: str, cache_text: str = "", include_profile: bool = True,
              max_tokens: Optional[int] = None) -> str:
    """One generation. `cache_text` (the base resume) is placed in the cached
    system prefix; `prompt` carries the per-job instruction + job text."""
    client = _get_client()
    system = _build_system(cache_text, include_profile)
    mt = max_tokens or settings.claude_max_tokens
    chain = _model_chain()
    last_err: Exception | None = None

    for idx, model_name in enumerate(chain):
        if model_name in _unavailable_models:
            continue
        try:
            return _call_one(client, model_name, system, prompt, mt)
        except Exception as e:  # noqa: BLE001
            last_err = e
            if _is_overload_error(e) and idx + 1 < len(chain):
                _unavailable_models.add(model_name)
                log.warning("Claude model %s overloaded (%s); trying next: %s",
                            model_name, type(e).__name__, chain[idx + 1])
                continue
            raise
    if last_err:
        raise last_err
    raise RuntimeError("Claude chain exhausted with no usable model")


def _call_one(client, model_name: str, system: list[dict], prompt: str,
              max_tokens: int) -> str:
    req = dict(
        model=model_name,
        max_tokens=max_tokens,
        thinking={"type": "adaptive"},
        output_config={"effort": settings.claude_effort},
        system=system,
        messages=[{"role": "user", "content": prompt}],
    )
    # Stream above ~16k to dodge the SDK's long-request HTTP timeout guard.
    if max_tokens > 16000:
        with client.messages.stream(**req) as stream:
            msg = stream.get_final_message()
    else:
        msg = client.messages.create(**req)
    return _text_of(msg).strip()


def _text_of(msg) -> str:
    parts = []
    for block in msg.content:
        if getattr(block, "type", None) == "text":
            parts.append(block.text)
    return "".join(parts)


def reset_fallback_state() -> None:
    """Forget that a model was overloaded (used by tests)."""
    _unavailable_models.clear()


def is_available() -> bool:
    """True if an Anthropic API key is configured. Does NOT make a network
    call — for that, use `probe()`."""
    return bool(settings.anthropic_api_key)


_probe_cache: dict[str, bool] = {}


def probe(timeout_seconds: float = 5.0) -> bool:
    """Cheap one-shot Claude call to verify the key works. Cached per process."""
    if "ok" in _probe_cache:
        return _probe_cache["ok"]
    if not is_available():
        _probe_cache["ok"] = False
        return False
    try:
        client = _get_client()
        msg = client.messages.create(
            model=settings.claude_model,
            max_tokens=16,
            messages=[{"role": "user", "content": "Reply with: OK"}],
        )
        ok = bool(_text_of(msg).strip())
        _probe_cache["ok"] = ok
        return ok
    except Exception as e:  # noqa: BLE001
        log.warning("Claude probe failed (%s) — Claude features will be skipped.", e)
        _probe_cache["ok"] = False
        return False


def _parse_json_block(text: str) -> dict:
    m = re.search(r"```(?:json)?\s*(\{.*?\})\s*```", text, re.DOTALL)
    raw = m.group(1) if m else text
    start = raw.find("{")
    end = raw.rfind("}")
    if start == -1 or end == -1:
        return {}
    try:
        return json.loads(raw[start:end + 1])
    except json.JSONDecodeError:
        return {}


# The tailoring contract is shared with gemini_client / ollama_client so all three
# providers hold the candidate to the SAME truthfulness + no-hedging rules. This
# module used to keep its own 401-char paraphrase against the shared 2.9k contract,
# so switching provider silently swapped in a far weaker one. Do not re-fork it.
SYSTEM_RESUME_TAILOR = _g.SYSTEM_RESUME_TAILOR


# ----- Public API (mirrors gemini_client) -----------------------------------

def analyze_match(resume: str, job_title: str, company: str, job_description: str) -> dict:
    prompt = f"""Analyze the match between the candidate (resume in system context) and this job posting.

JOB TITLE: {job_title}
COMPANY: {company}
JOB DESCRIPTION:
{job_description[:6000]}

Return strict JSON with this shape (no prose outside JSON):
{{
  "score": <float 0-1, how well candidate matches>,
  "strengths": ["bullet", ...],
  "gaps": ["bullet", ...],
  "missing_keywords": ["keyword", ...],
  "summary": "one-paragraph honest assessment",
  "recommendation": "apply" | "tailor_first" | "skip"
}}"""
    out = _generate(prompt, cache_text=resume)
    return _parse_json_block(out) or {"score": 0.0, "summary": out[:500]}


def tailor_resume(resume: str, job_title: str, company: str, job_description: str) -> str:
    prompt = f"""Tailor the candidate's base resume (in system context) for the target role below. Rules:
- Keep all content TRUTHFUL — do not invent jobs, dates, degrees, or skills.
- Reorder bullets to put the most relevant achievements first.
- Rephrase bullets to mirror the job's language and keywords (ATS-friendly).
- Quantify impact where the original resume contains numbers.
- Keep the standard sections: Summary, Experience, Skills, Education, Projects.
- Output VALID GitHub-flavored Markdown only — no commentary.

JOB TITLE: {job_title}
COMPANY: {company}
JOB DESCRIPTION:
{job_description[:6000]}"""
    return _generate(prompt, cache_text=resume)


def generate_cover_letter(resume: str, job_title: str, company: str, job_description: str, applicant_name: str) -> str:
    prompt = f"""Write a concise, compelling cover letter (250-350 words) from {applicant_name}, drawing ONLY on the candidate's resume in system context.

Rules:
- Open with a specific hook tied to the company/role.
- 2 short paragraphs of relevant experience drawn ONLY from the resume.
- Close with a forward-looking call to interview.
- Sound human, not AI-generic. No clichés ("dynamic team player").
- Plain text, no markdown headers. Sign off with "Sincerely,\\n{applicant_name}".

JOB TITLE: {job_title}
COMPANY: {company}
JOB DESCRIPTION:
{job_description[:5000]}"""
    return _generate(prompt, cache_text=resume)


def interview_prep(resume: str, job_title: str, company: str, job_description: str) -> str:
    prompt = f"""You are a senior technical hiring manager preparing the candidate (resume in system context) for an upcoming interview. Produce a comprehensive prep doc.

JOB TITLE: {job_title}
COMPANY: {company}
JOB DESCRIPTION:
{job_description[:6500]}

Output Markdown with these sections:

## Role decoded
2-3 sentences on what this team likely actually does day-to-day and the unstated priorities behind the JD.

## Top 10 predicted interview questions (technical + behavioral, mixed)
For each question include:
- **Q**: the question
- **Why they'd ask**: what they're probing for
- **STAR-ready answer hook**: a 1-sentence pointer to which of the candidate's experiences to use

## 5 hard technical questions you should be ready for
Specific to the techniques / disease areas / methods in the JD. Include the answer outline (3-4 bullets each) — accurate, expert-level.

## 5 behavioral / culture-fit questions
Tailored to the company's stated values or what's typical for this role's seniority.

## Questions YOU should ask them
6 sharp, role-specific questions that signal scientific maturity and genuine interest.

## Red flags to watch for in the interview
3-5 signs this team may not be a great fit.

## Study list (24-hour cram)
A bulleted list of papers, methods, products, or concepts to brush up on, prioritized.

## Salary / negotiation talking points
2-3 specific data points to anchor the comp conversation for this role + location.

Be specific, scientific, and honest. No fluff."""
    return _generate(prompt, cache_text=resume)


def company_intel(company: str, job_title: str, job_description: str) -> str:
    prompt = f"""Brief the candidate on this company before their interview for the role below.

COMPANY: {company}
ROLE: {job_title}
JD (for context only):
{job_description[:3000]}

Output Markdown:

## What they do (2-3 sentences)
## Recent science / products / pipeline (3-5 bullets)
## Org / culture quick-take (size, location, public/private, recent funding or layoffs if relevant)
## How to position yourself for THIS company (3 bullets)
## One specific hook to mention (a paper, a product launch, a value)

Honest, factual, specific. If you genuinely don't know, say so rather than invent."""
    # No resume needed -> separate (small) cache key.
    return _generate(prompt, cache_text="", include_profile=True)


def interview_question_bank(resume: str, job_title: str, company: str,
                            job_description: str, company_intel: str,
                            contact: Optional[dict] = None) -> dict:
    spec = _g._contact_bank_spec(contact) if contact else _g._BANK_SPEC
    prompt = f"""You are the hiring manager for the role below, listing the questions
you would actually ask this candidate.

JOB TITLE: {job_title}
COMPANY: {company}
JOB DESCRIPTION:
{job_description[:5000]}

COMPANY INTEL BRIEF:
{(company_intel or '')[:3000]}

{spec}"""
    out = _generate(prompt, cache_text=resume[:6000])
    return _g._normalize_bank(_g._parse_json_block(out))


def score_answer(question: str, question_kind: str, answer: str,
                 job_title: str, company: str, resume: str) -> dict:
    out = _generate(_g._score_prompt(question, question_kind, answer,
                                     job_title, company, ""),
                    cache_text=resume[:4000])
    return _g._normalize_rubric(_g._parse_json_block(out))


def interview_session_summary(job_title: str, company: str, transcript: str) -> str:
    return _generate(_g._summary_prompt(job_title, company, transcript))


def advise_resume(resume: str, focus: str = "") -> str:
    prompt = f"""Review the candidate's base resume (in system context) and give targeted, actionable improvement advice.
Focus area (optional): {focus or "general impact, clarity, ATS-readiness"}

Output Markdown with sections:
## Quick wins
## Bullet rewrites (before -> after for 3-5 weakest bullets)
## Missing sections / additions
## Keyword & ATS suggestions
## Overall score (out of 10) + one-line verdict"""
    return _generate(prompt, cache_text=resume)


def _align_answer_keys(parsed: dict, questions: list[str]) -> dict:
    """Map the model's answer keys back to the EXACT question strings.

    Models often key by "1. <question>", "1)", a bare "1", or a slightly
    reworded question. Normalise all of these so the returned dict is keyed by
    the original question text — otherwise downstream lookups (e.g. the
    blank-detection in browser_apply) silently miss every answer.
    """
    if not parsed:
        return {}

    def _norm(s: str) -> str:
        return re.sub(r"[^a-z0-9]+", "", str(s).lower())

    by_norm = {_norm(q): q for q in questions}
    out: dict = {}
    for k, v in parsed.items():
        ks = str(k).strip()
        # Strip a leading "N." / "N)" / "N -" / "N:" numbering prefix.
        m = re.match(r"^\s*(\d+)\s*[.)\-:]\s*(.*)$", ks)
        if m:
            rest = m.group(2).strip()
            if rest and _norm(rest) in by_norm:
                out[by_norm[_norm(rest)]] = v
                continue
            idx = int(m.group(1))
            if 0 < idx <= len(questions):
                out[questions[idx - 1]] = v
                continue
        if ks.isdigit() and 0 < int(ks) <= len(questions):
            out[questions[int(ks) - 1]] = v
            continue
        if _norm(ks) in by_norm:
            out[by_norm[_norm(ks)]] = v
            continue
        out[k] = v  # unrecognised — keep as-is
    return out


def answer_application_questions(resume: str, job_title: str, company: str,
                                 job_description: str, questions: list[str]) -> dict:
    """Answer free-text application-form questions truthfully from the resume.

    Returns {question: answer}. Used by the assisted browser-apply flow.
    """
    if not questions:
        return {}
    numbered = "\n".join(f"{i+1}. {q}" for i, q in enumerate(questions))
    prompt = f"""You are filling out a job application form for the candidate (resume in system context).
Answer each question below TRUTHFULLY using only the resume/profile. Keep answers
concise and professional (1-4 sentences unless the question implies an essay).
If the resume genuinely lacks the information, answer with an empty string "".

JOB TITLE: {job_title}
COMPANY: {company}
JOB DESCRIPTION:
{job_description[:4000]}

QUESTIONS:
{numbered}

Return strict JSON: an object mapping the EXACT question text to the answer string.
No prose outside JSON."""
    out = _generate(prompt, cache_text=resume)
    parsed = _parse_json_block(out)
    return _align_answer_keys(parsed, questions)


SYSTEM_NETWORKING = """You are a sharp, warm professional networking coach helping a job-seeking scientist reach out to real people. You write outreach that sounds genuinely human — specific, concise, and never templated. You are scrupulously HONEST: you never invent shared history, mutual connections, prior conversations, or experience the candidate does not have. Ground every claim about the candidate in their resume/profile provided in the system context."""


def networking_message(resume: str, job_title: str, company: str,
                       job_description: str, contact: dict,
                       channel: str = "linkedin") -> dict:
    """Draft a tailored outreach message to a specific professional.

    channel="linkedin" -> short connection note, <=300 chars (subject empty).
    channel="email"    -> longer outreach email with a subject line.

    Returns {"subject": str, "body": str}.
    """
    name = (contact.get("name") or "").strip()
    role = (contact.get("role") or "").strip()
    generic_roles = {"", "linkedin", "from posting", "unknown"}
    if name and role.lower() not in generic_roles:
        who = f"{name} ({role} at {company})" if company else f"{name} ({role})"
    elif name:
        who = f"{name} at {company}" if company else name
    else:
        who = f"a professional at {company}" if company else "a professional at the company"

    if channel == "email":
        spec = (
            'Write a warm, specific networking EMAIL of about 120-170 words. '
            'Return STRICT JSON: {"subject": "...", "body": "..."}. The subject '
            'line is under 60 characters. The body greets the contact by first '
            'name, introduces the candidate in one honest line using their real '
            'background, expresses genuine interest in the role/team with ONE '
            'concrete hook drawn from the job or company, and asks for a brief '
            'chat or to be pointed to the right person. Sign off with the '
            "candidate's name. Plain-text body; use \\n for line breaks."
        )
    else:
        spec = (
            'Write a LinkedIn connection note with a HARD LIMIT of 280 '
            'characters including spaces. Return STRICT JSON: '
            '{"subject": "", "body": "..."}. Address the contact by first name '
            'if known, state in one line who the candidate is and why '
            'connecting is relevant to this role/team, and end with a light '
            'ask to connect. Warm and specific, never generic. No hashtags, '
            'no links.'
        )

    prompt = f"""Help the candidate (resume/profile in system context) network toward a job.

TARGET ROLE: {job_title} at {company}
JOB CONTEXT (use only for one specific hook):
{job_description[:2500]}

CONTACT TO REACH OUT TO: {who}
CONTACT ROLE: {role or 'unknown'}

{spec}

Rules:
- Be HONEST: never claim shared history, mutual connections, or experience the candidate lacks.
- Ground the candidate's one-line intro in their real resume/profile.
- Sound like a real person. No clichés, no "I hope this finds you well", no AI throat-clearing."""
    out = _generate(prompt, cache_text=resume)
    return _coerce_networking(out, channel)


def _coerce_networking(out: str, channel: str) -> dict:
    parsed = _parse_json_block(out)
    body = (parsed.get("body") or "").strip() if parsed else ""
    subject = (parsed.get("subject") or "").strip() if parsed else ""
    if not body:
        body = (out or "").strip()
    if channel != "email" and len(body) > 300:
        body = body[:297].rstrip() + "..."
    return {"subject": subject, "body": body}
