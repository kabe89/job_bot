"""Gemini client — google-genai SDK (current, supported).

Replaces the deprecated `google.generativeai` package with `google.genai`.
Adds quota/rate-limit fallback to a secondary model.
"""
from __future__ import annotations

import json
import logging
import re
from typing import Optional

from tenacity import retry, stop_after_attempt, wait_exponential

from .config import settings

log = logging.getLogger("jobbot.gemini")


def _profile_context() -> str:
    from .profile_context import applicant_context
    return applicant_context()


def refresh_profile() -> None:
    from .profile_context import refresh
    refresh()


def _template_block(kind: str = "resume") -> str:
    """House style template injected into tailoring/cover prompts (or "")."""
    from .resume import template_block
    return template_block(kind)


# ----- SDK glue (google-genai) ---------------------------------------------
_client = None
_quota_exhausted: set[str] = set()  # primary models that have 429'd this process


def _get_client():
    global _client
    if _client is not None:
        return _client
    if not settings.gemini_api_key:
        raise RuntimeError("GEMINI_API_KEY not set in .env")
    from google import genai
    _client = genai.Client(api_key=settings.gemini_api_key)
    return _client


def _is_quota_error(exc: Exception) -> bool:
    s = f"{type(exc).__name__} {exc}".lower()
    return any(tok in s for tok in (
        "resourceexhausted", "resource exhausted", "quota", "rate limit",
        "ratelimit", "429", "too many requests",
    ))


def _call_one(model_name: str, full: str, temperature: float) -> str:
    """One generation against a specific model via the google-genai SDK."""
    from google.genai import types
    client = _get_client()
    resp = client.models.generate_content(
        model=model_name,
        contents=full,
        config=types.GenerateContentConfig(
            temperature=temperature,
            max_output_tokens=30000,
        ),
    )
    return (resp.text or "").strip()


def _model_chain() -> list[str]:
    """Build the ordered model list: primary first, then fallback chain
    (de-duplicated, in order)."""
    primary = settings.gemini_model
    chain_raw = (settings.gemini_fallback_chain or "")
    # Back-compat: honor the single gemini_fallback_model too.
    extras = [m.strip() for m in chain_raw.split(",")] + [settings.gemini_fallback_model]
    out: list[str] = [primary]
    for m in extras:
        if m and m not in out:
            out.append(m)
    return out


@retry(stop=stop_after_attempt(2), wait=wait_exponential(min=2, max=20), reraise=True)
def _generate(prompt: str, system: Optional[str] = None, temperature: float = 0.4,
              include_profile: bool = True) -> str:
    parts = []
    if system:
        parts.append(system)
    if include_profile and _profile_context():
        parts.append("ADDITIONAL APPLICANT CONTEXT (treat as ground truth — do not invent beyond this):\n" + _profile_context())
    parts.append(prompt)
    full = "\n\n".join(parts)

    chain = _model_chain()
    last_err: Exception | None = None
    for idx, model_name in enumerate(chain):
        if model_name in _quota_exhausted:
            continue
        try:
            return _call_one(model_name, full, temperature)
        except Exception as e:  # noqa: BLE001
            last_err = e
            if _is_quota_error(e) and idx + 1 < len(chain):
                _quota_exhausted.add(model_name)
                log.warning(
                    "Model %s hit quota/rate-limit (%s); trying next: %s",
                    model_name, type(e).__name__, chain[idx + 1],
                )
                continue
            raise
    if last_err:
        raise last_err
    raise RuntimeError("Gemini chain exhausted with no usable model")


def reset_fallback_state() -> None:
    """Forget that the primary model was rate-limited (used by tests)."""
    _quota_exhausted.clear()


def is_available() -> bool:
    """True if a Gemini API key is configured. Does NOT make a network call;
    a True result doesn't guarantee the key is still valid — for that, call
    `probe()` (which actually hits the API and is slow)."""
    return bool(settings.gemini_api_key)


_probe_cache: dict[str, bool] = {}


def probe(timeout_seconds: float = 5.0) -> bool:
    """Cheap one-shot Gemini call to verify the key works.

    Cached per-process: if a previous probe failed we won't re-hit the API
    until `reset_fallback_state()` clears it.
    """
    if "ok" in _probe_cache:
        return _probe_cache["ok"]
    if not is_available():
        _probe_cache["ok"] = False
        return False
    try:
        out = _call_one(settings.gemini_model, "Reply with: OK", 0.0)
        ok = bool(out)
        _probe_cache["ok"] = ok
        return ok
    except Exception as e:  # noqa: BLE001
        log.warning("Gemini probe failed (%s) — Gemini features will be skipped.", e)
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


SYSTEM_RESUME_TAILOR = """You are an expert resume and technical document editor with deep knowledge of engineering, software architecture, data systems, and modern technical hiring practices. Reorganize, rephrase, and emphasize EXISTING material to align with the target role.

HUMAN VOICE & ANTI-AI CLICHÉ CONTRACT:
- Write in an authentic, natural professional voice as if the candidate wrote it themselves for an expert hiring peer.
- STRICTLY BAN robotic AI buzzwords and corporate filler: NEVER use "spearheaded", "leveraged", "pioneered", "utilized", "demonstrated proficiency in", "synergized", "executed end-to-end workflows", "fostered a collaborative environment", "results-driven", "seasoned".
- Use direct, natural, active technical verbs: Designed, Architected, Developed, Optimized, Characterized, Profiled, Modeled, Benchmarked, Formulated, Built, Validated, Troubleshot, Quantified, Analyzed.
- VARY sentence structures across bullets. Do NOT mechanically repeat the formulaic "Action verb + Object + by Method + resulting in Metric%" frame on every single line. Mix technical protocols, architectural insights, computational models, and concrete findings naturally.
- Explicitly name real technical systems, frameworks, instruments, and workflows from the candidate's verified record.
- Professional Summary: Write 3–4 natural sentences with authentic authority summarizing core identity, primary technical strengths, and targeted fit.

TRUTHFULNESS CONTRACT (overrides every other instruction, including keyword matching):
- The candidate's resume/profile is the ONLY source of truth. If a fact, skill, tool, technique, course, employer type, or metric is not present there, you MUST NOT add it — not even to match the job description.
- Never invent or imply: programming languages, software/tools, omics or lab techniques, coursework, certifications, industry/cross-functional collaboration, team structures, or quantified outcomes that the source does not state.
- Do not upgrade the nature of a role (e.g. "sample preparation" must not become "statistical analysis of large datasets"; an academic lab must not be described as partnering with "medicinal chemistry teams" unless the source says so).
- A Skills section may ONLY list skills with direct evidence in the source (named in a bullet, publication, presentation, or degree). If the source has little skills evidence, keep the Skills section short and honest rather than padding it.
- Incorporate a job-description keyword ONLY when the source already supports it. It is correct and expected to leave real gaps unfilled — recruiters and interviewers will test fabricated claims.
- You may reorder, reframe, and sharpen real content freely; you may NOT manufacture content. When in doubt, omit.

NO HEDGING (a gap is omitted, never confessed):
- The truthfulness rules above mean you will be unable to claim some things the job wants. The ONLY correct response is to leave them out SILENTLY.
- NEVER write aspirational, in-progress, or self-deprecating qualifiers on the resume: no "(learning phase)", "exploring X", "expanding proficiency", "familiar with", "basic knowledge of", "currently learning", "working toward". A resume states what the candidate HAS done. Anything else is omitted, not downgraded in place.
- Never annotate a skill with a caveat that weakens it. If a skill is too weak to state plainly, drop it entirely.
- Never editorialize about relevance ("relevant to bioinformatics workflows", "which demonstrates transferable skills"). State the real work; let the reader judge.

PRESERVE THE RECORD:
- Never drop a publication, presentation, degree, award, or role that is in the source. Tailoring reorders and reframes; it does not discard the candidate's record.
- Never restate a credential more strongly than the source: if the source says a degree is expected/in progress, it stays expected/in progress.
- Copy institution names, locations, dates, and technique names EXACTLY as the source gives them. Do not "correct", relocate, or substitute a near-neighbour (e.g. RT-PCR is not RT-qPCR; a home town is not an employer's location)."""


def analyze_match(resume: str, job_title: str, company: str, job_description: str) -> dict:
    prompt = f"""Analyze the match between this candidate's resume and the job posting.

JOB TITLE: {job_title}
COMPANY: {company}
JOB DESCRIPTION:
{job_description[:6000]}

CANDIDATE RESUME:
{resume[:8000]}

Return strict JSON with this shape (no prose outside JSON):
{{
  "score": <float 0-1, how well candidate matches>,
  "strengths": ["bullet", ...],
  "gaps": ["bullet", ...],
  "missing_keywords": ["keyword", ...],
  "summary": "one-paragraph honest assessment",
  "recommendation": "apply" | "tailor_first" | "skip"
}}"""
    out = _generate(prompt, system=SYSTEM_RESUME_TAILOR, temperature=0.2)
    return _parse_json_block(out) or {"score": 0.0, "summary": out[:500]}


def tailor_resume(resume: str, job_title: str, company: str, job_description: str) -> str:
    prompt = f"""Tailor this resume for the target role below. Rules:
- TRUTHFULNESS FIRST (see the truthfulness contract): use ONLY facts, skills,
  tools, techniques, and experience present in the ORIGINAL RESUME / profile
  context. Do NOT invent or imply anything to match the job description.
- Reorder and reframe REAL content to put the most relevant achievements first.
- Use a job keyword ONLY where the source already supports it. Leave genuine
  gaps unfilled rather than fabricating — never add skills/tools/coursework the
  source doesn't show.
- Do not inflate a role's scope (e.g. "sample preparation" stays sample
  preparation; an academic lab is not described as industry/cross-functional
  unless the source says so).
- Quantify impact ONLY using numbers that appear in the original resume.
- Build the Skills section solely from skills evidenced in the source; keep it
  short and honest rather than padded.
- HUMAN TONE & SCIENTIFIC RIGOR: Write in the natural voice of an authentic bench & computational scientist.
  Strictly NEVER use AI clichés or corporate jargon ("spearheaded", "leveraged", "pioneered", "utilized", "streamlined", "fostered", "orchestrated", "passionate", "seasoned").
  Use direct active verbs describing concrete lab actions, assays, targets, and data ("purified", "developed", "expressed", "characterized", "optimized", "assayed", "modeled", "automated", "built", "designed", "isolated").
- Keep the standard sections that apply: Summary, Experience, Skills, Education,
  Projects/Publications. Keep real entries (don't silently drop experience).
- Output VALID GitHub-flavored Markdown only — no commentary, no disclaimers
  inside the resume text.

JOB TITLE: {job_title}
COMPANY: {company}
JOB DESCRIPTION:
{job_description[:6000]}

ORIGINAL RESUME (Markdown):
{resume[:10000]}{_template_block("resume")}"""
    return _generate(prompt, system=SYSTEM_RESUME_TAILOR, temperature=0.4)


COVER_LETTER_NO_IMPORT_RULE = """THE JOB DESCRIPTION IS NOT THE CANDIDATE'S HISTORY. It tells you what they WANT;
the resume tells you what the candidate HAS. Never read a requirement in the job description
and write it back as the applicant's experience. If the candidate has not done it, the letter does not
claim it -- a letter that invents a qualification the applicant lacks is worse than a
letter that omits it, because that is the claim the screener will probe first.
Address a gap honestly or leave it alone."""


def generate_cover_letter(resume: str, job_title: str, company: str, job_description: str, applicant_name: str) -> str:
    from datetime import datetime
    from .resume import get_canonical_letterhead
    letterhead = get_canonical_letterhead(applicant_name)
    current_date = datetime.now().strftime("%B %d, %Y")

    prompt = f"""Write a formal, authoritative, peer-to-peer scientific and technical cover letter for {letterhead['name']} applying for the {job_title} position at {company}.

CRITICAL FORMAT REQUIREMENT: Follow the formal executive/scientist letterhead format:
1. SENDER HEADER:
{letterhead['name']}
{letterhead['address_line1']}
{letterhead['city_state_zip']}
{current_date}

2. RECIPIENT BLOCK:
Search Committee / Hiring Team for {job_title}
{company}

3. SALUTATION:
Dear Search Committee Members and Hiring Team, [open with a specific, authentic hook tied to {company} / the mission]

4. BODY PARAGRAPHS (4 well-developed, authoritative paragraphs, peer-to-peer scientist tone):
- Paragraph 1: Strategic Alignment & Enthusiasm. Express strong enthusiasm for the role and mission at {company}. Directly align your core background (integrating predictive deep learning / computational modeling with rigorous biochemical validation) with their specific pipeline, disease area, or technical initiative.
- Paragraph 2: Deep Technical & Domain Rigor. Frame the core challenge, engineering problem, or scientific target. Detail your concrete computational frameworks, methodologies, and technical tools, and directly connect them to practical validation, experimental design, or end-to-end implementation. Emphasize verification and reproducible results.
- Paragraph 3: Quantitative Translation, Scalable Computing & Mentoring. Detail how your algorithmic design translates into quantitative clinical digital biomarkers, high-performance computing pipelines, or scalable production tools without user/patient testing burdens. Highlight commitment to expanding technical literacy across the team and fostering an inclusive lab culture.
- Paragraph 4: Value Proposition & Zero-Onboarding Continuity. Emphasize immediate continuity with zero onboarding friction, intimate familiarity with HPC clusters and shared core instrumentation, readiness from day one to generate preliminary data for grant proposals/milestones, and a forward-looking call to discuss advancing strategic goals.

5. FORMAL CLOSING & SIGN-OFF:
Thank you very much for your time, consideration, and continued leadership of our scientific community.

Sincerely,

{letterhead['name']}
{letterhead['department']}
{letterhead['institution']}
{letterhead['email']} | {letterhead['phone']}

TRUTHFULNESS & VOICE RULES:
- All claims MUST be grounded 100% in the candidate's real record below. Zero hallucination tolerance.
- Tone: authentic, senior bench and computational scientist speaking peer-to-peer.
- BANNED AI filler: spearheaded, leveraged, pioneered, utilized, streamlined, fostered, orchestrated, passionate, seasoned, beacon, dynamic team player, results-driven, executed workflows.
- USE active scientific verbs: purified, developed, expressed, characterized, optimized, assayed, modeled, automated, built, designed, isolated, synthesized, profiled, screened, cultured, troubleshot, quantified.

{COVER_LETTER_NO_IMPORT_RULE}

JOB TITLE: {job_title}
COMPANY: {company}
JOB DESCRIPTION (what they want -- NOT the applicant's experience):
{job_description[:5000]}

RESUME (the candidate's verified experience -- the source of truth for every claim):
{resume[:8000]}{_template_block("cover")}"""
    return _generate(prompt, system=SYSTEM_RESUME_TAILOR, temperature=0.4)



def interview_prep(resume: str, job_title: str, company: str, job_description: str) -> str:
    prompt = f"""You are a senior technical hiring manager and domain expert preparing the candidate for an upcoming interview.
Use the job description, candidate resume, and candidate profile context to produce a comprehensive prep doc.

JOB TITLE: {job_title}
COMPANY: {company}
JOB DESCRIPTION:
{job_description[:6500]}

CANDIDATE RESUME:
{resume[:8000]}

Output Markdown with these sections:

## Role decoded
2-3 sentences on what this team likely actually does day-to-day and the unstated priorities behind the JD.

## Top 10 predicted interview questions (technical + behavioral, mixed)
For each question include:
- **Q**: the question
- **Why they'd ask**: what they're probing for
- **STAR-ready answer hook**: a 1-sentence pointer to which of the candidate's experiences (drawn from resume/profile) to use

## 5 hard technical questions you should be ready for
Specific to the techniques / disease areas / methods in the JD. Include the answer outline (3-4 bullets each) — accurate, expert-level.

## 5 behavioral / culture-fit questions
Tailored to the company's stated values or what's typical for this role's seniority.

## Questions YOU should ask them
6 sharp, role-specific questions that signal scientific maturity and genuine interest.

## Red flags to watch for in the interview
3-5 signs this team may not be a great fit (e.g., siloed work, no publication culture, unclear scope).

## Study list (24-hour cram)
A bulleted list of papers, methods, products, or concepts to brush up on before the interview, prioritized.

## Salary / negotiation talking points
2-3 specific data points to anchor the comp conversation for this role + location.

Be specific, scientific, and honest. No fluff."""
    return _generate(prompt, system=SYSTEM_RESUME_TAILOR, temperature=0.45)


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
    return _generate(prompt, temperature=0.3)


# ----- Interview coach seam (shared helpers live here; other providers import
# this module as `_g`). See jobbot/interview_coach.py. --------------------------

_RUBRIC_AXES = ("relevance", "specificity", "structure", "fit")
_BANK_KEYS = ("behavioral", "technical", "research", "contact")

_BANK_SPEC = """Return STRICT JSON, no prose outside it, with exactly these keys:
{
  "behavioral": [{"q": "...", "why": "...", "answer_hook": "..."}],
  "technical":  [{"q": "...", "why": "...", "answer_hook": "..."}],
  "research":   [{"q": "...", "grounded_in": "company_intel", "answer_hook": "..."}],
  "contact":    []
}
Rules:
- 5 behavioral, 5 technical, 4 research questions.
- "why" = why an interviewer for THIS role asks it.
- "answer_hook" = the specific thing from the candidate's own background they
  should lead with (name the project/technique/result - never generic advice).
- RESEARCH questions must reference this company's actual pipeline / products /
  recent work from the intel below. A research question that would fit any
  company is WRONG. Set "grounded_in" to "company_intel" when it comes from the
  intel brief, or "jd" when it comes from the job description.
- Leave "contact" as an empty list."""


def _normalize_bank(parsed: dict) -> dict:
    """Coerce an LLM bank blob into the canonical 4-key shape. Never raises."""
    out: dict = {k: [] for k in _BANK_KEYS}
    if not isinstance(parsed, dict):
        return out
    for key in _BANK_KEYS:
        items = parsed.get(key) or []
        if not isinstance(items, list):
            continue
        for item in items:
            if not isinstance(item, dict):
                continue
            q = str(item.get("q") or "").strip()
            if not q:
                continue
            out[key].append({
                "q": q,
                "why": str(item.get("why") or "").strip(),
                "answer_hook": str(item.get("answer_hook") or "").strip(),
                "grounded_in": str(item.get("grounded_in") or "").strip(),
            })
    return out


def _normalize_rubric(parsed: dict) -> dict:
    """Coerce an LLM score blob into the canonical rubric shape, clamped 0-5."""
    out = {a: 3.0 for a in _RUBRIC_AXES}
    out["critique"] = ""
    out["follow_up"] = ""
    if not isinstance(parsed, dict):
        return out
    for axis in _RUBRIC_AXES:
        try:
            out[axis] = max(0.0, min(5.0, float(parsed.get(axis, 3.0))))
        except (TypeError, ValueError):
            out[axis] = 3.0
    out["critique"] = str(parsed.get("critique") or "").strip()
    out["follow_up"] = str(parsed.get("follow_up") or "").strip()
    return out


def _contact_bank_spec(contact: dict) -> str:
    """Bank spec that makes the LLM role-play a REAL known interviewer."""
    name = str(contact.get("name") or "the interviewer").strip()
    role = str(contact.get("title") or "").strip()
    org = str(contact.get("company") or "").strip()
    research = str(contact.get("research") or "").strip()
    who = ", ".join(p for p in (name, role, org) if p)
    return f"""You are {who}. This is YOUR interview.

WHAT IS KNOWN ABOUT YOUR WORK:
{research[:2500] or '(no research available - rely on the role and company only)'}

Return STRICT JSON, no prose outside it, with exactly these keys:
{{
  "behavioral": [{{"q": "...", "why": "...", "answer_hook": "..."}}],
  "technical":  [{{"q": "...", "why": "...", "answer_hook": "..."}}],
  "research":   [{{"q": "...", "grounded_in": "company_intel", "answer_hook": "..."}}],
  "contact":    [{{"q": "...", "grounded_in": "contact", "answer_hook": "..."}}]
}}
Rules:
- 5 behavioral, 5 technical, 4 research, 4 contact questions.
- CONTACT questions are what {name} specifically would probe given THEIR work
  above - name their techniques/papers/problems. A contact question that any
  interviewer could ask is WRONG.
- RESEARCH questions must reference this company's actual pipeline/products.
- "answer_hook" = the specific thing from the candidate's own background they
  should lead with (name the project/technique/result)."""


def _score_prompt(question: str, question_kind: str, answer: str,
                  job_title: str, company: str, resume: str) -> str:
    structure_rule = ("logical STAR flow (Situation, Task, Action, Result)"
                      if question_kind == "behavioral"
                      else "technical correctness and clarity of explanation")
    return f"""You are a demanding interviewer grading one answer. Be honest - a
vague answer scores low even if it is well spoken.

ROLE: {job_title} at {company}
QUESTION ({question_kind}): {question}

CANDIDATE ANSWER:
{answer[:4000]}

CANDIDATE BACKGROUND (for judging whether they used their real evidence):
{resume[:4000]}

Score each axis 0-5 (0 = absent, 3 = adequate, 5 = excellent):
- relevance: did they answer THIS question?
- specificity: concrete examples, techniques, numbers, outcomes?
- structure: {structure_rule}
- fit: did they tie it to this role/company/interviewer?

Return STRICT JSON, no prose outside it:
{{"relevance": 0-5, "specificity": 0-5, "structure": 0-5, "fit": 0-5,
  "critique": "2-4 sentences: what worked, what to fix, concretely",
  "follow_up": "one probing follow-up question IF a real gap remains, else \\"\\""}}"""


def _summary_prompt(job_title: str, company: str, transcript: str) -> str:
    return f"""Summarize this mock interview for the candidate.

ROLE: {job_title} at {company}

TRANSCRIPT:
{transcript[:9000]}

Write 4-8 sentences of plain Markdown: their two strongest moments, their two
weakest, and the single highest-leverage thing to fix before the real interview.
Be specific and reference their actual answers. No preamble."""


def interview_question_bank(resume: str, job_title: str, company: str,
                            job_description: str, company_intel: str,
                            contact: Optional[dict] = None) -> dict:
    """Machine-readable predicted-question bank for the mock interviewer."""
    spec = _contact_bank_spec(contact) if contact else _BANK_SPEC
    prompt = f"""You are the hiring manager for the role below, listing the questions
you would actually ask this candidate.

JOB TITLE: {job_title}
COMPANY: {company}
JOB DESCRIPTION:
{job_description[:5000]}

COMPANY INTEL BRIEF:
{(company_intel or '')[:3000]}

CANDIDATE RESUME:
{resume[:6000]}

{spec}"""
    out = _generate(prompt, system=SYSTEM_RESUME_TAILOR, temperature=0.5)
    return _normalize_bank(_parse_json_block(out))


def score_answer(question: str, question_kind: str, answer: str,
                 job_title: str, company: str, resume: str) -> dict:
    out = _generate(_score_prompt(question, question_kind, answer,
                                  job_title, company, resume),
                    system=SYSTEM_RESUME_TAILOR, temperature=0.2)
    return _normalize_rubric(_parse_json_block(out))


def interview_session_summary(job_title: str, company: str, transcript: str) -> str:
    return _generate(_summary_prompt(job_title, company, transcript),
                     system=SYSTEM_RESUME_TAILOR, temperature=0.3)


def advise_resume(resume: str, focus: str = "") -> str:
    prompt = f"""Review this resume and give targeted, actionable improvement advice.
Focus area (optional): {focus or "general impact, clarity, ATS-readiness"}

Output Markdown with sections:
## Quick wins
## Bullet rewrites (before -> after for 3-5 weakest bullets)
## Missing sections / additions
## Keyword & ATS suggestions
## Overall score (out of 10) + one-line verdict

RESUME:
{resume[:10000]}"""
    return _generate(prompt, system=SYSTEM_RESUME_TAILOR, temperature=0.4)


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
        out[k] = v
    return out


def answer_application_questions(resume: str, job_title: str, company: str,
                                 job_description: str, questions: list[str]) -> dict:
    """Answer free-text application-form questions truthfully from the resume.

    Returns {question: answer}. Used by the assisted browser-apply flow.
    """
    if not questions:
        return {}
    numbered = "\n".join(f"{i+1}. {q}" for i, q in enumerate(questions))
    prompt = f"""You are filling out a job application form for this candidate.
Answer each question below TRUTHFULLY using only the resume/profile. Keep answers
concise and professional (1-4 sentences unless the question implies an essay).
If the resume genuinely lacks the information, answer with an empty string "".

JOB TITLE: {job_title}
COMPANY: {company}
JOB DESCRIPTION:
{job_description[:4000]}

CANDIDATE RESUME:
{resume[:8000]}

QUESTIONS:
{numbered}

Return strict JSON: an object mapping the EXACT question text to the answer string.
No prose outside JSON."""
    out = _generate(prompt, system=SYSTEM_RESUME_TAILOR, temperature=0.3)
    parsed = _parse_json_block(out)
    return _align_answer_keys(parsed, questions)


SYSTEM_NETWORKING = """You are a sharp, warm professional networking coach helping a job-seeking scientist reach out to real people. You write outreach that sounds genuinely human — specific, concise, and never templated. You are scrupulously HONEST: you never invent shared history, mutual connections, prior conversations, or experience the candidate does not have. Ground every claim about the candidate in their provided resume/profile. The candidate's profile is provided as ground truth context."""


def networking_message(resume: str, job_title: str, company: str,
                       job_description: str, contact: dict,
                       channel: str = "linkedin") -> dict:
    """Draft a tailored outreach message to a specific professional.

    channel="linkedin" -> short connection note, <=300 chars (subject empty).
    channel="email"    -> longer outreach email with a subject line.

    Returns {"subject": str, "body": str}. Personalised to the contact's
    name/role, the target role, and the candidate's REAL background.
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

    prompt = f"""Help the candidate (resume/profile in context) network toward a job.

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
    out = _generate(prompt, system=SYSTEM_NETWORKING, temperature=0.6)
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


def list_models() -> list[str]:
    """List models available to the current API key via the new SDK."""
    client = _get_client()
    out = []
    for m in client.models.list():
        name = getattr(m, "name", "")
        actions = getattr(m, "supported_actions", None) or getattr(m, "supported_generation_methods", None) or []
        if not actions or "generateContent" in actions:
            out.append(name)
    return out
