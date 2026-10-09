"""AI-free tailoring + per-job prompt-engineering packs.

Two cloud-free deliverables for each job, neither of which calls any LLM:

1.  ``tailor_resume_offline`` — a fully deterministic resume tailor. It builds a
    clean, ATS-friendly resume straight from the ground-truth ``data/profile.md``,
    re-ordering and emphasising the candidate's REAL skills/experience by how well
    they match the job description. It cannot fabricate anything because every line
    comes verbatim from the profile.

2.  ``build_prompt_pack`` — a "prompt engineer for each job". It writes a folder
    containing a single copy-paste-ready prompt plus all the supporting materials
    (base resume, profile, job description, keyword analysis). Paste the prompt
    into any AI chat (ChatGPT / Claude / Gemini) and you get a tailored resume +
    cover letter without this bot ever needing a working API key.

Both share one keyword engine so the offline resume and the pasted-prompt advice
agree on what the job actually wants.
"""
from __future__ import annotations

import logging
import re
from collections import Counter
from pathlib import Path
from typing import Iterable

from .config import settings
from .matcher import STOP, _tokens
from .models import Job, init_db, session
from .resume import load_resume, load_template, markdown_to_docx, markdown_to_pdf

log = logging.getLogger("jobbot.prompt_pack")

# Generic JD filler that carries no signal for matching — pruned on top of the
# matcher's small STOP set so the extracted keywords are actually skills/topics.
EXTRA_STOP = {
    "experience", "experiences", "work", "working", "team", "teams", "role",
    "roles", "job", "jobs", "position", "positions", "candidate", "candidates",
    "ability", "abilities", "skill", "skills", "including", "etc", "year",
    "years", "strong", "excellent", "required", "require", "requires",
    "requirements", "preferred", "plus", "using", "use", "used", "within",
    "across", "must", "day", "days", "help", "support", "new", "join", "joining",
    "looking", "seeking", "ideal", "responsibilities", "responsibility",
    "qualifications", "qualified", "duties", "what", "who", "how", "why", "when",
    "where", "company", "companies", "opportunity", "opportunities", "well",
    "highly", "ideally", "etc.", "able", "also", "may", "per", "via", "etc",
    "good", "great", "best", "more", "most", "than", "such", "into", "out",
    "one", "two", "three", "first", "second", "full", "time", "part", "based",
    "located", "location", "remote", "hybrid", "onsite", "office", "benefits",
    "salary", "apply", "applicant", "applicants", "please", "department",
    "group", "groups", "function", "functions", "functional", "level", "senior",
    "junior", "principal", "associate", "lead", "manager", "director", "staff",
    "degree", "phd", "bs", "ms", "msc", "bsc", "field", "fields", "related",
    "demonstrated", "proven", "track", "record", "ensure", "ensuring", "make",
    "made", "making", "provide", "providing", "develop", "developing", "perform",
    "performing", "include", "includes", "drive", "driving", "deliver",
    "delivering", "build", "building", "create", "creating", "manage",
    "managing", "lead", "leading", "support", "supporting",
    # HTML-entity / scrape artifacts that leak in from job descriptions
    "nbsp", "amp", "quot", "gt", "lt", "rsquo", "ldquo", "rdquo", "mdash",
    "ndash", "apos", "hellip", "span", "div", "href",
}

# Multi-word skills are far higher-signal than single tokens. These come from the
# user's own search config (.env SEARCH_KEYWORDS) so they always reflect the
# techniques the candidate actually cares about / can do.
def _configured_phrases() -> list[str]:
    phrases = []
    for k in settings.keywords:
        k = (k or "").strip().lower()
        if k and " " in k or "-" in k:
            phrases.append(k)
        elif k:
            phrases.append(k)
    # De-dupe, longest first so "computer-aided drug design" beats "drug".
    seen, out = set(), []
    for p in sorted(set(phrases), key=len, reverse=True):
        if p not in seen:
            seen.add(p)
            out.append(p)
    return out


def _stem(tok: str) -> str:
    """Crude singular/plural fold so 'assays' matches 'assay'."""
    t = tok.lower()
    for suf in ("ies", "es", "s"):
        if len(t) > 4 and t.endswith(suf):
            return t[: -len(suf)]
    return t


# --------------------------------------------------------------------------- #
# Keyword engine
# --------------------------------------------------------------------------- #
def extract_jd_keywords(job_title: str, job_description: str,
                        top_n: int = 40) -> list[tuple[str, int]]:
    """Weighted keywords pulled from the JD.

    Single tokens weighted by frequency; title terms get a heavy boost (the title
    is the single most honest statement of what the role is). Returns
    ``[(term, weight), ...]`` highest-weight first.
    """
    title_tokens = [t for t in _tokens(job_title) if t not in EXTRA_STOP]
    body_tokens = [t for t in _tokens(job_description) if t not in EXTRA_STOP]
    weights: Counter[str] = Counter()
    for t in body_tokens:
        if len(t) >= 3:
            weights[t] += 1
    for t in title_tokens:
        if len(t) >= 3:
            weights[t] += 4  # title terms dominate
    return weights.most_common(top_n)


def keyword_match(resume_text: str, profile_text: str, job_title: str,
                  job_description: str) -> dict:
    """Compare what the JD asks for against what the candidate can evidence.

    Returns ``{matched, missing, matched_phrases, missing_phrases, score}`` where
    every "matched" item genuinely appears in the resume/profile (no guessing).
    """
    have_blob = f"{resume_text}\n{profile_text}".lower()
    have_stems = {_stem(t) for t in _tokens(have_blob)}

    jd_kw = extract_jd_keywords(job_title, job_description)
    matched, missing = [], []
    for term, _w in jd_kw:
        if _stem(term) in have_stems:
            matched.append(term)
        else:
            missing.append(term)

    jd_blob = f"{job_title}\n{job_description}".lower()
    matched_phrases, missing_phrases = [], []
    for ph in _configured_phrases():
        if " " not in ph and "-" not in ph:
            continue  # single-token phrases handled above
        in_jd = ph in jd_blob
        in_have = ph in have_blob
        if in_jd and in_have:
            matched_phrases.append(ph)
        elif in_jd and not in_have:
            missing_phrases.append(ph)

    denom = max(len(jd_kw), 1)
    base = len(matched) / denom
    phrase_bonus = 0.05 * len(matched_phrases)
    score = round(min(1.0, 0.85 * base + phrase_bonus), 3)
    return {
        "matched": matched,
        "missing": missing,
        "matched_phrases": matched_phrases,
        "missing_phrases": missing_phrases,
        "score": score,
    }


# --------------------------------------------------------------------------- #
# Profile parsing (ground-truth source for the offline resume)
# --------------------------------------------------------------------------- #
def load_profile_text() -> str:
    """Raw profile.md text with the HTML editing-note comment stripped."""
    p = Path(settings.profile_path)
    if not p.exists():
        return ""
    text = p.read_text(encoding="utf-8")
    return re.sub(r"<!--.*?-->", "", text, flags=re.DOTALL).strip()


def _parse_profile_sections(profile_text: str) -> dict[str, list[str]]:
    """Split profile markdown into ``{section_title_lower: [content_lines]}``."""
    sections: dict[str, list[str]] = {}
    current = None
    for raw in profile_text.splitlines():
        line = raw.rstrip()
        m = re.match(r"^##\s+(.*)$", line)
        if m:
            current = m.group(1).strip().lower()
            sections[current] = []
        elif current is not None:
            sections[current].append(line)
    return sections


def _get_section(sections: dict[str, list[str]], *aliases: str) -> list[str]:
    """Find section lines by checking multiple case-insensitive alias names."""
    for a in aliases:
        al = a.strip().lower()
        if al in sections:
            return sections[al]
    return []


def _identity(sections: dict[str, list[str]]) -> dict[str, str]:
    out: dict[str, str] = {}
    lines = _get_section(sections, "identity", "contact", "contact info")
    for line in lines:
        m = re.match(r"^\s*-\s*\*\*(.+?):\*\*\s*(.+)$", line)
        if m:
            out[m.group(1).strip().lower()] = m.group(2).strip()

    # Fallback to settings / personal_info when Identity section is absent
    if "name" not in out and settings.applicant_name:
        out["name"] = settings.applicant_name
    if "email" not in out and settings.applicant_email:
        out["email"] = settings.applicant_email
    if "phone" not in out and settings.applicant_phone:
        out["phone"] = settings.applicant_phone
    if "location" not in out and settings.applicant_location:
        out["location"] = settings.applicant_location
    if "linkedin" not in out and getattr(settings, "applicant_linkedin", ""):
        out["linkedin"] = settings.applicant_linkedin
    return out


def _bullets(lines: Iterable[str]) -> list[str]:
    """Top-level bullets ('- ' or '* '), joining wrapped continuation lines."""
    out: list[str] = []
    for raw in lines:
        line = raw.rstrip()
        if re.match(r"^\s*[-*]\s+", line):
            out.append(re.sub(r"^\s*[-*]\s+", "", line).strip())
        elif re.match(r"^\s*\d+\.\s+", line):
            out.append(re.sub(r"^\s*\d+\.\s+", "", line).strip())
        elif line.strip() and out and not line.startswith(">"):
            out[-1] += " " + line.strip()
    return out


def _relevance(text: str, jd_stems: set[str]) -> int:
    return sum(1 for t in {_stem(x) for x in _tokens(text)} if t in jd_stems)


# --------------------------------------------------------------------------- #
# 1) Deterministic, AI-free resume tailor
# --------------------------------------------------------------------------- #
def tailor_resume_offline(job_title: str, company: str, job_description: str,
                          profile_text: str | None = None) -> str:
    """Build a tailored resume from the profile with NO AI call.

    Strategy (truthful by construction — only profile content is used):
      * header + factual summary verbatim from the profile,
      * a "Most relevant to this role" line listing skills the candidate has that
        the JD asks for (verbatim overlap),
      * Skills section with categories and items re-ordered to surface JD matches,
      * Experience / Education / Publications kept intact (real, reverse-chron).
    """
    profile_text = profile_text if profile_text is not None else load_profile_text()
    sections = _parse_profile_sections(profile_text)
    ident = _identity(sections)
    match = keyword_match("", profile_text, job_title, job_description)
    jd_stems = {_stem(t) for t, _ in extract_jd_keywords(job_title, job_description)}

    lines: list[str] = []

    # ---- header ----
    name = ident.get("name") or settings.applicant_name or "Candidate"
    title = ident.get("title") or getattr(settings, "applicant_title", "")
    loc = ident.get("location") or settings.applicant_location
    contact_bits = [b for b in (ident.get("email") or settings.applicant_email,
                                ident.get("phone") or settings.applicant_phone,
                                ident.get("linkedin") or getattr(settings, "applicant_linkedin", "")) if b]
    lines.append(f"# {name}")
    subtitle = " | ".join(b for b in (title, loc) if b)
    if subtitle:
        lines.append(subtitle + "  ")
    if contact_bits:
        lines.append(" | ".join(contact_bits))
    lines.append("")
    lines.append("---")
    lines.append("")

    # ---- summary ----
    summary_lines = _get_section(sections, "summary (factual)", "summary", "who i am", "about")
    summary = " ".join(l.strip() for l in summary_lines if l.strip() and not l.startswith("#"))
    if summary:
        lines.append("### Professional Summary")
        lines.append(summary)
        lines.append("")

    # ---- most relevant (honest keyword surfacing) ----
    highlight = match["matched_phrases"][:8] or match["matched"][:10]
    if highlight:
        pretty = ", ".join(h if not h.islower() else h for h in highlight)
        lines.append(f"**Most relevant to {job_title} @ {company}:** {pretty}.")
        lines.append("")
        lines.append("---")
        lines.append("")

    # ---- experience (kept in profile order = reverse chronological) ----
    exp_lines = _get_section(sections, "research experience", "experience highlights", "experience", "work experience")
    exp = _bullets(exp_lines)
    if exp:
        lines.append("### Research Experience")
        for b in exp:
            lines.append(f"- {b}")
        lines.append("")

    teach_lines = _get_section(sections, "teaching", "teaching experience")
    teach = _bullets(teach_lines)
    if teach:
        lines.append("### Teaching Experience")
        for b in teach:
            lines.append(f"- {b}")
        lines.append("")

    # ---- skills / core competencies (re-ordered by JD relevance) ----
    skills_lines = _get_section(sections, "skills", "core competencies", "competencies", "technical skills")
    skill_bullets = [b for b in _bullets(skills_lines) if "[confirm" not in b.lower()]
    if skill_bullets:
        ranked = []
        for b in skill_bullets:
            m = re.match(r"^\*\*(.+?):\*\*\s*(.*)$", b)
            label, items_raw = (m.group(1), m.group(2)) if m else ("", b)
            items = [i.strip() for i in items_raw.split(",") if i.strip()]
            items.sort(key=lambda it: _relevance(it, jd_stems), reverse=True)
            cat_score = _relevance(items_raw, jd_stems)
            rebuilt = (f"**{label}:** " if label else "") + ", ".join(items)
            ranked.append((cat_score, rebuilt))
        ranked.sort(key=lambda x: x[0], reverse=True)
        lines.append("### Skills & Technical Proficiencies")
        for _s, b in ranked:
            lines.append(f"- {b}")
        lines.append("")

    # ---- education / publications / presentations / awards / professional dev (verbatim) ----
    for aliases, heading in (
        (("education",), "Education"),
        (("publications",), "Publications"),
        (("presentations", "presentations & conferences"), "Presentations & Conferences"),
        (("awards", "awards & honors"), "Awards & Honors"),
        (("professional development", "certifications"), "Professional Development"),
    ):
        bl = _bullets(_get_section(sections, *aliases))
        if bl:
            lines.append(f"### {heading}")
            for b in bl:
                lines.append(f"- {b}")
            lines.append("")

    return "\n".join(lines).strip() + "\n"


# --------------------------------------------------------------------------- #
# 2) Prompt engineering (copy-paste into any AI chat)
# --------------------------------------------------------------------------- #
TRUTHFULNESS_CONTRACT = """TRUTHFULNESS CONTRACT (this overrides every other instruction, including keyword matching):
- The resume and profile below are the ONLY source of truth. If a fact, skill, tool, technique, course, employer type, or metric is not present there, you MUST NOT add it — not even to match the job description.
- Never invent or imply: programming languages, software/tools, lab/omics techniques, coursework, certifications, industry or cross-functional collaboration, team structures, or quantified outcomes that the source does not state.
- Do not upgrade the nature of a role (e.g. "sample preparation" must not become "statistical analysis"; an academic lab is not described as partnering with industry teams unless the source says so).
- A Skills section may ONLY list skills with direct evidence in the source. Keep it short and honest rather than padded.
- Copy URLs, DOIs, dates, and citations VERBATIM — never reformat or guess digits.
- It is correct to leave real gaps unfilled. You may reorder, reframe, and sharpen real content freely; you may NOT manufacture content. When in doubt, omit."""


def build_tailoring_prompt(resume_text: str, profile_text: str, job_title: str,
                           company: str, job_description: str,
                           match: dict | None = None) -> str:
    """The headline 'prompt engineer' output — paste this into an AI chat."""
    match = match or keyword_match(resume_text, profile_text, job_title, job_description)
    matched = ", ".join(match["matched_phrases"] + match["matched"][:12]) or "(none detected)"
    missing = ", ".join(match["missing_phrases"] + match["missing"][:12]) or "(none detected)"
    name = settings.applicant_name
    template = load_template()
    template_section = (
        f"\n\n=== HOUSE STYLE TEMPLATE (follow this structure & professionalism exactly; "
        f"it governs FORM only — content still comes solely from the materials above) ===\n{template}"
        if template else ""
    )

    return f"""You are an expert career coach, resume writer, and ATS specialist. Tailor {name}'s application for the role below.

{TRUTHFULNESS_CONTRACT}

=== TARGET ROLE ===
TITLE: {job_title}
COMPANY: {company}

JOB DESCRIPTION:
{job_description.strip()}

=== CANDIDATE PROFILE (ground truth) ===
{profile_text.strip()}

=== CANDIDATE BASE RESUME (ground truth) ===
{resume_text.strip()}

=== KEYWORD ANALYSIS (pre-computed, for your guidance) ===
ALREADY EVIDENCED in the candidate's materials (safe to emphasise): {matched}
ASKED FOR BUT NOT EVIDENCED (do NOT fabricate these — only mention if the source truly supports them): {missing}{template_section}

=== YOUR TASK ===
Produce THREE deliverables, each under its own clear heading:

1. ## Tailored Resume
   Valid GitHub-flavored Markdown. Reorder and reframe REAL content so the most
   relevant experience and skills appear first. Surface the "already evidenced"
   keywords naturally. Keep standard sections (Summary, Experience, Skills,
   Education, Publications). No commentary inside the resume.

2. ## Cover Letter
   250-350 words, first person as {name}. Specific hook tied to {company}/the
   role, two short paragraphs of relevant REAL experience, forward-looking close.
   Human, not generic. No clichés.

3. ## Honest Match Assessment
   - 3-5 genuine strengths for this role
   - 3-5 real gaps and how to honestly address them in an interview
   - The 5 most important keywords to weave in (only ones the source supports)

Do not invent anything. If a requested qualification is missing, say so in the
assessment rather than faking it in the resume."""


def build_cover_letter_prompt(resume_text: str, profile_text: str, job_title: str,
                              company: str, job_description: str) -> str:
    from datetime import datetime
    from .resume import get_canonical_letterhead
    name = settings.applicant_name
    letterhead = get_canonical_letterhead(name)
    current_date = datetime.now().strftime("%B %d, %Y")

    return f"""Write a formal, authoritative, peer-to-peer scientific and technical cover letter for {letterhead['name']} applying for the {job_title} position at {company}. {TRUTHFULNESS_CONTRACT}

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
- Paragraph 1: Strategic Alignment & Enthusiasm. Express strong enthusiasm for the role and mission at {company}. If a TARGET COMPANY / PLATFORM INITIATIVE or specific modality is mentioned, anchor the opening directly to their active initiatives or pipeline. Directly align your core background (integrating predictive deep learning / computational modeling with rigorous biochemical validation) with their specific pipeline, disease area, or technical initiative.
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

ROLE: {job_title} at {company}

JOB DESCRIPTION:
{job_description.strip()}

CANDIDATE PROFILE (ground truth):
{profile_text.strip()}

CANDIDATE RESUME (ground truth):
{resume_text.strip()}"""



def build_networking_prompt(profile_text: str, job_title: str, company: str,
                            job_description: str) -> str:
    name = settings.applicant_name
    return f"""Help {name} network toward a job. {TRUTHFULNESS_CONTRACT}

TARGET ROLE: {job_title} at {company}

JOB CONTEXT (use only for one specific, genuine hook):
{job_description[:2500].strip()}

CANDIDATE PROFILE (ground truth):
{profile_text.strip()}

Produce:
1. A LinkedIn connection note (<= 280 characters) to a scientist/hiring manager
   on this team.
2. A 120-170 word outreach email (with a subject line under 60 chars) to the same
   person.
Both must be warm, specific, and HONEST — never claim shared history, mutual
connections, or experience {name} does not have. Ground the one-line intro in the
real profile above. No "I hope this finds you well", no AI throat-clearing."""


# --------------------------------------------------------------------------- #
# Orchestrator: write a full per-job pack to disk
# --------------------------------------------------------------------------- #
def _slug(text: str) -> str:
    return re.sub(r"[^a-z0-9]+", "-", (text or "").lower()).strip("-")[:48]


def build_prompt_pack(job_id: int, out_dir: str | Path | None = None,
                      make_docx: bool = True) -> dict:
    """Write a complete, copy-paste prompt pack + AI-free resume for one job.

    Creates ``output/prompt_packs/<id>_<company>_<title>/`` containing:
      * ``README.md``                 — how to use the pack
      * ``PASTE_THIS_PROMPT.md``      — the engineered tailoring prompt (the deliverable)
      * ``cover_letter_prompt.md``    — standalone cover-letter prompt
      * ``networking_prompt.md``      — standalone outreach prompt
      * ``job_description.md``        — the JD
      * ``base_resume.md`` / ``profile.md`` — the source materials
      * ``keyword_match.md``          — matched vs missing keyword analysis
      * ``offline_tailored_resume.md`` (+ ``.docx``) — the no-AI resume

    Returns a dict of written paths + the match summary. No network/LLM calls.
    """
    init_db()
    base = Path(out_dir) if out_dir else Path(settings.output_dir) / "prompt_packs"

    with session() as db:
        job = db.get(Job, job_id)
        if not job:
            raise ValueError(f"Job {job_id} not found")
        job_title, company = job.title, job.company
        job_description = job.description or ""
        job_url = job.url or ""

    try:
        resume_text = load_resume(settings.base_resume_path)
    except FileNotFoundError:
        log.warning("Base resume not found at %s — using profile only.",
                    settings.base_resume_path)
        resume_text = ""
    profile_text = load_profile_text()

    match = keyword_match(resume_text, profile_text, job_title, job_description)

    pack_dir = base / f"{job_id}_{_slug(company)}_{_slug(job_title)}"[:120]
    pack_dir.mkdir(parents=True, exist_ok=True)

    written: dict[str, str] = {}

    def _w(fname: str, text: str) -> None:
        p = pack_dir / fname
        p.write_text(text, encoding="utf-8")
        written[fname] = str(p)

    # JD + sources
    jd_md = (f"# {job_title} @ {company}\n\n"
             f"{'Source: ' + job_url if job_url else ''}\n\n"
             f"---\n\n{job_description}\n")
    _w("job_description.md", jd_md)
    _w("base_resume.md", f"# Base Resume (source of truth)\n\n{resume_text}\n")
    _w("profile.md", profile_text + "\n")

    # Keyword analysis
    km = [
        f"# Keyword match — {job_title} @ {company}\n",
        f"**Offline match score:** {match['score']:.0%}\n",
        "## Evidenced in your materials (emphasise these)",
        *(f"- {k}" for k in (match["matched_phrases"] + match["matched"]) or ["(none)"]),
        "",
        "## Asked for but NOT evidenced (do NOT fabricate)",
        *(f"- {k}" for k in (match["missing_phrases"] + match["missing"]) or ["(none)"]),
        "",
    ]
    _w("keyword_match.md", "\n".join(km))

    # The prompts
    _w("PASTE_THIS_PROMPT.md",
       build_tailoring_prompt(resume_text, profile_text, job_title, company,
                              job_description, match))
    _w("cover_letter_prompt.md",
       build_cover_letter_prompt(resume_text, profile_text, job_title, company,
                                 job_description))
    _w("networking_prompt.md",
       build_networking_prompt(profile_text, job_title, company, job_description))

    # AI-free tailored resume
    offline_md = tailor_resume_offline(job_title, company, job_description, profile_text)
    _w("offline_tailored_resume.md", offline_md)
    docx_path = None
    if make_docx:
        try:
            docx_path = pack_dir / "offline_tailored_resume.docx"
            markdown_to_docx(offline_md, docx_path)
            written["offline_tailored_resume.docx"] = str(docx_path)
        except Exception as e:  # noqa: BLE001
            log.warning("Offline resume .docx render failed: %s", e)
            docx_path = None

    # README last so it can reference everything
    readme = f"""# Application pack — {job_title} @ {company}

Job ID {job_id}{(' · ' + job_url) if job_url else ''}
Offline keyword-match score: **{match['score']:.0%}** (no AI used)

This pack gives you TWO ways to produce a tailored application, neither of which
needs a paid API key:

## A) Fully automatic, no AI
`offline_tailored_resume.docx` / `.md` is a ready resume built straight from your
verified profile, with skills re-ordered to match this job. Nothing was invented.
Open the `.docx`, glance over it, and send.

## B) Best quality — paste into any AI chat (ChatGPT / Claude / Gemini)
1. Open **`PASTE_THIS_PROMPT.md`** and copy the WHOLE file.
2. Paste it into a fresh AI chat and send.
3. You'll get back a tailored resume + cover letter + an honest match assessment.
   The prompt already contains your resume, profile, the job description, and a
   strict no-fabrication contract, so the AI has everything it needs.

Standalone prompts are also included if you only want one piece:
- `cover_letter_prompt.md` — just the cover letter
- `networking_prompt.md`   — LinkedIn note + outreach email

## Reference files
- `job_description.md` — the posting
- `keyword_match.md`   — what to emphasise vs. what NOT to fake
- `base_resume.md`, `profile.md` — your source materials
"""
    _w("README.md", readme)

    return {
        "job_id": job_id,
        "dir": str(pack_dir),
        "prompt": written.get("PASTE_THIS_PROMPT.md"),
        "offline_resume_md": written.get("offline_tailored_resume.md"),
        "offline_resume_docx": str(docx_path) if docx_path else None,
        "match_score": match["score"],
        "matched": match["matched_phrases"] + match["matched"],
        "missing": match["missing_phrases"] + match["missing"],
        "files": written,
    }
