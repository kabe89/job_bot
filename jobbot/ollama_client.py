"""Ollama client — local open-source models (free, no API key, no quota).

Mirrors the public surface of `gemini_client` / `claude_client` so `ai_client`
can dispatch to it interchangeably:

    analyze_match, tailor_resume, generate_cover_letter, interview_prep,
    company_intel, advise_resume, answer_application_questions,
    networking_message, is_available, probe, refresh_profile,
    reset_fallback_state

Talks to the local Ollama server over plain HTTP (default
http://localhost:11434) via /api/chat — no extra dependency beyond `requests`.
Prompts and JSON parsing are shared with `gemini_client` to avoid drift.
"""
from __future__ import annotations

import logging
import os
import shutil
import subprocess
import time
import urllib.parse
from typing import Optional

import requests

from .config import settings
from . import gemini_client as _g  # reuse prompts, system strings, parsers
from . import source_fidelity as _source_fidelity

log = logging.getLogger("jobbot.ollama")


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


def reset_fallback_state() -> None:
    _probe_cache.clear()


def _host() -> str:
    return (getattr(settings, "ollama_host", "") or "http://localhost:11434").rstrip("/")


def _cloud_base() -> str:
    return (getattr(settings, "ollama_cloud_base", "") or "https://ollama.com").rstrip("/")


def _is_cloud_model(name: Optional[str] = None) -> bool:
    """True for Ollama *cloud* models, identified by the `-cloud` name suffix."""
    name = (name if name is not None else _model()) or ""
    return name.strip().endswith("-cloud")


def _chat_endpoint() -> tuple[str, dict]:
    """Return (url, headers) for /api/chat against the active model.

    Cloud models route to the hosted base with a bearer token; local models
    hit the configured (localhost) host with no auth.
    """
    if _is_cloud_model():
        return (f"{_cloud_base()}/api/chat",
                {"Authorization": f"Bearer {getattr(settings, 'ollama_api_key', '')}"})
    return f"{_host()}/api/chat", {}


def _model() -> str:
    return getattr(settings, "ollama_model", "") or "qwen3.5:latest"


def _timeout() -> int:
    return int(getattr(settings, "ollama_timeout", 300) or 300)


# ----- Server / model auto-start (run the model when the bot needs it) ------

_server_proc: Optional[subprocess.Popen] = None


def _is_local_host() -> bool:
    """Only auto-manage a server we own — never a remote/tunneled host."""
    try:
        host = urllib.parse.urlparse(_host()).hostname or ""
    except Exception:  # noqa: BLE001
        host = ""
    return host in ("", "localhost", "127.0.0.1", "0.0.0.0", "::1")


def _ollama_bin() -> Optional[str]:
    return shutil.which("ollama")


def _can_autostart() -> bool:
    return (bool(getattr(settings, "ollama_autostart", True))
            and _is_local_host() and _ollama_bin() is not None)


def _server_reachable(timeout: float = 3.0) -> bool:
    try:
        return requests.get(f"{_host()}/api/tags", timeout=timeout).status_code == 200
    except Exception:  # noqa: BLE001
        return False


def _installed_models() -> list[str]:
    try:
        r = requests.get(f"{_host()}/api/tags", timeout=3)
        return [m.get("name", "") for m in (r.json().get("models") or [])]
    except Exception:  # noqa: BLE001
        return []


def _model_present() -> bool:
    want = _model()
    base = want.split(":")[0]
    return any(m == want or m.split(":")[0] == base for m in _installed_models())


def _start_server() -> bool:
    """Launch `ollama serve` detached and wait for it to accept connections."""
    global _server_proc
    binp = _ollama_bin()
    if not binp:
        return False
    try:
        kwargs: dict = {"stdout": subprocess.DEVNULL, "stderr": subprocess.DEVNULL}
        if os.name == "nt":
            kwargs["creationflags"] = (
                getattr(subprocess, "CREATE_NO_WINDOW", 0)
                | getattr(subprocess, "DETACHED_PROCESS", 0))
        else:
            kwargs["start_new_session"] = True
        _server_proc = subprocess.Popen([binp, "serve"], **kwargs)
    except Exception as e:  # noqa: BLE001
        log.warning("Could not launch `ollama serve` (%s).", e)
        return False
    for _ in range(30):  # up to ~30s for the server to bind
        if _server_reachable(timeout=2):
            log.info("Ollama server is up at %s.", _host())
            return True
        time.sleep(1)
    log.warning("Launched `ollama serve` but it did not become reachable in time.")
    return False


def _pull_model() -> bool:
    """Best-effort `ollama pull <model>` (large models can take a while)."""
    binp = _ollama_bin()
    if not binp:
        return False
    log.info("Pulling Ollama model %s (first run may take a while)...", _model())
    try:
        subprocess.run([binp, "pull", _model()], check=True, timeout=7200)
        return True
    except Exception as e:  # noqa: BLE001
        log.warning("`ollama pull %s` failed (%s).", _model(), e)
        return False


def _heal_windows_manifest_symlinks() -> None:
    """Fix Windows Ollama issue where manifest relative symlinks fail with:
    'The path cannot be traversed because it contains an untrusted mount point.'
    Replacing relative symlinks with copies of the target blob resolves this.
    """
    if os.name != "nt":
        return
    home = os.path.expanduser("~")
    manifest_dir = os.path.join(home, ".ollama", "models", "manifests-v2")
    if not os.path.isdir(manifest_dir):
        return
    for root, _, files in os.walk(manifest_dir):
        for f in files:
            p = os.path.join(root, f)
            if os.path.islink(p):
                try:
                    target = os.path.realpath(p)
                    if os.path.exists(target):
                        os.remove(p)
                        shutil.copyfile(target, p)
                        log.info("Fixed Windows Ollama symlink for manifest %s", f)
                except Exception as e:
                    log.debug("Could not convert symlink %s: %s", p, e)


def ensure_ready(pull: bool = True) -> bool:
    """Make the local model usable: start the server and pull the model if
    needed. Returns True if a reachable server is ready to serve.

    No-op success if the server is already reachable. For remote/tunneled
    hosts (not local) we only check reachability — we never spawn or pull.
    """
    if os.name == "nt":
        _heal_windows_manifest_symlinks()
    if _server_reachable():
        if (pull and _is_local_host() and getattr(settings, "ollama_autostart_pull", True)
                and _ollama_bin() and not _model_present()):
            _pull_model()
            _probe_cache.clear()
        return True
    if not _can_autostart():
        return False
    if not _start_server():
        return False
    if (pull and getattr(settings, "ollama_autostart_pull", True)
            and not _model_present()):
        _pull_model()
    _probe_cache.clear()
    return _server_reachable()


def is_available() -> bool:
    """True if the Ollama model is usable now or can be started on demand.

    Reachable server -> True. Otherwise True only if we *can* auto-start one
    locally (server gets launched lazily by `_generate`/`ensure_ready`), so the
    dispatcher will still route here as a backup. Cached per process.
    """
    if "available" in _probe_cache:
        return _probe_cache["available"]
    ok = ((_is_cloud_model() and bool(getattr(settings, "ollama_api_key", "")))
          or _server_reachable() or _can_autostart())
    _probe_cache["available"] = ok
    return ok


_probe_cache: dict[str, bool] = {}


def probe(timeout_seconds: float = 5.0) -> bool:
    """Verify the server answers a tiny generation."""
    if not is_available():
        return False
    try:
        out = _generate("Reply with: OK", temperature=0.0, include_profile=False)
        return bool(out)
    except Exception as e:  # noqa: BLE001
        log.warning("Ollama probe failed (%s) — Ollama features skipped.", e)
        return False


def _generate(prompt: str, system: Optional[str] = None, temperature: float = 0.4,
              include_profile: bool = True, _ctx_override: Optional[int] = None) -> str:
    """One chat completion against the local Ollama model.

    Auto-starts the server (and pulls the model on first use) when possible so
    the bot "just runs" the model whenever it needs it.
    """
    if _is_cloud_model():
        if not getattr(settings, "ollama_api_key", ""):
            raise RuntimeError(
                "Selected an Ollama cloud model but OLLAMA_API_KEY is not set. "
                "Add OLLAMA_API_KEY to your .env or pick a local model.")
    elif not _server_reachable(timeout=2):
        if not ensure_ready():
            raise RuntimeError(
                f"Ollama server not reachable at {_host()} and could not be "
                "auto-started (is the `ollama` binary installed / is the host "
                "remote?).")
    messages = []
    sys_text = system or ""
    if include_profile and _profile_context():
        sys_text = (sys_text + "\n\n" if sys_text else "") + (
            "ADDITIONAL APPLICANT CONTEXT (treat as ground truth — do not "
            "invent beyond this):\n" + _profile_context())
    if sys_text:
        messages.append({"role": "system", "content": sys_text})
    messages.append({"role": "user", "content": prompt})

    configured_ctx = int(getattr(settings, "ollama_num_ctx", 8192) or 8192)
    configured_predict = int(getattr(settings, "ollama_num_predict", 4096) or 4096)
    num_ctx = _ctx_override if _ctx_override is not None else configured_ctx

    payload = {
        "model": _model(),
        "messages": messages,
        "stream": False,
        # Disable reasoning so the answer lands in `content`, not `thinking`.
        "think": bool(getattr(settings, "ollama_think", False)),
        "options": {
            "temperature": temperature,
            # Big enough to fit our long prompts (default 4096 truncates them).
            "num_ctx": num_ctx,
            "num_predict": configured_predict,
        },
    }
    log.debug("Ollama generate: num_ctx=%d num_predict=%d prompt_chars=%d",
              num_ctx, configured_predict, len(prompt))
    _url, _headers = _chat_endpoint()
    try:
        resp = requests.post(_url, json=payload, headers=_headers or None, timeout=_timeout())
        resp.raise_for_status()
    except requests.HTTPError as exc:
        if exc.response is not None and "untrusted mount point" in (exc.response.text or "").lower():
            log.warning("Detected Windows Ollama untrusted mount point error; healing symlinks and retrying...")
            _heal_windows_manifest_symlinks()
            resp = requests.post(_url, json=payload, headers=_headers or None, timeout=_timeout())
            resp.raise_for_status()
        else:
            raise
    data = resp.json()
    msg = data.get("message", {}) or {}
    out = (msg.get("content") or "").strip()
    # Safety net: if a reasoning model still returned empty content but put text
    # in `thinking`, salvage that rather than emitting an empty document.
    if not out:
        out = (msg.get("thinking") or "").strip()

    # Retry with a reduced context window if the response came back empty.
    # This handles OOM / VRAM-spill cases on 8 GB GPUs where a large num_ctx
    # causes the model to stall and return nothing.
    if not out and _ctx_override is None:
        fallback_ctx = max(4096, num_ctx // 2)
        log.warning(
            "Ollama returned empty content at num_ctx=%d — retrying with "
            "num_ctx=%d (possible OOM/VRAM pressure on this GPU).",
            num_ctx, fallback_ctx,
        )
        return _generate(prompt, system=system, temperature=temperature,
                         include_profile=include_profile,
                         _ctx_override=fallback_ctx)

    if not out:
        log.error(
            "Ollama returned empty content even at fallback num_ctx=%d. "
            "Full response: %s", num_ctx, str(data)[:500],
        )

    # A cut-off document is worse than an obviously broken one, because it looks
    # finished. Ollama tells us when it stopped at a limit rather than because
    # the model was done -- say so loudly instead of shipping half a resume.
    if data.get("done_reason") == "length":
        p_tok = data.get("prompt_eval_count") or 0
        g_tok = data.get("eval_count") or 0
        if p_tok + g_tok >= num_ctx - 8:
            why = ("The prompt is consuming the context window — raise "
                   "OLLAMA_NUM_CTX or shrink the prompt.")
        elif g_tok >= configured_predict - 8:
            why = "Generation hit num_predict — raise OLLAMA_NUM_PREDICT."
        else:
            # Don't guess: naming the wrong knob sends the reader tuning a
            # setting that was never the constraint.
            why = "Neither limit explains the stop; inspect the raw response."
        log.warning(
            "Ollama stopped at a LIMIT, not at the end of its answer — output "
            "is truncated (prompt %d + generated %d tokens; num_ctx=%d, "
            "num_predict=%d). %s",
            p_tok, g_tok, num_ctx, configured_predict, why,
        )
    return out


# ----- Public API (mirrors gemini_client, reusing its prompts) --------------

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
    out = _generate(prompt, system=_g.SYSTEM_RESUME_TAILOR, temperature=0.2)
    return _g._parse_json_block(out) or {"score": 0.0, "summary": out[:500]}


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
    return _generate(prompt, system=_g.SYSTEM_RESUME_TAILOR, temperature=0.4)


def _critique_tailored(tailored_md: str, job_title: str, company: str,
                       job_description: str, source_resume: str = "") -> dict:
    """Self-critique pass: score the draft and list concrete fixes.

    Returns {"score": float 0-1, "issues": [...], "missing_keywords": [...],
    "fabrication_risks": [...]}. Used by the iterative refinement loop. The
    ORIGINAL resume is provided so the reviewer can detect content that isn't
    supported by the candidate's real background.
    """
    # The drafter is given resume + profile as ground truth (see
    # SYSTEM_RESUME_TAILOR and _generate(include_profile=True)), so it may
    # legitimately use a skill that is in the profile but not on the shorter base
    # resume. The critic must audit against that SAME ground truth -- judging
    # against the resume alone reported every profile-sourced skill as
    # fabrication, which pinned the score under 0.5 forever and stopped the
    # refinement loop from ever converging.
    try:
        profile_text = _profile_context() or ""
    except Exception:  # noqa: BLE001 - a missing profile must not break critique
        profile_text = ""
    profile_block = (
        f"\nCANDIDATE PROFILE (ALSO source of truth — skills here are REAL even if "
        f"the shorter resume omits them):\n{profile_text[:6000]}\n"
        if profile_text.strip() else ""
    )

    prompt = f"""You are a ruthless ATS reviewer AND an integrity auditor. Critique the
TAILORED resume against the target job AND against the candidate's real background.
Be specific and harsh.

You audit in BOTH directions — content invented, and content lost.

(a) FABRICATION — flag every skill, tool, technique, course, metric, employer
type, or responsibility that appears in the tailored version but is NOT supported
by the SOURCE OF TRUTH below. Inflated scope (e.g. "sample prep" rewritten as
"statistical analysis") counts as fabrication.

The SOURCE OF TRUTH is the original resume AND the candidate profile together. A
skill drawn from the profile but absent from the resume is NOT fabrication — the
profile is the candidate's fuller record and the writer is entitled to surface it.
Only flag content supported by NEITHER.

(b) OMISSIONS & ALTERED FACTS — flag every publication, presentation, award,
degree, honor, or role that IS in the source but is MISSING from the tailored
version, and every fact that has been CHANGED. Dropping a peer-reviewed
publication is a severe defect, not a stylistic trim. Also flag any fact made
MORE SPECIFIC than the source (e.g. source says "Expected 2027", draft invents
"Expected January 2027"), any renamed institution or relocated employer, and any
credential upgraded (a degree "expected" must not read as conferred).

JOB TITLE: {job_title}
COMPANY: {company}
JOB DESCRIPTION:
{job_description[:5000]}

ORIGINAL RESUME (source of truth):
{source_resume[:8000]}
{profile_block}
TAILORED RESUME (under review):
{tailored_md[:10000]}

Scoring rule: start from how well it matches the job, then HEAVILY penalize any
fabrication — a resume with invented content must score below 0.5 no matter how
well it matches the JD. Penalize omissions and altered facts the same way: a
draft that drops a real publication/award/role, or restates a fact more
specifically than the source, must score below 0.7 however polished it reads.

Return STRICT JSON only (no prose outside JSON):
{{
  "score": <float 0-1>,
  "issues": ["specific, actionable problem to fix", ...],
  "missing_keywords": ["JD keyword the SOURCE supports but the draft under-uses", ...],
  "fabrication_risks": ["claim in the draft supported by NEITHER the resume nor the profile", ...],
  "omissions": ["publication/award/role/fact in the source that the draft dropped or altered", ...]
}}"""
    out = _generate(prompt, system=_g.SYSTEM_RESUME_TAILOR, temperature=0.2)
    parsed = _g._parse_json_block(out)
    if not isinstance(parsed, dict):
        parsed = {"score": None, "issues": [], "missing_keywords": [],
                  "fabrication_risks": [], "omissions": []}
    parsed["score"] = _coerce_score(parsed.get("score"))
    for key in ("issues", "missing_keywords", "fabrication_risks", "omissions"):
        if not isinstance(parsed.get(key), list):
            parsed[key] = []

    # Deterministic backstop. LLMs can occasionally miss omissions or fabrication
    # defects during self-critique. The deterministic fidelity check runs in code
    # to guarantee that the candidate's verified record is preserved. See source_fidelity.
    fid = _source_fidelity.check_fidelity(f"{source_resume}\n{profile_text}", tailored_md)
    if not fid.ok:
        parsed["omissions"] = list(parsed["omissions"]) + fid.omissions
        parsed["fabrication_risks"] = list(parsed["fabrication_risks"]) + fid.fabrications
        all_defects = fid.omissions + fid.fabrications + fid.buzzwords
        parsed["issues"] = list(parsed["issues"]) + [
            f"VERIFIED DEFECT (checked in code, not opinion): {m}"
            for m in all_defects]
        # The model's own score is unreliable exactly when this fires -- it scored
        # 0.95 with a deleted publication. Cap it below target so the loop cannot
        # early-stop on a draft that misrepresents the record or contains banned buzzwords.
        if fid.severe or fid.buzzwords:
            known = parsed["score"]
            parsed["score"] = min(0.5, known) if known is not None else 0.5
    return parsed


def _coerce_score(raw) -> Optional[float]:
    """A critique score in 0-1, or None when the model didn't give a usable one.

    None means UNKNOWN and must never be conflated with 0.0 ("this resume is
    worthless"): a missing/garbled score would otherwise both defeat the
    refinement loop's early-stop and report a clean resume as scoring zero.
    """
    if raw is None or isinstance(raw, bool):
        return None
    try:
        val = float(raw)
    except (TypeError, ValueError):
        return None
    if val != val:                      # NaN
        return None
    if not 0.0 <= val <= 1.0:
        # Out of contract (85? 1.4?). We cannot know what was meant, and guessing
        # risks inventing a HIGH score that early-stops refinement. Unknown.
        return None
    return val


def _refine_tailored(tailored_md: str, critique: dict, resume: str,
                     job_title: str, company: str, job_description: str) -> str:
    """Revise the draft to address a critique, staying truthful to the source."""
    def _fmt(items) -> str:
        return "\n".join(f"- {x}" for x in (items or [])) or "- (none)"

    # Same ground truth the drafter and critic get: judging the revision against
    # the resume alone made it STRIP real, profile-sourced skills as "fabrication".
    try:
        profile_text = _profile_context() or ""
    except Exception:  # noqa: BLE001
        profile_text = ""
    profile_block = (
        f"\nCANDIDATE PROFILE (ALSO source of truth — skills here are REAL even if "
        f"the shorter resume omits them):\n{profile_text[:6000]}\n"
        if profile_text.strip() else ""
    )

    prompt = f"""Revise the tailored resume below to fix the reviewer's critique. Rules:
- FIRST, remove or correct EVERY flagged fabrication risk — delete or downgrade
  any claim supported by NEITHER the ORIGINAL RESUME nor the CANDIDATE PROFILE.
  This takes priority over matching the job description; a smaller, truthful
  resume beats an impressive fabricated one. A skill drawn from the profile is
  NOT fabrication — do not strip it.
- Use ONLY material supported by the ORIGINAL RESUME or the CANDIDATE PROFILE.
  Never invent jobs, dates, degrees, skills, tools, coursework, or metrics.
- Address the other issues and add a missing keyword ONLY when the original
  resume genuinely supports it; otherwise leave the gap.
- RESTORE every listed omission verbatim from the source. Never drop a real
  publication, presentation, award, degree or role to save space, and never
  restate a fact more specifically than the source does.
- Preserve the candidate's real sections and entries (don't drop real experience).
- Keep the house style template's structure, section order, and bullet form
  (Action -> Method -> Result; ATS-safe Markdown; no pronouns).
- Output VALID GitHub-flavored Markdown only — the full revised resume, no
  commentary and no disclaimer notes inside the text.

JOB TITLE: {job_title}
COMPANY: {company}
JOB DESCRIPTION:
{job_description[:5000]}

REVIEWER CRITIQUE:
Issues to fix:
{_fmt(critique.get("issues"))}
Missing keywords to incorporate (only if truthful):
{_fmt(critique.get("missing_keywords"))}
Fabrication risks to remove/soften:
{_fmt(critique.get("fabrication_risks"))}
Omissions / altered facts to RESTORE verbatim from the source:
{_fmt(critique.get("omissions"))}

ORIGINAL RESUME (source of truth):
{resume[:8000]}
{profile_block}
CURRENT TAILORED DRAFT (to revise):
{tailored_md[:10000]}{_template_block("resume")}"""
    return _generate(prompt, system=_g.SYSTEM_RESUME_TAILOR, temperature=0.4)


def tailor_resume_iterative(resume: str, job_title: str, company: str,
                            job_description: str, rounds: Optional[int] = None,
                            target_score: Optional[float] = None,
                            progress=None) -> dict:
    """Tailor a resume with iterative draft -> critique -> revise refinement.

    Returns {"resume": <final markdown>, "rounds": <int run>,
    "score": <final self-score>, "history": [{round, score, issues}, ...]}.

    `progress` (optional) is a callable(str) for status updates (CLI/dashboard).
    Stops early once a critique self-scores >= target_score with no fabrication risks.
    """
    if rounds is None:
        rounds = int(getattr(settings, "ollama_refine_rounds", 2) or 0)
    if target_score is None:
        target_score = float(getattr(settings, "ollama_refine_target_score", 0.9) or 0.9)

    def _say(msg: str) -> None:
        if progress:
            try:
                progress(msg)
            except Exception:  # noqa: BLE001
                pass

    _say("Drafting initial tailored resume...")
    current = tailor_resume(resume, job_title, company, job_description)
    if not current or not current.strip():
        raise RuntimeError(
            "Ollama returned empty content for the initial resume draft. "
            "Check Ollama server logs — possible causes: VRAM OOM, model "
            "crashed, or context window too large for available GPU memory. "
            "Try reducing OLLAMA_NUM_CTX in .env (current default: 8192)."
        )
    history: list[dict] = []
    last_score: Optional[float] = None

    for i in range(max(0, rounds)):
        _say(f"Refinement round {i + 1}/{rounds}: critiquing...")
        crit = _critique_tailored(current, job_title, company, job_description,
                                  source_resume=resume)
        last_score = crit.get("score")
        history.append({
            "round": i + 1,
            "score": last_score,
            "issues": crit.get("issues", []),
            "fabrication_risks": crit.get("fabrication_risks", []),
            "omissions": crit.get("omissions", []),
        })
        shown = f"{last_score:.2f}" if last_score is not None else "unknown"
        # An unknown score is NOT permission to stop -- only a real one is. A high
        # score does not license stopping while critical content is still missing.
        if (last_score is not None and last_score >= target_score
                and not crit.get("fabrication_risks")
                and not crit.get("omissions")
                and not any("banned AI buzzword" in x for x in crit.get("issues", []))):
            _say(f"Round {i + 1}: self-score {shown} >= target; stopping early.")
            break
        _say(f"Round {i + 1}: self-score {shown}; revising...")
        revised = _refine_tailored(current, crit, resume, job_title, company, job_description)
        if revised and revised.strip():
            current = revised

    # Final gate. Verify that fidelity and precision constraints are satisfied
    # before outputting. Repair mechanical discrepancies and report remaining issues.
    try:
        profile_text = _profile_context() or ""
    except Exception:  # noqa: BLE001
        profile_text = ""
    try:
        current, defects = _source_fidelity.repair_fidelity(
            f"{resume}\n{profile_text}", current)
    except Exception:  # noqa: BLE001 - never fail the tailor over the audit
        defects = []
    if defects:
        log.warning("Tailored resume ships with %d verified defect(s): %s",
                    len(defects), "; ".join(defects))

    return {
        "resume": current,
        "rounds": len(history),
        "score": last_score,
        "history": history,
        "defects": defects,
    }


def generate_cover_letter(resume: str, job_title: str, company: str,
                          job_description: str, applicant_name: str) -> str:
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
- Tone: authentic, senior professional speaking peer-to-peer.
- BANNED AI filler: spearheaded, leveraged, pioneered, utilized, streamlined, fostered, orchestrated, passionate, seasoned, beacon, dynamic team player, results-driven, executed workflows.
- USE active technical verbs: designed, architected, developed, optimized, characterized, profiled, modeled, automated, built, synthesized, troubleshot, quantified, benchmarked.

{_g.COVER_LETTER_NO_IMPORT_RULE}

JOB TITLE: {job_title}
COMPANY: {company}
JOB DESCRIPTION (what they want -- NOT the applicant's experience):
{job_description[:5000]}

RESUME (the candidate's verified experience -- the source of truth for every claim):
{resume[:8000]}{_template_block("cover")}"""
    return _generate(prompt, system=_g.SYSTEM_RESUME_TAILOR, temperature=0.4)


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
For each: **Q**, **Why they'd ask**, **STAR-ready answer hook** (point to a real experience).

## 5 hard technical questions you should be ready for
Specific to the techniques / disease areas / methods in the JD, with answer outlines (3-4 bullets each).

## 5 behavioral / culture-fit questions
Tailored to the company's values / role seniority.

## Questions YOU should ask them
6 sharp, role-specific questions.

## Red flags to watch for in the interview
3-5 signs this team may not be a great fit.

## Study list (24-hour cram)
Prioritized bullets of papers, methods, products, concepts.

## Salary / negotiation talking points
2-3 specific data points for this role + location.

Be specific, scientific, and honest. No fluff."""
    return _generate(prompt, system=_g.SYSTEM_RESUME_TAILOR, temperature=0.45)


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


def interview_question_bank(resume: str, job_title: str, company: str,
                            job_description: str, company_intel: str,
                            contact: Optional[dict] = None) -> dict:
    """Machine-readable predicted-question bank for the mock interviewer."""
    spec = _g._contact_bank_spec(contact) if contact else _g._BANK_SPEC
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
    out = _generate(prompt, system=_g.SYSTEM_RESUME_TAILOR, temperature=0.5)
    return _g._normalize_bank(_g._parse_json_block(out))


def score_answer(question: str, question_kind: str, answer: str,
                 job_title: str, company: str, resume: str) -> dict:
    out = _generate(_g._score_prompt(question, question_kind, answer,
                                     job_title, company, resume),
                    system=_g.SYSTEM_RESUME_TAILOR, temperature=0.2)
    return _g._normalize_rubric(_g._parse_json_block(out))


def interview_session_summary(job_title: str, company: str, transcript: str) -> str:
    return _generate(_g._summary_prompt(job_title, company, transcript),
                     system=_g.SYSTEM_RESUME_TAILOR, temperature=0.3)


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
    return _generate(prompt, system=_g.SYSTEM_RESUME_TAILOR, temperature=0.4)


def answer_application_questions(resume: str, job_title: str, company: str,
                                 job_description: str, questions: list[str]) -> dict:
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
    out = _generate(prompt, system=_g.SYSTEM_RESUME_TAILOR, temperature=0.3)
    parsed = _g._parse_json_block(out)
    return _g._align_answer_keys(parsed, questions)


def networking_message(resume: str, job_title: str, company: str,
                       job_description: str, contact: dict,
                       channel: str = "linkedin") -> dict:
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
    out = _generate(prompt, system=_g.SYSTEM_NETWORKING, temperature=0.6)
    return _g._coerce_networking(out, channel)


def polish_bullet(bullet: str, action: str = "action_verb", job_context: str = "") -> dict:
    """Polish a single resume bullet point according to SOTA scientist voice invariants.

    Actions:
    - 'action_verb': Replaces weak verbs with precise laboratory / engineering verbs.
    - 'quantify': Tightens technical metrics and construct parameters without fabricating data.
    - 'tighten': Shortens word count / trims filler to preserve single-line page budgeting.
    - 'align_jd': Reframes emphasis towards key job description requirements while remaining strictly truthful.

    Enforces Rule 1 (0% hallucination) and Rule 2 (Anti-AI Cliché / active verbs only).
    """
    clean_bullet = (bullet or "").strip()
    if not clean_bullet:
        return {"original": "", "polished": "", "action": action, "notes": "Empty bullet provided"}

    action_instructions = {
        "action_verb": (
            "Rewrite this bullet using a powerful, precise active technical/scientific verb at the start "
            "(e.g. purified, developed, expressed, characterized, optimized, modeled, automated, built, designed, isolated, synthesized, profiled, screened, cultured, troubleshot, quantified). "
            "NEVER use: spearheaded, leveraged, pioneered, utilized, streamlined, fostered, orchestrated, passionate, seasoned, beacon, testament, dynamic team player, results-driven, executed workflows."
        ),
        "quantify": (
            "Clarify and emphasize technical specificity, construct parameters, and operational scope "
            "WITHOUT inventing or fabricating new unverified metrics or percentages. Name real instruments, protocols, or constructs."
        ),
        "tighten": (
            "Make this bullet concise and punchy so it fits comfortably on 1 line (under 130 characters if possible). "
            "Remove passive filler words, wordy transitions, and redundant descriptions while preserving key technical substance."
        ),
        "align_jd": (
            "Reframe emphasis towards key competencies needed for the target role while strictly preserving truthful facts:\n"
            f"TARGET JOB CONTEXT: {job_context[:1000] if job_context else 'Technical role'}\n"
            "Do NOT invent experience the candidate does not have."
        ),
    }

    instruction = action_instructions.get(action, action_instructions["action_verb"])
    system = (
        "You are an expert resume polisher for scientists and engineers.\n"
        "MANDATORY INVARIANTS:\n"
        "1. Never invent or hallucinate credentials, degrees, dates, employers, or skills.\n"
        "2. Strict Anti-AI Buzzword Invariant: NEVER use 'spearheaded', 'leveraged', 'pioneered', 'utilized', "
        "'streamlined', 'fostered', 'orchestrated', 'passionate', 'seasoned', 'beacon', 'testament', "
        "'dynamic team player', 'results-driven', or 'executed workflows'.\n"
        "3. Use direct laboratory and technical verbs.\n"
        "4. Return ONLY the polished bullet point text (e.g. '- Purified ...' or 'Purified ...'), nothing else. No commentary."
    )

    prompt = f"{instruction}\n\nORIGINAL BULLET:\n{clean_bullet}\n\nPOLISHED BULLET:"

    try:
        if ensure_ready(pull=False):
            raw_polished = _generate(prompt, system=system, temperature=0.2).strip()
            # Clean any prompt echo or markdown header artifact
            for prefix in ("POLISHED BULLET:", "Polished Bullet:", "Polished:", "Here is the polished bullet:"):
                if raw_polished.startswith(prefix):
                    raw_polished = raw_polished[len(prefix):].strip()
            # Strip outer quotes if model added them
            if (raw_polished.startswith('"') and raw_polished.endswith('"')) or (raw_polished.startswith("'") and raw_polished.endswith("'")):
                raw_polished = raw_polished[1:-1].strip()
            # Ensure bullet prefix consistency
            is_bulleted = clean_bullet.startswith("- ") or clean_bullet.startswith("* ")
            if is_bulleted and not (raw_polished.startswith("- ") or raw_polished.startswith("* ")):
                raw_polished = f"- {raw_polished}"
            elif not is_bulleted and (raw_polished.startswith("- ") or raw_polished.startswith("* ")):
                raw_polished = raw_polished[2:].strip()
            polished = raw_polished
        else:
            polished = clean_bullet
    except Exception as exc:
        log.warning("Local Ollama polish_bullet call failed (%s); applying deterministic repair.", exc)
        polished = clean_bullet

    # Always pass through deterministic buzzword and fidelity repair
    fixed_polished, defects = _source_fidelity.repair_fidelity("", polished)
    return {
        "original": clean_bullet,
        "polished": fixed_polished.strip(),
        "action": action,
        "defects": defects,
    }

