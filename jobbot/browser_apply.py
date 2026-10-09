"""Assisted browser auto-apply (Claude-in-Chrome).

This module does the *deterministic* half of a browser application:
  1. Ensures AI-tailored materials exist for the job (resume PDF + cover letter).
  2. Assembles an "application package" — every value a form is likely to ask
     for (identity fields, links, file paths, cover-letter text) plus AI-drafted
     answers to common free-text questions.
  3. Provides the **CAPTCHA / human-checkpoint guard** used by the live flow.
  4. Generates a machine-followable **fill plan** — an ordered list of browser
     steps derived from a completed Apply Kit. The plan is a pure data transform
     (no network I/O, no browser automation) that a Claude-in-Chrome session
     can execute step by step.

The *interactive* half — navigating the page and typing into fields — is driven
by Claude-in-Chrome (the `mcp__claude-in-chrome__*` tools). That half is
deliberately NOT automated end-to-end: per the operator's instruction, whenever
a CAPTCHA, login wall, or other human checkpoint appears, the flow STOPS and
hands control back to the user. Final submission also pauses for review unless
`BROWSER_APPLY_AUTOSUBMIT=true`.

Safety model:
  * Never invents answers — `answer_application_questions` returns "" when the
    resume lacks the info, and those blanks are surfaced for the user to fill.
  * Never solves CAPTCHAs. `captcha_in_text()` / `checkpoint_reason()` detect
    them; the caller must yield to the user.
  * Dry-run by default: fills + reviews, submits only when explicitly enabled.

OPERATOR PROTOCOL — how a Claude-in-Chrome session executes a fill plan:
  See the FILL_PROTOCOL constant below for the canonical rules.
"""
from __future__ import annotations

import json
import logging
from dataclasses import dataclass, field, asdict
from datetime import datetime, timezone
from pathlib import Path
from typing import Dict, List, Optional

from .config import settings
from .models import Application, Job, OutcomeEvent, init_db, session
from .pipeline import tailor_for_job
from .apply_questions import (
    FormQuestion,
    answer_questions,
    fetch_form_questions,
    kit_markdown,
    needs_attention,
    questions_to_json,
    questions_from_json,
)

log = logging.getLogger("jobbot.browser_apply")

# ---------------------------------------------------------------------------
# Operator protocol — canonical execution rules for a Claude-in-Chrome session
# ---------------------------------------------------------------------------
FILL_PROTOCOL: str = (
    "FILL PLAN EXECUTION PROTOCOL: "
    "(1) Follow steps in the exact sequential order given. "
    "(2) After every navigate action, call checkpoint_reason(page_text); "
    "if the result is non-None, STOP immediately and report the reason to the user — "
    "do NOT attempt to solve CAPTCHAs or bypass login walls. "
    "(3) For 'fill' actions, type the value into the labelled field; "
    "for 'select'/'multiselect', choose the option matching the value. "
    "(4) For 'upload' actions, use the file path in 'value'; "
    "if 'exists' is False, pause and ask the user to supply the file first. "
    "(5) At any 'pause-user' step, STOP and surface the label + hint to the user; "
    "resume only after they provide a value. "
    "(6) At 'pause-review', STOP and let the user inspect + confirm before submitting. "
    "(7) At 'submit', proceed only if autosubmit is True; "
    "still halt on any CAPTCHA (checkpoint_reason). "
    "(8) Never invent or modify answers — use values exactly as given in the plan."
)


# Questions almost every application form asks in some form. The AI answers
# these up front so the live flow can paste them in without round-trips.
COMMON_QUESTIONS: List[str] = [
    "Why are you interested in this role?",
    "Why do you want to work at this company?",
    "What relevant experience makes you a strong fit for this position?",
    "Are you legally authorized to work in this country?",
    "Will you now or in the future require visa sponsorship?",
    "What are your salary expectations?",
    "What is your earliest available start date?",
    "Are you willing to work on-site / hybrid at the listed location?",
]


def applicant_fields() -> Dict[str, str]:
    """Standard identity fields most forms request, drawn from config."""
    name = settings.applicant_name or ""
    first, _, last = name.partition(" ")
    return {
        "full_name": name,
        "first_name": first,
        "last_name": last or first,
        "email": settings.applicant_email or settings.smtp_user or "",
        "phone": settings.applicant_phone or "",
        "location": settings.applicant_location or "",
        "linkedin": settings.applicant_linkedin or "",
        "github": settings.applicant_github or "",
        "portfolio": settings.applicant_portfolio or "",
        "current_title": settings.applicant_title or "",
    }


def _ensure_resume_question(questions: List[FormQuestion]) -> List[FormQuestion]:
    """Guarantee the package offers the tailored resume for upload. If the
    fetched form exposed no file question, prepend a synthetic 'Resume/CV' one so
    the fill plan / driver uploads the tailored resume."""
    if any((q.qtype == "file") or (q.kind == "file") for q in questions):
        return questions
    return [FormQuestion(text="Resume/CV", qtype="file", kind="file",
                         required=True)] + questions


@dataclass
class ApplicationPackage:
    job_id: int
    title: str
    company: str
    url: str
    application_id: Optional[int]
    resume_path: str
    cover_letter_path: str
    cover_letter_text: str
    fields: Dict[str, str]
    answers: Dict[str, str]
    blanks: List[str] = field(default_factory=list)   # questions AI couldn't answer
    autosubmit: bool = False
    notes: str = ""
    # Structured real-form questions (apply_questions.FormQuestion dicts) and
    # which extractor produced them ("greenhouse"/"lever"/"ashby"/"html"/"").
    form_questions: List[dict] = field(default_factory=list)
    questions_source: str = ""
    kit_markdown_path: str = ""

    def to_json(self) -> str:
        return json.dumps(asdict(self), indent=2, ensure_ascii=False)


def build_package(job_id: int, extra_questions: Optional[List[str]] = None,
                  fetch_form: bool = True) -> ApplicationPackage:
    """Assemble everything needed to fill the job's application form.

    Ensures tailored materials exist (runs `tailor_for_job` if the job has no
    Application yet), grabs the REAL questions from the job's application form
    (Greenhouse / Lever / Ashby APIs, generic HTML fallback), auto-answers them
    (answer bank -> profile/config -> AI), and falls back to the common
    question set when the form isn't readable (e.g. login-walled Workday).
    Writes the package JSON + a copy/paste markdown kit to OUTPUT_DIR, persists
    the Q&A on the Application row, and returns the package.
    """
    init_db()
    with session() as db:
        job = db.get(Job, job_id)
        if not job:
            raise ValueError(f"Job {job_id} not found")
        app_row = job.applications[0] if job.applications else None
        title, company, url, description = job.title, job.company, job.url, job.description

    # Make sure we have a tailored resume + cover letter on disk.
    if not app_row:
        result = tailor_for_job(job_id)
        application_id = result.get("application_id")
        resume_path = result.get("resume", "")
        cover_letter_path = result.get("cover_letter", "")
    else:
        application_id = app_row.id
        resume_path = app_row.tailored_resume_path or ""
        cover_letter_path = app_row.cover_letter_path or ""

    cover_text = ""
    if cover_letter_path and Path(cover_letter_path).exists():
        try:
            cover_text = Path(cover_letter_path).read_text(encoding="utf-8")
        except Exception as e:  # noqa: BLE001
            log.warning("Could not read cover letter %s: %s", cover_letter_path, e)

    # 1) Pull the REAL questions off the application form when possible.
    questions: List[FormQuestion] = []
    q_source = ""
    if fetch_form:
        questions, q_source = fetch_form_questions(url)
    # 2) Fall back to the generic set when the form isn't readable.
    if not questions:
        questions = [FormQuestion(text=q, qtype="textarea") for q in COMMON_QUESTIONS]
    questions = _ensure_resume_question(questions)
    for extra in extra_questions or []:
        questions.append(FormQuestion(text=extra, qtype="textarea"))

    # 3) Answer everything: answer bank -> profile/config -> AI.
    base_resume = ""
    try:
        from .resume import load_resume
        base_resume = load_resume(settings.base_resume_path)
    except Exception as e:  # noqa: BLE001
        # This is more than a missed resume load: an empty base_resume is
        # passed straight through to answer_questions as the fidelity source,
        # so every free-text AI answer on this application fails the
        # gate_free_text check closed (apply_questions.gate_free_text refuses
        # to verify a draft against nothing) and needs_user instead of being
        # answered. Call that consequence out, not just the load failure.
        log.warning("Could not load base resume (%s) - free-text AI answers "
                    "on this application will have no fidelity source to "
                    "verify against and will be held for the user.", e)
    answer_questions(questions, base_resume, title, company, description,
                     identity_fields=applicant_fields(),
                     resume_path=resume_path, cover_letter_path=cover_letter_path)

    answers = {q.text: q.answer for q in questions if q.answer}
    # "Needs you" is broader than "blank": it also covers answers we filled from
    # an unverified model guess on a required question, which must be confirmed
    # before submit rather than shipped silently.
    blanks = [q.text for q in needs_attention(questions)]

    pkg = ApplicationPackage(
        job_id=job_id,
        title=title,
        company=company,
        url=url,
        application_id=application_id,
        resume_path=resume_path,
        cover_letter_path=cover_letter_path,
        cover_letter_text=cover_text,
        fields=applicant_fields(),
        answers=answers,
        blanks=blanks,
        autosubmit=settings.browser_apply_autosubmit,
        notes=("Auto-submit is ON." if settings.browser_apply_autosubmit
               else "Auto-submit is OFF — fill fields, then pause for human review before submit."),
        form_questions=[asdict(q) for q in questions],
        questions_source=q_source,
    )

    out_dir = Path(settings.output_dir)
    out_dir.mkdir(parents=True, exist_ok=True)

    md_path = out_dir / f"{job_id}_apply_kit.md"
    md_path.write_text(
        kit_markdown(title, company, url, pkg.fields, questions, q_source,
                     resume_path, cover_letter_path),
        encoding="utf-8")
    pkg.kit_markdown_path = str(md_path)

    out_path = out_dir / f"{job_id}_application_package.json"
    out_path.write_text(pkg.to_json(), encoding="utf-8")

    # Persist the Q&A on the Application row so the dashboard can show/edit it.
    if application_id:
        with session() as db:
            app = db.get(Application, application_id)
            if app:
                app.questions_json = questions_to_json(questions, q_source)
                db.commit()

    log.info("Application package for job %d written to %s (questions via %s)",
             job_id, out_path, q_source or "common defaults")
    return pkg


# ----- CAPTCHA / human-checkpoint guard -------------------------------------

def captcha_in_text(text: str) -> bool:
    """True if the page text/HTML contains any configured CAPTCHA marker."""
    if not text:
        return False
    low = text.lower()
    return any(marker in low for marker in settings.captcha_markers)


def checkpoint_reason(page_text: str) -> Optional[str]:
    """Return a human-readable reason to PAUSE (CAPTCHA / login / verification),
    or None if it's safe to continue. The live browser flow must call this
    after every navigation and before any submit; on a non-None result it must
    stop and hand control to the user."""
    if not page_text:
        return None
    low = page_text.lower()
    for marker in settings.captcha_markers:
        if marker in low:
            return f"CAPTCHA / bot-check detected (matched '{marker}')"
    login_markers = ("sign in to continue", "log in to apply", "please sign in",
                     "create an account to apply")
    for m in login_markers:
        if m in low:
            return f"Login wall detected (matched '{m}')"
    if "verify your email" in low or "two-factor" in low or "one-time code" in low:
        return "Email/2FA verification step detected"
    return None


def captcha_reason(page_text: str) -> Optional[str]:
    """CAPTCHA-only subset of checkpoint_reason. Returns a reason to HARD-STOP
    on a bot-check, or None. Unlike checkpoint_reason it does NOT treat login
    walls as a stop — profile-backed drivers pause for login and resume."""
    if not page_text:
        return None
    low = page_text.lower()
    for marker in settings.captcha_markers:
        if marker in low:
            return f"CAPTCHA / bot-check detected (matched '{marker}')"
    return None


def mark_applied(application_id: int, method: str = "browser",
                 when: Optional[datetime] = None) -> None:
    """Mark an application + its job as applied after a confirmed submission.

    Also appends the 'applied' OutcomeEvent that the ranking priors count. That
    row used to be written only by manual CLI/web calls, so every automated
    application was missing from the funnel history.

    `when` back-dates a hand-sent application. Idempotent: a second call on an
    application that already has an 'applied' event is a no-op.
    """
    from .outcomes import record_outcome

    init_db()
    with session() as db:
        app = db.get(Application, application_id)
        if not app:
            raise ValueError(f"Application {application_id} not found")
        already = (db.query(OutcomeEvent)
                   .filter_by(application_id=application_id, stage="applied")
                   .first())
        # Set the status BEFORE record_outcome. With _RANK["sent"] == 1, the
        # comparison stage_rank("applied") > stage_rank("sent") is then false, so
        # record_outcome appends the event and leaves the status alone. Called in
        # the other order the status is still "draft" (rank 0) and would be
        # overwritten with "applied", breaking followup.py's status == "sent".
        app.status = "sent"
        # Set the date only when explicitly given or not yet recorded. A redundant
        # bare re-mark must not clobber a previously back-dated sent_at with "now";
        # an explicit `when` on a later call is an intentional correction and wins.
        if when is not None or app.sent_at is None:
            app.sent_at = when or datetime.now(timezone.utc).replace(tzinfo=None)
        if app.job:
            app.job.status = "applied"
        db.commit()

    if not already:
        record_outcome(application_id, "applied", note=f"auto: {method}")


# ---------------------------------------------------------------------------
# Fill-plan generator (pure data transform — no network I/O, no browser I/O)
# ---------------------------------------------------------------------------

# Kind ordering for deterministic step sequencing that matches real form layouts:
#   identity + short + screening come first (top of every ATS form),
#   then file uploads (usually just below identity on Greenhouse/Lever),
#   then essays/textareas (the long free-text section),
#   then EEO last (compliance section, always at the bottom).
_KIND_ORDER = {
    "identity": 0,
    "short": 1,
    "screening": 2,
    "file": 3,
    "essay": 4,
    "eeo": 5,
    "": 1,  # unclassified short questions treated as "short"
}


def _question_sort_key(q: FormQuestion) -> int:
    """Return a sort integer so questions are ordered for real ATS form layouts."""
    # textarea / essay-like questions go after simple fields even when kind is unset
    if q.kind:
        base = _KIND_ORDER.get(q.kind, 1)
    elif q.qtype in ("textarea",):
        base = _KIND_ORDER["essay"]
    elif q.qtype == "file":
        base = _KIND_ORDER["file"]
    else:
        base = _KIND_ORDER["short"]
    return base


def _resolve_file_path(label: str, resume_path: str,
                       cover_letter_path: str) -> str:
    """Return the absolute path for a file-upload question.

    Heuristic: if the label mentions "cover" it maps to cover_letter_path,
    otherwise it defaults to the tailored resume.
    """
    low = label.lower()
    if "cover" in low and cover_letter_path:
        target = cover_letter_path
    elif resume_path:
        target = resume_path
    elif cover_letter_path:
        target = cover_letter_path
    else:
        return ""
    # Resolve to an absolute POSIX-style string via pathlib.
    try:
        return str(Path(target).resolve())
    except Exception:  # noqa: BLE001
        return target


def fill_plan(job_id: int) -> dict:
    """Convert a completed Apply Kit into an ordered list of browser steps.

    Loads the job and its Application.questions_json from the database.
    Raises ValueError (with a helpful build-kit hint) when the kit is missing.

    Returns a dict with keys:
      job_id, title, company, url, source, autosubmit, generated_at,
      resume_path, cover_letter_path,
      steps: [ordered step dicts],
      needs_user: [labels of steps that require human input]

    Step schemas
    ------------
    navigate  : {step, action, value, note}
    fill      : {step, action, label, value, multiline, required}
    select    : {step, action, label, value, options, required}
    upload    : {step, action, label, value, exists, required}
    pause-user: {step, action, label, hint, options, required}
    pause-review: {step, action, note}
    submit    : {step, action, note}
    """
    init_db()
    with session() as db:
        job = db.get(Job, job_id)
        if not job:
            raise ValueError(f"Job {job_id} not found")
        app_row = job.applications[0] if job.applications else None
        title = job.title
        company = job.company
        url = job.url
        questions_json_raw = app_row.questions_json if app_row else ""
        resume_path = app_row.tailored_resume_path or "" if app_row else ""
        cover_letter_path = app_row.cover_letter_path or "" if app_row else ""
        application_id = app_row.id if app_row else None

    if not questions_json_raw or not questions_json_raw.strip():
        raise ValueError(
            f"No Apply Kit found for job {job_id} ({title} @ {company}). "
            f"Build it first with: jobbot apply-kit {job_id}"
        )

    questions, source = questions_from_json(questions_json_raw)
    if not questions:
        raise ValueError(
            f"Apply Kit for job {job_id} is empty or unreadable. "
            f"Rebuild it with: jobbot apply-kit {job_id}"
        )

    autosubmit = settings.browser_apply_autosubmit

    # Stable sort: preserve original question order within the same kind bucket
    # so the plan matches the actual ATS form layout.
    sorted_qs = sorted(enumerate(questions), key=lambda t: _question_sort_key(t[1]))

    steps: List[dict] = []
    needs_user_labels: List[str] = []

    # Step 1: navigate
    steps.append({
        "step": 1,
        "action": "navigate",
        "value": url,
        "note": "Open posting, click Apply if needed",
    })

    step_num = 2
    for _orig_idx, q in sorted_qs:
        # Determine whether this step needs a human pause.
        # Optional EEO with no answer is silently skipped (unanswered EEO is fine
        # unless required=True).
        unanswered = not (q.answer or "").strip()
        optional_eeo_blank = (
            q.kind == "eeo" and not q.required and unanswered
        )

        # needs_review means a POPULATED but unverified answer (an AI guess on a
        # required question) -- it must pause too, or with autosubmit on it would
        # be filled and submitted to the employer without review. Gating only on
        # needs_user / blank let it through as a normal fill step.
        if q.needs_user or q.needs_review or (unanswered and not optional_eeo_blank):
            review = q.needs_review and not q.needs_user and bool((q.answer or "").strip())
            step = {
                "step": step_num,
                "action": "pause-user",
                "label": q.text,
                "hint": (f"Unverified guess — confirm before submitting: {q.answer}"
                         if review else "No confident answer available — ask the user"),
                "options": q.options,
                "required": q.required,
            }
            needs_user_labels.append(q.text)

        elif q.qtype == "file":
            abs_path = _resolve_file_path(q.text, resume_path, cover_letter_path)
            step = {
                "step": step_num,
                "action": "upload",
                "label": q.text,
                "value": abs_path,
                "exists": bool(abs_path and Path(abs_path).exists()),
                "required": q.required,
            }
            if not step["exists"]:
                # Treat a missing file as needing user attention.
                step["action"] = "pause-user"
                step["hint"] = f"File not found: {abs_path or '(no path configured)'}"
                needs_user_labels.append(q.text)

        elif q.qtype in ("select", "multiselect"):
            step = {
                "step": step_num,
                "action": "select",
                "label": q.text,
                "value": q.answer,
                "options": q.options,
                "required": q.required,
            }

        else:
            # text / textarea / date / number / boolean / everything else
            step = {
                "step": step_num,
                "action": "fill",
                "label": q.text,
                "value": q.answer,
                "multiline": q.qtype == "textarea",
                "required": q.required,
            }

        steps.append(step)
        step_num += 1

    # Final step: pause-review or submit
    if autosubmit:
        steps.append({
            "step": step_num,
            "action": "submit",
            "note": (
                "Autosubmit is enabled — click the final Submit button. "
                "Still stop if checkpoint_reason() returns non-None (CAPTCHA / login)."
            ),
        })
    else:
        steps.append({
            "step": step_num,
            "action": "pause-review",
            "note": (
                "Autosubmit is OFF — review every field before submitting. "
                "When satisfied, click Submit manually and call mark_applied()."
            ),
        })

    return {
        "job_id": job_id,
        "title": title,
        "company": company,
        "url": url,
        "source": source,
        "autosubmit": autosubmit,
        "generated_at": datetime.utcnow().isoformat() + "Z",
        "resume_path": resume_path,
        "cover_letter_path": cover_letter_path,
        "steps": steps,
        "needs_user": needs_user_labels,
    }


def write_fill_plan(job_id: int) -> str:
    """Generate and write the fill plan for *job_id* to OUTPUT_DIR.

    Calls fill_plan(), serialises to pretty JSON (UTF-8), writes to
    ``{settings.output_dir}/{job_id}_fill_plan.json``, and returns the
    absolute path string.

    Raises ValueError (propagated from fill_plan) when no kit exists.
    """
    plan = fill_plan(job_id)
    out_dir = Path(settings.output_dir)
    out_dir.mkdir(parents=True, exist_ok=True)
    out_path = out_dir / f"{job_id}_fill_plan.json"
    out_path.write_text(json.dumps(plan, indent=2, ensure_ascii=False),
                        encoding="utf-8")
    log.info("Fill plan for job %d written to %s (%d steps, %d need user)",
             job_id, out_path, len(plan["steps"]), len(plan["needs_user"]))
    return str(out_path.resolve())
