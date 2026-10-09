"""Filling the gaps an application form leaves behind — by asking the user.

The auto-answer pipeline (:mod:`jobbot.apply_questions`) deliberately refuses to
invent answers: anything it can't source confidently comes back flagged, either
as an outright blank (``needs_user``) or as a model guess held for confirmation
(``needs_review``). This module turns those flags into an actual conversation.

The I/O is injected (``ask``) so the same loop drives the terminal prompt, and
so the decision logic is testable without a TTY. Nothing here writes to the
database; see :func:`fill_job_gaps` for the persisting wrapper.
"""
from __future__ import annotations

import logging
from dataclasses import dataclass, field
from typing import Callable, List, Optional, Tuple

from .apply_questions import FormQuestion, _snap_to_options, needs_attention

log = logging.getLogger("jobbot.apply_fill")


@dataclass
class Gap:
    """One question awaiting the user, with everything the prompt needs."""
    index: int                  # position in the original question list
    question: FormQuestion
    reason: str                 # "blank" (nothing to submit) | "confirm" (guess held)
    suggestion: str = ""        # a value to offer as the default, if we have one
    error: str = ""             # why the previous reply was rejected, when re-asking


@dataclass
class Reply:
    """What the user said. Exactly one of these modes is meaningful."""
    value: str = ""
    remember: bool = False      # also save to the answer bank for future forms
    skip: bool = False          # leave this gap open, move on
    quit: bool = False          # stop asking entirely
    accept: bool = False        # keep the existing (held) answer as-is


def gaps(questions: List[FormQuestion]) -> List[Gap]:
    """Every question the user still has to deal with, in form order."""
    out: List[Gap] = []
    for i, q in enumerate(questions):
        if q.needs_user:
            out.append(Gap(index=i, question=q, reason="blank",
                           suggestion=q.guess))
        elif q.needs_review:
            out.append(Gap(index=i, question=q, reason="confirm",
                           suggestion=q.answer))
    return out


def validate(q: FormQuestion, raw: str) -> Tuple[str, str]:
    """Turn a raw reply into a submittable answer.

    Returns ``(answer, error)``; exactly one is non-empty. On an options
    question the reply may be the option text, a loose variant of it ("yes"),
    or its 1-based number. Anything that doesn't resolve is an error rather
    than a value — the whole point is to stop unsubmittable answers."""
    raw = (raw or "").strip()
    if not raw:
        if q.required:
            return "", "This question is required - please give an answer."
        return "", ""

    if not q.options:
        return raw, ""

    # The reply may name an option outright. Try that BEFORE reading a bare
    # number as a position: on a numeric dropdown ("0/1/2/3" years) the digit
    # the user typed is the option itself, and treating it as an index quietly
    # submitted its neighbour.
    snapped = _snap_to_options(raw, q.options)
    if snapped:
        return snapped, ""

    # Otherwise a bare number picks the option at that position.
    if raw.isdigit():
        idx = int(raw) - 1
        if 0 <= idx < len(q.options):
            return q.options[idx], ""
        return "", f"Pick 1-{len(q.options)}, or type the option text."
    return "", ("Not one of the options. Choose from: "
                + " | ".join(q.options))


def fill_gaps(questions: List[FormQuestion],
              ask: Callable[[Gap], Reply],
              remember: Optional[Callable[[str, str], None]] = None,
              confirm_remember: Optional[Callable[[Gap, str], bool]] = None) -> dict:
    """Walk every gap, asking until each is answered, skipped, or the user quits.

    Mutates `questions` in place. `ask` receives a :class:`Gap` and returns a
    :class:`Reply`; an invalid reply re-asks the same question. `remember`, when
    given, is called for answers the user chose to save to the answer bank.

    `confirm_remember` is asked *after* an answer validates — so the user is
    never offered the chance to save a value that is about to be rejected.
    """
    report = {"answered": 0, "skipped": 0, "remembered": 0, "quit": False,
              "errors": []}

    for gap in gaps(questions):
        q = gap.question
        while True:
            try:
                reply = ask(gap)
            except (EOFError, KeyboardInterrupt):
                report["quit"] = True
                return report

            if reply.quit:
                report["quit"] = True
                return report
            if reply.skip:
                report["skipped"] += 1
                break
            if reply.accept and q.answer:
                # Keep the held guess, but it's a human decision now.
                q.answer_source = "user"
                q.needs_user = False
                q.needs_review = False
                report["answered"] += 1
                break

            value, err = validate(q, reply.value)
            if err:
                report["errors"].append({"question": q.text, "error": err})
                # Carry the reason forward so the re-ask can explain itself.
                gap = Gap(index=gap.index, question=q, reason=gap.reason,
                          suggestion=gap.suggestion, error=err)
                continue
            if not value:
                # Empty reply on an optional question — treat as a skip.
                report["skipped"] += 1
                break

            q.answer = value
            q.answer_source = "user"
            q.guess = ""
            q.needs_user = False
            q.needs_review = False
            report["answered"] += 1

            want_remember = reply.remember
            if not want_remember and confirm_remember is not None:
                try:
                    want_remember = bool(confirm_remember(gap, value))
                except (EOFError, KeyboardInterrupt):
                    want_remember = False
            if want_remember and remember is not None:
                try:
                    remember(q.text, value)
                    report["remembered"] += 1
                except Exception as e:  # noqa: BLE001
                    log.warning("Could not save %r to the answer bank: %s",
                                q.text, e)
            break

    return report


def recheck(questions: List[FormQuestion]) -> int:
    """Re-validate already-stored answers against each question's real options.

    Kits built before the option-snapping fix can carry answers that were never
    valid for the form (the classic: a gender answer sitting in a Yes/No
    transgender field). Demote any such answer to a gap, keeping the old value
    as the suggestion. Returns how many were demoted.
    """
    changed = 0
    for q in questions:
        if not q.answer or not q.options:
            continue
        if _snap_to_options(q.answer, q.options):
            continue
        q.guess, q.answer, q.answer_source = q.answer, "", ""
        q.needs_user, q.needs_review = True, False
        changed += 1
    return changed


def recheck_job(job_id: int) -> int:
    """`recheck` against a job's stored kit, persisting any demotions."""
    from .apply_questions import questions_from_json, questions_to_json
    from .models import Job, init_db, session

    init_db()
    with session() as db:
        job = db.get(Job, job_id)
        if not job or not job.applications:
            return 0
        app_row = job.applications[0]
        questions, source = questions_from_json(app_row.questions_json)
        if not questions:
            return 0
        changed = recheck(questions)
        if changed:
            app_row.questions_json = questions_to_json(questions, source)
            db.commit()
    if changed:
        _rewrite_kit(job_id, questions, source)
    return changed


def fill_job_gaps(job_id: int, ask: Callable[[Gap], Reply],
                  confirm_remember: Optional[Callable[[Gap, str], bool]] = None
                  ) -> dict:
    """`fill_gaps` for a job's stored kit, persisting the result.

    Loads the questions off the Application row, fills the gaps, then writes
    them back and regenerates the copy/paste sheet so the kit on disk matches
    what the user just answered.
    """
    from .apply_questions import (questions_from_json, questions_to_json,
                                  remember_answer)
    from .models import Job, init_db, session

    init_db()
    with session() as db:
        job = db.get(Job, job_id)
        if not job or not job.applications:
            return {"error": f"No application kit for job {job_id}.",
                    "answered": 0, "skipped": 0, "remembered": 0, "quit": False}
        app_row = job.applications[0]
        questions, source = questions_from_json(app_row.questions_json)
        if not questions:
            return {"error": f"Job {job_id} has no stored form questions - "
                             f"build a kit first (`jobbot apply-kit {job_id}`).",
                    "answered": 0, "skipped": 0, "remembered": 0, "quit": False}

        report = fill_gaps(questions, ask=ask, remember=remember_answer,
                           confirm_remember=confirm_remember)
        app_row.questions_json = questions_to_json(questions, source)
        db.commit()

    report["remaining"] = len(needs_attention(questions))
    _rewrite_kit(job_id, questions, source)
    return report


def _rewrite_kit(job_id: int, questions: List[FormQuestion], source: str) -> None:
    """Refresh the on-disk copy sheet after the user fills gaps. Best-effort:
    a stale sheet is not worth failing an otherwise-successful fill."""
    try:
        import json
        from pathlib import Path

        from .config import settings
        pkg_path = Path(settings.output_dir) / f"{job_id}_application_package.json"
        if not pkg_path.exists():
            return
        pkg = json.loads(pkg_path.read_text(encoding="utf-8"))

        from dataclasses import asdict
        pkg["form_questions"] = [asdict(q) for q in questions]
        pkg["answers"] = {q.text: q.answer for q in questions if q.answer}
        pkg["blanks"] = [q.text for q in needs_attention(questions)]
        pkg_path.write_text(json.dumps(pkg, indent=2, ensure_ascii=False),
                            encoding="utf-8")

        from .apply_questions import kit_markdown
        md = kit_markdown(pkg.get("title", ""), pkg.get("company", ""),
                          pkg.get("url", ""), pkg.get("fields", {}), questions,
                          source, pkg.get("resume_path", ""),
                          pkg.get("cover_letter_path", ""))
        (Path(settings.output_dir) / f"{job_id}_apply_kit.md").write_text(
            md, encoding="utf-8")
    except Exception as e:  # noqa: BLE001
        log.warning("Could not refresh the apply kit for job %d: %s", job_id, e)


def queue_gap_counts() -> List[dict]:
    """Per-job gap counts across every prepared-but-unsent kit.

    Powers both the dashboard's queue-wide "N answers needed" figure and the
    CLI's whole-queue fill, so the two can never disagree.
    """
    from .apply_questions import questions_from_json
    from .models import Application, Job, init_db, session

    init_db()
    out: List[dict] = []
    with session() as db:
        rows = (db.query(Application).join(Job)
                .filter(Application.questions_json != "",
                        Application.status != "sent",
                        Job.status.notin_(["applied", "rejected", "skipped"]))
                .all())
        for app_row in rows:
            try:
                questions, _ = questions_from_json(app_row.questions_json)
            except Exception:  # noqa: BLE001
                continue
            pending = needs_attention(questions)
            if not pending:
                continue
            out.append({
                "job_id": app_row.job_id,
                "title": app_row.job.title,
                "company": app_row.job.company,
                "blanks": sum(1 for q in pending if q.needs_user),
                "confirms": sum(1 for q in pending if q.needs_review),
                "total": len(pending),
            })
    out.sort(key=lambda r: -r["total"])
    return out
