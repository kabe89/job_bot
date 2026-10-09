"""Interview coach: structured prep pack + question bank + mock-interview engine.

Upgrades the one-shot `pipeline.generate_interview_prep` text blob into four
reviewable sections plus a machine-readable question bank, and drives a DB-backed
practice loop (one engine, two front-ends: the web chat panel and
`jobbot interview practice`).

Fail-open by contract: no function here raises into a caller. LLM down -> the
existing text prep is preserved, the bank is empty-but-valid, and scoring returns
a deterministic neutral rubric so practice never dead-ends.
"""
from __future__ import annotations

import json
import logging
from datetime import datetime
from pathlib import Path
from typing import Optional

from . import ai_client as gc
from . import referrals, research_enricher
from .config import settings
from .models import Job, init_db, session

log = logging.getLogger("jobbot.interview_coach")

BANK_KEYS = ("behavioral", "technical", "research", "contact")


def _empty_bank() -> dict:
    return {k: [] for k in BANK_KEYS}


def load_bank(job) -> dict:
    """Parse a job's stored question bank. Fail-open -> empty 4-key shape."""
    try:
        parsed = json.loads(job.interview_question_bank or "")
    except Exception:  # noqa: BLE001
        return _empty_bank()
    if not isinstance(parsed, dict):
        return _empty_bank()
    out = _empty_bank()
    for key in BANK_KEYS:
        items = parsed.get(key)
        if isinstance(items, list):
            out[key] = [i for i in items if isinstance(i, dict) and i.get("q")]
    return out


def _bridges(bank: dict) -> str:
    """Section 3: background -> JD bridges, derived from the bank's answer hooks
    (no extra LLM call — the hooks already name the candidate's real evidence)."""
    lines = []
    for kind in BANK_KEYS:
        for item in bank.get(kind, []):
            hook = (item.get("answer_hook") or "").strip()
            if hook:
                lines.append(f"- **{item['q']}** -> lead with: {hook}")
    return "\n".join(lines) or "_No bridges available (question bank is empty)._"


def _questions_to_ask(bank: dict, company: str) -> str:
    """Section 4: questions the candidate should ask, mirrored off the research
    questions so they land on this company's actual work."""
    qs = [i["q"] for i in bank.get("research", []) if i.get("q")][:4]
    if not qs:
        return (f"- What does success in this role look like at {company} in the "
                "first 6 months?\n"
                "- Who would I work with most closely, and on what?\n"
                "- What is the biggest technical risk on this program right now?")
    out = ["- What is the biggest technical risk on this program right now?",
           f"- How does this team measure success at {company}?"]
    for q in qs[:2]:
        out.append(f"- (Turn their own ground back on them) They may ask: \"{q}\" "
                   "-- ask how the team is currently attacking that.")
    return "\n".join(out)


def render_pack(job, bank: dict) -> str:
    """The four structured sections, as Markdown."""
    return (
        f"# Interview Prep - {job.title} @ {job.company}\n\n"
        f"## 1. Predicted questions + tailored answers\n\n"
        f"{job.interview_prep or '_Not available._'}\n\n"
        f"## 2. Company & role research\n\n"
        f"{job.company_intel or '_Not available._'}\n\n"
        f"## 3. Your background -> this JD\n\n"
        f"{_bridges(bank)}\n\n"
        f"## 4. Questions to ask them\n\n"
        f"{_questions_to_ask(bank, job.company)}\n"
    )


def generate_pack(job_id: int) -> dict:
    """Build (or rebuild) the structured prep pack + question bank for a job.

    Always clears `interview_prep_pending` -- even on LLM failure -- so a dead
    Ollama cannot leave a job in a permanent retry loop with the background
    worker. Returns a result dict; never raises.
    """
    from .pipeline import _augment_description, _slug, load_resume

    init_db()
    result = {"job_id": job_id, "prep": "", "intel": "", "bank": _empty_bank(),
              "path": "", "error": ""}

    if not settings.interview_coach_enabled:
        _clear_pending(job_id)
        result["error"] = "interview coach disabled"
        return result

    with session() as db:
        job = db.get(Job, job_id)
        if job is None:
            result["error"] = f"job {job_id} not found"
            return result
        title, company = job.title, job.company
        description = job.description
        embedding = job.embedding
        prior_prep = job.interview_prep or ""
        prior_intel = job.company_intel or ""

    try:
        resume = load_resume(settings.base_resume_path)
    except Exception as e:  # noqa: BLE001
        resume = ""
        log.warning("generate_pack: resume unreadable (%s)", e)

    aug = _augment_description(description, embedding)
    prep, intel, bank, err = prior_prep, prior_intel, _empty_bank(), ""

    try:
        prep = gc.interview_prep(resume, title, company, aug) or prior_prep
        intel = gc.company_intel(company, title, aug) or prior_intel
    except Exception as e:  # noqa: BLE001
        err = f"prep generation failed: {e}"
        log.warning("generate_pack(%s): %s", job_id, err)

    if not err:
        contact = _contact_payload(job_id)
        try:
            bank = _normalize(gc.interview_question_bank(
                resume, title, company, aug, intel, contact=contact))
        except Exception as e:  # noqa: BLE001
            err = f"question bank failed: {e}"
            log.warning("generate_pack(%s): %s", job_id, err)
        else:
            bank = _stamp_provenance(bank, contact)

    path = ""
    with session() as db:
        job = db.get(Job, job_id)
        if job is None:
            result["error"] = err or f"job {job_id} vanished"
            return result
        job.interview_prep = prep
        job.company_intel = intel
        job.interview_question_bank = json.dumps(bank)
        job.interview_prep_generated_at = datetime.utcnow()
        job.interview_prep_pending = False
        db.commit()
        try:
            out_dir = Path(settings.output_dir)
            out_dir.mkdir(parents=True, exist_ok=True)
            slug = f"{job.id}_{_slug(job.company)}_{_slug(job.title)}"[:120]
            p = out_dir / f"{slug}_interview_prep.md"
            p.write_text(render_pack(job, bank), encoding="utf-8")
            path = str(p)
        except Exception as e:  # noqa: BLE001
            log.warning("generate_pack(%s): could not write pack file (%s)", job_id, e)

    result.update(prep=prep, intel=intel, bank=bank, path=path, error=err)
    return result


def generate_pending(limit: int = 5) -> list[int]:
    """Build packs for jobs flagged by the interview-stage auto-trigger.

    Called from a background thread (web) and on demand. Fail-open per job: one
    bad job never stops the batch. Returns the job ids processed.
    """
    init_db()
    try:
        with session() as db:
            ids = [j.id for j in db.query(Job)
                   .filter(Job.interview_prep_pending.is_(True))
                   .order_by(Job.match_score.desc())
                   .limit(max(0, limit)).all()]
    except Exception as e:  # noqa: BLE001
        log.warning("generate_pending: could not list pending jobs (%s)", e)
        return []
    done: list[int] = []
    for jid in ids:
        try:
            generate_pack(jid)
            done.append(jid)
        except Exception as e:  # noqa: BLE001
            log.warning("generate_pending: job %s failed (%s)", jid, e)
    return done


def _normalize(bank) -> dict:
    out = _empty_bank()
    if not isinstance(bank, dict):
        return out
    for key in BANK_KEYS:
        items = bank.get(key)
        if isinstance(items, list):
            out[key] = [i for i in items if isinstance(i, dict) and i.get("q")]
    return out


def _clear_pending(job_id: int) -> None:
    try:
        with session() as db:
            job = db.get(Job, job_id)
            if job is not None:
                job.interview_prep_pending = False
                db.commit()
    except Exception as e:  # noqa: BLE001
        log.warning("could not clear pending on job %s (%s)", job_id, e)


def _stamp_provenance(bank: dict, contact: Optional[dict]) -> dict:
    """Authoritative provenance. The LLM's own `grounded_in` is advisory only:
    research questions get company_intel/jd, contact questions get contact:<id>,
    and contact questions are DROPPED entirely when no contact matched (so the
    model cannot invent an interviewer the user does not actually know)."""
    for item in bank.get("research", []):
        if item.get("grounded_in") not in ("company_intel", "jd"):
            item["grounded_in"] = "company_intel"
    if contact and contact.get("id"):
        for item in bank.get("contact", []):
            item["grounded_in"] = f"contact:{contact['id']}"
    else:
        bank["contact"] = []
    for kind in ("behavioral", "technical"):
        for item in bank.get(kind, []):
            item["grounded_in"] = item.get("grounded_in") or "jd"
    return bank


def _research_text(pkg: dict) -> str:
    """Flatten a research_package into prompt-ready text."""
    if not isinstance(pkg, dict):
        return ""
    parts = []
    for key in ("hook", "skill_connection", "health_impact"):
        val = str(pkg.get(key) or "").strip()
        if val:
            parts.append(val)
    papers = pkg.get("key_papers") or []
    if isinstance(papers, list):
        parts.extend(str(p).strip() for p in papers if str(p).strip())
    return "\n".join(parts)


def matched_contact(job) -> Optional[dict]:
    """The warmest known contact at this job's company, with their research.

    Returns None when no contact matches. Fail-open: if research enrichment
    fails, the contact is still returned with research="" so the interviewer can
    role-play them from name/title alone.
    """
    try:
        hits = referrals.contacts_for_job(job, referrals.build_index())
    except Exception as e:  # noqa: BLE001
        log.warning("matched_contact: referral lookup failed (%s)", e)
        return None
    if not hits:
        return None
    best = hits[0]
    payload = {"id": best.id, "name": best.name or "",
               "title": best.title or "", "company": best.company or "",
               "research": ""}
    try:
        enriched = research_enricher.enrich_contact(
            {"name": payload["name"], "role": payload["title"],
             "company": payload["company"], "notes": best.notes or ""})
        payload["research"] = _research_text(enriched.get("research_package") or {})
    except Exception as e:  # noqa: BLE001
        log.warning("matched_contact: enrichment failed for %s (%s)",
                    payload["name"], e)
    return payload


def _contact_payload(job_id: int) -> Optional[dict]:
    try:
        with session() as db:
            job = db.get(Job, job_id)
            return matched_contact(job) if job is not None else None
    except Exception as e:  # noqa: BLE001
        log.warning("_contact_payload(%s) failed (%s)", job_id, e)
        return None


# ---- practice engine ------------------------------------------------------

MODES = ("general", "technical", "behavioral", "research", "contact")

MODE_KINDS: dict[str, tuple[str, ...]] = {
    # "general" deliberately excludes contact questions: role-playing a real
    # person is an explicit choice, not something a mixed session drops in.
    "general": ("behavioral", "technical", "research"),
    "behavioral": ("behavioral",),
    "technical": ("technical",),
    "research": ("research",),
    "contact": ("contact",),
}

RUBRIC_AXES = ("relevance", "specificity", "structure", "fit")
# A base answer at or below this triggers one adaptive follow-up (if the scorer
# named a real gap). 3/5 is "adequate" -- we drill on adequate-or-worse.
FOLLOWUP_SCORE_THRESHOLD = 3.0


def ask_dict(turn) -> dict:
    """The canonical question wire shape shared by the web routes and the CLI."""
    return {"turn_id": turn.id, "idx": turn.idx, "parent_idx": turn.parent_idx,
            "question": turn.question, "question_kind": turn.question_kind,
            "grounded_in": turn.grounded_in}


def _pool(bank: dict, mode: str) -> list[dict]:
    """Bank questions for a mode, interleaved across kinds so a general session
    alternates rather than front-loading every behavioral question."""
    kinds = MODE_KINDS[mode]
    lanes = [[dict(i, question_kind=k) for i in bank.get(k, [])] for k in kinds]
    out: list[dict] = []
    for row in range(max((len(l) for l in lanes), default=0)):
        for lane in lanes:
            if row < len(lane):
                out.append(lane[row])
    return out


def _draw(db, session_id: int):
    """Create the next unasked InterviewTurn for a session, or None if spent."""
    from .models import InterviewSession, InterviewTurn

    sess = db.get(InterviewSession, session_id)
    if sess is None or sess.ended_at is not None:
        return None
    job = db.get(Job, sess.job_id)
    if job is None:
        return None
    turns = (db.query(InterviewTurn)
             .filter(InterviewTurn.session_id == session_id).all())
    asked = {t.question for t in turns}
    nxt_idx = max((t.idx for t in turns), default=-1) + 1
    for item in _pool(load_bank(job), sess.mode):
        if item["q"] in asked:
            continue
        turn = InterviewTurn(session_id=session_id, idx=nxt_idx, parent_idx=None,
                             question=item["q"],
                             question_kind=item["question_kind"],
                             grounded_in=item.get("grounded_in") or "")
        db.add(turn)
        db.commit()
        db.refresh(turn)
        return turn
    return None


def start(db, job_id: int, mode: str = "general",
          contact_id: Optional[int] = None) -> dict:
    """Open a practice session and draw its first question.

    Returns {"session_id": int, "question": dict | None, "error": str}.
    session_id is 0 when the session could not start. Never raises.
    """
    from .models import InterviewSession

    out = {"session_id": 0, "question": None, "error": ""}
    try:
        if mode not in MODES:
            out["error"] = f"unknown mode {mode!r} (use: {', '.join(MODES)})"
            return out
        job = db.get(Job, job_id)
        if job is None:
            out["error"] = f"job {job_id} not found"
            return out

        if mode == "contact" and contact_id is None:
            matched = matched_contact(job)
            if not matched:
                out["error"] = ("no contact matches this company - "
                                "contact mode needs someone you actually know")
                return out
            contact_id = matched["id"]

        if not _pool(load_bank(job), mode):
            out["error"] = (f"no questions in the {mode} bank - "
                            "run `jobbot interview prep` for this job first")
            return out

        sess = InterviewSession(job_id=job_id, mode=mode, contact_id=contact_id)
        db.add(sess)
        db.commit()
        db.refresh(sess)
        out["session_id"] = sess.id
        turn = _draw(db, sess.id)
        out["question"] = ask_dict(turn) if turn is not None else None
        return out
    except Exception as e:  # noqa: BLE001
        log.warning("start(job=%s, mode=%s) failed (%s)", job_id, mode, e)
        out["error"] = f"could not start practice: {e}"
        return out


def next_question(db, session_id: int) -> Optional[dict]:
    """Draw the next unasked question, or None when the bank is exhausted."""
    try:
        turn = _draw(db, session_id)
        return ask_dict(turn) if turn is not None else None
    except Exception as e:  # noqa: BLE001
        log.warning("next_question(%s) failed (%s)", session_id, e)
        return None


def rubric_score(rubric: dict) -> float:
    """Mean of the 4 rubric axes, rounded to the nearest 0.5, clamped to 0-5."""
    vals = []
    for axis in RUBRIC_AXES:
        try:
            vals.append(max(0.0, min(5.0, float(rubric.get(axis, 0.0)))))
        except (TypeError, ValueError):
            vals.append(0.0)
    return round((sum(vals) / len(vals)) * 2.0) / 2.0


def submit_answer(db, turn_id: int, text: str) -> dict:
    """Score an answer against the rubric and, on a weak answer with a named gap,
    emit at most `settings.interview_followup_max` follow-ups per base question.

    Fail-open: an LLM error still saves the answer and returns a neutral
    (score=None) critique so the session can continue.
    """
    from .models import InterviewSession, InterviewTurn
    from .pipeline import load_resume

    out = {"turn_id": turn_id, "score": None, "rubric": {}, "critique": "",
           "follow_up": None, "error": ""}
    try:
        text = (text or "").strip()
        if not text:
            out["error"] = "an answer is required"
            return out
        turn = db.get(InterviewTurn, turn_id)
        if turn is None:
            out["error"] = f"turn {turn_id} not found"
            return out
        sess = db.get(InterviewSession, turn.session_id)
        job = db.get(Job, sess.job_id) if sess is not None else None
        if job is None:
            out["error"] = "session's job no longer exists"
            return out

        turn.answer = text
        db.commit()

        try:
            resume = load_resume(settings.base_resume_path)
        except Exception:  # noqa: BLE001
            resume = ""

        try:
            raw = gc.score_answer(turn.question, turn.question_kind, text,
                                  job.title, job.company, resume)
        except Exception as e:  # noqa: BLE001
            log.warning("submit_answer(%s): scoring failed (%s)", turn_id, e)
            turn.critique = ("This answer could not be scored right now (the "
                             "model was unavailable). It is saved - keep going.")
            db.commit()
            out["critique"] = turn.critique
            return out

        rubric = {a: max(0.0, min(5.0, float(raw.get(a, 3.0) or 0.0)))
                  for a in RUBRIC_AXES}
        score = rubric_score(rubric)
        turn.rubric_json = json.dumps(rubric)
        turn.score = score
        turn.critique = str(raw.get("critique") or "").strip()
        db.commit()

        out.update(score=score, rubric=rubric, critique=turn.critique)

        follow_up_q = str(raw.get("follow_up") or "").strip()
        if (follow_up_q and score <= FOLLOWUP_SCORE_THRESHOLD
                and _followups_left(db, turn)):
            base_idx = turn.parent_idx if turn.parent_idx is not None else turn.idx
            nxt = (db.query(InterviewTurn)
                   .filter(InterviewTurn.session_id == turn.session_id).all())
            fu = InterviewTurn(session_id=turn.session_id,
                               idx=max(t.idx for t in nxt) + 1,
                               parent_idx=base_idx, question=follow_up_q,
                               question_kind=turn.question_kind,
                               grounded_in="follow_up")
            db.add(fu)
            db.commit()
            db.refresh(fu)
            out["follow_up"] = ask_dict(fu)
        return out
    except Exception as e:  # noqa: BLE001
        log.warning("submit_answer(%s) failed (%s)", turn_id, e)
        out["error"] = f"could not record that answer: {e}"
        return out


def _followups_left(db, turn) -> bool:
    """True while this base question is under the follow-up cap."""
    from .models import InterviewTurn

    base_idx = turn.parent_idx if turn.parent_idx is not None else turn.idx
    used = (db.query(InterviewTurn)
            .filter(InterviewTurn.session_id == turn.session_id,
                    InterviewTurn.parent_idx == base_idx).count())
    return used < max(0, settings.interview_followup_max)


def transcript(db, session_id: int) -> list[dict]:
    """Ordered read model of a session. Shared by the chat UI, CLI and history."""
    from .models import InterviewTurn

    try:
        turns = (db.query(InterviewTurn)
                 .filter(InterviewTurn.session_id == session_id)
                 .order_by(InterviewTurn.idx).all())
    except Exception as e:  # noqa: BLE001
        log.warning("transcript(%s) failed (%s)", session_id, e)
        return []
    rows = []
    for t in turns:
        try:
            rubric = json.loads(t.rubric_json or "{}")
        except Exception:  # noqa: BLE001
            rubric = {}
        rows.append({**ask_dict(t), "answer": t.answer, "critique": t.critique,
                     "score": t.score, "rubric": rubric})
    return rows


def _transcript_text(rows: list[dict]) -> str:
    out = []
    for r in rows:
        if not r["answer"]:
            continue
        out.append(f"Q ({r['question_kind']}): {r['question']}\n"
                   f"A: {r['answer']}\n"
                   f"Score: {r['score']}/5")
    return "\n\n".join(out)


def end(db, session_id: int) -> dict:
    """Close a session: average the scored turns and ask for a summary.

    Idempotent -- re-ending an already-closed session recomputes nothing and
    leaves ended_at alone. Fail-open: a dead summarizer still closes and scores
    the session, it just leaves summary empty.
    """
    from .models import InterviewSession

    out = {"session_id": session_id, "overall_score": None, "summary": "",
           "turns": 0, "error": ""}
    try:
        sess = db.get(InterviewSession, session_id)
        if sess is None:
            out["error"] = f"session {session_id} not found"
            return out
        rows = transcript(db, session_id)
        scored = [r["score"] for r in rows if r["score"] is not None and r["answer"]]
        out["turns"] = len(scored)

        if sess.ended_at is not None:      # already closed - report, do not redo
            out["overall_score"] = sess.overall_score
            out["summary"] = sess.summary
            return out

        overall = (round((sum(scored) / len(scored)) * 2.0) / 2.0) if scored else None
        summary = ""
        if scored:
            job = db.get(Job, sess.job_id)
            try:
                summary = (gc.interview_session_summary(
                    job.title if job else "", job.company if job else "",
                    _transcript_text(rows)) or "").strip()
            except Exception as e:  # noqa: BLE001
                log.warning("end(%s): summary failed (%s)", session_id, e)

        sess.ended_at = datetime.utcnow()
        sess.overall_score = overall
        sess.summary = summary
        db.commit()
        out.update(overall_score=overall, summary=summary)
        return out
    except Exception as e:  # noqa: BLE001
        log.warning("end(%s) failed (%s)", session_id, e)
        out["error"] = f"could not close the session: {e}"
        return out


def history(job_id: Optional[int] = None) -> dict:
    """Read model for the history dashboard and `jobbot interview history`.

    Open sessions are listed but excluded from every statistic -- an unfinished
    session has no score to average. Fail-open -> the empty/neutral shape.
    """
    from .models import InterviewSession, InterviewTurn

    empty = {"sessions": [], "by_mode": {}, "trend": [], "mean": None}
    try:
        init_db()
        with session() as db:
            q = db.query(InterviewSession)
            if job_id is not None:
                q = q.filter(InterviewSession.job_id == job_id)
            rows = q.order_by(InterviewSession.created_at.desc(),
                              InterviewSession.id.desc()).all()
            if not rows:
                return empty
            sessions = []
            for s in rows:
                job = db.get(Job, s.job_id)
                answered = (db.query(InterviewTurn)
                            .filter(InterviewTurn.session_id == s.id,
                                    InterviewTurn.answer != "").count())
                sessions.append({
                    "session_id": s.id, "job_id": s.job_id,
                    "title": job.title if job else "",
                    "company": job.company if job else "",
                    "mode": s.mode, "created_at": s.created_at,
                    "ended_at": s.ended_at, "overall_score": s.overall_score,
                    "turns": answered})
    except Exception as e:  # noqa: BLE001
        log.warning("history(%s) failed (%s)", job_id, e)
        return empty

    scored = [s for s in sessions if s["ended_at"] and s["overall_score"] is not None]
    by_mode: dict[str, float] = {}
    for mode in {s["mode"] for s in scored}:
        vals = [s["overall_score"] for s in scored if s["mode"] == mode]
        by_mode[mode] = round(sum(vals) / len(vals), 2)
    trend = [s["overall_score"] for s in reversed(scored)]   # oldest -> newest
    mean = round(sum(trend) / len(trend), 2) if trend else None
    return {"sessions": sessions, "by_mode": by_mode, "trend": trend, "mean": mean}


# A 2.5/5 session is "adequate" -- the neutral point. Below it earns nothing
# (practice is never a penalty); above it scales linearly to the cap.
READINESS_NEUTRAL = 2.5
# Sessions needed for full confidence in the signal. One good session is
# encouraging; three is evidence.
READINESS_FULL_CONFIDENCE = 3


def readiness_bonus(job_id: int) -> tuple[float, str]:
    """Bounded ranking bonus earned by practicing for this job.

    bonus = cap * quality * confidence, where
      quality    = (best ended session score - 2.5) / 2.5, clamped to 0..1
      confidence = min(1, ended session count / 3)

    Zero sessions -> (0.0, ""). Never negative, never above
    settings.interview_practice_weight. Monotonic in the best score.
    """
    from .models import InterviewSession

    try:
        with session() as db:
            rows = (db.query(InterviewSession)
                    .filter(InterviewSession.job_id == job_id,
                            InterviewSession.ended_at.isnot(None),
                            InterviewSession.overall_score.isnot(None)).all())
            scores = [s.overall_score for s in rows]
    except Exception as e:  # noqa: BLE001
        log.warning("readiness_bonus(%s) failed (%s)", job_id, e)
        return 0.0, ""
    if not scores:
        return 0.0, ""
    cap = max(0.0, float(settings.interview_practice_weight))
    quality = max(0.0, min(1.0, (max(scores) - READINESS_NEUTRAL) / READINESS_NEUTRAL))
    confidence = min(1.0, len(scores) / READINESS_FULL_CONFIDENCE)
    bonus = cap * quality * confidence
    if bonus <= 0.0:
        return 0.0, ""
    why = (f"+{bonus:.3f}: mock-interview practice "
           f"(best {max(scores)}/5 over {len(scores)} session(s))")
    return bonus, why
