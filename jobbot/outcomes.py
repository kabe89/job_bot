# jobbot/outcomes.py
"""Application-outcome funnel tracking + Bayesian-smoothed ranking priors.

Records each application's funnel progress (applied -> responded -> screen ->
interview -> offer, or terminal rejected/ghosted/withdrawn) and turns the
history into a small, explainable, sparse-data-safe nudge on job ranking.
Fail-open: any error in the prior math yields a neutral (0.0) adjustment.
"""
from __future__ import annotations

import logging
import math
from datetime import datetime, timedelta
from typing import Dict, List, Optional, Tuple

from .config import settings
from .models import Application, Job, OutcomeEvent, session

log = logging.getLogger("jobbot.outcomes")

STAGES = ["applied", "responded", "screen", "interview", "offer"]
TERMINAL = {"rejected", "ghosted", "withdrawn"}
# "sent" is the value mark_applied writes to Application.status; "applied" is the
# funnel's name for the same stage. Both rank 1 so record_outcome never demotes
# or overwrites a sent application (followup.py and predict.py filter on "sent").
_RANK = {"draft": 0, "applied": 1, "sent": 1, "responded": 2, "screen": 3,
         "interview": 4, "offer": 5}


def stage_rank(stage: str) -> int:
    """Ordinal of a positive funnel stage; -1 for terminal/unknown stages."""
    return _RANK.get(stage, -1)


def record_outcome(application_id: int, stage: str, note: str = "") -> None:
    """Append an OutcomeEvent and advance Application.status to the furthest
    positive stage. Terminal stages are sticky; status never regresses.
    Unknown stage -> ValueError."""
    if stage not in _RANK and stage not in TERMINAL:
        raise ValueError(f"unknown outcome stage: {stage!r}")
    with session() as db:
        app = db.get(Application, application_id)
        if app is None:
            raise ValueError(f"no application {application_id}")
        db.add(OutcomeEvent(application_id=application_id, stage=stage, note=note))
        if app.status not in TERMINAL:
            if stage in TERMINAL:
                app.status = stage
            elif stage_rank(stage) > stage_rank(app.status):
                app.status = stage
        # Interview-coach auto-trigger: flag ONLY. Generation happens in a
        # background worker / lazily on the /interview page -- keeping this
        # function pure means advancing a stage never blocks on Ollama.
        if stage == "interview":
            job = db.get(Job, app.job_id)
            if job is not None and not (job.interview_question_bank or ""):
                job.interview_prep_pending = True
        db.commit()


def sweep_ghosted(now: Optional[datetime] = None) -> int:
    """Mark 'applied'/'sent' applications older than settings.ghost_after_days
    with no response-or-later event as 'ghosted'. Idempotent. Returns count."""
    now = now or datetime.utcnow()
    cutoff = now - timedelta(days=settings.ghost_after_days)
    marked = 0
    with session() as db:
        apps = (db.query(Application)
                .filter(Application.status.in_(("applied", "sent")))
                .filter(Application.sent_at.isnot(None))
                .filter(Application.sent_at < cutoff).all())
        ids = []
        for a in apps:
            responded = any(stage_rank(e.stage) >= _RANK["responded"]
                            for e in a.outcome_events)
            if not responded:
                ids.append(a.id)
    for aid in ids:
        record_outcome(aid, "ghosted", note="auto: no response within window")
        marked += 1
    return marked


# ---- feature extraction -----------------------------------------------------

_ARCHETYPE_KW = [
    ("computational", ("computational", "cheminform", "bioinformatic", "in silico",
                       "modeling", "molecular dynamics", "machine learning", "ml ")),
    ("scientist", ("scientist", "biochem", "chemist", "biolog")),
    ("research", ("research", "postdoc", "fellow", "associate")),
    ("engineer", ("engineer", "developer", "software")),
]
_SENIORITY_KW = [
    ("senior", ("senior", "staff", "principal", "lead", "iii", "iv")),
    ("junior", ("junior", "assistant", "intern", "entry", " i ", " ii")),
]


def _source_key(source: str) -> str:
    return (source or "").split(":")[0].strip().lower() or "unknown"


def _archetype(title: str) -> str:
    t = f" {(title or '').lower()} "
    for name, kws in _ARCHETYPE_KW:
        if any(k in t for k in kws):
            return name
    return "other"


def _seniority(title: str) -> str:
    t = f" {(title or '').lower()} "
    for name, kws in _SENIORITY_KW:
        if any(k in t for k in kws):
            return name
    return "mid"


def _norm_company(name: str) -> str:
    return " ".join((name or "").lower().replace(",", " ").split()) or "unknown"


def job_features(job) -> Dict[str, str]:
    """The coarse feature values a job contributes to / is scored against."""
    return {
        "source": _source_key(getattr(job, "source", "")),
        "remote": "remote" if getattr(job, "is_remote", False) else "onsite",
        "archetype": _archetype(getattr(job, "title", "")),
        "seniority": _seniority(getattr(job, "title", "")),
        "company": _norm_company(getattr(job, "company", "")),
    }


# ---- priors -----------------------------------------------------------------

def _reached_rank(app) -> int:
    """Furthest positive stage an application reached (via status + events)."""
    r = stage_rank(getattr(app, "status", "applied"))
    for e in getattr(app, "outcome_events", []) or []:
        r = max(r, stage_rank(e.stage))
    return r


def _smoothed_rate(successes: int, trials: int, base: float) -> float:
    s = settings.outcome_prior_strength
    return (successes + s * base) / (trials + s) if (trials + s) > 0 else base


def compute_priors(apps=None) -> dict:
    """Aggregate response/interview counts per coarse feature value. When `apps`
    is None, read all non-draft applications from the DB. Always recomputed
    (cheap for a single-user SQLite app) so the live process never serves stale
    priors — Applications are created/mutated in several modules (pipeline,
    apply flow) that don't know about this feature, so a cache here would go
    stale and silently undercount the learning loop."""
    if apps is None:
        with session() as db:
            from sqlalchemy.orm import joinedload
            apps = (db.query(Application)
                    .options(joinedload(Application.job),
                             joinedload(Application.outcome_events))
                    .filter(Application.status != "draft").all())
            apps = [_snapshot(a) for a in apps]  # detach before session closes
    dims = {d: {} for d in ("source", "remote", "archetype", "seniority", "company")}
    tot = resp = intv = 0
    for a in apps:
        job = getattr(a, "job", None)
        if job is None:
            continue
        reached = _reached_rank(a)
        r_ok = 1 if reached >= _RANK["responded"] else 0
        i_ok = 1 if reached >= _RANK["interview"] else 0
        tot += 1; resp += r_ok; intv += i_ok
        for dim, val in job_features(job).items():
            cell = dims[dim].setdefault(val, {"trials": 0, "responses": 0, "interviews": 0})
            cell["trials"] += 1; cell["responses"] += r_ok; cell["interviews"] += i_ok
    return {
        "base_response": (resp / tot) if tot else 0.15,
        "base_interview": (intv / tot) if tot else 0.05,
        "dims": dims,
    }


class _Snap:
    __slots__ = ("status", "job", "outcome_events")


def _snapshot(app):
    """Detach the fields compute_priors needs so it survives session close."""
    s = _Snap()
    s.status = app.status
    s.job = app.job
    s.outcome_events = list(app.outcome_events)
    return s


# ---- adjustment / blend -----------------------------------------------------

def _logit(p: float) -> float:
    p = min(max(p, 1e-3), 1 - 1e-3)
    return math.log(p / (1 - p))


def adjustment_for_job(job, priors: dict) -> Tuple[float, str]:
    """Bounded, explainable ranking nudge for `job` given `priors`. Fail-open:
    any malformed prior / error -> (0.0, "")."""
    try:
        base_r = priors["base_response"]
        base_i = priors["base_interview"]
        dims = priors["dims"]
        cap = settings.outcome_adjustment_cap
        iw = settings.outcome_interview_weight
        contribs: List[Tuple[float, str]] = []
        for dim, val in job_features(job).items():
            cell = dims.get(dim, {}).get(val)
            if not cell or cell["trials"] < settings.outcome_min_trials:
                continue
            rr = _smoothed_rate(cell["responses"], cell["trials"], base_r)
            ri = _smoothed_rate(cell["interviews"], cell["trials"], base_i)
            c = (_logit(rr) - _logit(base_r)) + iw * (_logit(ri) - _logit(base_i))
            if abs(c) > 1e-6:
                contribs.append((c, f"{dim}={val}"))
        total = sum(c for c, _ in contribs)
        adj = cap * math.tanh(0.5 * total)
        contribs.sort(key=lambda x: abs(x[0]), reverse=True)
        top = " & ".join(name for _, name in contribs[:2])
        if not top:
            return 0.0, ""
        sign = "+" if adj >= 0 else ""
        why = f"{sign}{adj:.3f}: {top} vs your baseline"
        return adj, why
    except Exception as e:  # noqa: BLE001 — fail-open
        log.warning("outcome adjustment failed (%s) — neutral.", e)
        return 0.0, ""


def blended(job, priors: dict) -> float:
    base = getattr(job, "rerank_score", None)
    if base is None:
        base = getattr(job, "match_score", 0.0) or 0.0
    adj, _ = adjustment_for_job(job, priors)
    return min(1.0, max(0.0, base + adj))


def insights(apps=None) -> dict:
    """Per-dimension learned rates for the Insights panel."""
    p = compute_priors(apps=apps)
    base = p["base_response"]
    out = {}
    for dim, values in p["dims"].items():
        rows = []
        for val, cell in values.items():
            rows.append({
                "value": val, "trials": cell["trials"],
                "responses": cell["responses"], "interviews": cell["interviews"],
                "response_rate": round(_smoothed_rate(cell["responses"], cell["trials"], base), 3),
            })
        rows.sort(key=lambda r: r["trials"], reverse=True)
        out[dim] = rows
    return out
