# jobbot/ranking.py
"""Two-stage profile-driven ranking.

Stage 1 (`semantic_score`): embed a job and cosine it against the profile
embedding — runs on every scraped job in the pipeline. Stage 2 (`rerank_topk`,
added in Task 6): an Ollama judge re-ranks only the top-K for precision.
"""
from __future__ import annotations

import json
import logging
from dataclasses import dataclass
from typing import List, Tuple

from . import embeddings
from . import gemini_client as _g
from . import ollama_client as oc

log = logging.getLogger("jobbot.ranking")


def semantic_score(job_text: str, profile_embedding: List[float]) -> Tuple[float, List[float]]:
    """Return (score 0..1, job_embedding). Raises if embedding fails so the
    caller can fall back to the legacy bag-of-words score."""
    job_emb = embeddings.embed(job_text)
    raw = embeddings.cosine(job_emb, profile_embedding)
    score = max(0.0, min(1.0, raw))
    return round(score, 4), job_emb


_RERANK_PROMPT = """You decide whether this candidate should SPEND AN APPLICATION on this job.
You are not measuring whether the candidate can do the job. Almost any job is a
"match" by that test, which is useless. Score DESIRABILITY AND FIT AS A CAREER MOVE.
Return STRICT JSON only: {{"score": <float 0-1>, "rationale": "one concise sentence"}}.

HOW TO SCORE (apply every rule):

1. SENIORITY IS TWO-SIDED. Being OVERQUALIFIED IS A PENALTY, EXACTLY LIKE being
   underqualified. If the candidate's experience EXCEEDS the role, the score goes
   DOWN, not up. A PhD-level scientist applying to a technician, QC sample-handling,
   lab-assistant, intern, or entry-level role is a BAD fit -- score it 0.0-0.15 even
   if the candidate covers every listed requirement. "Their background exceeds this
   role" is a reason to score LOW. Never a reason to score high.

2. VOCABULARY OVERLAP IS NOT METHOD OVERLAP. Shared keywords do not mean shared
   science. Two fields can share every noun and share no technique. The candidate is
   already scored by keyword similarity elsewhere; your job is to catch what keyword
   matching gets WRONG. Ask whether the candidate has actually DONE this work, not
   whether they recognize the words.

3. DEALBREAKERS ARE A HARD GATE. If the job hits any listed dealbreaker, cap the
   score at 0.1 regardless of every other strength.

4. SCORE ANCHORS -- calibrate to these, do not cluster:
     0.9-1.0  squarely the candidate's field AND right seniority. Rare. Apply today.
     0.7-0.85 real fit, one clear gap (adjacent subfield, or a stretch in level).
     0.4-0.6  plausible reach: transferable skills, but they have not done this work.
     0.2-0.35 weak: shared vocabulary only, or a noticeable level mismatch.
     0.0-0.15 wrong field, wrong level, overqualified, or a dealbreaker hit.
   Most jobs are NOT above 0.5. If you are scoring most jobs high, you are wrong.

5. JUDGE THE DESCRIPTION YOU ARE GIVEN. If the description is missing, truncated, or
   is only scraper metadata (a few hundred characters, no responsibilities or
   qualifications), you CANNOT assess fit -- score 0.3 and say the description was
   insufficient. Do not infer fit from the job title alone.

CANDIDATE PROFILE:
titles: {titles}
skills: {skills}
domains: {domains}
dealbreakers: {dealbreakers}
summary: {summary}

JOB:
title: {job_title}
company: {company}
description: {description}"""


@dataclass
class RerankResult:
    job_id: int
    score: float
    rationale: str


def _judge_one(profile, job) -> "RerankResult | None":
    try:
        out = oc._generate(
            _RERANK_PROMPT.format(
                titles=", ".join(profile.role_titles),
                skills=", ".join(profile.skills[:20]),
                domains=", ".join(profile.domains),
                dealbreakers=", ".join(profile.dealbreakers),
                summary=profile.summary,
                job_title=job.title, company=job.company,
                description=(job.description or "")[:5000]),
            temperature=0.2, include_profile=False)
        data = _g._parse_json_block(out) or {}
        if "score" not in data:
            # No parseable score → fall back to the stage-1 score (drop the row).
            return None
        score = float(data.get("score", 0.0) or 0.0)
        score = max(0.0, min(1.0, score))
        # Keep a valid numeric score even if the rationale came back empty —
        # don't discard the re-rank ordering over a missing sentence (review m3).
        rationale = str(data.get("rationale") or "").strip() or "(no rationale)"
        return RerankResult(job_id=job.id, score=round(score, 4), rationale=rationale)
    except Exception as e:  # noqa: BLE001
        log.warning("re-rank failed for job %s (%s) — keeping stage-1 score.",
                    getattr(job, "id", "?"), e)
        return None


def rerank_topk(profile, jobs: List, k: int) -> List[RerankResult]:
    """Judge the top-`k` jobs by match_score. Per-job failures are skipped."""
    ordered = sorted(jobs, key=lambda j: (j.match_score or 0.0), reverse=True)[:max(0, k)]
    results: List[RerankResult] = []
    for job in ordered:
        r = _judge_one(profile, job)
        if r is not None:
            results.append(r)
    return results


def rescore_jobs(profile, *, force_reembed: bool = False, on_progress=None) -> dict:
    """Recompute match_score for every active (non-expired) job against the CURRENT
    profile embedding.

    match_score is otherwise frozen at scrape time, so jobs scraped before semantic
    ranking went live keep a legacy bag-of-words score and no embedding — leaving
    genuinely relevant roles underrated versus newer, semantically-scored jobs. This
    backfill embeds each job that lacks an embedding (persisting it for cheap future
    re-scores) and rewrites match_score = cosine(job, profile), mirroring the
    pipeline's watchlist boost so the two paths agree.

    Fail-open per job: on an embed failure the prior score is kept. Returns
    {'total','rescored','embedded','failed'}.
    """
    from .models import Job, session
    from .matcher import company_in_watchlist
    from .watchlist import load_companies

    stats = {"total": 0, "rescored": 0, "embedded": 0, "failed": 0}
    prof_emb = getattr(profile, "embedding", None)
    if not prof_emb:
        log.warning("rescore: profile has no embedding — nothing to score against.")
        return stats
    try:
        watch = load_companies()
    except Exception:  # noqa: BLE001
        watch = []

    with session() as db:
        jobs = db.query(Job).filter(Job.status != "expired").all()
        stats["total"] = len(jobs)
        for i, job in enumerate(jobs):
            try:
                emb = None
                if job.embedding and not force_reembed:
                    try:
                        emb = json.loads(job.embedding)
                    except Exception:  # noqa: BLE001
                        emb = None
                if emb is None:
                    emb = embeddings.embed(f"{job.title} {job.description}")
                    job.embedding = json.dumps(emb)
                    stats["embedded"] += 1
                s = max(0.0, min(1.0, embeddings.cosine(emb, prof_emb)))
                if company_in_watchlist(job.company, watch):
                    s = min(1.0, s + 0.15)
                job.match_score = round(s, 4)
                stats["rescored"] += 1
            except Exception as e:  # noqa: BLE001
                log.warning("rescore: job %s failed (%s) — keeping prior score.",
                            getattr(job, "id", "?"), e)
                stats["failed"] += 1
            if on_progress:
                try:
                    on_progress(i + 1, stats["total"])
                except Exception:  # noqa: BLE001
                    pass
        db.commit()
    return stats
