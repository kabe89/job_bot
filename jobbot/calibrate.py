# jobbot/calibrate.py
"""Threshold calibration. After a real scrape has stored job embeddings, sample
the cosine-vs-profile distribution and recommend `semantic_recall_threshold`
(keep ~the top 75%) and `min_match_score` (a stricter cut for downstream
consumers). Reuses stored `Job.embedding`; only embeds rows that lack one.
"""
from __future__ import annotations

import json
import logging
from typing import List

import numpy as np

from . import embeddings
from .models import Job, session

log = logging.getLogger("jobbot.calibrate")


def cosine_distribution(profile, limit: int = 500) -> dict:
    prof_emb = profile.embedding
    sims: List[float] = []
    with session() as db:
        jobs = db.query(Job).limit(limit).all()
        for j in jobs:
            emb = None
            if j.embedding:
                try:
                    emb = json.loads(j.embedding)
                except Exception:  # noqa: BLE001
                    emb = None
            if emb is None:
                try:
                    emb = embeddings.embed(f"{j.title} {j.description}")
                except Exception:  # noqa: BLE001
                    continue
            if prof_emb:
                sims.append(embeddings.cosine(emb, prof_emb))
    if not sims:
        return {"n": 0, "p10": 0.0, "p25": 0.0, "p50": 0.0, "p75": 0.0, "p90": 0.0}
    arr = np.asarray(sims, dtype=float)
    return {
        "n": int(arr.size),
        "p10": round(float(np.percentile(arr, 10)), 4),
        "p25": round(float(np.percentile(arr, 25)), 4),
        "p50": round(float(np.percentile(arr, 50)), 4),
        "p75": round(float(np.percentile(arr, 75)), 4),
        "p90": round(float(np.percentile(arr, 90)), 4),
    }


def recommend_thresholds(dist: dict) -> dict:
    """Recall threshold ~= p25 (keep roughly the top 75%); min_match_score ~= p50
    (stricter cut for downstream consumers). Guaranteed min_match >= recall."""
    recall = float(dist.get("p25", 0.0))
    strict = max(float(dist.get("p50", 0.0)), recall)
    return {
        "semantic_recall_threshold": round(recall, 3),
        "min_match_score": round(strict, 3),
    }
