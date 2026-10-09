"""Success-rate prediction.

Estimates per-job callback probability and time-to-offer given:
  - job features (match score, watchlist hit, source, freshness, salary visibility)
  - application features (tailored? cover letter personalized? interview-prep done?)
  - applicant calibration (historical application outcomes)
  - timing relative to a designated start date

Model:
  logit(p) = w * features
  p = sigmoid(logit(p))
  callback_in_T_days ~= p * CDF(T; lambda=expected_response_time)

Industry calibration anchors (technical hiring benchmarks):
  - Cold ATS apply, no tailoring:           ~3% callback
  - Cold ATS + tailored resume + cover:     ~9-12% callback
  - Watchlist company + tailored:           ~15-20% callback
  - Strong match (>0.6) + watchlist + prep: ~25-35% callback
  - Referral / warm intro:                   ~35-50% (out of model scope)

These priors are conservative and re-calibrate as the user logs outcomes.
"""
from __future__ import annotations

import json
import logging
import math
import os
from dataclasses import dataclass
from datetime import datetime, timedelta
from typing import Optional

import numpy as np

from .config import settings
from .models import Application, Interview, Job, session

log = logging.getLogger("jobbot.predict")

# Base log-odds for a "vanilla cold application" -> ~3% callback => logit(0.03) ~= -3.48
BASE_LOGIT = -3.48

# Feature weights (additive log-odds contributions). Tunable.
WEIGHTS = {
    "match_score_per_0.1":   0.45,   # +0.45 logit per +0.1 score above 0.30
    "watchlist_company":     0.70,   # company is on user's curated list
    "tailored_resume":       0.55,   # resume was tailored for this job
    "cover_letter":          0.35,   # personalized cover letter present
    "interview_prep":        0.15,   # signals investment (small)
    "fresh_posting":         0.40,   # posted within 7 days
    "stale_posting":        -0.55,   # posted >30 days ago
    "salary_visible":        0.10,   # company shows comp band (transparency signal)
    "source_ats_direct":     0.20,   # Greenhouse/Lever (no recruiter middleman)
    "source_aggregator":    -0.10,   # RSS aggregator (noisier)
    "high_volume_company":  -0.15,   # 100+ open reqs -> harder to stand out
    "small_focused_company": 0.25,   # <25 reqs -> letter likely read
}

# Canonical feature order for the learned model (must be stable across saves/loads).
FEATURE_NAMES: list[str] = [
    "match_score_per_0.1",
    "watchlist_company",
    "tailored_resume",
    "cover_letter",
    "interview_prep",
    "fresh_posting",
    "stale_posting",
    "salary_visible",
    "source_ats_direct",
    "source_aggregator",
    "high_volume_company",
    "small_focused_company",
]

# Expected response distribution (typical industry hiring):
#   median ~14 days, 90th percentile ~45 days. Use Gamma-like CDF with k=2, theta=10.
MEDIAN_DAYS = 14.0
P95_DAYS = 45.0

# ---------------------------------------------------------------------------
# Module-level model cache: (mtime_or_None, model_dict_or_None)
# ---------------------------------------------------------------------------
_cached_model: tuple[float | None, dict | None] = (None, None)


def _sigmoid(x: float) -> float:
    return 1.0 / (1.0 + math.exp(-x))


def _gamma_like_cdf(t_days: float, median: float = MEDIAN_DAYS) -> float:
    """Approximate CDF of hiring response time. Returns P(response <= t)."""
    if t_days <= 0:
        return 0.0
    # Calibrate so median maps to 0.5 -- exponential with rate = ln(2)/median
    rate = math.log(2) / median
    return 1.0 - math.exp(-rate * t_days)


@dataclass
class Prediction:
    job_id: int
    callback_probability: float
    confidence: str                       # low | medium | high
    logit_contributions: dict
    expected_response_days: float          # E[days | response happens]
    p_by_day: dict[int, float]             # cumulative P(callback) at +7, +14, +30, +60
    interview_probability: float           # P(advance past phone screen | callback)
    offer_probability: float               # P(offer | interview)
    recommended_actions: list[str]
    days_from_start: Optional[int]


def _user_calibration() -> tuple[float, str]:
    """Adjust base rate using user's actual history. Returns (delta_logit, confidence)."""
    with session() as db:
        total = (db.query(Application)
                 .filter(Application.sent_at.isnot(None), Application.status != "draft")
                 .count())
        # "callback" proxy = job status moved past 'applied' to anything richer.
        # Currently we lack interview/offer status capture for sent apps without manual entry.
        # If user has logged interviews, we use them.
        interviews = db.query(Interview).count()
        if total < 10:
            return 0.0, "low"
        rate = max(min(interviews / total, 0.6), 0.005)
        # logit-shift toward observed
        observed_logit = math.log(rate / (1.0 - rate))
        # Pull base toward observed by 50% if total>=10, 80% if >=30
        weight = 0.5 if total < 30 else 0.8
        return weight * (observed_logit - BASE_LOGIT), ("high" if total >= 30 else "medium")


# ---------------------------------------------------------------------------
# Feature extraction
# ---------------------------------------------------------------------------

def _company_size_flags(company: str) -> tuple[bool, bool]:
    """Return (high_volume, small_focused) for a company by counting open jobs in DB."""
    with session() as db:
        same_co = db.query(Job).filter(Job.company == company).count()
    high_volume = same_co >= 25
    small_focused = same_co <= 5
    return high_volume, small_focused


def _features(job: Job, application: Optional[Application] = None) -> dict[str, float]:
    """Compute raw feature values (0/1 indicators, except match_score_per_0.1 which is continuous).

    Returns a dict keyed by FEATURE_NAMES.  The heuristic logit contribution
    for each feature is WEIGHTS[k] * raw[k], which is what predict_job uses today.
    """
    raw: dict[str, float] = {}

    # match_score band: continuous
    raw["match_score_per_0.1"] = max(0.0, (job.match_score - 0.30) / 0.10)

    # 0/1 indicators
    raw["watchlist_company"] = 1.0 if "watchlist" in (job.tags or "") else 0.0

    if application:
        raw["tailored_resume"] = 1.0 if application.tailored_resume_path else 0.0
        raw["cover_letter"] = 1.0 if application.cover_letter_path else 0.0
    else:
        raw["tailored_resume"] = 0.0
        raw["cover_letter"] = 0.0

    raw["interview_prep"] = 1.0 if job.interview_prep else 0.0

    if job.posted_at:
        age_days = (datetime.utcnow() - job.posted_at).days
        raw["fresh_posting"] = 1.0 if age_days <= 7 else 0.0
        raw["stale_posting"] = 1.0 if age_days > 30 else 0.0
    else:
        raw["fresh_posting"] = 0.0
        raw["stale_posting"] = 0.0

    raw["salary_visible"] = 1.0 if job.salary else 0.0

    src = (job.source or "").lower()
    raw["source_ats_direct"] = 1.0 if src.startswith(("greenhouse", "lever")) else 0.0
    raw["source_aggregator"] = 1.0 if (src.startswith("rss") or src in {"nystate", "suny", "nih"}) else 0.0

    high_volume, small_focused = _company_size_flags(job.company)
    raw["high_volume_company"] = 1.0 if high_volume else 0.0
    raw["small_focused_company"] = 1.0 if small_focused else 0.0

    return raw


# ---------------------------------------------------------------------------
# Learned model: load / save / train
# ---------------------------------------------------------------------------

def _load_model() -> dict | None:
    """Load trained callback model from disk, with mtime-based cache invalidation.

    Returns the model dict if valid, or None if file absent/invalid.
    """
    global _cached_model
    path = settings.callback_model_path
    try:
        mtime = os.path.getmtime(path)
    except OSError:
        _cached_model = (None, None)
        return None

    cached_mtime, cached_data = _cached_model
    if cached_mtime == mtime and cached_data is not None:
        return cached_data

    try:
        with open(path, "r", encoding="utf-8") as f:
            data = json.load(f)
        # Validate required keys
        required = {"feature_names", "weights", "bias", "n_samples"}
        if not required.issubset(data.keys()):
            log.warning("callback_model.json missing required keys; ignoring")
            _cached_model = (mtime, None)
            return None
        _cached_model = (mtime, data)
        return data
    except Exception as exc:
        log.warning("Failed to load callback model: %s", exc)
        _cached_model = (mtime, None)
        return None


def collect_training_data() -> tuple[list[dict], list[int], int, int]:
    """Collect labeled training samples from the database.

    Returns:
        X       -- list of raw-feature dicts (one per labeled application)
        y       -- parallel list of labels (1=callback, 0=no-callback)
        n_pos   -- count of positive labels
        n_neg   -- count of negative labels
    """
    cutoff_days = 45
    now = datetime.utcnow()
    X: list[dict] = []
    y: list[int] = []

    with session() as db:
        apps = (
            db.query(Application)
            # "Was ever sent", not "is currently sent": record_outcome advances
            # status to responded/screen/interview and sweep_ghosted writes
            # ghosted, and all of those are still real, labelable applications.
            .filter(Application.sent_at.isnot(None), Application.status != "draft")
            .all()
        )
        for app in apps:
            job = app.job
            if job is None:
                continue

            # Determine label
            has_interview = len(app.interviews) > 0
            positive_status = job.status in ("interview", "offer", "callback")

            label: int | None = None
            if has_interview or positive_status:
                label = 1
            elif job.status == "rejected":
                label = 0
            elif app.sent_at and (now - app.sent_at).days > cutoff_days:
                # Old enough with no signal -> negative
                label = 0
            # else: too recent, no signal yet -> skip

            if label is None:
                continue

            feats = _features(job, app)
            X.append(feats)
            y.append(label)

    n_pos = sum(y)
    n_neg = len(y) - n_pos
    return X, y, n_pos, n_neg


def _logistic_fit(
    X_arr: np.ndarray,
    y_arr: np.ndarray,
    lr: float = 0.1,
    n_iter: int = 2000,
    l2_lambda: float = 0.01,
) -> tuple[np.ndarray, float]:
    """Fit logistic regression via gradient descent (numpy only).

    Args:
        X_arr     -- (N, F) feature matrix
        y_arr     -- (N,) binary labels
        lr        -- learning rate
        n_iter    -- gradient descent iterations
        l2_lambda -- L2 regularisation coefficient

    Returns:
        weights   -- (F,) weight vector
        bias      -- scalar bias term
    """
    n, f = X_arr.shape
    weights = np.zeros(f, dtype=np.float64)
    bias = 0.0

    for _ in range(n_iter):
        logits = X_arr @ weights + bias
        # Numerically stable sigmoid
        preds = 1.0 / (1.0 + np.exp(-np.clip(logits, -500, 500)))
        error = preds - y_arr  # (N,)
        grad_w = (X_arr.T @ error) / n + l2_lambda * weights
        grad_b = error.mean()
        weights -= lr * grad_w
        bias -= lr * grad_b

    return weights, float(bias)


def train_callback_model() -> dict:
    """Train a logistic regression model on real DB outcomes and save it.

    Returns a status dict with keys: trained, reason (if not trained), n,
    train_accuracy (if trained).
    """
    X, y, n_pos, n_neg = collect_training_data()
    n_total = len(y)

    if n_total < settings.callback_model_min_outcomes:
        return {
            "trained": False,
            "reason": f"insufficient labeled data: {n_total} samples (need {settings.callback_model_min_outcomes})",
            "n": n_total,
            "n_pos": n_pos,
            "n_neg": n_neg,
        }
    if n_pos < 3:
        return {
            "trained": False,
            "reason": f"too few positive samples: {n_pos} (need >= 3)",
            "n": n_total,
            "n_pos": n_pos,
            "n_neg": n_neg,
        }
    if n_neg < 3:
        return {
            "trained": False,
            "reason": f"too few negative samples: {n_neg} (need >= 3)",
            "n": n_total,
            "n_pos": n_pos,
            "n_neg": n_neg,
        }

    # Build feature matrix in canonical order
    X_arr = np.array(
        [[row.get(k, 0.0) for k in FEATURE_NAMES] for row in X],
        dtype=np.float64,
    )
    y_arr = np.array(y, dtype=np.float64)

    weights, bias = _logistic_fit(X_arr, y_arr)

    # Compute training accuracy
    logits = X_arr @ weights + bias
    preds_bin = (1.0 / (1.0 + np.exp(-np.clip(logits, -500, 500)))) >= 0.5
    train_accuracy = float((preds_bin == y_arr.astype(bool)).mean())

    model = {
        "feature_names": FEATURE_NAMES,
        "weights": weights.tolist(),
        "bias": bias,
        "n_samples": n_total,
        "n_pos": n_pos,
        "n_neg": n_neg,
        "trained_at": datetime.utcnow().isoformat(),
        "train_accuracy": round(train_accuracy, 4),
    }

    path = settings.callback_model_path
    os.makedirs(os.path.dirname(os.path.abspath(path)), exist_ok=True)
    with open(path, "w", encoding="utf-8") as f:
        json.dump(model, f, indent=2)
    log.info("Saved callback model to %s (n=%d, acc=%.3f)", path, n_total, train_accuracy)

    # Invalidate module cache so next predict_job picks up new model
    global _cached_model
    _cached_model = (None, None)

    return {
        "trained": True,
        "n": n_total,
        "n_pos": n_pos,
        "n_neg": n_neg,
        "train_accuracy": round(train_accuracy, 4),
    }


def model_status() -> dict:
    """Return status dict about the trained callback model and available data."""
    path = settings.callback_model_path
    model = _load_model()
    _, _, n_pos, n_neg = collect_training_data()
    n_labeled = n_pos + n_neg

    if model:
        return {
            "model_exists": True,
            "n_samples": model.get("n_samples"),
            "n_pos": model.get("n_pos"),
            "n_neg": model.get("n_neg"),
            "trained_at": model.get("trained_at"),
            "train_accuracy": model.get("train_accuracy"),
            "feature_names": model.get("feature_names"),
            "path": path,
            "current_labeled_data": n_labeled,
            "current_n_pos": n_pos,
            "current_n_neg": n_neg,
        }
    else:
        return {
            "model_exists": False,
            "path": path,
            "current_labeled_data": n_labeled,
            "current_n_pos": n_pos,
            "current_n_neg": n_neg,
            "min_outcomes_needed": settings.callback_model_min_outcomes,
        }


# ---------------------------------------------------------------------------
# Main prediction entry point
# ---------------------------------------------------------------------------

def predict_job(job: Job, application: Optional[Application] = None,
                start_date: Optional[datetime] = None) -> Prediction:
    """Predict outcome for a single job (+ optional application)."""

    # --- Compute raw features and heuristic contributions ---
    raw = _features(job, application)

    contribs: dict[str, float] = {}
    for k in FEATURE_NAMES:
        v = raw.get(k, 0.0)
        if v != 0.0:
            contribs[k] = WEIGHTS[k] * v

    user_delta, confidence = _user_calibration()
    base = BASE_LOGIT + user_delta
    logit = base + sum(contribs.values())
    p_heuristic = _sigmoid(logit)
    p_heuristic = max(0.005, min(0.85, p_heuristic))

    # --- Blend with learned model if available ---
    trained_model = _load_model()
    if trained_model is not None:
        try:
            feat_names = trained_model["feature_names"]
            w = np.array(trained_model["weights"], dtype=np.float64)
            b = float(trained_model["bias"])
            x_vec = np.array([raw.get(fn, 0.0) for fn in feat_names], dtype=np.float64)
            logit_model = float(x_vec @ w) + b
            p_model = float(1.0 / (1.0 + math.exp(-max(-500, min(500, logit_model)))))
            p_model = max(0.005, min(0.85, p_model))

            p_callback = 0.35 * p_heuristic + 0.65 * p_model

            n_samples = trained_model.get("n_samples", 0)
            confidence = "high" if n_samples >= 40 else "medium"

            contribs["learned_model"] = round(p_model, 4)
        except Exception as exc:
            log.warning("Learned model prediction failed, falling back to heuristic: %s", exc)
            p_callback = p_heuristic
    else:
        p_callback = p_heuristic

    # --- Interview-coach readiness: bounded, never a penalty -----------------
    # Applied post-hoc rather than as a WEIGHTS/FEATURE_NAMES entry so the
    # learned model's saved feature schema stays stable and the user-set cap
    # means what it says: a max absolute lift on the probability.
    try:
        from .interview_coach import readiness_bonus
        practice_bonus, _why = readiness_bonus(job.id)
        if practice_bonus:
            p_callback += practice_bonus
            contribs["interview_practice"] = round(practice_bonus, 3)
    except Exception as exc:  # noqa: BLE001
        log.warning("readiness bonus skipped: %s", exc)

    p_callback = max(0.005, min(0.85, p_callback))

    # --- Funnel: callback -> interview -> offer ---
    p_interview_given_callback = 0.55
    p_offer_given_interview = 0.20 + 0.25 * job.match_score  # better fit -> better offer odds

    days_offsets = (7, 14, 30, 60)
    p_by_day = {d: round(p_callback * _gamma_like_cdf(d), 4) for d in days_offsets}

    # Apply start-date shift
    days_from_start = None
    if start_date and application and application.sent_at:
        days_from_start = (application.sent_at - start_date).days
    elif start_date:
        days_from_start = (datetime.utcnow() - start_date).days

    actions: list[str] = []
    if not contribs.get("tailored_resume"):
        actions.append("Run resume tailoring (+0.55 logit ~= +9% absolute callback)")
    if not contribs.get("cover_letter"):
        actions.append("Generate a personalized cover letter (+0.35 logit ~= +5% absolute callback)")
    if not contribs.get("interview_prep"):
        actions.append("Generate interview prep -- small lift but high upside if invited")
    if contribs.get("stale_posting"):
        actions.append("Posting is >30 days old -- apply only if you have a referral angle")
    if not contribs.get("watchlist_company"):
        actions.append("Not on your watchlist -- verify this company is a real target before investing time")
    if job.match_score < 0.45:
        actions.append("Match score is low -- consider whether to apply or skip")

    return Prediction(
        job_id=job.id,
        callback_probability=round(p_callback, 4),
        confidence=confidence,
        logit_contributions={k: round(v, 3) for k, v in contribs.items()},
        expected_response_days=MEDIAN_DAYS,
        p_by_day=p_by_day,
        interview_probability=round(p_callback * p_interview_given_callback, 4),
        offer_probability=round(p_callback * p_interview_given_callback * p_offer_given_interview, 4),
        recommended_actions=actions,
        days_from_start=days_from_start,
    )


# ---------------------------------------------------------------------------
# Portfolio forecast
# ---------------------------------------------------------------------------

def forecast(start_date: datetime, horizon_days: int = 90,
             apply_rate_per_week: int = 5) -> dict:
    """Project pipeline outcomes from a given start date.

    Aggregates predictions across all 'new' or 'tailored' jobs in the DB
    and a projected application throughput.
    """
    with session() as db:
        jobs = db.query(Job).filter(Job.status.in_(["new", "tailored"])).order_by(Job.match_score.desc()).all()

    horizon_end = start_date + timedelta(days=horizon_days)
    total_applies = max(0, int((horizon_days / 7.0) * apply_rate_per_week))

    # Take top-N by score as the realistic apply set
    top_n = jobs[:total_applies] if jobs else []
    preds = []
    for j in top_n:
        # Simulate a "fully-prepped" application: tailored + cover letter + prep
        fake_app = Application(tailored_resume_path="sim", cover_letter_path="sim")
        p = predict_job(j, application=fake_app, start_date=start_date)
        preds.append(p)

    sum_callback = sum(p.callback_probability for p in preds)
    sum_interview = sum(p.interview_probability for p in preds)
    sum_offer = sum(p.offer_probability for p in preds)

    # P(at least one offer) using independence approximation
    if preds:
        p_no_offer = 1.0
        for p in preds:
            p_no_offer *= (1.0 - p.offer_probability)
        p_at_least_one_offer = 1.0 - p_no_offer
    else:
        p_at_least_one_offer = 0.0

    return {
        "start_date": start_date.isoformat(),
        "horizon_days": horizon_days,
        "apply_rate_per_week": apply_rate_per_week,
        "total_applications_projected": len(preds),
        "expected_callbacks": round(sum_callback, 2),
        "expected_first_round_interviews": round(sum_interview, 2),
        "expected_offers": round(sum_offer, 2),
        "p_at_least_one_offer": round(p_at_least_one_offer, 4),
        "horizon_end": horizon_end.isoformat(),
        "top_predictions": [
            {
                "job_id": p.job_id,
                "callback": p.callback_probability,
                "interview": p.interview_probability,
                "offer": p.offer_probability,
            }
            for p in preds[:10]
        ],
    }
