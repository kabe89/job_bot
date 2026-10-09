"""Strategic Tailoring Angles Engine.

Provides extensible, user-adaptable strategy angles for tailoring resumes and
cover letters. Can load custom profiles from `data/strategic_angles.json` or fall
back to domain archetypes.
"""
from __future__ import annotations

import json
import logging
from pathlib import Path
from typing import Any, Dict, List, Optional

from .config import settings

log = logging.getLogger("jobbot.strategic_angles")

STRATEGIC_ANGLES_FILE = Path("data/strategic_angles.json")

# Default domain archetypes (adaptable across technical and scientific fields)
DEFAULT_ANGLES: Dict[str, Dict[str, str]] = {
    "balanced": {
        "id": "balanced",
        "label": "Balanced Profile (Full-Scope Overview)",
        "description": "Evenly balances technical architecture, hands-on implementation, and quantitative outcomes.",
        "guidance": (
            "TARGETING ANGLE: Balanced Profile. Present a well-rounded balance of technical design, "
            "hands-on implementation, and verified project outcomes without skewing toward any single specialty."
        ),
    },
    "software_engineering": {
        "id": "software_engineering",
        "label": "Software Engineering & Backend Architecture",
        "description": "Prioritizes distributed systems, REST/gRPC APIs, clean architecture, and testing.",
        "guidance": (
            "TARGETING ANGLE: Software Engineering & Backend Architecture. Prioritize distributed systems, "
            "performant APIs (REST/gRPC), concurrency, scalable database designs, and test-driven reliability. "
            "Emphasize concrete technical design and production code quality."
        ),
    },
    "machine_learning": {
        "id": "machine_learning",
        "label": "Machine Learning, AI & Applied Data Science",
        "description": "Prioritizes deep learning models, PyTorch, evaluation pipelines, and vector embeddings.",
        "guidance": (
            "TARGETING ANGLE: Machine Learning & Applied AI. Prioritize model development, PyTorch architectures, "
            "vector embeddings, retrieval-augmented pipelines, dataset preprocessing, and rigorous validation metrics. "
            "Emphasize quantitative problem-solving and model performance."
        ),
    },
    "data_engineering": {
        "id": "data_engineering",
        "label": "Data Engineering & High-Throughput Pipelines",
        "description": "Prioritizes streaming pipelines, ETL, data warehouses, and database optimization.",
        "guidance": (
            "TARGETING ANGLE: Data Engineering. Prioritize data pipeline architecture, ETL workflows, "
            "database query optimization, schema migration reliability, and data warehouse modeling."
        ),
    },
    "computational": {
        "id": "computational",
        "label": "Computational Science & Modeling Focus",
        "description": "Prioritizes machine learning models, simulation pipelines, and high-performance computing.",
        "guidance": (
            "TARGETING ANGLE: Computational Modeling & Data Science. Prioritize numerical models, "
            "algorithmic optimization, exploratory data analysis, and HPC cluster workflows. "
            "Emphasize predictive rigor and quantitative validation."
        ),
    },
    "research_scientific": {
        "id": "research_scientific",
        "label": "Applied Research & Quantitative Analysis",
        "description": "Prioritizes mechanistic hypotheses, experimental design, and statistical benchmarking.",
        "guidance": (
            "TARGETING ANGLE: Applied Research & Quantitative Analysis. Prioritize hypothesis testing, "
            "rigorous experimental validation, statistical benchmarking, and translating theoretical concepts into practice."
        ),
    },
}


def load_strategic_angles() -> Dict[str, Dict[str, str]]:
    """Load strategic angles from `data/strategic_angles.json` or fallback to defaults."""
    angles = dict(DEFAULT_ANGLES)
    if STRATEGIC_ANGLES_FILE.exists():
        try:
            custom = json.loads(STRATEGIC_ANGLES_FILE.read_text(encoding="utf-8"))
            if isinstance(custom, dict):
                for k, v in custom.items():
                    if isinstance(v, dict):
                        item = dict(v)
                        item["id"] = k
                        angles[k] = item
        except Exception as e:
            log.warning("Failed to parse custom strategic angles file (%s): %s", STRATEGIC_ANGLES_FILE, e)
    return angles


def get_available_angles() -> List[Dict[str, str]]:
    """Return an ordered list of strategic angle metadata for UI selectors."""
    angles_dict = load_strategic_angles()
    preferred_order = [
        "balanced", "software_engineering", "machine_learning",
        "data_engineering", "computational", "research_scientific"
    ]
    ordered: List[Dict[str, str]] = []
    seen = set()
    for k in preferred_order:
        if k in angles_dict:
            ordered.append(angles_dict[k])
            seen.add(k)
    for k, v in angles_dict.items():
        if k not in seen:
            ordered.append(v)
    return ordered


def get_angle_guidance(angle_key: str) -> str:
    """Retrieve prompt guidance for a specific angle key, falling back gracefully."""
    angles = load_strategic_angles()
    # Support aliases / backwards compatibility
    alias_map = {
        "swe": "software_engineering",
        "ai": "machine_learning",
        "ml": "machine_learning",
        "data": "data_engineering",
        "research": "research_scientific",
        "comp": "computational",
    }
    resolved_key = alias_map.get(angle_key, angle_key)
    angle = angles.get(resolved_key) or angles.get("balanced") or {}
    return angle.get("guidance", "")


def auto_detect_angle(job_title: str, job_description: str) -> str:
    """Intelligently recommend the best strategic angle based on JD keywords."""
    text = f"{job_title} {job_description}".lower()

    scores = {
        "software_engineering": 0,
        "machine_learning": 0,
        "data_engineering": 0,
        "computational": 0,
        "research_scientific": 0,
    }

    swe_keywords = ["software engineering", "software engineer", "backend", "microservices", "grpc", "fullstack", "golang", "api", "frontend"]
    data_keywords = ["data engineering", "kafka", "spark", "etl", "data warehouse", "streaming", "sql"]
    ml_keywords = ["machine learning", "deep learning", "transformer", "vector embeddings", "llm", "neural network", "pytorch"]
    comp_keywords = ["computational", "simulation", "modeling", "hpc", "algorithm", "numerical", "docking", "bioinformatics"]
    research_keywords = ["research scientist", "applied research", "experimental design", "hypothesis", "statistical analysis", "benchmarking"]

    for kw in swe_keywords:
        if kw in text:
            scores["software_engineering"] += 1
    for kw in data_keywords:
        if kw in text:
            scores["data_engineering"] += 1
    for kw in ml_keywords:
        if kw in text:
            scores["machine_learning"] += 1
    for kw in comp_keywords:
        if kw in text:
            scores["computational"] += 1
    for kw in research_keywords:
        if kw in text:
            scores["research_scientific"] += 1

    best_key = max(scores, key=scores.get)
    if scores[best_key] > 0:
        return best_key
    return "balanced"
