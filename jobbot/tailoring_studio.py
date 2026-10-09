"""Resume Tailoring Studio & Job Intelligence Engine.

Provides:
- Keyword & skill matching analysis (possessed skills vs missing JD requirements)
- Diff generation between base resume and tailored draft
- Multi-tier tailoring orchestrator (Offline Zero-AI / Local Ollama / Cloud)
- In-situ section revision and truthfulness verification
"""
from __future__ import annotations

import difflib
import logging
import re
from pathlib import Path
from typing import Any, Dict, List, Optional, Tuple

from .config import settings
from .models import Application, Job, session
from .resume import load_resume, markdown_to_docx
from .skills import known_skills, load_skills, Skill

log = logging.getLogger("jobbot.tailoring_studio")


def analyze_job_skills(job_description: str, tailored_resume_md: str = "") -> Dict[str, Any]:
    """Analyze job description against candidate's verified skill inventory.

    Returns:
    - matched_skills: list of dicts {name, level, category} present in JD
    - transferable_skills: list of dicts {jd_term, candidate_skill, explanation}
    - honest_gaps: list of strings (JD requirements absent from record, kept unfilled per Rule 1)
    - match_percentage: int (0-100) estimated keyword alignment
    - section_heatmap: dict mapping sections ('summary', 'competencies', 'experience') to matched terms
    """
    desc_low = (job_description or "").lower()
    pool = known_skills(load_skills(), min_rank=1)
    if not pool:
        ref_text = tailored_resume_md or get_base_resume()
        if ref_text:
            extracted = []
            for line in ref_text.splitlines():
                line_str = line.strip()
                if line_str.startswith("- ") or line_str.startswith("* "):
                    raw_items = line_str[2:].split(",")
                    for itm in raw_items:
                        name_clean = itm.strip().strip(".")
                        if 1 < len(name_clean) < 35 and not any(p in name_clean.lower() for p in ["developed", "purified", "managed", "worked", "led", "using"]):
                            extracted.append(Skill(name=name_clean, level="Proficient", notes="", category="Technical Skills"))
            pool = extracted
    candidate_skills_by_name = {s.name.lower(): s for s in pool}

    matched: List[Dict[str, Any]] = []
    seen_matched_names = set()

    # 1. Match verified candidate skills present in JD
    for s in pool:
        s_low = s.name.lower()
        pattern = rf"(?:\b|_){re.escape(s_low)}(?:\b|_)"
        if re.search(pattern, desc_low):
            if s_low not in seen_matched_names:
                seen_matched_names.add(s_low)
                matched.append({
                    "name": s.name,
                    "level": s.level or "Proficient",
                    "category": s.category,
                    "rank": s.rank,
                })

    matched.sort(key=lambda x: -x["rank"])

    # 2. Curated domain technical terms & transferability mapping
    transferable_map: Dict[str, Tuple[List[str], str]] = {
        "spr": (["binding", "spectroscopy", "uv-vis", "fluorescence"], "Adjacent biophysical binding & kinetic affinity assay background"),
        "biacore": (["binding", "spectroscopy", "uv-vis", "fluorescence"], "Extensive optical & spectroscopic binding assay experience"),
        "bli": (["binding", "spectroscopy", "uv-vis", "fluorescence"], "Adjacent optical binding kinetics & affinity assay experience"),
        "gmp": (["sop", "standard operating procedures", "protocol optimization"], "Rigorous protocol documentation & assay reproducibility"),
        "glp": (["sop", "standard operating procedures", "reproducible protocol"], "Controlled protocol documentation & analytical reproducibility"),
        "tensorflow": (["pytorch", "deep learning", "machine learning"], "Directly transferable deep learning framework architecture (PyTorch)"),
        "kafka": (["message queues", "distributed systems", "streaming"], "Scalable streaming pipeline architecture background"),
        "aws": (["slurm", "linux", "hpc clusters", "bash"], "High-performance cluster compute & Linux systems experience"),
        "docker": (["linux", "conda", "reproducible environments"], "Reproducible compute environment configuration"),
        "react": (["javascript", "typescript", "frontend", "html"], "Directly transferable component UI framework experience"),
        "vue": (["react", "typescript", "frontend"], "Directly transferable modern reactive frontend architecture"),
        "postgres": (["sql", "sqlite", "relational databases"], "Relational database schema and query design experience"),
        "spark": (["pandas", "python", "data pipelines", "distributed computing"], "Scalable data processing and distributed pipeline experience"),
    }

    # Technical terms library across laboratory and computing domains
    tech_terms = [
        "hplc", "fplc", "akta", "mass spectrometry", "lc-ms", "nmr", "spr", "biacore", "bli",
        "elisa", "pcr", "qpcr", "western blot", "sds-page", "cloning", "crispr",
        "cell culture", "mammalian", "fermentation", "purification", "assay development",
        "flow cytometry", "bioconjugation", "docking", "molecular dynamics",
        "machine learning", "deep learning", "pytorch", "tensorflow", "nextflow", "bioinformatics",
        "glp", "gmp", "ind", "fda", "sop", "doe", "automation", "liquid handling",
        "dna sequencing", "rna-seq", "proteomics", "crystallography", "cryo-em",
        "docker", "kubernetes", "aws", "gcp", "slurm", "kafka", "spark", "grpc",
        "react", "vue", "postgres"
    ]

    transferable: List[Dict[str, str]] = []
    honest_gaps: List[str] = []
    seen_gap_terms = set()

    for term in tech_terms:
        if term in desc_low and term not in seen_matched_names:
            # Check if candidate has directly matched this
            if any(term in m["name"].lower() for m in matched):
                continue
            if term in seen_gap_terms:
                continue
            seen_gap_terms.add(term)

            # Check if this term has a transferable candidate skill
            display_term = term.upper() if len(term) <= 4 else term.title()
            has_transfer = False
            if term in transferable_map:
                adjacent_skills, explanation = transferable_map[term]
                for adj in adjacent_skills:
                    found_cand_name = None
                    for cand_name in candidate_skills_by_name:
                        if adj in cand_name or cand_name in adj:
                            found_cand_name = candidate_skills_by_name[cand_name].name
                            break
                    if not found_cand_name and tailored_resume_md and adj in tailored_resume_md.lower():
                        found_cand_name = adj.title()

                    if found_cand_name:
                        transferable.append({
                            "jd_keyword": display_term,
                            "jd_term": display_term,
                            "candidate_skill": found_cand_name,
                            "transfer_from": found_cand_name,
                            "explanation": explanation,
                        })
                        has_transfer = True
                        break
            if not has_transfer:
                honest_gaps.append(display_term)

    # 3. Calculate honest match percentage
    total_signals = len(matched) + (len(transferable) * 0.5) + len(honest_gaps)
    if total_signals > 0:
        score_val = (len(matched) + (len(transferable) * 0.5)) / total_signals
        pct = int(round(score_val * 100))
    else:
        pct = 75

    # 4. Section Placement Heatmap (Header / Competencies / Experience)
    heatmap: Dict[str, List[str]] = {"summary": [], "competencies": [], "experience": []}
    if tailored_resume_md:
        sections = re.split(r"\n##+\s+", tailored_resume_md)
        for s_chunk in sections:
            header_line = s_chunk.split("\n", 1)[0].lower() if s_chunk else ""
            content_low = s_chunk.lower()
            target_key = "experience"
            if any(k in header_line for k in ["summary", "profile", "overview", "objective"]):
                target_key = "summary"
            elif any(k in header_line for k in ["skill", "competenc", "technolog", "proficienc"]):
                target_key = "competencies"
            elif any(k in header_line for k in ["experience", "employment", "history", "work"]):
                target_key = "experience"

            for m in matched:
                if m["name"].lower() in content_low and m["name"] not in heatmap[target_key]:
                    heatmap[target_key].append(m["name"])

    return {
        "matched_skills": matched,
        "transferable_skills": transferable,
        "honest_gaps": honest_gaps[:10],
        "match_percentage": max(15, min(100, pct)),
        "section_heatmap": heatmap,
    }


def compute_resume_diff(base_md: str, tailored_md: str) -> List[Dict[str, Any]]:
    """Compute line-by-line diff with tags for visualization in the Tailor Studio."""
    base_lines = base_md.splitlines(keepends=True)
    tailored_lines = tailored_md.splitlines(keepends=True)
    
    diff = difflib.ndiff(base_lines, tailored_lines)
    result = []
    for line in diff:
        tag = line[0]
        text = line[2:].rstrip("\r\n")
        if tag == " ":
            result.append({"type": "equal", "text": text})
        elif tag == "-":
            result.append({"type": "removed", "text": text})
        elif tag == "+":
            result.append({"type": "added", "text": text})
        elif tag == "?":
            continue
    return result


def get_base_resume() -> str:
    """Load canonical base resume markdown."""
    try:
        return load_resume(settings.base_resume_path)
    except Exception as e:
        log.warning("Could not load base resume: %s", e)
        return ""
