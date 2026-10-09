"""Unit tests for SOTA Tailoring Engine enhancements:
- Dynamic, extensible strategic tailoring angles
- Page budgeting & typography (1-page condensed vs 2-page scientific)
- Local bullet polisher with strict Rule 1/Rule 2 invariants
- Web endpoints for bullet polishing and tailoring studio
"""
from pathlib import Path
import pytest
from jobbot.strategic_angles import (
    load_strategic_angles,
    get_available_angles,
    get_angle_guidance,
    auto_detect_angle,
)
from jobbot.resume import markdown_to_docx
from jobbot.ollama_client import polish_bullet
from jobbot.models import Job, init_db, session
from jobbot.web import create_app


def test_strategic_angles_loading_and_fallback():
    angles = load_strategic_angles()
    assert isinstance(angles, dict)
    assert "balanced" in angles
    assert "computational" in angles
    assert "software_engineering" in angles

    avail = get_available_angles()
    assert len(avail) >= 4
    assert any(a["id"] == "balanced" for a in avail)

    guidance = get_angle_guidance("computational")
    assert "Computational" in guidance or "docking" in guidance.lower() or "cadd" in guidance.lower()

    # Fallback to balanced if unknown angle requested
    fallback_guidance = get_angle_guidance("unknown_nonexistent_angle")
    assert "Balanced" in fallback_guidance or "balance" in fallback_guidance.lower()


def test_auto_detect_strategic_angle():
    angle_comp = auto_detect_angle("Computational Biologist", "Machine learning, AutoDock, simulation pipelines")
    assert angle_comp == "computational"

    angle_swe = auto_detect_angle("Backend Engineer", "Software engineering, microservices, gRPC, APIs")
    assert angle_swe == "software_engineering"

    angle_data = auto_detect_angle("Data Engineer", "Kafka, Spark, ETL data pipeline architecture")
    assert angle_data == "data_engineering"

    angle_bal = auto_detect_angle("General Analyst", "Cross-functional operations")
    assert angle_bal == "balanced"


def test_page_budget_docx_typography(tmp_path: Path):
    sample_md = """# Candidate Name
candidate@example.com | 555-0100

## Professional Experience
### Senior Research Scientist — Acme Corp
- Built automated data analysis pipelines in Python.
- Developed automated analysis pipelines in Python.

## Education
### Ph.D. in Chemistry — State University
"""
    docx_1p = tmp_path / "resume_1p.docx"
    docx_2p = tmp_path / "resume_2p.docx"

    markdown_to_docx(sample_md, docx_1p, page_budget="1-page")
    markdown_to_docx(sample_md, docx_2p, page_budget="2-page")

    assert docx_1p.exists()
    assert docx_2p.exists()
    assert docx_1p.stat().st_size > 1000
    assert docx_2p.stat().st_size > 1000


def test_bullet_polishing_cleans_banned_buzzwords():
    # Input has banned AI corporate buzzwords
    buzzword_bullet = "- Spearheaded and leveraged cross-functional pipelines to utilize machine learning."
    result = polish_bullet(buzzword_bullet, action="action_verb")

    assert result["original"] == buzzword_bullet
    polished = result["polished"]
    assert "spearheaded" not in polished.lower()
    assert "leveraged" not in polished.lower()
    assert "utilize" not in polished.lower()


def test_api_polish_bullet_endpoint():
    app = create_app()
    app.config["TESTING"] = True
    client = app.test_client()

    # Test empty payload returns 400
    res_empty = client.post("/api/tailor/polish-bullet", json={})
    assert res_empty.status_code == 400

    # Test valid bullet
    res = client.post(
        "/api/tailor/polish-bullet",
        json={
            "bullet": "- Built automated data processing pipelines in Python.",
            "action": "action_verb",
        },
    )
    assert res.status_code == 200
    data = res.get_json()
    assert data["ok"] is True
    assert "result" in data
    assert "polished" in data["result"]


def test_extract_company_hook():
    from jobbot.pipeline import _extract_company_hook
    desc_adc = "We are seeking a scientist to advance antibody-drug conjugates (ADCs) in oncology."
    hook_adc = _extract_company_hook("Acme Bio", desc_adc)
    assert "antibody-drug conjugate" in hook_adc
    assert "Acme Bio" in hook_adc

    desc_cadd = "Work on in silico drug design and molecular docking for novel small molecules."
    hook_cadd = _extract_company_hook("InSilico Inc", desc_cadd)
    assert "in silico drug design" in hook_cadd
    assert "InSilico Inc" in hook_cadd


def test_ats_skills_analysis_transferable_and_honest_gaps():
    from jobbot.tailoring_studio import analyze_job_skills
    jd = "Requirements: Must have experience with Biacore, Cryo-EM, Python, and PyTorch."
    tailored_md = """# Candidate
## Core Competencies
- Python, PyTorch, UV-Vis spectroscopy, Chromatography
## Experience
- Developed PyTorch deep learning models.
"""
    analysis = analyze_job_skills(jd, tailored_resume_md=tailored_md)
    assert "match_percentage" in analysis
    assert "matched_skills" in analysis
    assert "transferable_skills" in analysis
    assert "honest_gaps" in analysis
    assert "section_heatmap" in analysis

    # Python / PyTorch should be matched
    matched_names = [s["name"].lower() for s in analysis["matched_skills"]]
    assert any("python" in m for m in matched_names)

    # Biacore should map to transferable UV-Vis/spectroscopy
    transferable_kws = [t["jd_keyword"].lower() for t in analysis["transferable_skills"]]
    assert any("biacore" in kw or "spr" in kw for kw in transferable_kws)

    # Cryo-EM should be an honest gap per Rule 1
    gaps_lower = [g.lower() for g in analysis["honest_gaps"]]
    assert any("cryo" in g for g in gaps_lower)

    # Section heatmap should have detected competencies and experience
    assert len(analysis["section_heatmap"]["competencies"]) >= 1


def test_markdown_to_pdf_export(tmp_path: Path):
    from jobbot.resume import markdown_to_pdf
    sample_md = """# Dr. Jane Doe
jane@example.com | (555) 012-3456 | New York, NY

## Professional Experience
### Senior Research Scientist — Global Tech
- Developed automated data ingestion pipelines in Python.
- Developed automated analysis pipelines in Python.

## Core Competencies
- Python, PyTorch, SDS-PAGE, UV-Vis spectroscopy, Docker
"""
    pdf_1p = tmp_path / "test_resume_1p.pdf"
    pdf_2p = tmp_path / "test_resume_2p.pdf"

    markdown_to_pdf(sample_md, pdf_1p, page_budget="1-page")
    markdown_to_pdf(sample_md, pdf_2p, page_budget="2-page")

    assert pdf_1p.exists()
    assert pdf_2p.exists()
    assert pdf_1p.stat().st_size > 500
    assert pdf_2p.stat().st_size > 500


def test_api_job_ats_breakdown_endpoint():
    init_db()
    with session() as db:
        job = Job(
            hash="test_ats_h101",
            source="test",
            title="Senior Biochemist",
            company="BioTest Corp",
            url="https://example.com/jobs/101",
            description="Seeking expertise in Docker, Biacore SPR, Cryo-EM, and Python.",
        )
        db.add(job)
        db.commit()
        job_id = job.id

    app = create_app()
    app.config["TESTING"] = True
    client = app.test_client()

    res = client.get(f"/api/job/{job_id}/ats-breakdown")
    assert res.status_code == 200
    data = res.get_json()
    assert data["ok"] is True
    assert data["job_id"] == job_id
    assert "match_percentage" in data
    assert "matched_skills" in data
    assert "transferable_skills" in data
    assert "honest_gaps" in data
    assert "section_heatmap" in data

