"""End-to-end orchestration: scrape -> dedupe -> filter -> score -> store -> digest."""
from __future__ import annotations

import logging
from datetime import datetime, timedelta
from pathlib import Path
from typing import List, Optional

from .config import settings
from .emailer import digest_html, send
from .models import Application, Job, backup_db, init_db, session
from .matcher import REMOTE_TOKENS, company_in_watchlist, excluded, location_match, score
from .profile import load_profile
from .query_expansion import expand_queries
from .ranking import semantic_score, rerank_topk
import json as _json
from .resume import load_resume, markdown_to_docx, markdown_to_pdf
from .skills import relevant_markdown
from .liveness import sweep_expired
from .scrape_progress import progress as scrape_progress
from .scrapers import registry, RawJob
from .user_prefs import load_tags, load_locations, includes_remote, scrape_locations
from .watchlist import load_companies

log = logging.getLogger("jobbot.pipeline")


def run_harvest() -> dict:
    """Lazy wrapper — defers the discovery import to break the contacts circular chain.
    Module-level so tests can monkeypatch ``pipeline.run_harvest``."""
    from .discovery.service import run_harvest as _rh
    return _rh()


def run_scrape_cycle(send_digest: bool = True) -> dict:
    """Scrape all sources, store new biochem jobs filtered to location + watchlist."""
    init_db()
    # Snapshot the DB before a cycle mutates it (rotating auto-backups).
    try:
        bak = backup_db()
        if bak:
            log.info("DB backed up before scrape -> %s", bak)
    except Exception as e:  # noqa: BLE001
        log.warning("Pre-scrape DB backup failed (%s); continuing.", e)
    tags = load_tags()
    # Use the user-overridable locations (data/user_locations.txt > .env fallback)
    # and log what is actually effective so the user can verify at a glance.
    user_locs = load_locations()        # e.g. ["albany"] or ["Albany, NY", "remote"]
    effective_locs = scrape_locations() # geo-expanded canonical form
    remote_on = includes_remote()
    log.info(
        "Starting scrape cycle | user locations: %s | expanded cities: %s | remote: %s | tags: %s",
        user_locs, effective_locs, remote_on, tags,
    )
    # Auto-derived profile drives query expansion + semantic ranking (fail-open).
    profile = load_profile()
    profile_emb = profile.embedding if settings.semantic_ranking_enabled else None
    expanded = expand_queries(profile) if settings.semantic_ranking_enabled else []
    search_terms = list(dict.fromkeys(list(tags) + list(expanded)))
    log.info("Search terms: %d (tags=%d, expanded=%d)",
             len(search_terms), len(tags), len(expanded))
    try:
        raw: List[RawJob] = registry.run_all(search_terms)
    except Exception as e:  # noqa: BLE001
        scrape_progress.fail(str(e)[:300])
        raise
    log.info("Total raw jobs: %d", len(raw))

    try:
        resume_text = load_resume(settings.base_resume_path)
    except FileNotFoundError:
        log.warning("Base resume not found at %s — scoring will degrade.", settings.base_resume_path)
        resume_text = ""

    watch = load_companies()
    log.info("Watchlist loaded: %d companies", len(watch))

    stats = {"scraped": len(raw), "new": 0, "filtered_excluded": 0,
             "filtered_freelance": 0,
             "duplicates": 0, "out_of_range_stored": 0, "in_range_stored": 0,
             "watchlist_boosted": 0}

    new_jobs: List[Job] = []
    new_in_range_jobs: List[Job] = []
    with session() as db:
        # Single bulk fetch of existing hashes — avoids N round-trips when raw
        # scrape volume is large (was the dominant cost of run_scrape_cycle).
        existing_hashes = {h for (h,) in db.query(Job.hash).all()}
        seen_this_run: set[str] = set()
        for r in raw:
            if not r.url or not r.title:
                continue
            h = Job.make_hash(r.source, r.url, r.title)
            if h in existing_hashes or h in seen_this_run:
                stats["duplicates"] += 1
                continue
            seen_this_run.add(h)
            combined = f"{r.title} {r.description}"
            # Only filter the obvious junk (sales/intern) — keep everything biochem-relevant
            if excluded(combined, settings.excludes):
                stats["filtered_excluded"] += 1
                continue
            # Freelance/gig work is judged on the title alone -- "contract" in a
            # description is usually a CRO or a contracting team, not the
            # employment type. See Settings.search_exclude_title.
            if excluded(r.title, settings.excludes_title):
                stats["filtered_freelance"] += 1
                continue
            in_loc = location_match(r.location, r.description, user_locs)
            is_remote = any(tok in f"{r.location} {r.description}".lower() for tok in REMOTE_TOKENS)

            # Stage-1 semantic score (fail-open to legacy bag-of-words).
            emb_json = None
            used_semantic = False
            if settings.semantic_ranking_enabled and profile_emb:
                try:
                    s, job_emb = semantic_score(combined, profile_emb)
                    emb_json = _json.dumps(job_emb)
                    used_semantic = True
                except Exception as e:  # noqa: BLE001
                    log.warning("semantic score failed for %r (%s) — bag-of-words.",
                                r.title[:60], e)
                    s = score(resume_text, combined, settings.keywords)
            else:
                s = score(resume_text, combined, settings.keywords)

            in_watch = company_in_watchlist(r.company, watch)
            if in_watch:
                s = min(1.0, s + 0.15)
                stats["watchlist_boosted"] += 1
            job_tags = r.tags
            if in_watch:
                job_tags = (job_tags + ",watchlist") if job_tags else "watchlist"
            job = Job(
                hash=h,
                source=r.source,
                title=r.title[:500],
                company=r.company[:240],
                location=r.location[:240],
                url=r.url[:1000],
                description=r.description[:20000],
                salary=r.salary[:120],
                tags=job_tags[:500],
                posted_at=r.posted_at,
                discovered_at=datetime.utcnow(),
                match_score=s,
                is_local=in_loc,
                is_remote=is_remote,
                embedding=emb_json,
            )
            db.add(job)
            new_jobs.append(job)
            # In-range gate. When a REAL semantic score was produced, relevance
            # gates membership and location is de-gated (a strong-fit far role is
            # kept; location only sets is_local/is_remote). When semantic scoring
            # was unavailable (Ollama down, or kill-switch off) `s` is the legacy
            # bag-of-words value on a different scale, so we MUST fall back to
            # today's gate (`in_loc and s >= min_match_score`) — otherwise the
            # cosine-calibrated threshold silently empties the digest in exactly
            # the failure mode fail-open exists to cover (review M1).
            if used_semantic:
                in_range = s >= settings.semantic_recall_threshold
            else:
                in_range = in_loc and s >= settings.min_match_score
            if in_range:
                new_in_range_jobs.append(job)
                stats["in_range_stored"] += 1
            else:
                stats["out_of_range_stored"] += 1
        db.commit()
        stats["new"] = len(new_jobs)
        log.info("Pipeline stats: %s", stats)

        # Capture in-range ids while the objects are live for the post-store
        # stage-2 re-rank (which runs in its own session after this block).
        in_range_ids = [j.id for j in new_in_range_jobs]

        if send_digest and new_in_range_jobs and settings.digest_email_to:
            top = sorted(
                new_in_range_jobs,
                key=lambda j: (j.rerank_score if j.rerank_score is not None else j.match_score),
                reverse=True)[:25]
            try:
                send(
                    to=settings.digest_email_to,
                    subject=f"JobBot — {len(new_in_range_jobs)} new biochem matches",
                    body_html=digest_html(top),
                )
            except Exception as e:  # noqa: BLE001
                log.warning("Digest email failed: %s", e)

    # Stage-2 re-rank runs at cycle end (after scraping/storing/digest) so the
    # slow Ollama judge never blocks the cycle. Fail-open. The persisted
    # rerank_score then improves the dashboard / apply-queue ordering.
    if settings.rerank_enabled and in_range_ids:
        try:
            n = _apply_rerank(profile, in_range_ids)
            stats["reranked"] = n
            log.info("Re-ranked %d of %d in-range job(s).", n, len(in_range_ids))
        except Exception as e:  # noqa: BLE001
            log.warning("re-rank step failed (%s).", e)

    scrape_progress.finish(stats)

    # Kit-mode auto-apply: after each cycle, prepare Apply Kits for the best
    # new prospects so the queue is always stocked (AUTO_PREPARE_KITS=0 = off).
    if settings.auto_prepare_kits > 0:
        try:
            from .auto_apply import AutoApplySettings, prepare_apply_queue
            cfg = AutoApplySettings(
                min_match_score=settings.auto_apply_min_score,
                min_callback_probability=settings.auto_apply_min_callback_prob,
                require_watchlist_or_score=settings.auto_apply_require_watchlist_or_score,
                watchlist_only=settings.auto_apply_watchlist_only,
                skip_stale_days=settings.auto_apply_skip_stale_days,
            )
            report = prepare_apply_queue(cfg, build_limit=settings.auto_prepare_kits)
            stats["kits_built"] = len(report["built"])
            log.info("Auto-prepared %d apply kit(s) (tier=%s, %d errors)",
                     len(report["built"]), report.get("selection_tier"),
                     len(report["errors"]))
        except Exception as e:  # noqa: BLE001
            log.warning("Auto kit preparation failed: %s", e)

    # Bulk pre-fetch contacts for the top new prospects so networking targets
    # are ready before you open them (AUTO_PREFETCH_CONTACTS=0 = off).
    if settings.auto_prefetch_contacts > 0:
        try:
            from .contacts import prefetch_contacts
            cstats = prefetch_contacts(
                limit=settings.auto_prefetch_contacts,
                min_score=settings.min_match_score,
                delay=settings.contacts_prefetch_delay,
            )
            stats["contacts_prefetched"] = cstats["total_contacts"]
            log.info("Pre-fetched contacts: %d job(s) -> %d contact(s)",
                     cstats["jobs_with_contacts"], cstats["total_contacts"])
        except Exception as e:  # noqa: BLE001
            log.warning("Auto contact prefetch failed: %s", e)

    # Harvest new ATS boards from the jobs we just scraped (fills the discovery
    # review queue; never touches live target files). Fail-open.
    if settings.auto_discover_harvest:
        try:
            rep = run_harvest()
            stats["discovered"] = rep.get("added", 0)
            log.info("Discovery harvest: %d new board(s) queued for review.", rep.get("added", 0))
        except Exception as e:  # noqa: BLE001
            log.warning("Auto discovery harvest failed (%s).", e)

    # Liveness sweep: soft-expire postings that have since gone dead. Fail-open.
    if settings.liveness_check_per_cycle > 0:
        try:
            rep = sweep_expired(limit=settings.liveness_check_per_cycle)
            stats["expired"] = rep.get("expired", 0)
            log.info("Liveness: expired %d dead posting(s).", rep.get("expired", 0))
        except Exception as e:  # noqa: BLE001
            log.warning("Liveness sweep failed (%s).", e)

    return stats


def _apply_rerank(profile, job_ids: List[int]) -> int:
    """Stage-2: re-rank the given jobs and persist rerank_score + rationale.
    Fail-open and a no-op when disabled. Returns the number of jobs updated."""
    if not settings.rerank_enabled or not job_ids:
        return 0
    try:
        with session() as db:
            jobs = [db.get(Job, jid) for jid in job_ids]
            jobs = [j for j in jobs if j is not None]
            results = rerank_topk(profile, jobs, settings.rerank_top_k)
            by_id = {r.job_id: r for r in results}
            for j in jobs:
                r = by_id.get(j.id)
                if r is not None:
                    j.rerank_score = r.score
                    j.rerank_rationale = r.rationale
            db.commit()
            return len(by_id)
    except Exception as e:  # noqa: BLE001
        log.warning("stage-2 re-rank failed (%s) — keeping stage-1 scores.", e)
        return 0


def _augment_description(description: str, embedding_json) -> str:
    """Append this job's most relevant verified skills to the JD the model sees,
    so tailoring emphasizes truthful, role-relevant strengths. Fail-open."""
    try:
        emb = _json.loads(embedding_json) if embedding_json else None
    except Exception:  # noqa: BLE001
        emb = None
    try:
        block = relevant_markdown(description or "", job_embedding=emb)
    except Exception as e:  # noqa: BLE001
        log.warning("relevant-skills augmentation failed (%s).", e)
        block = ""
    return f"{description}\n\n{block}" if block else (description or "")


def _augment_resume(resume_text: str) -> str:
    """Append authoritative candidate facts from the personal-info sheet
    (contact, work experience, education, languages) to the resume the model
    tailors from, so it keeps names, dates, employers and degrees accurate and
    doesn't omit detail. Fail-open — returns the base resume unchanged if the
    sheet is missing/unreadable."""
    try:
        from .personal_info import resume_context
        ctx = resume_context()
    except Exception as e:  # noqa: BLE001
        log.warning("personal-info resume context unavailable (%s).", e)
        ctx = ""
    if not ctx:
        return resume_text
    return (f"{resume_text}\n\n"
            "--- CANDIDATE FACTS (authoritative personal profile; ground names, "
            "dates, employers, degrees and contact details on this — do NOT "
            "invent anything beyond it) ---\n"
            f"{ctx}")


def _write_cover_letter_docx(cover_text: str, cover_txt_path, recipient_info: dict | None = None) -> "Optional[Path]":
    """Write the cover letter as a formal Word document beside its .txt, or None.

    Employers ask for an uploadable cover letter, and a bare .txt meant
    hand-converting it every time while the resume already produced .docx.
    Gated on settings.output_docx like the resume, and fail-open: a render
    problem must not sink an otherwise successful tailor run.
    """
    if not (cover_text or "").strip():
        return None
    if not getattr(settings, "output_docx", True):
        return None
    try:
        from .resume import cover_letter_to_docx
        p = Path(cover_txt_path)
        out = p.with_name(p.name.replace("_cover.txt", "_cover_letter.docx"))
        cover_letter_to_docx(cover_text, out, recipient_info=recipient_info)
        return out
    except Exception as e:  # noqa: BLE001
        log.warning("Cover-letter .docx render failed (%s); .txt still written", e)
        return None


def _extract_company_hook(company: str, description: str) -> str:
    """Extract concrete pipeline, platform, or technical domain hooks for Paragraph 1 of cover letters."""
    import re
    desc_low = (description or "").lower()
    hooks = []

    focus_patterns = [
        (r"\b(antibody[- ]drug conjugates?|adcs?)\b", "antibody-drug conjugate (ADC) development"),
        (r"\b(rna therapeutics?|mrna|oligonucleotides?|aptamers?)\b", "RNA therapeutics and targeted oligonucleotide engineering"),
        (r"\b(gene therapy|aav|viral vectors?)\b", "gene therapy vector engineering and delivery platforms"),
        (r"\b(protein engineering|directed evolution|biologics?)\b", "recombinant biologics and protein engineering initiatives"),
        (r"\b(structural biology|cryo[- ]em|crystallography|x-ray)\b", "structural biology and target structure characterization"),
        (r"\b(cadd|computational chemistry|molecular docking|in silico)\b", "in silico drug design and predictive computational screening"),
        (r"\b(oncology|immuno[- ]oncology|cancer)\b", "innovative oncology and therapeutic target discovery"),
        (r"\b(neuroscience|neurodegeneration|cns)\b", "neurodegenerative disease research and receptor mechanisms"),
        (r"\b(bioprocess|fermentation|downstream purification|cdmo)\b", "scalable bioprocess development and downstream purification pipelines"),
        (r"\b(distributed systems?|cloud infrastructure|streaming data)\b", "high-reliability distributed systems and scalable data infrastructure"),
        (r"\b(machine learning|deep learning|mlops|ai models?)\b", "production machine learning systems and algorithmic model architectures"),
    ]
    for pattern, label in focus_patterns:
        if re.search(pattern, desc_low):
            hooks.append(label)
            if len(hooks) >= 2:
                break

    if hooks:
        return f"TARGET COMPANY / PLATFORM INITIATIVE: Directly tie enthusiasm to {company}'s ongoing work in {', and '.join(hooks)}."
    return ""


def tailor_for_job(job_id: int, angle_guidance: str = "", page_budget: str = "2-page") -> dict:
    """Run Gemini tailoring for one job. Returns paths + analysis.

    Runs analyze_match, tailor_resume, and generate_cover_letter concurrently
    (they are independent) — reduces per-job latency from ~40s to ~15s.

    If Gemini is denied / errors out and `gemini_skip_on_error` is True,
    returns a partial result with `error` instead of raising.
    """
    from concurrent.futures import ThreadPoolExecutor as _TPE
    from . import ai_client as gc
    init_db()
    out_dir = Path(settings.output_dir)
    out_dir.mkdir(parents=True, exist_ok=True)

    with session() as db:
        job = db.get(Job, job_id)
        if not job:
            raise ValueError(f"Job {job_id} not found")
        resume_text = _augment_resume(load_resume(settings.base_resume_path))
        aug_desc = _augment_description(job.description, job.embedding)
        if angle_guidance:
            aug_desc += f"\n\n{angle_guidance}"
        company_hook = _extract_company_hook(job.company, job.description)
        if company_hook:
            aug_desc += f"\n\n{company_hook}"

        log.info("Tailoring resume for %s @ %s", job.title, job.company)
        try:
            # Fire all three Gemini calls simultaneously — they share no state.
            with _TPE(max_workers=3, thread_name_prefix="gemini_tailor") as ex:
                f_analysis = ex.submit(gc.analyze_match,
                                       resume_text, job.title, job.company, aug_desc)
                f_resume = ex.submit(gc.tailor_resume,
                                     resume_text, job.title, job.company, aug_desc)
                f_cover = ex.submit(gc.generate_cover_letter,
                                    resume_text, job.title, job.company,
                                    aug_desc, settings.applicant_name)
                analysis = f_analysis.result()
                tailored_md = f_resume.result()
                cover = f_cover.result()
        except Exception as e:  # noqa: BLE001
            if not settings.gemini_skip_on_error:
                raise
            log.warning("Gemini tailoring failed (%s) — skipping with stub output.", e)
            return {
                "application_id": None,
                "resume": "",
                "cover_letter": "",
                "analysis": {"score": 0.0, "summary": f"Gemini unavailable: {e}"},
                "error": str(e),
            }

        # SOTA Deterministic Fidelity & Anti-AI Buzzword Repair Gate
        from . import source_fidelity as _sf
        tailored_md, defects = _sf.repair_fidelity(resume_text, tailored_md)
        if defects:
            log.warning("Tailored resume ships with %d verified defect(s): %s",
                        len(defects), "; ".join(defects))

        slug = f"{job.id}_{_slug(job.company)}_{_slug(job.title)}"[:120]
        md_path = out_dir / f"{slug}_resume.md"
        pdf_path = out_dir / f"{slug}_resume.pdf"
        docx_path = out_dir / f"{slug}_resume.docx"
        cover_path = out_dir / f"{slug}_cover.txt"
        md_path.write_text(tailored_md, encoding="utf-8")
        cover_path.write_text(cover, encoding="utf-8")
        _write_cover_letter_docx(cover, cover_path, recipient_info={"company": job.company, "title": job.title, "location": job.location})
        try:
            markdown_to_pdf(tailored_md, pdf_path, page_budget=page_budget)
        except Exception as e:  # noqa: BLE001
            log.warning("PDF render failed (%s); using markdown only", e)
            pdf_path = md_path
        primary_resume_path = pdf_path
        if settings.output_docx:
            try:
                markdown_to_docx(tailored_md, docx_path, page_budget=page_budget)
                if docx_path.exists():
                    primary_resume_path = docx_path
            except Exception as e:  # noqa: BLE001
                log.warning("DOCX render failed: %s", e)

        app = Application(
            job_id=job.id,
            tailored_resume_path=str(primary_resume_path),
            cover_letter_path=str(cover_path),
            ai_match_analysis=str(analysis),
            status="draft",
        )
        db.add(app)
        job.status = "tailored"
        db.commit()
        return {
            "application_id": app.id,
            "resume": str(primary_resume_path),
            "docx": str(docx_path) if docx_path.exists() else None,
            "cover_letter": str(cover_path),
            "analysis": analysis,
        }


def tailor_for_job_local(job_id: int, rounds: int | None = None, progress=None, angle_guidance: str = "", page_budget: str = "2-page") -> dict:
    """Tailor a job locally with Ollama + iterative smart refinement -> Word doc.

    This is the free, offline backup path for resume tailoring: it runs the
    local Ollama model through a draft -> self-critique -> revise loop, then
    always emits a .docx (plus .md and a best-effort .pdf) and records an
    Application. Use it when cloud credits are depleted or you want a fully
    local run. `rounds` overrides settings.ollama_refine_rounds.

    Raises RuntimeError if no local Ollama server is reachable (so the caller
    can fall back to the cloud path rather than silently produce nothing).
    """
    from . import ollama_client as oc
    init_db()
    out_dir = Path(settings.output_dir)
    out_dir.mkdir(parents=True, exist_ok=True)

    def _say(msg: str) -> None:
        if progress:
            try:
                progress(msg)
            except Exception:  # noqa: BLE001
                pass

    _say("Ensuring local Ollama server + model are running...")
    if not oc.ensure_ready():
        raise RuntimeError(
            "Ollama model unavailable: server at "
            f"{settings.ollama_host} isn't running and couldn't be auto-started. "
            "Install Ollama (`ollama serve`) locally, or point OLLAMA_HOST at a "
            "cluster tunnel (see cluster/README.md)."
        )

    with session() as db:
        job = db.get(Job, job_id)
        if not job:
            raise ValueError(f"Job {job_id} not found")
        _say("Preparing candidate profile and matching verified skills...")
        resume_text = _augment_resume(load_resume(settings.base_resume_path))
        aug_desc = _augment_description(job.description, job.embedding)
        if angle_guidance:
            aug_desc += f"\n\n{angle_guidance}"
        company_hook = _extract_company_hook(job.company, job.description)
        if company_hook:
            aug_desc += f"\n\n{company_hook}"

        log.info("Local (Ollama) iterative tailoring for %s @ %s", job.title, job.company)
        tailor = oc.tailor_resume_iterative(
            resume_text, job.title, job.company, aug_desc,
            rounds=rounds, progress=_say,
        )
        tailored_md = tailor["resume"]
        _say("Drafting cover letter...")
        cover = oc.generate_cover_letter(
            resume_text, job.title, job.company, aug_desc, settings.applicant_name)
        _say("Scoring match...")
        try:
            analysis = oc.analyze_match(resume_text, job.title, job.company, aug_desc)
        except Exception as e:  # noqa: BLE001
            log.warning("Local analyze_match failed (%s); continuing.", e)
            analysis = {"score": 0.0, "summary": "match analysis unavailable"}
        analysis = dict(analysis or {})
        analysis["refinement"] = {
            "provider": "ollama",
            "model": settings.ollama_model,
            "rounds": tailor.get("rounds"),
            "final_self_score": tailor.get("score"),
            "history": tailor.get("history"),
        }

        slug = f"{job.id}_{_slug(job.company)}_{_slug(job.title)}"[:120]
        md_path = out_dir / f"{slug}_resume.md"
        docx_path = out_dir / f"{slug}_resume.docx"
        pdf_path = out_dir / f"{slug}_resume.pdf"
        cover_path = out_dir / f"{slug}_cover.txt"
        md_path.write_text(tailored_md, encoding="utf-8")
        cover_path.write_text(cover, encoding="utf-8")
        _write_cover_letter_docx(cover, cover_path, recipient_info={"company": job.company, "title": job.title, "location": job.location})

        # Word doc is the headline deliverable here — always produce it.
        _say("Rendering Word document...")
        markdown_to_docx(tailored_md, docx_path, page_budget=page_budget)
        primary_path = docx_path
        try:
            markdown_to_pdf(tailored_md, pdf_path, page_budget=page_budget)
        except Exception as e:  # noqa: BLE001
            log.warning("PDF render failed (%s); docx + md still written.", e)

        app = Application(
            job_id=job.id,
            tailored_resume_path=str(primary_path),
            cover_letter_path=str(cover_path),
            ai_match_analysis=str(analysis),
            status="draft",
        )
        db.add(app)
        job.status = "tailored"
        db.commit()
        return {
            "application_id": app.id,
            "resume": str(primary_path),
            "docx": str(docx_path),
            "cover_letter": str(cover_path),
            "analysis": analysis,
            "rounds": tailor.get("rounds"),
            "self_score": tailor.get("score"),
            # Verified defects (checked in code, not the model's opinion) that
            # survived refinement. Callers MUST surface these -- a resume that
            # misstates the candidate's record must never ship quietly behind a
            # "self-score 0.50" line.
            "defects": tailor.get("defects", []),
        }


def _record_sent(application_id: int, when: Optional[datetime] = None) -> None:
    """Mark an application sent. Delegates so this path logs the same funnel
    event as the browser path -- it used to set the status directly and the
    application never appeared in the outcome history."""
    from .browser_apply import mark_applied
    mark_applied(application_id, method="pipeline", when=when)


def send_application(application_id: int, to_email: str, custom_note: str = "") -> None:
    init_db()
    with session() as db:
        app = db.get(Application, application_id)
        if not app:
            raise ValueError(f"Application {application_id} not found")
        job = app.job
        cover = Path(app.cover_letter_path).read_text(encoding="utf-8") if app.cover_letter_path else ""
        body_text = (custom_note + "\n\n" if custom_note else "") + cover
        body_html = f"<pre style='font-family:Georgia,serif;font-size:13px;white-space:pre-wrap'>{body_text}</pre>"
        send(
            to=to_email,
            subject=f"Application for {job.title} — {settings.applicant_name}",
            body_html=body_html,
            body_text=body_text,
            attachments=[p for p in [app.tailored_resume_path, app.cover_letter_path] if p],
        )
        _sent_app_id = app.id
        db.commit()

    # Outside the session: mark_applied opens its own, and nesting two sessions
    # on the same SQLite file deadlocks.
    _record_sent(_sent_app_id)


def generate_interview_prep(job_id: int) -> dict:
    """Generate interview prep + company intel for a job. Stores on Job, returns dict.
    Returns partial result with `error` if Gemini is denied (when gemini_skip_on_error)."""
    from . import ai_client as gc
    init_db()
    out_dir = Path(settings.output_dir)
    out_dir.mkdir(parents=True, exist_ok=True)
    with session() as db:
        job = db.get(Job, job_id)
        if not job:
            raise ValueError(f"Job {job_id} not found")
        resume_text = load_resume(settings.base_resume_path)
        aug_desc = _augment_description(job.description, job.embedding)
        log.info("Generating interview prep for %s @ %s", job.title, job.company)
        try:
            prep = gc.interview_prep(resume_text, job.title, job.company, aug_desc)
            intel = gc.company_intel(job.company, job.title, aug_desc)
        except Exception as e:  # noqa: BLE001
            if not settings.gemini_skip_on_error:
                raise
            log.warning("Gemini interview-prep failed (%s) — returning stub.", e)
            return {"prep": "", "intel": "", "path": "", "error": str(e)}
        job.interview_prep = prep
        job.company_intel = intel
        job.interview_prep_generated_at = datetime.utcnow()
        db.commit()
        slug = f"{job.id}_{_slug(job.company)}_{_slug(job.title)}"[:120]
        prep_path = out_dir / f"{slug}_interview_prep.md"
        prep_path.write_text(f"# Interview Prep — {job.title} @ {job.company}\n\n{intel}\n\n---\n\n{prep}", encoding="utf-8")
        return {"prep": prep, "intel": intel, "path": str(prep_path)}


def _slug(text: str) -> str:
    import re
    return re.sub(r"[^a-z0-9]+", "-", (text or "").lower()).strip("-")[:48]


def daily_apply_quota_remaining() -> int:
    with session() as db:
        since = datetime.utcnow() - timedelta(days=1)
        # No status clause: sent_at already implies the application went out,
        # and record_outcome advances status past "sent" (responded/screen/
        # interview). Filtering on status would hand back rate-limit headroom
        # every time someone replied.
        sent_today = db.query(Application).filter(
            Application.sent_at >= since
        ).count()
        return max(settings.apply_rate_limit_per_day - sent_today, 0)


def batch_tailor_jobs(job_ids: List[int], max_workers: int = 2) -> List[dict]:
    """Tailor multiple jobs concurrently.

    Uses a small thread pool so multiple `tailor_for_job` calls run in parallel.
    max_workers is intentionally kept at 2 by default to avoid exhausting the
    Gemini rate-limit (each call itself fires 3 concurrent Gemini requests, so
    2 outer workers = 6 simultaneous API calls).

    Returns a list of result dicts in the same order as job_ids.
    """
    from concurrent.futures import ThreadPoolExecutor, as_completed

    results: dict[int, dict] = {}
    with ThreadPoolExecutor(max_workers=max_workers, thread_name_prefix="batch_tailor") as ex:
        futures = {ex.submit(tailor_for_job, jid): jid for jid in job_ids}
        for fut in as_completed(futures, timeout=600):
            jid = futures[fut]
            try:
                results[jid] = fut.result()
                log.info("batch_tailor: job %d done", jid)
            except Exception as e:  # noqa: BLE001
                log.warning("batch_tailor: job %d failed: %s", jid, e)
                results[jid] = {"application_id": None, "error": str(e)}
    return [results.get(jid, {"application_id": None, "error": "not started"}) for jid in job_ids]
