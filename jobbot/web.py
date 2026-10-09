"""Flask dashboard — browse jobs, tailor, send, get AI advice."""
from __future__ import annotations

import json
import logging
import re
from pathlib import Path

from flask import Flask, abort, flash, jsonify, make_response, redirect, render_template, request, send_file, url_for

from .config import settings
from .discovery import service as discovery_service
from .models import Application, Contact, Job, init_db, session
from .pipeline import (
    daily_apply_quota_remaining,
    generate_interview_prep,
    run_scrape_cycle,
    send_application,
    tailor_for_job,
    tailor_for_job_local,
)
from . import web_apply

log = logging.getLogger("jobbot.web")

import threading as _threading
_discover_run = {"running": False, "kind": None, "result": None, "error": None}
_discover_run_lock = _threading.Lock()

_pull_run = {"running": False, "model": None, "lines": [], "ok": None}
_pull_lock = _threading.Lock()


def _md_to_html(md: str) -> str:
    if not md:
        return ""
    import markdown as md_mod
    return md_mod.markdown(md, extensions=["extra", "sane_lists"])


def _localdt(value, fmt: str = "%Y-%m-%d") -> str:
    """Render a stored naive-UTC datetime in the user's local zone.

    Fail-open: an unknown zone name renders UTC rather than 500-ing a page.
    """
    if not value:
        return ""
    from datetime import timezone as _tz
    from zoneinfo import ZoneInfo, ZoneInfoNotFoundError
    try:
        local = ZoneInfo(settings.timezone)
    except (ZoneInfoNotFoundError, ValueError):
        local = _tz.utc
    return value.replace(tzinfo=_tz.utc).astimezone(local).strftime(fmt)


def create_app() -> Flask:
    init_db()
    app = Flask(__name__, template_folder=str(Path(__file__).parent / "templates"),
                static_folder=str(Path(__file__).parent / "static"))
    app.secret_key = settings.dashboard_secret
    app.jinja_env.filters["localdt"] = _localdt

    @app.route("/")
    def index():
        from datetime import datetime as _dt, timedelta
        from . import user_prefs
        from .matcher import location_match
        with session() as db:
            status = request.args.get("status", "")
            scope = request.args.get("scope", "in-range")  # in-range|out-of-range|all|anywhere
            min_score = float(request.args.get("min_score") or 0)
            source = request.args.get("source", "")
            company_q = (request.args.get("company") or "").strip().lower()
            text_q = (request.args.get("q") or "").strip().lower()
            watchlist_only = request.args.get("watchlist_only") == "on"
            remote_ok = request.args.get("remote_ok") == "on"
            starred_only = request.args.get("starred") == "on"
            preset = request.args.get("preset", "")
            # Resolve effective locations: query param override > user prefs file > baseline .env
            user_locations = user_prefs.load_locations()
            qs_locs = request.args.get("locations", "").strip()
            effective_locations = user_prefs._normalize(qs_locs) if qs_locs else list(user_locations)
            try:
                radius_miles = float(request.args.get("radius_miles") or 0)
            except ValueError:
                radius_miles = 0.0
            anywhere = user_prefs.is_anywhere_mode(effective_locations) or scope == "anywhere"
            # Quick filter presets — set sane defaults if requested
            if preset == "near-me":
                scope, min_score, starred_only = "in-range", 0.30, False
            elif preset == "remote-watchlist":
                scope, watchlist_only, remote_ok = "all", True, True
            elif preset == "today":
                scope = "all"
            elif preset == "high-match":
                scope, min_score = "all", 0.45
            elif preset == "starred":
                scope, starred_only = "all", True

            q = db.query(Job)
            if status:
                q = q.filter(Job.status == status)
            # Expired jobs are hidden everywhere unless explicitly requested via
            # ?scope=expired or ?status=expired (soft-expire, restorable).
            show_expired = (scope == "expired") or (status == "expired")
            if scope == "expired":
                # Push the expired filter into SQL so the top-2000 window below
                # is drawn from expired rows only — otherwise, once the table
                # exceeds 2000 jobs, higher-scoring active jobs fill the window
                # and the Expired tab silently truncates (while its badge count,
                # a separate COUNT(*), still shows the true total).
                q = q.filter(Job.status == "expired")
            elif not show_expired:
                q = q.filter(Job.status != "expired")
            # Dismissed jobs are hidden the same way, unless asked for by name
            # via ?status=skipped. Skipping a job has to make it leave the list:
            # the skip button and `jobbot clean freelance` both write this
            # status, and without the filter every dismissed posting kept
            # occupying the pool the user is trying to read.
            if status != "skipped":
                q = q.filter(Job.status != "skipped")
            # NOTE: scope filtering against effective_locations is applied below
            # in-Python (after DB query), since user can re-define "in range" at runtime.
            if remote_ok:
                q = q.filter((Job.is_remote == True) | (Job.is_local == True))  # noqa: E712
            if min_score > 0:
                q = q.filter(Job.match_score >= min_score)
            if source:
                q = q.filter(Job.source.like(f"%{source}%"))
            if company_q:
                q = q.filter(Job.company.ilike(f"%{company_q}%"))
            if text_q:
                like = f"%{text_q}%"
                q = q.filter((Job.title.ilike(like)) | (Job.description.ilike(like)))
            if watchlist_only:
                q = q.filter(Job.tags.like("%watchlist%"))
            if starred_only:
                q = q.filter(Job.starred == True)  # noqa: E712
            if preset == "today":
                cutoff = _dt.utcnow() - timedelta(hours=24)
                q = q.filter(Job.discovered_at >= cutoff)

            raw_jobs = q.order_by(Job.match_score.desc(), Job.discovered_at.desc()).limit(2000).all()

            # Runtime location filter — uses the user's CURRENT effective_locations
            # plus an optional mile-radius around each. Lets you add "Rockville MD"
            # or set "75 miles around Albany" without rescraping.
            from .geo import location_in_radius
            if scope == "expired":
                jobs = [j for j in raw_jobs if j.status == "expired"]
            elif anywhere or scope == "anywhere":
                jobs = raw_jobs
            else:
                def _matches(job):
                    in_loc = location_match(job.location, job.description, effective_locations)
                    if not in_loc and radius_miles > 0:
                        in_loc = location_in_radius(job.location, effective_locations, radius_miles)
                    if scope == "in-range":
                        return in_loc or (remote_ok and job.is_remote)
                    if scope == "out-of-range":
                        return not in_loc
                    return True  # scope == "all"
                jobs = [j for j in raw_jobs if _matches(j)]

            # Outcome- and referral-aware scoring + warm-intro tagging (fail-open).
            # CRITICAL: score the FULL location-filtered set BEFORE the display cap
            # below. A warm-intro (or outcome-boosted) job whose raw match_score
            # isn't top-tier can be the genuine #1 by blended score; truncating
            # first would silently cut it off. We sort the whole set, then slice.
            try:
                from . import outcomes, referrals
                outcomes.sweep_ghosted()
                _priors = outcomes.compute_priors()
                _ref_index = referrals.build_index()
                for j in jobs:
                    b, why = referrals.referral_bonus(j, _ref_index)
                    j.warm_intro = why           # "" when no match
                    j._refbonus = b
                    j._blend = outcomes.blended(j, _priors) + b
            except Exception:
                for j in jobs:
                    if not hasattr(j, "_blend"):
                        j.warm_intro = getattr(j, "warm_intro", "")
                        j._refbonus = 0.0
                        j._blend = j.match_score or 0.0

            if preset == "warm-intros":
                jobs = [j for j in jobs if getattr(j, "warm_intro", "")]

            # Sort selector. Because we sort the full filtered set here (before the
            # display cap / pagination), the top job for the chosen order always
            # survives. Every column header is a clickable sort key (see `columns`
            # below); the toolbar dropdown reuses the same keys.
            _sort_keys = {
                "smart":    lambda j: getattr(j, "_blend", 0.0),
                "match":    lambda j: (j.match_score or 0.0, j.discovered_at or _dt.min),
                "new":      lambda j: j.discovered_at or _dt.min,
                "warm":     lambda j: (1 if getattr(j, "warm_intro", "") else 0,
                                       getattr(j, "_blend", 0.0)),
                "star":     lambda j: (1 if j.starred else 0, getattr(j, "_blend", 0.0)),
                "title":    lambda j: (j.title or "").lower(),
                "company":  lambda j: ((j.company or "").lower(), -getattr(j, "_blend", 0.0)),
                "location": lambda j: (j.location or "").lower(),
                "source":   lambda j: (j.source or "").lower(),
                "status":   lambda j: (j.status or "").lower(),
            }
            # Columns whose natural order is high->low (score-like); everything else
            # defaults to A->Z ascending.
            _desc_default = {"smart", "match", "new", "warm", "star"}
            sort = request.args.get("sort", "smart")
            if sort not in _sort_keys:
                sort = "smart"
            _dir = request.args.get("dir", "")
            if _dir in ("asc", "desc"):
                reverse = _dir == "desc"
            else:
                reverse = sort in _desc_default
            jobs.sort(key=_sort_keys[sort], reverse=reverse)
            sort_dir = "desc" if reverse else "asc"
            shown_count = len(jobs)

            # Clickable, direction-toggling column headers. Clicking the active
            # column flips its direction; clicking a new one uses that column's
            # natural default. Sorting always resets to page 1.
            _cur = request.args.to_dict()
            _cur.pop("page", None)
            _col_defs = [("star", "★"), ("smart", "Score"), ("title", "Title"),
                         ("company", "Company"), ("location", "Location"),
                         ("source", "Source"), ("status", "Status")]

            def _col_href(key):
                if key == sort:
                    nd = "asc" if sort_dir == "desc" else "desc"
                else:
                    nd = "desc" if key in _desc_default else "asc"
                return url_for("index", **dict(_cur, sort=key, dir=nd, page=1))

            columns = [{"key": k, "label": lbl, "href": _col_href(k),
                        "active": k == sort,
                        "arrow": ("▼" if sort_dir == "desc" else "▲") if k == sort else ""}
                       for k, lbl in _col_defs]

            # Pagination — replaces the old hard 300-row cap so nothing past the
            # first page is silently dropped.
            page_size = 100
            try:
                page = int(request.args.get("page", 1))
            except (TypeError, ValueError):
                page = 1
            num_pages = max(1, (shown_count + page_size - 1) // page_size)
            page = min(max(1, page), num_pages)
            _start = (page - 1) * page_size
            jobs = jobs[_start:_start + page_size]

            # Score column shows the *blended* AI score the ranking actually uses
            # (raw match + outcome prior + warm-intro bonus), not the bare
            # match_score — so the number lines up with the sort order.
            for j in jobs:
                bl = getattr(j, "_blend", None)
                if bl is None:
                    bl = j.match_score or 0.0
                j._blend_pct = max(0, min(100, int(round(bl * 100))))
                j._score_class = max(0, min(10, int(bl * 10)))

            # Counts for tab badges — compute against the user's effective locations.
            from .geo import location_in_radius as _lir
            all_jobs = db.query(Job).all()
            total = len(all_jobs)
            remote_count = sum(1 for j in all_jobs if j.is_remote)
            if anywhere:
                local_count = total
                out_count = 0
            else:
                def _in(j):
                    if location_match(j.location, j.description, effective_locations):
                        return True
                    return radius_miles > 0 and _lir(j.location, effective_locations, radius_miles)
                local_count = sum(1 for j in all_jobs if _in(j))
                out_count = total - local_count
            starred_count = sum(1 for j in all_jobs if j.starred)
            expired_count = db.query(Job).filter(Job.status == "expired").count()

            # Daily summary stats
            since_24h = _dt.utcnow() - timedelta(hours=24)
            since_7d = _dt.utcnow() - timedelta(days=7)
            new_24h = db.query(Job).filter(Job.discovered_at >= since_24h).count()
            new_7d = db.query(Job).filter(Job.discovered_at >= since_7d).count()
            high_match = db.query(Job).filter(Job.match_score >= 0.5,
                                              Job.discovered_at >= since_7d,
                                              Job.is_local == True).count()  # noqa: E712
            watchlist_recent = db.query(Job).filter(Job.tags.like("%watchlist%"),
                                                    Job.discovered_at >= since_7d).count()
            applied_7d = db.query(Job).filter(Job.status == "applied",
                                              Job.discovered_at >= since_7d).count()

            # Distinct sources for dropdown
            sources = sorted({j.source.split(":")[0] for j in db.query(Job.source).all() if j.source})

        return render_template(
            "index.html",
            jobs=jobs,
            quota=daily_apply_quota_remaining(),
            status_filter=status,
            scope=scope,
            min_score=min_score,
            source=source,
            company_q=company_q,
            text_q=text_q,
            watchlist_only=watchlist_only,
            remote_ok=remote_ok,
            sources=sources,
            starred_only=starred_only,
            preset=preset,
            sort=sort,
            sort_dir=sort_dir,
            columns=columns,
            cur_args=_cur,
            page=page,
            num_pages=num_pages,
            shown_count=shown_count,
            effective_locations=effective_locations,
            anywhere=anywhere,
            radius_miles=radius_miles,
            counts={"total": total, "local": local_count, "remote": remote_count,
                    "out_of_range": out_count, "starred": starred_count,
                    "expired": expired_count},
            summary={"new_24h": new_24h, "new_7d": new_7d, "high_match": high_match,
                     "watchlist_recent": watchlist_recent, "applied_7d": applied_7d},
            now=_dt.utcnow(),
            stale_days=settings.stale_after_days,
        )

    @app.route("/search-builder")
    def search_builder():
        """Build clickable LinkedIn/Glassdoor/Google/Indeed search URLs from
        the configured keywords + custom location/keyword overrides."""
        from urllib.parse import quote_plus
        kw = (request.args.get("keywords") or ",".join(settings.keywords[:3])).strip()
        loc = (request.args.get("location") or settings.applicant_location).strip()
        kw_q = quote_plus(kw)
        loc_q = quote_plus(loc)
        links = {
            "LinkedIn": f"https://www.linkedin.com/jobs/search/?keywords={kw_q}&location={loc_q}&f_TPR=r604800",
            "LinkedIn (remote)": f"https://www.linkedin.com/jobs/search/?keywords={kw_q}&location=United%20States&f_WT=2&f_TPR=r604800",
            "Indeed": f"https://www.indeed.com/jobs?q={kw_q}&l={loc_q}&fromage=14",
            "Indeed (remote)": f"https://www.indeed.com/jobs?q={kw_q}&l=Remote&fromage=14",
            "Glassdoor": f"https://www.glassdoor.com/Job/jobs.htm?sc.keyword={kw_q}&locT=N&locName={loc_q}",
            "Google Jobs": f"https://www.google.com/search?q={kw_q}+jobs+near+{loc_q}&ibp=htl;jobs",
            "ZipRecruiter": f"https://www.ziprecruiter.com/jobs-search?search={kw_q}&location={loc_q}",
            "TechCareers": f"https://www.techcareers.com/jobs/?q={kw_q}&l={loc_q}",
            "Engineering Jobs": f"https://careers.engineering.org/jobs/?keywords={kw_q}&location={loc_q}",
            "Tech Staffing": f"https://jobs.techstaffing.org/jobs/?keywords={kw_q}&location={loc_q}",
            "Global Careers": f"https://www.globalcareers.com/jobs?q={kw_q}&where={loc_q}",
            "USAJobs.gov": f"https://www.usajobs.gov/Search/Results?k={kw_q}&l={loc_q}",
        }
        return render_template("search_builder.html", keywords=kw, location=loc, links=links)

    @app.route("/job/<int:job_id>")
    def job_detail(job_id: int):
        with session() as db:
            job = db.get(Job, job_id)
            if not job:
                abort(404)
            apps = list(job.applications)
        try:
            contacts = json.loads(job.contacts_json or "[]")
        except Exception:  # noqa: BLE001
            contacts = []
        # Auto-populate contacts on first view when Google CSE is configured.
        # Gated on cse_configured() so it never runs the slow (bot-blocked)
        # HTML fallback on a page load.
        if not contacts and getattr(settings, "contacts_autofind", True):
            try:
                from .contacts import search_configured, find_contacts
                if search_configured():
                    contacts = find_contacts(job_id, save=True)
            except Exception:  # noqa: BLE001
                contacts = []
        try:
            from .contacts import networking_targets
            targets = networking_targets(job_id)
        except Exception:  # noqa: BLE001
            targets = []
        from pathlib import Path
        tailored_resume_text = ""
        cover_letter_text = ""
        cover_letter_docx_rel = ""
        resume_docx_rel = ""
        latest_app = apps[-1] if apps else None
        latest_app_id = latest_app.id if latest_app else None
        out_dir = Path(settings.output_dir)
        resume_pdf_rel = None

        for a in reversed(apps):
            if a.tailored_resume_path:
                try:
                    p = Path(a.tailored_resume_path)
                    # Check for docx
                    if p.suffix == ".docx" and p.exists():
                        if not resume_docx_rel:
                            resume_docx_rel = p.name
                    else:
                        docx_adj = p.with_suffix(".docx")
                        if docx_adj.exists() and not resume_docx_rel:
                            resume_docx_rel = docx_adj.name
                        else:
                            name_adj = p.with_name(p.stem.replace("_resume", "") + "_resume.docx")
                            if name_adj.exists() and not resume_docx_rel:
                                resume_docx_rel = name_adj.name

                    # Check for pdf
                    pdf_adj = p.with_suffix(".pdf")
                    if pdf_adj.exists() and not resume_pdf_rel:
                        resume_pdf_rel = pdf_adj.name
                    else:
                        name_pdf = p.with_name(p.stem.replace("_resume", "") + "_resume.pdf")
                        if name_pdf.exists() and not resume_pdf_rel:
                            resume_pdf_rel = name_pdf.name

                    # Check for markdown text
                    if p.suffix == ".md" and p.exists():
                        if not tailored_resume_text:
                            tailored_resume_text = p.read_text(encoding="utf-8")
                    else:
                        md_adj = p.with_suffix(".md")
                        if md_adj.exists() and not tailored_resume_text:
                            tailored_resume_text = md_adj.read_text(encoding="utf-8")
                except Exception:
                    pass

            if a.cover_letter_path:
                try:
                    cp = Path(a.cover_letter_path)
                    if cp.exists() and not cover_letter_text:
                        cover_letter_text = cp.read_text(encoding="utf-8")
                    docx_adj = cp.with_name(cp.name.replace("_cover.txt", "_cover_letter.docx"))
                    if docx_adj.exists() and not cover_letter_docx_rel:
                        cover_letter_docx_rel = docx_adj.name
                    elif cp.suffix == ".docx" and cp.exists() and not cover_letter_docx_rel:
                        cover_letter_docx_rel = cp.name
                except Exception:
                    pass

        # Check output directory directly if not found on application records
        if out_dir.exists():
            if not resume_docx_rel:
                for match in out_dir.glob(f"{job.id}_*resume.docx"):
                    resume_docx_rel = match.name
                    break
            if not resume_pdf_rel:
                for match in out_dir.glob(f"{job.id}_*resume.pdf"):
                    resume_pdf_rel = match.name
                    break
            if not tailored_resume_text:
                for match in out_dir.glob(f"{job.id}_*resume.md"):
                    try:
                        tailored_resume_text = match.read_text(encoding="utf-8")
                    except Exception:
                        pass
                    break
            if not cover_letter_docx_rel:
                for match in out_dir.glob(f"{job.id}_*cover_letter.docx"):
                    cover_letter_docx_rel = match.name
                    break
            if not cover_letter_text:
                for match in out_dir.glob(f"{job.id}_*cover.txt"):
                    try:
                        cover_letter_text = match.read_text(encoding="utf-8")
                    except Exception:
                        pass
                    break

        # On-the-fly generation if resume markdown text exists but docx does not exist yet
        if tailored_resume_text and not resume_docx_rel:
            try:
                from .resume import markdown_to_docx
                from .prompt_pack import _slug
                slug = f"{job.id}_{_slug(job.company)}_{_slug(job.title)}"[:120]
                auto_docx = out_dir / f"{slug}_resume.docx"
                markdown_to_docx(tailored_resume_text, auto_docx)
                if auto_docx.exists():
                    resume_docx_rel = auto_docx.name
            except Exception:
                pass

        # On-the-fly generation if cover letter text exists but docx does not exist yet
        if cover_letter_text and not cover_letter_docx_rel:
            try:
                from .resume import cover_letter_to_docx
                from .prompt_pack import _slug
                slug = f"{job.id}_{_slug(job.company)}_{_slug(job.title)}"[:120]
                auto_cover_docx = out_dir / f"{slug}_cover_letter.docx"
                rec_info = {"company": job.company, "title": job.title, "location": job.location}
                cover_letter_to_docx(cover_letter_text, auto_cover_docx, recipient_info=rec_info)
                if auto_cover_docx.exists():
                    cover_letter_docx_rel = auto_cover_docx.name
            except Exception:
                pass

        return render_template("job.html", job=job, applications=apps,
                               contacts=contacts, networking_targets=targets,
                               tailored_resume_text=tailored_resume_text,
                               cover_letter_text=cover_letter_text,
                               cover_letter_docx_rel=cover_letter_docx_rel,
                               resume_docx_rel=resume_docx_rel,
                               resume_pdf_rel=resume_pdf_rel,
                               latest_app_id=latest_app_id)

    @app.route("/job/<int:job_id>/inspect")
    def job_inspect(job_id: int):
        """Asynchronously return the rich split-screen Inspector partial for a job."""
        from .tailoring_studio import analyze_job_skills
        with session() as db:
            job = db.get(Job, job_id)
            if not job:
                abort(404)
            apps = list(job.applications)
            latest_app = apps[-1] if apps else None
            skills_analysis = analyze_job_skills(job.description or "")
            return render_template(
                "partials/job_inspector_pane.html",
                job=job,
                skills_analysis=skills_analysis,
                latest_app=latest_app,
            )

    @app.route("/job/<int:job_id>/status-api", methods=["POST"])
    def job_status_api(job_id: int):
        """Quick status and star toggle via AJAX for keyboard triage & inspector."""
        data = request.get_json(silent=True) or request.form.to_dict() or {}
        new_status = data.get("status")
        with session() as db:
            job = db.get(Job, job_id)
            if not job:
                return jsonify({"error": "Job not found"}), 404
            if new_status:
                job.status = new_status
                if new_status == "starred":
                    job.starred = True
                elif new_status == "skipped":
                    job.starred = False
            db.commit()
            return jsonify({
                "ok": True,
                "job_id": job.id,
                "status": job.status,
                "starred": bool(job.starred),
            })

    @app.route("/tailor-studio/<int:job_id>")
    def tailor_studio(job_id: int):
        from .tailoring_studio import analyze_job_skills, compute_resume_diff, get_base_resume
        from . import source_fidelity
        from .strategic_angles import get_available_angles, auto_detect_angle
        with session() as db:
            job = db.get(Job, job_id)
            if not job:
                abort(404)
            app_record = db.query(Application).filter_by(job_id=job_id).order_by(Application.created_at.desc()).first()

        base_md = get_base_resume()
        tailored_md = ""
        if app_record and app_record.tailored_resume_path:
            p = Path(app_record.tailored_resume_path)
            if p.suffix == ".md" and p.exists():
                tailored_md = p.read_text(encoding="utf-8")
            elif p.suffix == ".docx":
                adj_md = p.with_suffix(".md")
                if adj_md.exists():
                    tailored_md = adj_md.read_text(encoding="utf-8")

        skills_analysis = analyze_job_skills(job.description or "", tailored_resume_md=tailored_md or base_md)
        diff_lines = compute_resume_diff(base_md, tailored_md) if tailored_md else []
        fidelity = source_fidelity.check_fidelity(base_md, tailored_md) if tailored_md else None
        default_mode = getattr(settings, "ai_provider", "ollama")
        available_angles = get_available_angles()
        detected_angle = auto_detect_angle(job.title or "", job.description or "")

        return render_template(
            "tailor_studio.html",
            job=job,
            app=app_record,
            base_md=base_md,
            tailored_md=tailored_md or base_md,
            skills_analysis=skills_analysis,
            diff_lines=diff_lines,
            fidelity=fidelity,
            default_mode=default_mode,
            strategic_angles=available_angles,
            detected_angle=detected_angle,
        )

    @app.route("/tailor-studio/<int:job_id>/generate", methods=["POST"])
    def tailor_studio_generate(job_id: int):
        data = request.get_json(silent=True) or {}
        mode = data.get("mode", "ollama")
        angle = data.get("angle", "balanced")
        page_budget = data.get("page_budget", "2-page")

        from .strategic_angles import get_angle_guidance
        angle_guidance = get_angle_guidance(angle)

        try:
            if mode == "offline":
                from .prompt_pack import tailor_resume_offline, _slug
                from .resume import markdown_to_docx
                with session() as db:
                    job = db.get(Job, job_id)
                    if not job:
                        return jsonify({"ok": False, "error": "Job not found"}), 404
                    title, company, jd = job.title, job.company, job.description or ""
                if angle_guidance:
                    jd = f"{jd}\n\n{angle_guidance}"
                md = tailor_resume_offline(title, company, jd)
                out_dir = Path(settings.output_dir)
                out_dir.mkdir(parents=True, exist_ok=True)
                slug = f"{job_id}_{_slug(company)}_{_slug(title)}"[:120]
                md_path = out_dir / f"{slug}_resume.md"
                docx_path = out_dir / f"{slug}_resume.docx"
                md_path.write_text(md, encoding="utf-8")
                markdown_to_docx(md, docx_path, page_budget=page_budget)
                with session() as db:
                    app_record = db.query(Application).filter_by(job_id=job_id).first()
                    if not app_record:
                        app_record = Application(job_id=job_id, status="tailored")
                        db.add(app_record)
                    app_record.tailored_resume_path = str(docx_path)
                    job_ref = db.get(Job, job_id)
                    if job_ref:
                        job_ref.status = "tailored"
                    db.commit()
                return jsonify({"ok": True, "mode": "offline", "angle": angle, "page_budget": page_budget})

            elif mode == "cloud":
                from .pipeline import tailor_for_job
                res = tailor_for_job(job_id, angle_guidance=angle_guidance, page_budget=page_budget)
                return jsonify({"ok": True, "mode": "cloud", "result": res, "angle": angle, "page_budget": page_budget})

            else:  # 'ollama' or local
                from .pipeline import tailor_for_job_local
                res = tailor_for_job_local(job_id, rounds=1, angle_guidance=angle_guidance, page_budget=page_budget)
                return jsonify({"ok": True, "mode": "ollama", "result": res, "angle": angle, "page_budget": page_budget})

        except Exception as e:
            log.exception("Tailoring in studio failed: %s", e)
            return jsonify({"ok": False, "error": str(e)}), 500

    @app.route("/tailor-studio/<int:job_id>/save", methods=["POST"])
    def tailor_studio_save(job_id: int):
        data = request.get_json(silent=True) or {}
        content = data.get("content", "").strip()
        page_budget = data.get("page_budget", "2-page")
        if not content:
            return jsonify({"ok": False, "error": "Content cannot be empty"}), 400

        from .prompt_pack import _slug
        from .resume import markdown_to_docx
        with session() as db:
            job = db.get(Job, job_id)
            if not job:
                return jsonify({"ok": False, "error": "Job not found"}), 404
            app_record = db.query(Application).filter_by(job_id=job_id).order_by(Application.created_at.desc()).first()
            if not app_record:
                app_record = Application(job_id=job_id, status="tailored")
                db.add(app_record)

            out_dir = Path(settings.output_dir)
            out_dir.mkdir(parents=True, exist_ok=True)
            slug = f"{job_id}_{_slug(job.company)}_{_slug(job.title)}"[:120]
            md_path = out_dir / f"{slug}_resume.md"
            docx_path = out_dir / f"{slug}_resume.docx"
            md_path.write_text(content, encoding="utf-8")
            try:
                markdown_to_docx(content, docx_path, page_budget=page_budget)
                app_record.tailored_resume_path = str(docx_path)
            except Exception as e:
                log.warning("markdown_to_docx failed: %s", e)
                app_record.tailored_resume_path = str(md_path)

            db.commit()

        return jsonify({"ok": True, "page_budget": page_budget})

    @app.route("/api/tailor/polish-bullet", methods=["POST"])
    def api_polish_bullet():
        data = request.get_json(silent=True) or {}
        bullet = data.get("bullet", "").strip()
        action = data.get("action", "action_verb")
        job_id = data.get("job_id")
        if not bullet:
            return jsonify({"ok": False, "error": "No bullet provided"}), 400

        job_context = ""
        if job_id:
            try:
                with session() as db:
                    j = db.get(Job, int(job_id))
                    if j:
                        job_context = f"{j.title} at {j.company}\n{j.description or ''}"[:1500]
            except Exception:
                pass

        from .ollama_client import polish_bullet
        res = polish_bullet(bullet, action=action, job_context=job_context)
        return jsonify({"ok": True, "result": res})

    @app.route("/api/job/<int:job_id>/ats-breakdown")
    def api_job_ats_breakdown(job_id: int):
        from .tailoring_studio import analyze_job_skills, get_base_resume
        with session() as db:
            job = db.get(Job, job_id)
            if not job:
                return jsonify({"ok": False, "error": "Job not found"}), 404
            app_record = db.query(Application).filter_by(job_id=job_id).order_by(Application.created_at.desc()).first()

        tailored_md = ""
        if app_record and app_record.tailored_resume_path:
            p = Path(app_record.tailored_resume_path)
            if p.suffix == ".md" and p.exists():
                tailored_md = p.read_text(encoding="utf-8")
            elif p.suffix == ".docx":
                adj_md = p.with_suffix(".md")
                if adj_md.exists():
                    tailored_md = adj_md.read_text(encoding="utf-8")

        base_md = get_base_resume()
        resume_md = tailored_md or base_md
        analysis = analyze_job_skills(job.description or "", tailored_resume_md=resume_md)
        return jsonify({
            "ok": True,
            "job_id": job.id,
            "title": job.title,
            "company": job.company,
            "match_percentage": analysis.get("match_percentage", 0),
            "matched_skills": analysis.get("matched_skills", []),
            "transferable_skills": analysis.get("transferable_skills", []),
            "honest_gaps": analysis.get("honest_gaps", []),
            "section_heatmap": analysis.get("section_heatmap", {}),
        })

    @app.route("/download/app/<int:app_id>/pdf")
    def download_app_pdf(app_id: int):
        with session() as db:
            app_record = db.get(Application, app_id)
            if not app_record or not app_record.tailored_resume_path:
                abort(404)
            out_dir = Path(settings.output_dir).resolve()
            raw_path = Path(app_record.tailored_resume_path)
            p = raw_path if raw_path.is_absolute() else (out_dir / raw_path.name).resolve()
            pdf_path = p.with_suffix(".pdf")
            if pdf_path.exists():
                return send_file(str(pdf_path), as_attachment=True, download_name=pdf_path.name)
            md_path = p.with_suffix(".md")
            if not md_path.exists():
                md_path = p.with_name(p.stem.replace("_resume", "") + "_resume.md")
            if md_path.exists():
                try:
                    from .resume import markdown_to_pdf
                    markdown_to_pdf(md_path.read_text(encoding="utf-8"), pdf_path)
                    if pdf_path.exists():
                        return send_file(str(pdf_path), as_attachment=True, download_name=pdf_path.name)
                except Exception as e:
                    log.warning("On-the-fly markdown_to_pdf failed: %s", e)
            abort(404)

    @app.route("/download/app/<int:app_id>/docx")
    def download_app_docx(app_id: int):
        with session() as db:
            app_record = db.get(Application, app_id)
            if not app_record or not app_record.tailored_resume_path:
                abort(404)
            out_dir = Path(settings.output_dir).resolve()
            raw_path = Path(app_record.tailored_resume_path)
            p = raw_path if raw_path.is_absolute() else (out_dir / raw_path.name).resolve()
            if p.suffix == ".docx" and p.exists():
                return send_file(str(p), as_attachment=True, download_name=p.name)
            docx_adj = p.with_suffix(".docx")
            if docx_adj.exists():
                return send_file(str(docx_adj), as_attachment=True, download_name=docx_adj.name)
            name_adj = p.with_name(p.stem.replace("_resume", "") + "_resume.docx")
            if name_adj.exists():
                return send_file(str(name_adj), as_attachment=True, download_name=name_adj.name)
            # On-the-fly fallback from markdown
            md_path = p if p.suffix == ".md" else p.with_suffix(".md")
            if not md_path.exists():
                md_path = p.with_name(p.stem.replace("_resume", "") + "_resume.md")
            if md_path.exists():
                try:
                    from .resume import markdown_to_docx
                    target_docx = p.with_suffix(".docx")
                    markdown_to_docx(md_path.read_text(encoding="utf-8"), target_docx)
                    if target_docx.exists():
                        return send_file(str(target_docx), as_attachment=True, download_name=target_docx.name)
                except Exception as e:
                    log.warning("On-the-fly markdown_to_docx failed: %s", e)
            abort(404)

    @app.route("/download/app/<int:app_id>/cover-docx")
    def download_app_cover_docx(app_id: int):
        with session() as db:
            app_record = db.get(Application, app_id)
            if not app_record or not app_record.cover_letter_path:
                abort(404)
            job = db.get(Job, app_record.job_id)
            out_dir = Path(settings.output_dir).resolve()
            raw_cp = Path(app_record.cover_letter_path)
            cp = raw_cp if raw_cp.is_absolute() else (out_dir / raw_cp.name).resolve()
            docx_path = cp.with_name(cp.name.replace("_cover.txt", "_cover_letter.docx"))
            if docx_path.exists():
                return send_file(str(docx_path), as_attachment=True, download_name=docx_path.name)
            if cp.suffix == ".docx" and cp.exists():
                return send_file(str(cp), as_attachment=True, download_name=cp.name)
            # On-the-fly fallback from cover.txt
            if cp.exists():
                try:
                    from .resume import cover_letter_to_docx
                    rec_info = {"company": job.company, "title": job.title, "location": job.location} if job else None
                    cover_letter_to_docx(cp.read_text(encoding="utf-8"), docx_path, recipient_info=rec_info)
                    if docx_path.exists():
                        return send_file(str(docx_path), as_attachment=True, download_name=docx_path.name)
                except Exception as e:
                    log.warning("On-the-fly cover_letter_to_docx failed: %s", e)
            abort(404)

    @app.route("/download/job/<int:job_id>/interview-docx")
    def download_job_interview_docx(job_id: int):
        with session() as db:
            job = db.get(Job, job_id)
            if not job:
                abort(404)
            from .prompt_pack import _slug
            from .resume import markdown_to_docx
            from .interview_coach import render_pack, _empty_bank
            out_dir = Path(settings.output_dir).resolve()
            out_dir.mkdir(parents=True, exist_ok=True)
            slug = f"{job.id}_{_slug(job.company)}_{_slug(job.title)}"[:120]
            docx_path = out_dir / f"{slug}_interview_prep.docx"
            md_path = out_dir / f"{slug}_interview_prep.md"
            if docx_path.exists():
                return send_file(str(docx_path), as_attachment=True, download_name=docx_path.name)
            if md_path.exists():
                try:
                    markdown_to_docx(md_path.read_text(encoding="utf-8"), docx_path)
                    if docx_path.exists():
                        return send_file(str(docx_path), as_attachment=True, download_name=docx_path.name)
                except Exception as e:
                    log.warning("interview_prep markdown_to_docx failed: %s", e)
            if job.interview_prep:
                try:
                    bank_raw = json.loads(job.interview_question_bank or "{}")
                    bank = bank_raw if isinstance(bank_raw, dict) else _empty_bank()
                except Exception:
                    bank = _empty_bank()
                try:
                    md_content = render_pack(job, bank)
                except Exception:
                    md_content = f"# Interview Prep — {job.title} @ {job.company}\n\n{job.interview_prep}"
                try:
                    markdown_to_docx(md_content, docx_path)
                    if docx_path.exists():
                        return send_file(str(docx_path), as_attachment=True, download_name=docx_path.name)
                except Exception as e:
                    log.warning("interview_prep render_pack docx failed: %s", e)
            abort(404)

    @app.route("/scrape", methods=["POST"])
    def scrape():
        """Kick off a scrape in the background and return immediately so the
        dashboard can poll /scrape/status for the progress bar."""
        import threading
        from .scrape_progress import progress as p
        if p.snapshot()["state"] == "running":
            flash("A scrape is already running.", "error")
            return redirect(url_for("index"))

        def _runner():
            try:
                run_scrape_cycle(send_digest=False)
            except Exception:  # progress.fail already wrote the error
                pass

        threading.Thread(target=_runner, daemon=True, name="jobbot-scrape").start()
        flash("Scrape started in the background. Refresh the page in a few minutes for new jobs.", "success")
        return redirect(url_for("index"))

    @app.route("/scrape/status")
    def scrape_status():
        from flask import jsonify
        from .scrape_progress import progress as p
        return jsonify(p.snapshot())

    @app.route("/liveness/sweep", methods=["POST"])
    def liveness_sweep():
        """Kick off a liveness sweep in the background so the dashboard can poll /liveness/status."""
        import threading
        from . import liveness
        if liveness.progress.snapshot()["state"] == "running":
            flash("A liveness sweep is already running.", "error")
            return redirect(request.referrer or url_for("index"))

        def _runner():
            try:
                liveness.sweep_all()
            except Exception as e:
                liveness.progress.fail(str(e))

        threading.Thread(target=_runner, daemon=True, name="jobbot-liveness").start()
        flash("Liveness check started in the background. Expired jobs will be flagged automatically.", "success")
        return redirect(request.referrer or url_for("index"))

    @app.route("/liveness/status")
    def liveness_status():
        from flask import jsonify
        from . import liveness
        return jsonify(liveness.progress.snapshot())

    @app.route("/interview/pending", methods=["POST"])
    def interview_pending():
        """Kick the background pack builder for interview-flagged jobs."""
        import threading

        from . import interview_coach as coach

        def _runner():
            try:
                coach.generate_pending()
            except Exception:  # noqa: BLE001 - coach is fail-open; never crash the thread
                pass

        threading.Thread(target=_runner, daemon=True, name="jobbot-interview-pack").start()
        flash("Building interview prep in the background.", "success")
        return redirect(request.referrer or url_for("index"))

    @app.route("/models")
    def models_route():
        from . import model_registry as mr
        rows = mr.list_selectable()
        provider, model = mr.current_selection()
        return render_template("models.html", rows=rows,
                               current_provider=provider, current_model=model,
                               pull=_pull_run)

    @app.route("/models/set", methods=["POST"])
    def models_set():
        from . import model_registry as mr
        provider, _, model = (request.form.get("selection", "")).partition("::")
        try:
            mr.set_selection(provider, model)
            flash(f"Active model set to {model} ({provider}).", "success")
        except Exception as exc:  # noqa: BLE001
            flash(f"Could not set model: {exc}", "error")
        return redirect(url_for("models_route"))

    @app.route("/models/pull", methods=["POST"])
    def models_pull():
        import threading
        from . import model_registry as mr
        name = (request.form.get("model_name", "") or "").strip()
        if not name:
            flash("Enter a model name to pull.", "error")
            return redirect(url_for("models_route"))
        with _pull_lock:
            if _pull_run["running"]:
                flash("A pull is already running.", "error")
                return redirect(url_for("models_route"))
            _pull_run.update(running=True, model=name, lines=[], ok=None)

        def _runner():
            def _prog(line: str) -> None:
                _pull_run["lines"] = (_pull_run["lines"] + [line])[-50:]
            ok = mr.pull_ollama_model(name, progress=_prog)
            _pull_run.update(running=False, ok=ok)

        threading.Thread(target=_runner, daemon=True, name="jobbot-pull").start()
        flash(f"Pulling {name} in the background. Watch progress below.", "success")
        return redirect(url_for("models_route"))

    @app.route("/models/pull/status")
    def models_pull_status():
        return jsonify(_pull_run)

    @app.route("/tailor/<int:job_id>", methods=["POST"])
    def tailor(job_id: int):
        try:
            result = tailor_for_job(job_id)
            # Cloud path can return a stub (no application) when credits are
            # depleted / all providers error. Guarantee a real resume by
            # falling back to the local Ollama iterative path (-> Word doc).
            if not result.get("application_id"):
                log.warning("Cloud tailoring produced no resume (%s); using local Ollama.",
                            result.get("error", "no provider"))
                result = tailor_for_job_local(job_id)
                flash(f"Cloud AI unavailable — tailored locally with Ollama "
                      f"(app #{result['application_id']}, Word doc generated).", "success")
            else:
                flash(f"Tailored resume + cover letter generated "
                      f"(app #{result['application_id']}).", "success")
        except Exception as e:  # noqa: BLE001
            flash(f"Tailoring failed: {e}", "error")
        return redirect(url_for("job_detail", job_id=job_id))

    @app.route("/tailor-offline/<int:job_id>", methods=["POST"])
    def tailor_offline_route(job_id: int):
        from pathlib import Path
        from .prompt_pack import tailor_resume_offline, _slug
        from .resume import markdown_to_docx
        with session() as db:
            job = db.get(Job, job_id)
            if not job:
                abort(404)
            title, company, jd = job.title, job.company, job.description or ""

        try:
            md = tailor_resume_offline(title, company, jd)
            out_dir = Path(settings.output_dir)
            out_dir.mkdir(parents=True, exist_ok=True)
            slug = f"{job_id}_{_slug(company)}_{_slug(title)}"[:120]
            md_path = out_dir / f"{slug}_offline_resume.md"
            docx_path = out_dir / f"{slug}_offline_resume.docx"
            md_path.write_text(md, encoding="utf-8")
            try:
                markdown_to_docx(md, docx_path)
                docx_saved = True
            except Exception as e:
                docx_saved = False
                log.warning("markdown_to_docx failed: %s", e)

            with session() as db:
                app_record = db.query(Application).filter_by(job_id=job_id).first()
                if not app_record:
                    app_record = Application(job_id=job_id, status="tailored")
                    db.add(app_record)
                app_record.tailored_resume_path = str(docx_path if docx_saved else md_path)
                job_ref = db.get(Job, job_id)
                if job_ref and job_ref.status == "new":
                    job_ref.status = "tailored"
                db.commit()

            flash(f"Zero-AI offline resume generated ({'DOCX' if docx_saved else 'Markdown'}).", "success")
        except Exception as e:
            flash(f"Offline tailoring failed: {e}", "error")

        return redirect(url_for("job_detail", job_id=job_id))

    @app.route("/job/add", methods=["GET", "POST"])
    def job_add():
        if request.method == "GET":
            return render_template("add_job.html")

        url = (request.form.get("url") or "").strip()
        title = (request.form.get("title") or "").strip()
        company = (request.form.get("company") or "").strip()
        location = (request.form.get("location") or "").strip()
        description = (request.form.get("description") or "").strip()
        tailor_mode = (request.form.get("tailor_mode") or "none").strip()

        from .cli import _extract_meta_from_url, _fetch_description_from_url

        if not title or not company:
            inferred_title, inferred_company = _extract_meta_from_url(url)
            if not title:
                title = inferred_title or "Position"
            if not company:
                company = inferred_company or "Target Company"

        if url and not description:
            description = _fetch_description_from_url(url)

        if not description and not url:
            flash("Please provide at least a job URL or description.", "error")
            return redirect(url_for("job_add"))

        from .cli import _insert_manual_job
        job = _insert_manual_job(url=url, description=description, title=title, company=company, location=location)
        flash(f"Job #{job.id} added: {job.title} @ {job.company}", "success")

        if tailor_mode == "offline":
            return redirect(url_for("tailor_offline_route", job_id=job.id), code=307)
        elif tailor_mode == "cloud":
            return redirect(url_for("tailor", job_id=job.id), code=307)

        return redirect(url_for("job_detail", job_id=job.id))

    @app.route("/job/<int:job_id>/edit", methods=["POST"])
    def job_edit(job_id: int):
        with session() as db:
            job = db.get(Job, job_id)
            if not job:
                abort(404)
            if "title" in request.form:
                job.title = (request.form.get("title") or job.title).strip()
            if "company" in request.form:
                job.company = (request.form.get("company") or job.company).strip()
            if "location" in request.form:
                job.location = (request.form.get("location") or "").strip()
            if "salary" in request.form:
                job.salary = (request.form.get("salary") or "").strip()
            if "url" in request.form:
                job.url = (request.form.get("url") or job.url).strip()
            if "description" in request.form and request.form.get("description", "").strip():
                job.description = request.form.get("description", "").strip()
            if "notes" in request.form:
                job.notes = (request.form.get("notes") or "")[:5000]
            new_status = (request.form.get("status") or "").strip()
            if new_status:
                job.status = new_status
            if "starred" in request.form:
                job.starred = request.form.get("starred") in ("1", "true", "True", "on")
            db.commit()
        flash(f"Job #{job_id} updated.", "success")
        return redirect(url_for("job_detail", job_id=job_id))

    @app.route("/job/<int:job_id>/delete", methods=["POST"])
    def job_delete(job_id: int):
        with session() as db:
            job = db.get(Job, job_id)
            if not job:
                abort(404)
            title = job.title
            company = job.company
            from .models import OutreachDraft, OutreachThread, InterviewSession
            db.query(OutreachDraft).filter_by(job_id=job_id).delete()
            db.query(OutreachThread).filter_by(job_id=job_id).delete()
            db.query(InterviewSession).filter_by(job_id=job_id).delete()
            db.delete(job)
            db.commit()
        flash(f"Job #{job_id} ({title} @ {company}) deleted.", "success")
        return redirect(url_for("index"))

    @app.route("/chat")
    def chat_page():
        job_id_arg = request.args.get("job_id", "")
        selected_job = None
        with session() as db:
            jobs = db.query(Job).order_by(Job.starred.desc(), Job.discovered_at.desc()).limit(150).all()
            if job_id_arg and job_id_arg.isdigit():
                selected_job = db.get(Job, int(job_id_arg))
        return render_template("chat.html", jobs=jobs, selected_job=selected_job)

    @app.route("/chat/api/message", methods=["POST"])
    def chat_api_message():
        data = request.get_json(force=True) or {}
        messages = data.get("messages", [])
        job_id = data.get("job_id")
        enable_web = bool(data.get("enable_web", True))
        from . import chatbot
        try:
            result = chatbot.chat_turn(messages=messages, job_id=job_id, enable_web=enable_web)
            return jsonify(result)
        except Exception as exc:
            log.exception("Chatbot API error: %s", exc)
            return jsonify({"error": str(exc)}), 500

    @app.route("/chat/api/evaluate/<int:job_id>", methods=["POST"])
    def chat_api_evaluate(job_id: int):
        enable_web = request.args.get("enable_web", "true").lower() in ("true", "1", "yes")
        from . import chatbot
        try:
            result = chatbot.evaluate_job(job_id=job_id, enable_web=enable_web)
            return jsonify(result)
        except Exception as exc:
            log.exception("Job evaluation API error: %s", exc)
            return jsonify({"error": str(exc)}), 500

    @app.route("/chat/api/history/<int:job_id>")
    def chat_api_history(job_id: int):
        from . import chatbot
        return jsonify(chatbot.load_chat_history(job_id))

    @app.route("/apply/<int:app_id>", methods=["POST"])
    def apply(app_id: int):
        to = (request.form.get("to_email") or "").strip()
        note = request.form.get("note", "")
        if not to:
            flash("Recipient email required.", "error")
            return redirect(request.referrer or url_for("index"))
        try:
            send_application(app_id, to, note)
            flash("Application sent.", "success")
        except Exception as e:  # noqa: BLE001
            flash(f"Send failed: {e}", "error")
        return redirect(request.referrer or url_for("index"))

    @app.route("/interview/<int:job_id>", methods=["GET", "POST"])
    def interview(job_id: int):
        from . import interview_coach as coach
        with session() as db:
            job = db.get(Job, job_id)
            if not job:
                abort(404)
            pending = job.interview_prep_pending
            has_bank = bool(job.interview_question_bank)

        if request.method == "POST":
            result = coach.generate_pack(job_id)
            if result["error"]:
                flash(f"Prep generation had a problem: {result['error']}", "error")
            else:
                flash("Interview prep generated.", "success")
            return redirect(url_for("interview", job_id=job_id))

        # Lazy fallback: the background worker did not run (CLI-only flow, crash,
        # or a freshly-flagged job). Generate inline rather than show an empty page.
        if pending or not has_bank:
            coach.generate_pack(job_id)

        with session() as db:
            job = db.get(Job, job_id)
            bank = coach.load_bank(job)
            contact = coach.matched_contact(job)
            prep_html = _md_to_html(job.interview_prep or "")
            intel_html = _md_to_html(job.company_intel or "")
            bridges_html = _md_to_html(coach._bridges(bank))
            ask_html = _md_to_html(coach._questions_to_ask(bank, job.company))
        return render_template("interview.html", job=job, bank=bank,
                               contact=contact, prep_html=prep_html,
                               intel_html=intel_html, bridges_html=bridges_html,
                               ask_html=ask_html)

    @app.route("/interview/<int:job_id>/practice", methods=["POST"])
    def practice_start(job_id: int):
        from flask import jsonify

        from . import interview_coach as coach
        mode = (request.form.get("mode") or
                (request.get_json(silent=True) or {}).get("mode") or "general")
        raw_contact = (request.form.get("contact_id") or
                       (request.get_json(silent=True) or {}).get("contact_id"))
        contact_id = int(raw_contact) if raw_contact else None
        try:
            with session() as db:
                return jsonify(coach.start(db, job_id, mode, contact_id))
        except Exception as e:  # noqa: BLE001
            return jsonify({"session_id": 0, "question": None, "error": str(e)})

    @app.route("/practice/<int:session_id>")
    def practice_page(session_id: int):
        from . import interview_coach as coach
        from .models import InterviewSession
        with session() as db:
            sess = db.get(InterviewSession, session_id)
            if sess is None:
                abort(404)
            job = db.get(Job, sess.job_id)
            rows = coach.transcript(db, session_id)
            contact = coach.matched_contact(job) if job is not None else None
            open_session = sess.ended_at is None
            summary_html = _md_to_html(sess.summary or "")
            overall = sess.overall_score
            mode = sess.mode
        return render_template("practice.html", job=job, session_id=session_id,
                               mode=mode, rows=rows, contact=contact,
                               open_session=open_session, overall=overall,
                               summary_html=summary_html)

    @app.route("/practice/<int:session_id>/answer", methods=["POST"])
    def practice_answer(session_id: int):
        from flask import jsonify

        from . import interview_coach as coach
        body = request.get_json(silent=True) or {}
        try:
            with session() as db:
                out = coach.submit_answer(db, int(body.get("turn_id") or 0),
                                          body.get("answer") or "")
                # A follow-up is asked BEFORE the next base question.
                out["next"] = (None if (out["follow_up"] or out["error"])
                               else coach.next_question(db, session_id))
                return jsonify(out)
        except Exception as e:  # noqa: BLE001
            return jsonify({"turn_id": 0, "score": None, "rubric": {},
                            "critique": "", "follow_up": None, "next": None,
                            "error": str(e)})

    @app.route("/practice/<int:session_id>/end", methods=["POST"])
    def practice_end(session_id: int):
        from flask import jsonify

        from . import interview_coach as coach
        try:
            with session() as db:
                return jsonify(coach.end(db, session_id))
        except Exception as e:  # noqa: BLE001
            return jsonify({"session_id": session_id, "overall_score": None,
                            "summary": "", "turns": 0, "error": str(e)})

    @app.route("/interviews")
    def interviews():
        from . import interview_coach as coach
        data = coach.history()
        return render_template("interviews.html", **data)

    @app.route("/limits", methods=["GET", "POST"])
    def limits_route():
        """Adjust per-scraper max-job parameters at runtime. Persists to .env."""
        from pathlib import Path
        knobs = [
            ("max_linkedin_pages",        "LinkedIn pages per (kw × loc)",        1, 30),
            ("max_indeed_pages",          "Indeed pages per (kw × loc)",          1, 20),
            ("max_workday_offset",        "Workday max offset per tenant",       20, 1000),
            ("max_google_jobs_urls",      "Google Jobs detail-fetch cap",         10, 1000),
            ("max_search_engine_results", "Bing/CSE results per query",           10, 200),
            ("scrape_radius_miles",       "Scrape radius around each city (mi)",   0, 500),
            ("max_scrape_locations",      "Max scrape cities per cycle",           1, 50),
            ("apply_rate_limit_per_day",  "Auto-apply daily quota",                1, 50),
            ("min_match_score",           "Minimum match score (in-range only)",   0, 1),
            ("schedule_interval_hours",   "Scheduler interval (hours)",            1, 24),
        ]
        env_path = Path(".env")
        if request.method == "POST":
            current = env_path.read_text(encoding="utf-8") if env_path.exists() else ""
            new_lines = []
            for name, _label, _lo, _hi in knobs:
                upper = name.upper()
                raw = request.form.get(name, "").strip()
                if not raw:
                    continue
                line = f"{upper}={raw}"
                if re.search(rf"^{upper}=.*$", current, re.MULTILINE):
                    current = re.sub(rf"^{upper}=.*$", line, current, flags=re.MULTILINE)
                else:
                    new_lines.append(line)
                # also update settings live for this process
                try:
                    val = float(raw) if "." in raw else int(raw)
                except ValueError:
                    val = raw
                setattr(settings, name, val)
            if new_lines:
                current = current.rstrip() + "\n\n# Adjusted via /limits\n" + "\n".join(new_lines) + "\n"
            env_path.write_text(current, encoding="utf-8")
            flash("Limits updated. Restart `jobbot serve` to pick up env changes everywhere.", "success")
            return redirect(url_for("limits_route"))
        # Build view payload
        rows = [{"name": n, "label": l, "lo": lo, "hi": hi, "value": getattr(settings, n)}
                for n, l, lo, hi in knobs]
        return render_template("limits.html", rows=rows)

    @app.route("/profile", methods=["GET", "POST"])
    def profile():
        from . import gemini_client as gc
        p = Path(settings.profile_path)
        if request.method == "POST":
            p.parent.mkdir(parents=True, exist_ok=True)
            p.write_text(request.form.get("content", ""), encoding="utf-8")
            gc.refresh_profile()
            flash("Profile saved — will be used in next AI generation.", "success")
            return redirect(url_for("profile"))
        content = p.read_text(encoding="utf-8") if p.exists() else ""
        return render_template("profile.html", content=content, path=str(p))

    @app.route("/skills", methods=["GET", "POST"])
    def skills_route():
        from . import skills as sk
        from . import ai_client
        p = Path(settings.skills_path)
        if request.method == "POST":
            p.parent.mkdir(parents=True, exist_ok=True)
            p.write_text(request.form.get("content", ""), encoding="utf-8")
            ai_client.refresh_profile()  # clears the cached applicant context
            flash("Skills saved — verified skills will inform matching + tailoring.",
                  "success")
            return redirect(url_for("skills_route"))
        content = p.read_text(encoding="utf-8") if p.exists() else ""
        parsed = sk.known_skills(sk.parse_skills(content), min_rank=1)
        return render_template("skills.html", content=content, path=str(p),
                               parsed=parsed)

    @app.route("/skip/<int:job_id>", methods=["POST"])
    def skip(job_id: int):
        with session() as db:
            job = db.get(Job, job_id)
            if job:
                job.status = "skipped"
                db.commit()
        return redirect(request.referrer or url_for("index"))

    @app.route("/star/<int:job_id>", methods=["POST"])
    def star(job_id: int):
        with session() as db:
            job = db.get(Job, job_id)
            if job:
                job.starred = not bool(job.starred)
                db.commit()
        return redirect(request.referrer or url_for("index"))

    @app.route("/reopen/<int:job_id>", methods=["POST"])
    def reopen(job_id: int):
        with session() as db:
            job = db.get(Job, job_id)
            if job:
                job.status = "new"
                db.commit()
        return redirect(request.referrer or url_for("index"))

    @app.route("/job/<int:job_id>/restore", methods=["POST"])
    def restore_job_route(job_id: int):
        from .liveness import restore_job
        restore_job(job_id)
        flash("Job restored to active.", "success")
        return redirect(request.referrer or url_for("index"))

    @app.route("/job/<int:job_id>/check-live", methods=["POST"])
    def check_live_route(job_id: int):
        from .liveness import check_job_alive, expire_job
        with session() as db:
            job = db.get(Job, job_id)
            if not job:
                abort(404)
            url = job.url
        alive, reason = check_job_alive(url)
        if alive:
            flash(f"Still live ({reason}).", "success")
        else:
            expire_job(job_id, reason)
            flash(f"Marked expired — {reason}.", "success")
        return redirect(request.referrer or url_for("job_detail", job_id=job_id))

    @app.route("/note/<int:job_id>", methods=["POST"])
    def save_note(job_id: int):
        with session() as db:
            job = db.get(Job, job_id)
            if job:
                job.notes = (request.form.get("notes") or "")[:5000]
                db.commit()
        flash("Note saved.", "success")
        return redirect(url_for("job_detail", job_id=job_id))

    @app.route("/locations", methods=["GET", "POST"])
    def locations_route():
        from . import user_prefs
        if request.method == "POST":
            raw = request.form.get("locations", "")
            saved = user_prefs.save_locations(raw)
            flash(f"Saved {len(saved)} locations.", "success")
            return redirect(url_for("index"))
        current = user_prefs.load_locations()
        return render_template("locations.html",
                               current=current,
                               raw_text="\n".join(current),
                               baseline=settings.locations,
                               scrape_radius=getattr(settings, "scrape_radius_miles", 0),
                               scrape_cities=user_prefs.scrape_locations())

    @app.route("/tags", methods=["GET", "POST"])
    def tags_route():
        from . import user_prefs
        if request.method == "POST":
            raw = request.form.get("tags", "")
            saved = user_prefs.save_tags(raw)
            flash(f"Saved {len(saved)} tags.", "success")
            return redirect(url_for("tags_route"))
        current = user_prefs.load_tags()
        return render_template("tags.html",
                               current=current,
                               raw_text="\n".join(current),
                               baseline=settings.keywords)

    @app.route("/bulk-skip", methods=["POST"])
    def bulk_skip():
        ids = [int(x) for x in request.form.getlist("ids") if x.isdigit()]
        if not ids:
            return redirect(request.referrer or url_for("index"))
        with session() as db:
            for j in db.query(Job).filter(Job.id.in_(ids)).all():
                j.status = "skipped"
            db.commit()
        flash(f"Skipped {len(ids)} jobs.", "success")
        return redirect(request.referrer or url_for("index"))

    @app.route("/jobs/bulk", methods=["POST"])
    def jobs_bulk():
        action = (request.form.get("action") or "").strip()
        ids = [int(x) for x in request.form.getlist("ids") if x.isdigit()]
        if not ids:
            flash("No jobs selected.", "error")
            return redirect(request.referrer or url_for("index"))

        with session() as db:
            jobs = db.query(Job).filter(Job.id.in_(ids)).all()
            if action == "star":
                for j in jobs:
                    j.starred = True
                db.commit()
                flash(f"Starred {len(jobs)} jobs.", "success")
            elif action == "unstar":
                for j in jobs:
                    j.starred = False
                db.commit()
                flash(f"Unstarred {len(jobs)} jobs.", "success")
            elif action == "skip":
                for j in jobs:
                    j.status = "skipped"
                db.commit()
                flash(f"Skipped {len(jobs)} jobs.", "success")
            elif action == "reopen":
                for j in jobs:
                    j.status = "new"
                db.commit()
                flash(f"Reopened {len(jobs)} jobs.", "success")
            elif action == "delete":
                from .models import OutreachDraft, OutreachThread, InterviewSession
                db.query(OutreachDraft).filter(OutreachDraft.job_id.in_(ids)).delete(synchronize_session=False)
                db.query(OutreachThread).filter(OutreachThread.job_id.in_(ids)).delete(synchronize_session=False)
                db.query(InterviewSession).filter(InterviewSession.job_id.in_(ids)).delete(synchronize_session=False)
                for j in jobs:
                    db.delete(j)
                db.commit()
                flash(f"Permanently deleted {len(jobs)} jobs.", "success")
            elif action == "tailor_offline":
                from pathlib import Path
                from .prompt_pack import tailor_resume_offline, _slug
                from .resume import markdown_to_docx
                out_dir = Path(settings.output_dir)
                out_dir.mkdir(parents=True, exist_ok=True)
                tailored_count = 0
                for j in jobs:
                    try:
                        md = tailor_resume_offline(j.title, j.company, j.description or "")
                        slug = f"{j.id}_{_slug(j.company)}_{_slug(j.title)}"[:120]
                        md_path = out_dir / f"{slug}_offline_resume.md"
                        docx_path = out_dir / f"{slug}_offline_resume.docx"
                        md_path.write_text(md, encoding="utf-8")
                        docx_saved = False
                        try:
                            markdown_to_docx(md, docx_path)
                            docx_saved = True
                        except Exception:
                            pass
                        app_record = db.query(Application).filter_by(job_id=j.id).first()
                        if not app_record:
                            app_record = Application(job_id=j.id, status="tailored")
                            db.add(app_record)
                        app_record.tailored_resume_path = str(docx_path if docx_saved else md_path)
                        if j.status == "new":
                            j.status = "tailored"
                        tailored_count += 1
                    except Exception as exc:
                        log.warning("Offline tailor failed for job %s: %s", j.id, exc)
                db.commit()
                flash(f"Tailored {tailored_count} resumes offline with zero AI.", "success")
            elif action == "check_live":
                from .liveness import check_job_alive, expire_job
                dead_count = 0
                for j in jobs:
                    alive, reason = check_job_alive(j.url)
                    if not alive:
                        expire_job(j.id, reason)
                        dead_count += 1
                flash(f"Checked liveness for {len(jobs)} jobs ({dead_count} expired).", "success")
            else:
                flash(f"Unknown bulk action: {action}", "error")

        return redirect(request.referrer or url_for("index"))

    @app.route("/download/<path:filename>")
    def download(filename: str):
        """Serve a tailored resume / cover letter / interview-prep file.

        Accepts:
          - "output/123_company_resume.pdf"
          - "output\\123_company_resume.pdf" (url-encoded \\)
          - "123_company_resume.pdf"  (basename only)
        Resolves to an absolute path under settings.output_dir and refuses
        anything that would escape that directory (path-traversal guard).
        """
        out_dir = Path(settings.output_dir).resolve()
        # Normalize any backslashes that may have survived URL routing
        candidate = filename.replace("\\", "/")
        # Try as a path under output_dir, falling back to basename lookup
        primary = (out_dir / Path(candidate).name).resolve()
        if not primary.exists():
            # Maybe the candidate already starts with output_dir/
            alt = Path(candidate)
            if not alt.is_absolute():
                alt = (Path.cwd() / alt).resolve()
            else:
                alt = alt.resolve()
            if alt.exists() and str(alt).startswith(str(out_dir)):
                primary = alt
            else:
                abort(404)
        # Path-traversal guard: refuse anything outside output_dir
        try:
            primary.relative_to(out_dir)
        except ValueError:
            abort(404)
        return send_file(str(primary), as_attachment=True)

    @app.route("/advice", methods=["GET", "POST"])
    def advice():
        result = None
        focus = ""
        if request.method == "POST":
            from . import gemini_client as gc
            from .resume import load_resume
            focus = request.form.get("focus", "")
            try:
                resume_text = load_resume(settings.base_resume_path)
                result = gc.advise_resume(resume_text, focus)
            except Exception as e:  # noqa: BLE001
                flash(f"Advice failed: {e}", "error")
        return render_template("advice.html", result=result, focus=focus)

    @app.route("/predict/<int:job_id>")
    def predict_job_route(job_id: int):
        from .predict import predict_job
        with session() as db:
            job = db.get(Job, job_id)
            if not job:
                abort(404)
            app_row = job.applications[0] if job.applications else None
            p = predict_job(job, app_row)
        return render_template("predict.html", job=job, p=p)

    @app.route("/forecast", methods=["GET", "POST"])
    def forecast_route():
        from datetime import datetime as _dt
        from .predict import forecast as _forecast
        result = None
        start = request.values.get("start_date") or _dt.utcnow().strftime("%Y-%m-%d")
        horizon = int(request.values.get("horizon_days", 90))
        rate = int(request.values.get("apply_rate", 5))
        if request.method == "POST" or request.args.get("run"):
            try:
                sd = _dt.strptime(start, "%Y-%m-%d")
                result = _forecast(sd, horizon, rate)
            except Exception as e:  # noqa: BLE001
                flash(f"Forecast failed: {e}", "error")
        return render_template("forecast.html", result=result, start_date=start,
                               horizon_days=horizon, apply_rate=rate)

    @app.route("/auto-apply", methods=["GET", "POST"])
    def auto_apply_route():
        from .auto_apply import AutoApplySettings, run_auto_apply
        report = None
        cfg = AutoApplySettings(
            min_match_score=settings.auto_apply_min_score,
            min_callback_probability=settings.auto_apply_min_callback_prob,
            require_watchlist_or_score=settings.auto_apply_require_watchlist_or_score,
            watchlist_only=settings.auto_apply_watchlist_only,
            default_recipient=settings.auto_apply_default_recipient,
        )
        if request.method == "POST":
            try:
                cfg.confirm = bool(request.form.get("confirm"))
                cfg.max_applies = int(request.form.get("max_applies") or 0) or None
                cfg.min_match_score = float(request.form.get("min_score", cfg.min_match_score))
                cfg.min_callback_probability = float(request.form.get("min_callback", cfg.min_callback_probability))
                cfg.watchlist_only = bool(request.form.get("watchlist_only"))
                cfg.default_recipient = (request.form.get("default_recipient") or "").strip()
                report = run_auto_apply(cfg)
                if cfg.confirm:
                    flash(f"Sent {len(report['sent_ids'])} application(s).", "success")
                else:
                    flash(f"Dry-run: {report['selected']} candidates selected.", "success")
            except Exception as e:  # noqa: BLE001
                flash(f"Auto-apply failed: {e}", "error")
        return render_template("auto_apply.html", cfg=cfg, report=report,
                               quota=daily_apply_quota_remaining())

    @app.route("/apply-kit/<int:job_id>", methods=["GET"])
    def apply_kit(job_id: int):
        """Show the apply kit (real form questions + auto-answers) for a job."""
        from .apply_questions import needs_attention, questions_from_json
        from .browser_apply import applicant_fields
        with session() as db:
            job = db.get(Job, job_id)
            if not job:
                abort(404)
            app_row = job.applications[0] if job.applications else None
            questions, source = questions_from_json(
                app_row.questions_json if app_row else "")
            cover_text = ""
            if app_row and app_row.cover_letter_path and Path(app_row.cover_letter_path).exists():
                try:
                    cover_text = Path(app_row.cover_letter_path).read_text(encoding="utf-8")
                except Exception:  # noqa: BLE001
                    pass
            return render_template(
                "apply_kit.html", job=job, application=app_row,
                questions=questions, source=source,
                fields=applicant_fields(), cover_text=cover_text,
                blanks=len(needs_attention(questions)),
                unanswered=sum(1 for q in questions if q.needs_user),
                unconfirmed=sum(1 for q in questions if q.needs_review))

    @app.route("/apply-kit/<int:job_id>/generate", methods=["POST"])
    def apply_kit_generate(job_id: int):
        """Build (or rebuild) the apply kit — fetches the real form questions,
        tailors materials if needed, and auto-answers everything."""
        from .browser_apply import build_package
        try:
            pkg = build_package(job_id)
            n_blank = len(pkg.blanks)
            src = pkg.questions_source or "common defaults"
            flash(f"Apply kit ready ({len(pkg.form_questions)} questions via {src}; "
                  f"{n_blank} need your input).", "success")
        except Exception as e:  # noqa: BLE001
            flash(f"Apply kit failed: {e}", "error")
        return redirect(url_for("apply_kit", job_id=job_id))

    @app.route("/apply-queue")
    def apply_queue_page():
        """Show the apply queue — jobs with ready-to-submit application kits."""
        try:
            from .auto_apply import apply_queue_status
            rows = apply_queue_status()
        except Exception as e:  # noqa: BLE001
            flash(f"Could not load apply queue: {e}", "error")
            rows = []
        return render_template(
            "apply_queue.html", rows=rows,
            needs_you=sum(r.get("blanks", 0) for r in rows),
            kits_needing_you=sum(1 for r in rows if r.get("blanks")))

    @app.route("/apply-queue/<int:application_id>/mark-applied", methods=["POST"])
    def apply_queue_mark_applied(application_id: int):
        """Mark an application as applied and redirect back to the queue."""
        try:
            from .browser_apply import mark_applied
            mark_applied(application_id)
            flash("Marked applied.", "success")
        except Exception as e:  # noqa: BLE001
            flash(f"Could not mark applied: {e}", "error")
        return redirect(url_for("apply_queue_page"))

    @app.route("/job/<int:job_id>/launch-copilot", methods=["GET", "POST"])
    @app.route("/api/job/<int:job_id>/launch-copilot", methods=["GET", "POST", "OPTIONS"])
    def launch_copilot_route(job_id: int):
        """Launch In-Browser Copilot for a target job with auto-injected script."""
        if request.method == "OPTIONS":
            resp = make_response()
            resp.headers["Access-Control-Allow-Origin"] = "*"
            resp.headers["Access-Control-Allow-Methods"] = "GET, POST, OPTIONS"
            resp.headers["Access-Control-Allow-Headers"] = "Content-Type"
            return resp

        import threading
        from . import apply_runner

        def _bg():
            try:
                apply_runner.launch_copilot(job_id, browser_mode="headed", auto_fill=True, interactive_wait=False)
            except Exception as e:
                log.warning("Background copilot launch error: %s", e)

        t = threading.Thread(target=_bg, daemon=True)
        t.start()

        if request.is_json or request.path.startswith("/api/"):
            resp = jsonify({
                "ok": True,
                "job_id": job_id,
                "message": "Copilot browser session launched in background with script pre-injected.",
                "bookmarklet": apply_runner.BOOKMARKLET_TEMPLATE,
            })
            resp.headers["Access-Control-Allow-Origin"] = "*"
            return resp

        flash("🚀 JobBot Copilot launched in browser with auto-injected filling assistant!", "success")
        return redirect(url_for("apply_kit", job_id=job_id))

    @app.route("/apply-run/<int:job_id>", methods=["POST"])
    def browser_apply_start(job_id: int):
        try:
            token = web_apply.start_web_apply(job_id)
        except Exception as e:  # noqa: BLE001
            flash(f"Could not start browser apply: {e}", "error")
            return redirect(url_for("job_detail", job_id=job_id))
        return redirect(url_for("browser_apply_page", token=token))

    @app.route("/apply-run/view/<int:token>")
    def browser_apply_page(token: int):
        state = web_apply.get_state(token)
        return render_template("apply_run.html", state=state, token=token)

    @app.route("/apply-run/status/<int:token>")
    def browser_apply_status(token: int):
        return jsonify(web_apply.get_state(token))

    @app.route("/apply-run/decision/<int:token>", methods=["POST"])
    def browser_apply_decision(token: int):
        choice = (request.form.get("choice") or "").strip()
        ok = web_apply.submit_decision(token, choice)
        if not ok:
            flash("Decision not accepted (already decided or not ready).", "error")
        else:
            flash(f"Recorded: {choice}.", "success")
        return redirect(url_for("browser_apply_page", token=token))

    @app.route("/apply-queue/build", methods=["POST"])
    def apply_queue_build():
        """Build apply kits for the top N candidate jobs."""
        from .auto_apply import AutoApplySettings, prepare_apply_queue
        try:
            raw_count = request.form.get("count", "3")
            try:
                count = int(raw_count)
            except (ValueError, TypeError):
                count = 3
            count = max(1, min(10, count))
            cfg = AutoApplySettings(
                min_match_score=settings.auto_apply_min_score,
                min_callback_probability=settings.auto_apply_min_callback_prob,
                require_watchlist_or_score=settings.auto_apply_require_watchlist_or_score,
                watchlist_only=settings.auto_apply_watchlist_only,
                skip_stale_days=settings.auto_apply_skip_stale_days,
            )
            report = prepare_apply_queue(cfg, build_limit=count)
            built = len(report.get("built", []))
            errors = len(report.get("errors", []))
            flash(f"Built {built} kit(s) for top {count} jobs; {errors} error(s).", "success")
        except Exception as e:  # noqa: BLE001
            flash(f"Build failed: {e}", "error")
        return redirect(url_for("apply_queue_page"))

    @app.route("/apply-kit/<int:job_id>/save", methods=["POST"])
    def apply_kit_save(job_id: int):
        """Persist user-edited answers; optionally remember them in answer
        memory so every future application auto-fills them."""
        from .apply_questions import (questions_from_json, questions_to_json,
                                      remember_answer)
        with session() as db:
            job = db.get(Job, job_id)
            if not job or not job.applications:
                abort(404)
            app_row = job.applications[0]
            questions, source = questions_from_json(app_row.questions_json)
            remembered = 0
            for i, q in enumerate(questions):
                new_answer = request.form.get(f"answer_{i}")
                if new_answer is None:
                    continue
                new_answer = new_answer.strip()
                if new_answer != q.answer:
                    q.answer = new_answer
                    q.answer_source = "user"
                    q.guess = ""
                q.needs_user = not bool(q.answer)
                # Submitting this form means the user has seen every answer on
                # the page, so nothing here is awaiting review any more.
                q.needs_review = False
                if request.form.get(f"remember_{i}") and q.answer:
                    remember_answer(q.text, q.answer)
                    remembered += 1
            app_row.questions_json = questions_to_json(questions, source)
            db.commit()
        msg = "Answers saved."
        if remembered:
            msg += f" {remembered} remembered for future applications."
        flash(msg, "success")
        return redirect(url_for("apply_kit", job_id=job_id))

    @app.route("/api/job/<int:job_id>/copilot-package", methods=["GET", "OPTIONS"])
    @app.route("/api/application-kit/<int:job_id>", methods=["GET", "OPTIONS"])
    def api_copilot_package(job_id: int):
        """Full application data package for In-Browser Copilot and API clients."""
        if request.method == "OPTIONS":
            resp = jsonify({"ok": True})
            resp.headers["Access-Control-Allow-Origin"] = "*"
            resp.headers["Access-Control-Allow-Methods"] = "GET, OPTIONS"
            resp.headers["Access-Control-Allow-Headers"] = "Content-Type, Authorization"
            return resp

        from .browser_apply import applicant_fields, build_package
        from .apply_questions import questions_from_json

        with session() as db:
            job = db.get(Job, job_id)
            if not job:
                resp = jsonify({"error": f"Job {job_id} not found"})
                resp.status_code = 404
                resp.headers["Access-Control-Allow-Origin"] = "*"
                return resp

            app_row = job.applications[0] if job.applications else None
            title, company, url = job.title, job.company, job.url
            resume_path = ""
            cover_letter_text = ""

            if app_row:
                resume_path = app_row.tailored_resume_path or ""
                if app_row.cover_letter_path and Path(app_row.cover_letter_path).exists():
                    try:
                        cover_letter_text = Path(app_row.cover_letter_path).read_text(encoding="utf-8")
                    except Exception:
                        pass
                questions, _ = questions_from_json(app_row.questions_json or "")
            else:
                questions = []

        if not resume_path and settings.base_resume_path and Path(settings.base_resume_path).exists():
            resume_path = str(Path(settings.base_resume_path).resolve())

        fields = applicant_fields()
        applicant = {
            "first_name": fields.get("first_name", ""),
            "last_name": fields.get("last_name", ""),
            "full_name": fields.get("full_name", ""),
            "email": fields.get("email", ""),
            "phone": fields.get("phone", ""),
            "linkedin": fields.get("linkedin", ""),
            "github": fields.get("github", ""),
            "address": fields.get("address", ""),
            "city": fields.get("city", ""),
            "state": fields.get("state", ""),
            "zip": fields.get("zip", ""),
            "portfolio": fields.get("portfolio", ""),
            "current_title": fields.get("current_title", ""),
        }

        screening_answers = {
            "work_authorized": settings.work_authorized or "Yes",
            "requires_sponsorship": settings.requires_sponsorship or "No",
            "salary_expectation": settings.salary_expectation or "",
            "earliest_start_date": settings.earliest_start_date or "",
            "willing_onsite": settings.willing_onsite or "Yes",
            "willing_relocate": settings.willing_relocate or "",
            "how_heard": settings.how_heard or "Company careers site",
        }

        blanks = []
        if questions:
            for q in questions:
                if q.answer:
                    screening_answers[q.text] = q.answer
                else:
                    blanks.append(q.text)

        payload = {
            "job_id": job_id,
            "title": title,
            "company": company,
            "url": url,
            "applicant": applicant,
            "screening_answers": screening_answers,
            "resume_path": resume_path,
            "cover_letter_text": cover_letter_text,
            "blanks": blanks,
        }

        resp = jsonify(payload)
        resp.headers["Access-Control-Allow-Origin"] = "*"
        resp.headers["Access-Control-Allow-Methods"] = "GET, OPTIONS"
        resp.headers["Access-Control-Allow-Headers"] = "Content-Type, Authorization"
        return resp

    @app.route("/api/copilot-package", methods=["GET", "OPTIONS"])
    def api_copilot_package_by_url():
        """Lookup copilot package by job_id or target URL."""
        if request.method == "OPTIONS":
            resp = jsonify({"ok": True})
            resp.headers["Access-Control-Allow-Origin"] = "*"
            resp.headers["Access-Control-Allow-Methods"] = "GET, OPTIONS"
            return resp

        job_id_param = request.args.get("job_id")
        target_url = request.args.get("url")
        if job_id_param:
            try:
                return api_copilot_package(int(job_id_param))
            except ValueError:
                pass
        if target_url:
            with session() as db:
                clean = target_url.split("?")[0].rstrip("/")
                job = db.query(Job).filter((Job.url == target_url) | (Job.url.like(f"%{clean}%"))).first()
                if job:
                    return api_copilot_package(job.id)

        resp = jsonify({"error": "No matching job found for URL or job_id"})
        resp.status_code = 404
        resp.headers["Access-Control-Allow-Origin"] = "*"
        return resp

    @app.route("/api/copilot/answer-question", methods=["GET", "POST", "OPTIONS"])
    def api_copilot_answer_question():
        """On-the-fly grounded screening question answering for In-Browser Copilot.

        Accepts question text, optional options list, optional question type, and optional job_id.
        Returns a grounded answer using answer bank, candidate profile, and local LLM.
        """
        if request.method == "OPTIONS":
            resp = jsonify({"ok": True})
            resp.headers["Access-Control-Allow-Origin"] = "*"
            resp.headers["Access-Control-Allow-Methods"] = "GET, POST, OPTIONS"
            resp.headers["Access-Control-Allow-Headers"] = "Content-Type, Authorization"
            return resp

        if request.method == "POST":
            data = request.get_json(silent=True) or {}
            question = (data.get("question") or "").strip()
            job_id = data.get("job_id")
            options = data.get("options") or []
            qtype = data.get("qtype") or "text"
        else:
            question = (request.args.get("question") or "").strip()
            job_id = request.args.get("job_id")
            options = request.args.getlist("options")
            qtype = request.args.get("qtype") or "text"

        if not question:
            resp = jsonify({"ok": False, "error": "question parameter is required"})
            resp.status_code = 400
            resp.headers["Access-Control-Allow-Origin"] = "*"
            return resp

        job_id_int = None
        if job_id is not None:
            try:
                job_id_int = int(job_id)
            except ValueError:
                pass

        from .apply_questions import FormQuestion, answer_questions, _snap_to_options
        from .browser_apply import applicant_fields
        from .resume import load_resume

        title, company, description = "", "", ""
        if job_id_int:
            with session() as db:
                j = db.get(Job, job_id_int)
                if j:
                    title, company, description = j.title, j.company, j.description or ""

        base_resume = ""
        try:
            base_resume = load_resume(settings.base_resume_path)
        except Exception:
            pass

        fq = FormQuestion(
            text=question,
            qtype="select" if options else qtype,
            options=options,
            required=True,
        )

        answer_questions(
            [fq],
            resume=base_resume,
            job_title=title,
            company=company,
            job_description=description,
            identity_fields=applicant_fields(),
        )

        ans = fq.answer or fq.guess or ""
        if options and ans:
            snapped = _snap_to_options(ans, options)
            if snapped:
                ans = snapped

        resp = jsonify({
            "ok": True,
            "question": question,
            "answer": ans,
            "needs_review": fq.needs_review,
            "needs_user": fq.needs_user,
            "source": fq.answer_source or ("ai" if ans else "none"),
        })
        resp.headers["Access-Control-Allow-Origin"] = "*"
        return resp

    @app.route("/followups")
    def followups_page():
        from .followup import draft_followup, followups_due
        try:
            rows = followups_due()
        except Exception as e:  # noqa: BLE001
            flash(f"Could not load follow-ups: {e}", "error")
            rows = []
        enriched = []
        for r in rows:
            try:
                draft = draft_followup(r["application_id"])
            except Exception:  # noqa: BLE001
                draft = {"body": "", "subject": ""}
            enriched.append({**r, "draft_body": draft["body"],
                             "draft_subject": draft["subject"]})
        return render_template("followups.html", rows=enriched,
                               followup_after_days=settings.followup_after_days)

    @app.route("/followups/<int:application_id>/mark", methods=["POST"])
    def followup_mark(application_id: int):
        from .followup import record_followup
        try:
            record_followup(application_id)
            flash("Marked as followed up.", "success")
        except Exception as e:  # noqa: BLE001
            flash(f"Could not mark follow-up: {e}", "error")
        return redirect(url_for("followups_page"))

    @app.route("/followups/<int:application_id>/send", methods=["POST"])
    def followup_send(application_id: int):
        from .followup import send_followup
        to_email = (request.form.get("to_email") or "").strip()
        if not to_email:
            flash("Contact email is required to send a follow-up.", "error")
            return redirect(url_for("followups_page"))
        try:
            send_followup(application_id, to_email)
            flash(f"Follow-up sent to {to_email}.", "success")
        except Exception as e:  # noqa: BLE001
            flash(f"Send failed: {e}", "error")
        return redirect(url_for("followups_page"))

    @app.route("/job/<int:job_id>/find-contacts", methods=["POST"])
    def job_find_contacts(job_id: int):
        from .contacts import find_contacts
        try:
            found = find_contacts(job_id, save=True)
            if found:
                flash(f"Found {len(found)} contact(s).", "success")
            else:
                flash("No contacts found — free search engines often block "
                      "automated lookups; set GOOGLE_CSE_KEY/GOOGLE_CSE_CX "
                      "in .env for reliable results.", "error")
        except Exception as e:  # noqa: BLE001
            flash(f"Contact search failed: {e}", "error")
        return redirect(url_for("job_detail", job_id=job_id))

    @app.route("/job/<int:job_id>/network/<int:contact_idx>", methods=["POST"])
    def job_network_message(job_id: int, contact_idx: int):
        """Draft an AI-tailored networking message for one contact (JSON)."""
        from .contacts import load_contacts, networking_note, networking_email
        channel = (request.form.get("channel") or request.args.get("channel")
                   or "linkedin")
        contacts = load_contacts(job_id)
        if contact_idx < 0 or contact_idx >= len(contacts):
            return jsonify({"error": "contact not found"}), 404
        contact = contacts[contact_idx]
        try:
            if channel == "email":
                em = networking_email(job_id, contact)
                return jsonify({"channel": "email", "subject": em.get("subject", ""),
                                "body": em.get("body", "")})
            note = networking_note(job_id, contact)
            return jsonify({"channel": "linkedin", "subject": "", "body": note})
        except Exception as e:  # noqa: BLE001
            return jsonify({"error": str(e)}), 500

    @app.route("/applications/<int:application_id>/outcome", methods=["POST"])
    def application_outcome(application_id):
        from . import outcomes
        stage = (request.form.get("stage") or "").strip()
        try:
            outcomes.record_outcome(application_id, stage,
                                    note=request.form.get("note", ""))
            flash(f"Recorded '{stage}'.", "success")
        except ValueError as e:
            flash(str(e), "error")
        return redirect(url_for("applications"))

    @app.route("/applications")
    def applications():
        from sqlalchemy.orm import joinedload
        from . import outcomes
        # Fail-open on all outcome logic — the page must never 500 on it.
        try:
            outcomes.sweep_ghosted()
        except Exception:
            pass
        try:
            priors = outcomes.compute_priors()
        except Exception:
            priors = {"base_response": 0.15, "base_interview": 0.05, "dims": {}}
        try:
            panel = outcomes.insights()
        except Exception:
            panel = {}
        with session() as db:
            # Eager-load .job — the template reads app.job.* after the session
            # closes, which would otherwise raise DetachedInstanceError.
            apps = (db.query(Application)
                    .options(joinedload(Application.job))
                    .order_by(Application.created_at.desc()).limit(200).all())
            apps = list(apps)
            # Order by the job's blended score (outcome-aware) desc, newest first.
            apps.sort(key=lambda a: (outcomes.blended(a.job, priors) if a.job else 0.0),
                      reverse=True)
        return render_template("applications.html", applications=apps,
                               insights=panel,
                               stages=outcomes.STAGES, terminal=sorted(outcomes.TERMINAL))

    def _classify_kanban_column(job: Job, latest_app: Optional[Application]) -> str:
        js = (job.status or "").lower().strip()
        as_ = (latest_app.status if latest_app else "").lower().strip()

        # Terminal / Rejected / Archive
        if js in ("rejected", "archive", "archived") or as_ in ("rejected", "ghosted", "withdrawn"):
            return "rejected"

        # Offer
        if js in ("offer", "offered", "negotiation") or as_ in ("offer", "offered"):
            return "offer"

        # Interview / Screen
        if js in ("interview", "screen", "interviewing") or as_ in ("interview", "screen", "responded"):
            return "interview"

        # Applied
        if js in ("applied", "sent") or as_ in ("applied", "sent"):
            return "applied"

        # Tailored
        if js == "tailored" or (latest_app and (latest_app.tailored_resume_path or as_ == "tailored")):
            return "tailored"

        # Wishlist / Starred
        if job.starred or js in ("starred", "wishlist", "target"):
            return "wishlist"

        if latest_app:
            if latest_app.tailored_resume_path:
                return "tailored"
            return "wishlist"

        return "wishlist"

    @app.route("/kanban")
    def kanban_route():
        from sqlalchemy.orm import joinedload
        from datetime import datetime as _dt
        with session() as db:
            tracked_statuses = [
                "wishlist", "starred", "target", "tailored", "applied", "sent",
                "screen", "interview", "interviewing", "offer", "offered",
                "rejected", "archive", "archived"
            ]
            jobs = (
                db.query(Job)
                .options(joinedload(Job.applications))
                .filter(
                    (Job.applications.any())
                    | (Job.starred == True)
                    | (Job.status.in_(tracked_statuses))
                )
                .all()
            )

            columns_cfg = [
                {
                    "id": "wishlist",
                    "title": "Wishlist",
                    "icon": "📌",
                    "color": "var(--muted)",
                    "border_color": "rgba(139, 152, 173, 0.4)",
                    "desc": "Saved & target roles",
                },
                {
                    "id": "tailored",
                    "title": "Tailored",
                    "icon": "⚡",
                    "color": "var(--accent)",
                    "border_color": "rgba(34, 211, 238, 0.4)",
                    "desc": "Tailored resumes ready",
                },
                {
                    "id": "applied",
                    "title": "Applied",
                    "icon": "📨",
                    "color": "var(--accent2)",
                    "border_color": "rgba(167, 139, 250, 0.4)",
                    "desc": "Applications submitted",
                },
                {
                    "id": "interview",
                    "title": "Interview",
                    "icon": "🎯",
                    "color": "var(--warn)",
                    "border_color": "rgba(251, 191, 36, 0.4)",
                    "desc": "Screens & interviews",
                },
                {
                    "id": "offer",
                    "title": "Offer",
                    "icon": "🏆",
                    "color": "var(--ok)",
                    "border_color": "rgba(52, 211, 153, 0.4)",
                    "desc": "Active offers & talks",
                },
                {
                    "id": "rejected",
                    "title": "Archive",
                    "icon": "✕",
                    "color": "var(--err)",
                    "border_color": "rgba(248, 113, 113, 0.4)",
                    "desc": "Archived or passed",
                },
            ]

            board_cards = {col["id"]: [] for col in columns_cfg}

            for job in jobs:
                apps = sorted(job.applications, key=lambda a: a.created_at or _dt.min)
                latest_app = apps[-1] if apps else None
                col_id = _classify_kanban_column(job, latest_app)
                if col_id not in board_cards:
                    col_id = "wishlist"

                ms = job.match_score or 0.0
                score_class = max(0, min(10, int(ms * 10)))
                score_pct = int(round(ms * 100))

                activity_date = None
                if latest_app:
                    activity_date = latest_app.sent_at or latest_app.created_at
                if not activity_date:
                    activity_date = job.discovered_at or job.posted_at

                card_data = {
                    "job": job,
                    "latest_app": latest_app,
                    "column": col_id,
                    "score_class": score_class,
                    "score_pct": score_pct,
                    "activity_date": activity_date,
                }
                board_cards[col_id].append(card_data)

            for col_id in board_cards:
                board_cards[col_id].sort(
                    key=lambda c: (
                        c["activity_date"] or _dt.min,
                        c["job"].match_score or 0.0,
                    ),
                    reverse=True,
                )

            total_count = sum(len(cards) for cards in board_cards.values())

            return render_template(
                "kanban.html",
                columns=columns_cfg,
                board=board_cards,
                total_count=total_count,
            )

    @app.route("/api/kanban/move", methods=["POST"])
    def kanban_move_api():
        from datetime import datetime as _dt
        from .models import OutcomeEvent
        data = request.get_json(silent=True) or request.form.to_dict() or {}
        raw_job_id = data.get("job_id")
        raw_app_id = data.get("application_id") or data.get("app_id")
        raw_target = data.get("target_column") or data.get("column") or data.get("status")

        if not raw_target:
            return jsonify({"error": "target_column is required", "ok": False}), 400

        target = str(raw_target).strip().lower()
        alias_map = {
            "wishlist": "wishlist",
            "starred": "wishlist",
            "saved": "wishlist",
            "target": "wishlist",
            "tailored": "tailored",
            "draft": "tailored",
            "applied": "applied",
            "sent": "applied",
            "screen": "interview",
            "interview": "interview",
            "interviewing": "interview",
            "offer": "offer",
            "offered": "offer",
            "negotiation": "offer",
            "rejected": "rejected",
            "reject": "rejected",
            "archive": "rejected",
            "archived": "rejected",
            "ghosted": "rejected",
            "withdrawn": "rejected",
        }
        canonical_col = alias_map.get(target)
        if not canonical_col:
            return jsonify({"error": f"Invalid target column: {raw_target}", "ok": False}), 400

        with session() as db:
            app = None
            job_id = None
            if raw_job_id is not None:
                try:
                    job_id = int(raw_job_id)
                except (ValueError, TypeError):
                    return jsonify({"error": f"Invalid job_id: {raw_job_id}", "ok": False}), 400

            if raw_app_id is not None:
                try:
                    app_id = int(raw_app_id)
                    app = db.get(Application, app_id)
                    if app and not job_id:
                        job_id = app.job_id
                except (ValueError, TypeError):
                    return jsonify({"error": f"Invalid application_id: {raw_app_id}", "ok": False}), 400

            if not job_id:
                return jsonify({"error": "job_id or application_id is required", "ok": False}), 400

            job = db.get(Job, job_id)
            if not job:
                return jsonify({"error": f"Job {job_id} not found", "ok": False}), 404

            if not app and job.applications:
                apps = sorted(job.applications, key=lambda a: a.created_at or _dt.min)
                app = apps[-1] if apps else None

            now = _dt.utcnow()
            if canonical_col == "wishlist":
                job.status = "starred"
                job.starred = True
                if app and app.status in ("rejected", "ghosted", "withdrawn"):
                    app.status = "draft"

            elif canonical_col == "tailored":
                job.status = "tailored"
                if not app:
                    app = Application(job_id=job.id, status="draft", created_at=now)
                    db.add(app)
                elif app.status in ("rejected", "ghosted", "withdrawn", "applied", "sent"):
                    app.status = "draft"

            elif canonical_col == "applied":
                job.status = "applied"
                if not app:
                    app = Application(job_id=job.id, status="applied", sent_at=now, created_at=now)
                    db.add(app)
                    db.flush()
                else:
                    app.status = "applied"
                    if not app.sent_at:
                        app.sent_at = now
                db.add(OutcomeEvent(application_id=app.id, stage="applied", note="Kanban board move"))

            elif canonical_col == "interview":
                job.status = "interview"
                if not app:
                    app = Application(job_id=job.id, status="interview", sent_at=now, created_at=now)
                    db.add(app)
                    db.flush()
                else:
                    app.status = "interview"
                    if not app.sent_at:
                        app.sent_at = now
                db.add(OutcomeEvent(application_id=app.id, stage="interview", note="Kanban board move"))
                if not (job.interview_question_bank or ""):
                    job.interview_prep_pending = True

            elif canonical_col == "offer":
                job.status = "offer"
                if not app:
                    app = Application(job_id=job.id, status="offer", sent_at=now, created_at=now)
                    db.add(app)
                    db.flush()
                else:
                    app.status = "offer"
                    if not app.sent_at:
                        app.sent_at = now
                db.add(OutcomeEvent(application_id=app.id, stage="offer", note="Kanban board move"))

            elif canonical_col == "rejected":
                job.status = "rejected"
                if app:
                    app.status = "rejected"
                    db.add(OutcomeEvent(application_id=app.id, stage="rejected", note="Kanban board move"))

            db.commit()

            return jsonify({
                "ok": True,
                "job_id": job.id,
                "new_status": canonical_col,
                "column": canonical_col,
                "job_status": job.status,
                "app_status": app.status if app else None,
            })


    @app.route("/discover")
    def discover_page():
        rows = discovery_service.review()
        return render_template("discover.html", rows=rows, run=_discover_run)

    @app.route("/discover/<action>", methods=["POST"])
    def discover_action(action):
        if action not in ("approve", "approve_watchlist", "reject"):
            abort(404)
        # Fail-open: a raise in approve/reject (e.g. a target-file write error)
        # must flash an error, not 500 the request (review m1).
        try:
            coords = [discovery_service.parse_coord(c) for c in request.form.getlist("coord")]
            if action == "approve":
                discovery_service.approve(coords, watchlist=False)
                flash(f"Approved {len(coords)} board(s).", "success")
            elif action == "approve_watchlist":
                discovery_service.approve(coords, watchlist=True)
                flash(f"Approved {len(coords)} board(s) + watchlist.", "success")
            else:
                discovery_service.reject(coords)
                flash(f"Rejected {len(coords)} board(s).", "success")
        except Exception as e:  # noqa: BLE001
            flash(f"Discovery action failed: {e}", "error")
        return redirect(url_for("discover_page"))

    @app.route("/discover/run", methods=["POST"])
    def discover_run():
        import threading
        kind = request.form.get("kind", "harvest")
        # Claim the run slot SYNCHRONOUSLY under a lock before spawning the
        # worker, so a double-click can't pass the guard twice (review M1).
        with _discover_run_lock:
            if _discover_run["running"]:
                flash("A discovery run is already in progress.", "error")
                return redirect(url_for("discover_page"))
            _discover_run.update(running=True, kind=kind, result=None, error=None)

        def _runner():
            try:
                if kind == "discovery":
                    _discover_run["result"] = discovery_service.run_discovery(["websearch", "llm"])
                else:
                    _discover_run["result"] = discovery_service.run_harvest()
            except Exception as e:  # noqa: BLE001
                _discover_run["error"] = str(e)
            finally:
                _discover_run["running"] = False

        threading.Thread(target=_runner, daemon=True, name="jobbot-discover").start()
        flash(f"{kind.title()} run started in the background. Refresh in a moment.", "success")
        return redirect(url_for("discover_page"))

    @app.route("/discover/run/status")
    def discover_run_status():
        return jsonify(_discover_run)

    @app.route("/contacts")
    def contacts_page():
        from . import referrals
        with session() as db:
            rows = db.query(Contact).all()
            rows = list(rows)
        # warmth for display (fail-open)
        try:
            terms = referrals.profile_terms()
            rows.sort(key=lambda c: (c.pinned, referrals.warmth_score(c, terms)[0]),
                      reverse=True)
        except Exception:
            pass
        return render_template("contacts.html", contacts=rows)

    @app.route("/contacts", methods=["POST"])
    def contact_add():
        with session() as db:
            db.add(Contact(
                name=(request.form.get("name") or "").strip(),
                company=(request.form.get("company") or "").strip(),
                title=(request.form.get("title") or "").strip(),
                email=(request.form.get("email") or "").strip(),
                linkedin=(request.form.get("linkedin") or "").strip(),
                relationship=(request.form.get("relationship") or "unknown").strip(),
                notes=(request.form.get("notes") or "").strip(),
                source="manual",
                pinned=bool(request.form.get("pinned")),
            ))
            db.commit()
        flash("Contact added.", "success")
        return redirect(url_for("contacts_page"))

    @app.route("/contacts/<int:cid>/pin", methods=["POST"])
    def contact_pin(cid):
        with session() as db:
            c = db.get(Contact, cid)
            if c:
                c.pinned = not c.pinned
                db.commit()
        return redirect(url_for("contacts_page"))

    @app.route("/contacts/<int:cid>/delete", methods=["POST"])
    def contact_delete(cid):
        with session() as db:
            c = db.get(Contact, cid)
            if c:
                db.delete(c); db.commit()
        flash("Contact deleted.", "success")
        return redirect(url_for("contacts_page"))

    @app.route("/contacts/discover", methods=["POST"])
    def contacts_discover():
        from . import referrals
        company = (request.form.get("company") or "").strip()
        try:
            if company:
                n = len(referrals.discover_company(company))
                flash(f"Found {n} contact(s) at {company}.", "success")
            else:
                res = referrals.discover_top_companies()
                flash(f"Discovered contacts across {len(res)} companies.", "success")
        except Exception:
            flash("Discovery unavailable (no search provider configured).", "error")
        return redirect(url_for("contacts_page"))

    @app.route("/contacts/<int:cid>/kit", methods=["POST"])
    def contact_kit(cid):
        from . import networking_package as np
        job_id = request.form.get("job_id", type=int)
        with session() as db:
            c = db.get(Contact, cid)
            job = db.get(Job, job_id) if job_id else None
            cname = c.name if c else ""
            company = (job.company if job else "") or (c.company if c else "")
            role = job.title if job else ""
        if not cname:
            flash("Contact not found.", "error")
            return redirect(url_for("contacts_page"))
        try:
            np.build_package(name=cname, company=company, role=role)
            flash(f"Built outreach kit for {cname}.", "success")
        except Exception as e:  # noqa: BLE001
            flash(f"Kit build failed: {e}", "error")
        return redirect(url_for("contacts_page"))

    @app.route("/outreach")
    def outreach_page():
        from . import outreach as ox
        drafts = []
        try:
            drafts = ox.queue_list()
        except Exception:  # noqa: BLE001
            drafts = []
        grouped = {"warm_intro": [], "follow_up": [], "cold": [], "reply": []}
        for d in drafts:
            grouped.setdefault(d.kind, []).append(d)
        return render_template("outreach.html", grouped=grouped, total=len(drafts))

    @app.route("/outreach/scan", methods=["POST"])
    def outreach_scan():
        from . import outreach as ox
        try:
            res = ox.scan()
            flash(f"Scan queued: {res}", "success")
        except Exception as exc:  # noqa: BLE001
            flash(f"Scan failed: {exc}", "error")
        return redirect(url_for("outreach_page"))

    @app.route("/outreach/reply", methods=["POST"])
    def outreach_reply():
        from . import outreach as ox
        inbound = (request.form.get("inbound") or "").strip()
        sender = (request.form.get("sender") or "").strip()
        if inbound:
            try:
                ox.draft_reply(inbound, sender=sender)
                flash("Reply drafted.", "success")
            except Exception as exc:  # noqa: BLE001
                flash(f"Reply draft failed: {exc}", "error")
        return redirect(url_for("outreach_page"))

    @app.route("/outreach/<int:did>/approve", methods=["POST"])
    def outreach_approve(did: int):
        from . import outreach as ox
        try:
            ox.approve(did)
        except Exception as exc:  # noqa: BLE001
            flash(f"Approve failed: {exc}", "error")
        return redirect(url_for("outreach_page"))

    @app.route("/outreach/<int:did>/dismiss", methods=["POST"])
    def outreach_dismiss(did: int):
        from . import outreach as ox
        try:
            ox.dismiss(did)
        except Exception as exc:  # noqa: BLE001
            flash(f"Dismiss failed: {exc}", "error")
        return redirect(url_for("outreach_page"))

    @app.route("/outreach/thread/<int:tid>")
    def thread_page(tid: int):
        from . import conversations as cv
        from .models import session as _session
        try:
            with _session() as db:
                t = cv._get(db, tid)
                thread = {"id": t.id, "subject": t.subject, "status": t.status,
                          "scenario": t.scenario,
                          "counterpart_name": t.counterpart_name,
                          "counterpart_email": t.counterpart_email}
            rows = cv.history(tid)
        except ValueError:
            abort(404)
        return render_template("thread.html", thread=thread, rows=rows)

    @app.route("/outreach/<int:did>/edit", methods=["POST"])
    def outreach_edit(did: int):
        from . import outreach as ox
        try:
            ox.edit(did, request.form.get("subject", ""), request.form.get("body", ""))
        except Exception as exc:  # noqa: BLE001
            flash(f"Edit failed: {exc}", "error")
        return redirect(url_for("outreach_page"))

    @app.route("/outreach/<int:did>/send", methods=["POST"])
    def outreach_send(did: int):
        from . import outreach as ox
        try:
            ox.send(did)
            flash("Sent.", "success")
        except Exception as exc:  # noqa: BLE001
            flash(f"Send failed: {exc}", "error")
        return redirect(url_for("outreach_page"))

    return app


def serve() -> None:
    # Snapshot the DB on startup so an accidental wipe is always recoverable.
    try:
        from .models import backup_db
        bak = backup_db()
        if bak:
            log.info("DB backed up on serve startup -> %s", bak)
    except Exception as e:  # noqa: BLE001
        log.warning("Startup DB backup failed (%s); continuing.", e)
    app = create_app()
    app.run(host=settings.dashboard_host, port=settings.dashboard_port, debug=False)
