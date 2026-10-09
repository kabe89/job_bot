"""Command-line entry point: `python -m jobbot ...`"""
from __future__ import annotations

import logging
import sys
import time
import threading
import urllib.parse
from datetime import datetime
from pathlib import Path

import click
from rich.console import Console, Group
from rich.live import Live
from rich.panel import Panel
from rich.table import Table
from rich.text import Text

from .config import settings
from .logging_setup import setup_logging
from .models import Application, Job, init_db, session
from .pipeline import (generate_interview_prep, run_scrape_cycle, send_application,
                       tailor_for_job, tailor_for_job_local)
from . import apply_runner
from . import profile
from . import calibrate
from .discovery import service as discovery_service
from . import model_registry
from . import paper_podcast

console = Console(width=160)


def _cli_safe(text: str) -> str:
    """Make AI/dynamic text safe to print on a redirected cp1252 console.

    Maps common smart punctuation to ASCII and drops anything still
    non-encodable, so background/redirected runs don't crash on UnicodeEncodeError.
    """
    if not text:
        return text
    repl = {
        "’": "'", "‘": "'", "“": '"', "”": '"',
        "–": "-", "—": "-", "…": "...", "→": "->",
        " ": " ", "•": "*",
    }
    for k, v in repl.items():
        text = text.replace(k, v)
    return text.encode("cp1252", "ignore").decode("cp1252")


def _scrape_panel(snap: dict) -> Panel:
    """Build a Rich Panel from a scrape_progress snapshot for live display."""
    state   = snap["state"]
    done    = snap["done"]
    total   = snap["total"]
    pct     = snap["percent"]
    jobs    = snap["jobs_found"]
    elapsed = snap.get("elapsed_seconds") or 0
    current = snap.get("current", "")
    per     = snap.get("per_scraper", {})

    bar_width = 28
    filled = int(bar_width * pct / 100)
    bar = "#" * filled + "-" * (bar_width - filled)

    color = {"running": "cyan", "done": "green", "error": "red"}.get(state, "white")

    header = Text()
    header.append(f"[{bar}] {pct:3d}%  ", style=color)
    header.append(f"{done}/{total} scrapers  ", style="bold white")
    header.append(f"{jobs} raw jobs  ", style="yellow")
    header.append(f"{elapsed:.0f}s", style="dim")
    if current:
        header.append(f"\n  > {current}...", style=f"bold {color}")

    tbl = Table(box=None, show_header=False, padding=(0, 2), expand=False)
    tbl.add_column(style="dim", min_width=22)
    tbl.add_column(justify="right", style="yellow", min_width=5)
    for name, count in sorted(per.items()):
        tbl.add_row(f"+ {name}", str(count))

    return Panel(Group(header, tbl),
                 title="[bold]JobBot — scrape in progress[/]",
                 border_style=color, padding=(0, 1))


def _extract_meta_from_url(url: str) -> tuple[str, str]:
    """Attempt to extract sensible fallback company and title hints from a URL."""
    if not url:
        return "", ""
    try:
        p = urllib.parse.urlparse(url)
        netloc = p.netloc.lower()
        parts = [part for part in p.path.split("/") if part]
        company = ""
        title = ""

        if "greenhouse.io" in netloc and len(parts) >= 1:
            company = parts[0].replace("-", " ").title()
        elif "lever.co" in netloc and len(parts) >= 1:
            company = parts[0].replace("-", " ").title()
        elif "ashbyhq.com" in netloc and len(parts) >= 1:
            company = parts[0].replace("-", " ").title()
        elif "myworkdayjobs.com" in netloc:
            company = netloc.split(".")[0].replace("-", " ").title()
        else:
            host_parts = netloc.split(".")
            if len(host_parts) >= 2:
                company = host_parts[-2].replace("-", " ").title()

        if parts:
            candidate_title = parts[-1].replace("-", " ").replace("_", " ")
            if not candidate_title.isdigit() and len(candidate_title) > 3:
                title = candidate_title.title()

        return title, company
    except Exception:
        return "", ""


def _fetch_description_from_url(url: str) -> str:
    """Best-effort extraction of text description from a job posting URL."""
    if not url:
        return ""
    try:
        import requests
        resp = requests.get(
            url,
            timeout=12,
            headers={
                "User-Agent": (
                    "Mozilla/5.0 (Windows NT 10.0; Win64; x64) "
                    "AppleWebKit/537.36 (KHTML, like Gecko) "
                    "Chrome/120.0.0.0 Safari/537.36"
                )
            },
        )
        if resp.status_code != 200:
            return ""
        html = resp.text
        try:
            from bs4 import BeautifulSoup
            soup = BeautifulSoup(html, "html.parser")
            for tag in soup(["script", "style", "nav", "footer", "header", "noscript"]):
                tag.decompose()
            text = soup.get_text(separator="\n")
        except ImportError:
            import re
            text = re.sub(r"<(script|style).*?</\1>", "", html, flags=re.DOTALL | re.IGNORECASE)
            text = re.sub(r"<[^>]+>", " ", text)
        lines = [line.strip() for line in text.splitlines()]
        cleaned = "\n".join(line for line in lines if line)
        return cleaned[:15000]
    except Exception:
        return ""


def _insert_manual_job(
    url: str,
    description: str,
    title: str = "",
    company: str = "",
    location: str = "",
) -> Job:
    """Insert a manually provided job into the database and calculate its initial score."""
    init_db()
    url = (url or "").strip()
    description = (description or "").strip()
    title = (title or "").strip()
    company = (company or "").strip()
    location = (location or "").strip()

    if not title or not company:
        inferred_title, inferred_company = _extract_meta_from_url(url)
        if not title:
            title = inferred_title or "Position"
        if not company:
            company = inferred_company or "Target Company"

    with session() as db:
        job = Job(
            title=title,
            company=company,
            url=url,
            description=description,
            location=location or "Remote / Unspecified",
            source="manual",
            status="new",
            discovered_at=datetime.utcnow(),
            match_score=0.0,
        )
        db.add(job)
        db.commit()
        db.refresh(job)

        try:
            prof = profile.load_profile()
            if prof and prof.embedding:
                from . import ranking
                if hasattr(ranking, "score_job"):
                    job.match_score = float(ranking.score_job(job, prof) or 0.0)
                    db.commit()
                    db.refresh(job)
                elif hasattr(ranking, "rescore_jobs"):
                    ranking.rescore_jobs(prof)
                    db.refresh(job)
        except Exception:
            pass

        return job


def _execute_tailor(job_id: int, local: bool = False, offline: bool = False) -> None:
    """Execute resume tailoring for a specified job ID."""
    if offline:
        from .prompt_pack import tailor_resume_offline, _slug
        from .resume import markdown_to_docx
        with session() as db:
            job = db.get(Job, job_id)
            if not job:
                console.print(f"[red]Job {job_id} not found[/]")
                return
            title, company, jd = job.title, job.company, job.description or ""
        md = tailor_resume_offline(title, company, jd)
        out_dir = Path(settings.output_dir)
        out_dir.mkdir(parents=True, exist_ok=True)
        slug = f"{job_id}_{_slug(company)}_{_slug(title)}"[:120]
        md_path = out_dir / f"{slug}_offline_resume.md"
        docx_path = out_dir / f"{slug}_offline_resume.docx"
        md_path.write_text(md, encoding="utf-8")
        try:
            markdown_to_docx(md, docx_path)
            console.print(f"[green]AI-free resume written:[/] [bold]{docx_path}[/]")
        except Exception as e:
            console.print(f"[yellow]docx render failed ({e}); markdown at[/] {md_path}")
        console.print(f"Markdown: {md_path}")
        return

    if local:
        result = tailor_for_job_local(
            job_id, progress=lambda m: console.print(f"[dim]{_cli_safe(m)}[/]"))
        _score = result.get("self_score")
        _shown = f"{_score:.2f}" if isinstance(_score, (int, float)) else "unknown"
        console.print(f"[green]Tailored locally[/] ({result['rounds']} round(s), self-score {_shown}).")
        console.print(f"Word doc: [bold]{result['docx']}[/]")
        console.print(f"Cover letter: {result['cover_letter']}")
        return

    result = tailor_for_job(job_id)
    if not result.get("application_id"):
        console.print(f"[yellow]Cloud AI unavailable[/] "
                      f"({_cli_safe(str(result.get('error', 'no provider')))}); "
                      f"tailoring locally with Ollama...")
        result = tailor_for_job_local(
            job_id, progress=lambda m: console.print(f"[dim]{_cli_safe(m)}[/]"))
    console.print(f"[green]Tailored.[/] Resume: {result['resume']}")
    console.print(f"Cover letter: {result['cover_letter']}")
    console.print(f"Analysis: {result['analysis']}")


@click.group()
@click.option("-v", "--verbose", count=True,
              help="Verbose output (-v = INFO on stdout, -vv = DEBUG everywhere).")
@click.option("-q", "--quiet", is_flag=True, help="Suppress non-error console output.")
@click.pass_context
def cli(ctx, verbose, quiet):
    """JobBot — automated job search + AI resume tailoring (Claude / Gemini)."""
    if quiet:
        level = logging.WARNING
    elif verbose >= 2:
        level = logging.DEBUG
    elif verbose == 1:
        level = logging.INFO
    else:
        level = logging.INFO
    setup_logging(level=level, verbose=verbose >= 1, debug=verbose >= 2)
    init_db()
    ctx.ensure_object(dict)
    ctx.obj["verbose"] = verbose


@cli.group("profile")
def profile_group():
    """Inspect / rebuild the auto-derived candidate profile."""


@profile_group.command("show")
def profile_show():
    """Print the current candidate profile."""
    prof = profile.load_profile()
    click.echo(f"Seniority: {prof.seniority}")
    click.echo(f"Titles: {', '.join(prof.role_titles) or '(none)'}")
    click.echo(f"Archetypes: {', '.join(prof.role_archetypes) or '(none)'}")
    click.echo(f"Skills: {', '.join(prof.skills) or '(none)'}")
    click.echo(f"Domains: {', '.join(prof.domains) or '(none)'}")
    click.echo(f"Dealbreakers: {', '.join(prof.dealbreakers) or '(none)'}")
    click.echo(f"Summary: {prof.summary}")
    click.echo(f"Embedding: {'present' if prof.embedding else 'MISSING (Ollama down?)'}")


@profile_group.command("rebuild")
def profile_rebuild():
    """Force a rebuild from the current base resume."""
    prof = profile.load_profile(force=True)
    click.echo("Rebuilt candidate profile.")
    click.echo(f"  titles: {', '.join(prof.role_titles) or '(none)'}")
    click.echo(f"  skills: {len(prof.skills)} | embedding: "
               f"{'present' if prof.embedding else 'MISSING'}")


@profile_group.command("context")
def profile_context():
    """Print the freeform AI profile context (data/profile.md)."""
    p = Path(settings.profile_path)
    if not p.exists():
        console.print(f"[yellow]No profile at {p}[/]")
        return
    console.print(p.read_text(encoding="utf-8"))


@profile_group.command("calibrate")
def profile_calibrate():
    """Print the cosine distribution of stored jobs vs the profile and
    recommend semantic_recall_threshold + min_match_score."""
    prof = profile.load_profile()
    dist = calibrate.cosine_distribution(prof)
    if not dist["n"]:
        click.echo("No embeddable jobs yet — run a scrape cycle first.")
        return
    click.echo(f"Cosine distribution over {dist['n']} jobs:")
    for p in ("p10", "p25", "p50", "p75", "p90"):
        click.echo(f"  {p}: {dist[p]}")
    rec = calibrate.recommend_thresholds(dist)
    click.echo("\nRecommended (.env):")
    click.echo(f"  SEMANTIC_RECALL_THRESHOLD={rec['semantic_recall_threshold']}")
    click.echo(f"  MIN_MATCH_SCORE={rec['min_match_score']}")
    click.echo(f"  AUTO_APPLY_MIN_SCORE={dist['p75']}")
    click.echo(f"  AUTO_APPLY_REQUIRE_WATCHLIST_OR_SCORE={dist['p90']}")
    click.echo(
        "\nNOTE: semantic (cosine) match_score clusters differently than the old\n"
        "bag-of-words score. The auto-apply floors above are derived from THIS run's\n"
        "cosine distribution. The callback-predictor feature bands in predict.py were\n"
        "tuned to the OLD scale and should be reviewed after the scale change.")


@profile_group.command("rescore")
@click.option("--force-reembed", is_flag=True,
              help="Re-embed every job even if it already has a stored embedding.")
def profile_rescore(force_reembed):
    """Recompute match_score for all active jobs against the current profile.

    Fixes stale scores: jobs scraped before semantic ranking was live keep a legacy
    keyword score (and no embedding), so relevant roles stay underrated. This embeds
    those jobs and rewrites match_score = cosine(job, profile). Heavy (one Ollama
    embed call per un-embedded job) — let it run to completion."""
    from . import ranking as _ranking
    prof = profile.load_profile()
    if not prof.embedding:
        click.echo("Profile has no embedding (Ollama down / embed model not pulled). "
                   "Run `ollama pull nomic-embed-text`, then `jobbot profile rebuild`.")
        return
    click.echo("Re-scoring active jobs against the current profile...")
    stats = _ranking.rescore_jobs(prof, force_reembed=force_reembed)
    click.echo(f"  total active: {stats['total']}")
    click.echo(f"  re-scored:    {stats['rescored']}")
    click.echo(f"  newly embedded: {stats['embedded']}")
    click.echo(f"  failed (kept prior score): {stats['failed']}")


@cli.group("discover")
def discover_group():
    """Discover + review new company ATS boards to scrape."""


@discover_group.command("run")
@click.option("--source", default="websearch,llm",
              help="Comma list: harvest,websearch,llm (default websearch,llm).")
def discover_run(source):
    """Run discovery sources and queue results for review."""
    sources = [s.strip() for s in source.split(",") if s.strip()]
    rep = discovery_service.run_discovery(sources)
    click.echo(f"Discovery: {rep.get('added', 0)} new board(s) queued "
               f"({rep.get('candidates', 0)} candidates, {rep.get('invalid', 0)} invalid).")


@discover_group.command("review")
@click.option("--min-fit", type=float, default=None, help="Hide entries below this fit score.")
def discover_review(min_fit):
    """List pending discovered boards, best-fit first."""
    rows = discovery_service.review(min_fit=min_fit)
    if not rows:
        click.echo("No pending discoveries.")
        return
    for t in rows:
        click.echo(f"[{t.fit_score:.2f}] {t.provider}:{t.key}  {t.display_name}  "
                   f"({t.job_count} jobs, {t.source})")
        if t.fit_reason:
            click.echo(f"        {t.fit_reason}")


@discover_group.command("approve")
@click.argument("coords", nargs=-1, required=True)
@click.option("--watchlist", is_flag=True, help="Also add to the companies.md watchlist.")
def discover_approve(coords, watchlist):
    """Approve coordinates (provider:key ...) into the live target files."""
    parsed = [discovery_service.parse_coord(c) for c in coords]
    rep = discovery_service.approve(parsed, watchlist=watchlist)
    click.echo(f"Approved {rep.get('approved', 0)} board(s)"
               + (" (+ watchlist)" if watchlist else "") + ".")


@discover_group.command("reject")
@click.argument("coords", nargs=-1, required=True)
def discover_reject(coords):
    """Reject coordinates (provider:key ...) so they won't resurface."""
    parsed = [discovery_service.parse_coord(c) for c in coords]
    rep = discovery_service.reject(parsed)
    click.echo(f"Rejected {rep.get('rejected', 0)} board(s).")


@cli.command()
@click.option("--no-email", is_flag=True, help="Skip digest email")
def scrape(no_email):
    """Scrape all sources and store new matches."""
    from .scrape_progress import progress as _prog
    from .user_prefs import load_locations, scrape_locations, includes_remote

    user_locs = load_locations()
    effective = scrape_locations()
    remote = includes_remote()
    console.print(f"[bold cyan]Locations:[/] {', '.join(user_locs)}  "
                  f"[dim](expanded to {len(effective)} cities, remote={'on' if remote else 'off'})[/]")

    result_box: list = []
    err_box: list = []

    def _worker():
        try:
            result_box.append(run_scrape_cycle(send_digest=not no_email))
        except Exception as e:  # noqa: BLE001
            err_box.append(e)

    t = threading.Thread(target=_worker, daemon=True)
    t.start()

    is_tty = sys.stdout.isatty()

    if is_tty:
        with Live(_scrape_panel(_prog.snapshot()), console=console,
                  refresh_per_second=4, transient=False) as live:
            while t.is_alive():
                live.update(_scrape_panel(_prog.snapshot()))
                time.sleep(0.25)
            live.update(_scrape_panel(_prog.snapshot()))
    else:
        last_done = -1
        while t.is_alive():
            snap = _prog.snapshot()
            if snap["done"] != last_done:
                last_done = snap["done"]
                pct = snap["percent"]
                jobs = snap["jobs_found"]
                console.print(f"  [{pct:3d}%] {snap['done']}/{snap['total']} scrapers  {jobs} jobs found")
            time.sleep(1.0)

    if err_box:
        raise err_box[0]
    result = result_box[0] if result_box else {}
    stats_line = (
        f"[bold green]Done.[/]  "
        f"scraped=[yellow]{result.get('scraped', 0)}[/]  "
        f"new=[green]{result.get('new', 0)}[/]  "
        f"dupes=[dim]{result.get('duplicates', 0)}[/]  "
        f"excluded=[dim]{result.get('filtered_excluded', 0)}[/]  "
        f"in-range=[cyan]{result.get('in_range_stored', 0)}[/]"
    )
    console.print(stats_line)


@cli.command(name="list")
@click.option("--status", default=None, help="Filter by status")
@click.option("--limit", default=30)
def list_jobs(status, limit):
    """List jobs in the database."""
    with session() as db:
        q = db.query(Job).order_by(Job.match_score.desc(), Job.discovered_at.desc())
        if status:
            q = q.filter(Job.status == status)
        jobs = q.limit(limit).all()
    table = Table(show_header=True, header_style="bold cyan", show_lines=False,
                  pad_edge=False, expand=True)
    table.add_column("ID", width=5)
    table.add_column("Score", width=6)
    table.add_column("Title", ratio=4, overflow="fold")
    table.add_column("Company", ratio=2, overflow="fold")
    table.add_column("Location", ratio=2, overflow="fold")
    table.add_column("Source", width=14, overflow="fold")
    table.add_column("Status", width=10)
    for j in jobs:
        table.add_row(str(j.id), f"{j.match_score:.2f}", j.title, j.company,
                      j.location or "-", j.source, j.status)
    console.print(table)


@cli.command(name="add-job")
@click.argument("url_arg", required=False)
@click.option("--url", "--link", "url_opt", default="", help="Job posting URL / link.")
@click.option("--description", "-d", "--desc", "description", default="", help="Job description text.")
@click.option("--file", "-f", "desc_file", type=click.Path(exists=True), help="Read description from file.")
@click.option("--title", "-t", default="", help="Job title.")
@click.option("--company", "-c", default="", help="Company name.")
@click.option("--location", "-l", default="", help="Job location.")
@click.option("--tailor/--no-tailor", default=True, help="Immediately tailor resume after saving (default: True).")
@click.option("--local", "force_local", is_flag=True, help="Tailor locally using Ollama.")
@click.option("--offline", "force_offline", is_flag=True, help="Tailor deterministically offline (no AI).")
def add_job_cmd(url_arg, url_opt, description, desc_file, title, company, location,
                tailor, force_local, force_offline):
    """Manually add a job by link and description, store it in the database, and tailor your resume."""
    link = (url_opt or url_arg or "").strip()

    if desc_file:
        description = Path(desc_file).read_text(encoding="utf-8").strip()

    if not link and not description and sys.stdin.isatty():
        link = click.prompt("Job posting URL / Link", default="", show_default=False).strip()
        title = click.prompt("Job title", default=title or "", show_default=bool(title)).strip()
        company = click.prompt("Company name", default=company or "", show_default=bool(company)).strip()
        location = click.prompt("Location", default=location or "Remote", show_default=True).strip()

    if link and not description:
        console.print(f"[cyan]Attempting to fetch description from[/] {link}...")
        description = _fetch_description_from_url(link)
        if description:
            console.print(f"[green]Extracted {len(description)} chars from page.[/]")

    if not description and sys.stdin.isatty():
        console.print("[yellow]Paste or type the job description below (finish with Ctrl+D or Ctrl+Z on Windows):[/]")
        lines = []
        try:
            while True:
                lines.append(input())
        except (EOFError, KeyboardInterrupt):
            pass
        description = "\n".join(lines).strip()

    if not link and not description:
        console.print("[red]Error:[/] You must provide at least a job link or a description.")
        raise SystemExit(1)

    job = _insert_manual_job(url=link, description=description, title=title, company=company, location=location)
    console.print(f"[green]Job #{job.id} saved to database.[/] "
                  f"[bold]{job.title}[/] @ [bold]{job.company}[/] "
                  f"(score: {job.match_score:.2f})")

    if tailor:
        console.print(f"[cyan]Tailoring resume for Job #{job.id}...[/]")
        _execute_tailor(job.id, local=force_local, offline=force_offline)


@cli.command()
@click.argument("job_id", type=int, required=False)
@click.option("--url", "--link", "url", default="", help="Job posting URL/link to manually add and tailor.")
@click.option("--description", "-d", "--desc", "description", default="", help="Job description text.")
@click.option("--file", "-f", "desc_file", type=click.Path(exists=True), help="Read description from file.")
@click.option("--title", "-t", default="", help="Job title (for manual job).")
@click.option("--company", "-c", default="", help="Company name (for manual job).")
@click.option("--location", "-l", default="", help="Location (for manual job).")
@click.option("--local", "force_local", is_flag=True, help="Tailor locally using Ollama.")
@click.option("--offline", "force_offline", is_flag=True, help="Tailor deterministically offline (no AI).")
def tailor(job_id, url, description, desc_file, title, company, location, force_local, force_offline):
    """Run AI tailoring on a job (Claude or Gemini per AI_PROVIDER), or manually add a job to tailor."""
    if job_id is None:
        if desc_file:
            description = Path(desc_file).read_text(encoding="utf-8").strip()

        if url and not description:
            console.print(f"[cyan]Attempting to fetch description from[/] {url}...")
            description = _fetch_description_from_url(url)
            if description:
                console.print(f"[green]Extracted {len(description)} chars from page.[/]")

        if not url and not description and sys.stdin.isatty():
            url = click.prompt("Job posting URL", default="", show_default=False).strip()
            title = click.prompt("Job title", default=title or "", show_default=bool(title)).strip()
            company = click.prompt("Company name", default=company or "", show_default=bool(company)).strip()
            location = click.prompt("Location", default=location or "Remote", show_default=True).strip()
            console.print("[yellow]Paste or type the job description below (finish with Ctrl+D or Ctrl+Z on Windows):[/]")
            lines = []
            try:
                while True:
                    lines.append(input())
            except (EOFError, KeyboardInterrupt):
                pass
            description = "\n".join(lines).strip()

        if not url and not description:
            console.print("[red]Error:[/] Provide a JOB_ID or specify --url / --description.")
            raise SystemExit(1)

        job = _insert_manual_job(url=url, description=description, title=title, company=company, location=location)
        console.print(f"[green]Job #{job.id} saved to database.[/] "
                      f"[bold]{job.title}[/] @ [bold]{job.company}[/] "
                      f"(score: {job.match_score:.2f})")
        job_id = job.id

    _execute_tailor(job_id, local=force_local, offline=force_offline)


@cli.command(name="tailor-local")
@click.argument("job_id", type=int)
@click.option("--rounds", type=int, default=None,
              help="Refinement rounds (draft->critique->revise). Default: OLLAMA_REFINE_ROUNDS.")
def tailor_local(job_id, rounds):
    """Tailor locally with Ollama + iterative refinement -> Word doc (.docx).

    Free, offline backup path. Auto-starts the local Ollama server/model if
    needed. Use when cloud credits are out or you want a fully local run.
    """
    result = tailor_for_job_local(job_id, rounds=rounds,
                                  progress=lambda m: console.print(f"[dim]{_cli_safe(m)}[/]"))
    _score = result.get("self_score")
    _shown = f"{_score:.2f}" if isinstance(_score, (int, float)) else "unknown"
    console.print(f"[green]Tailored locally[/] "
                  f"({result['rounds']} round(s), self-score {_shown}).")
    console.print(f"Word doc: [bold]{result['docx']}[/]")
    console.print(f"Cover letter: {result['cover_letter']}")
    _defects = result.get("defects") or []
    if _defects:
        console.print(f"\n[bold red]DO NOT SEND AS-IS - "
                      f"{len(_defects)} verified defect(s):[/]")
        for d in _defects:
            console.print(f"  [red]x[/] {_cli_safe(d)}")
        console.print("[yellow]These were checked against your resume and "
                      "profile, not guessed. Fix them before applying.[/]")


@cli.command(name="prompt-pack")
@click.argument("job_id", type=int)
@click.option("--no-docx", is_flag=True, help="Skip the offline .docx render.")
def prompt_pack(job_id, no_docx):
    """Build a copy-paste AI prompt + materials pack for a job (no API key needed).

    Writes output/prompt_packs/<job>/ with a ready-to-paste tailoring prompt
    (PASTE_THIS_PROMPT.md), standalone cover-letter/networking prompts, a keyword
    analysis, and a fully AI-free tailored resume (.docx/.md).
    """
    from .prompt_pack import build_prompt_pack
    result = build_prompt_pack(job_id, make_docx=not no_docx)
    console.print(f"[green]Prompt pack built[/] (offline match {result['match_score']:.0%}).")
    console.print(f"Folder: [bold]{result['dir']}[/]")
    console.print(f"Paste into any AI chat: [bold]{result['prompt']}[/]")
    if result.get("offline_resume_docx"):
        console.print(f"AI-free resume: {result['offline_resume_docx']}")
    if result.get("matched"):
        console.print(f"[dim]Emphasise:[/] {_cli_safe(', '.join(result['matched'][:12]))}")
    if result.get("missing"):
        console.print(f"[dim]Don't fake:[/] {_cli_safe(', '.join(result['missing'][:12]))}")


@cli.command(name="tailor-offline")
@click.argument("job_id", type=int)
def tailor_offline(job_id):
    """Tailor a resume deterministically (NO AI) from your verified profile.

    Re-orders/emphasises your real skills to match the job and writes a .docx +
    .md. Cannot fabricate anything. Fast, free, fully offline.
    """
    _execute_tailor(job_id, offline=True)


@cli.command()
@click.argument("application_id", type=int)
@click.argument("to_email")
@click.option("--note", default="", help="Intro note to prepend")
def apply(application_id, to_email, note):
    """Send an application by email."""
    send_application(application_id, to_email, note)
    console.print(f"[green]Application {application_id} sent to {to_email}.[/]")


@cli.command()
def run():
    """Run one full scrape cycle (alias for scrape)."""
    ctx = click.get_current_context()
    ctx.invoke(scrape, no_email=False)


@cli.command()
def schedule():
    """Run the scheduler in the foreground (Ctrl+C to stop)."""
    from .scheduler import start_scheduler
    console.print(f"[cyan]Starting scheduler (every {settings.schedule_interval_hours}h).[/]")
    run_scrape_cycle(send_digest=True)
    start_scheduler(block=True)


@cli.command()
def serve():
    """Run the web dashboard."""
    from .web import serve as run_server
    console.print(f"[cyan]Dashboard at http://{settings.dashboard_host}:{settings.dashboard_port}[/]")
    run_server()


@cli.group("interview")
def interview_group():
    """Interview coach: prep packs, mock practice, history."""


@interview_group.command("prep")
@click.argument("job_id", type=int)
def interview_prep_cmd(job_id):
    """Generate the structured prep pack + question bank for a job."""
    from . import interview_coach as ic
    result = ic.generate_pack(job_id)
    if result["error"]:
        console.print(f"[yellow]Prep had a problem: {result['error']}[/]")
    bank = result.get("bank") or {}
    counts = ", ".join(f"{k}={len(bank.get(k) or [])}"
                       for k in ("behavioral", "technical", "research", "contact"))
    console.print(f"Question bank: {counts}")
    if result["path"]:
        console.print(f"[green]Saved to {result['path']}[/]")


@interview_group.command("practice")
@click.argument("job_id", type=int)
@click.option("--mode", default="general",
              type=click.Choice(["general", "technical", "behavioral",
                                 "research", "contact"]),
              help="Question mix. 'contact' role-plays a real contact you know.")
@click.option("--contact", "contact_id", type=int, default=None,
              help="Force a specific contact id for --mode contact.")
def interview_practice_cmd(job_id, mode, contact_id):
    """Interactive terminal mock interview (same engine as the dashboard)."""
    from . import interview_coach as ic
    from .models import session as db_session

    with db_session() as db:
        started = ic.start(db, job_id, mode, contact_id)
        if started["error"]:
            console.print(f"[yellow]{started['error']}[/]")
            return
        sid = started["session_id"]
        question = started["question"]
        console.print("[dim]Type your answer and press Enter. "
                      "Type 'quit' to end early.[/]\n")

        while question is not None:
            console.print(f"[bold]Q{question['idx'] + 1}"
                          f"{' (follow-up)' if question['parent_idx'] is not None else ''}"
                          f":[/] {question['question']}")
            if question["grounded_in"]:
                console.print(f"[dim]based on: {question['grounded_in']}[/]")
            answer = click.prompt("Your answer", default="", show_default=False)
            if answer.strip().lower() in ("quit", "exit", "q"):
                break
            scored = ic.submit_answer(db, question["turn_id"], answer)
            if scored["error"]:
                console.print(f"[yellow]{scored['error']}[/]\n")
                continue
            if scored["score"] is not None:
                console.print(f"[bold]Score:[/] {scored['score']}/5  "
                              + "  ".join(f"{a}={scored['rubric'][a]:.0f}"
                                          for a in ic.RUBRIC_AXES))
            if scored["critique"]:
                console.print(f"{scored['critique']}")
            console.print("")
            question = scored["follow_up"] or ic.next_question(db, sid)

        done = ic.end(db, sid)
        console.print("[bold]--- Session summary ---[/]")
        if done["overall_score"] is not None:
            console.print(f"Overall: {done['overall_score']}/5 "
                          f"over {done['turns']} answered question(s)")
        if done["summary"]:
            console.print(done["summary"])


@interview_group.command("history")
@click.argument("job_id", type=int, required=False)
@click.option("--session", "session_id", type=int, default=None,
              help="Print one session's transcript instead of the list.")
def interview_history_cmd(job_id, session_id):
    """List past practice sessions, or print one session's transcript."""
    from . import interview_coach as ic
    from .models import session as db_session

    if session_id is not None:
        with db_session() as db:
            rows = ic.transcript(db, session_id)
        if not rows:
            console.print(f"[yellow]No transcript for session {session_id}.[/]")
            return
        for r in rows:
            tag = " (follow-up)" if r["parent_idx"] is not None else ""
            console.print(f"[bold]Q{r['idx'] + 1}{tag}:[/] {r['question']}")
            if r["answer"]:
                console.print(f"A: {r['answer']}")
            if r["score"] is not None:
                console.print(f"Score: {r['score']}/5")
            if r["critique"]:
                console.print(f"[dim]{r['critique']}[/]")
            console.print("")
        return

    data = ic.history(job_id)
    if not data["sessions"]:
        console.print("[yellow]No practice sessions yet.[/]")
        return
    from rich import box as _box
    table = Table(title="Interview practice sessions", box=_box.ASCII)
    for col in ("id", "role", "company", "mode", "answers", "score", "when"):
        table.add_column(col)
    for s in data["sessions"]:
        score = f"{s['overall_score']}/5" if s["overall_score"] is not None else "open"
        table.add_row(str(s["session_id"]), s["title"], s["company"], s["mode"],
                      str(s["turns"]), score, s["created_at"].strftime("%Y-%m-%d %H:%M"))
    console.print(table)
    if data["mean"] is not None:
        console.print(f"Overall average: {data['mean']}/5")


@interview_group.command("config")
@click.option("--weight", type=float, default=None,
              help="Max ranking bonus from practice (0.0-0.5). Omit to show current.")
def interview_config_cmd(weight):
    """Show or set the interview-coach tunables."""
    from .config import settings as s
    from .model_registry import _write_env

    if weight is None:
        console.print(f"interview_coach_enabled    = {s.interview_coach_enabled}")
        console.print(f"interview_followup_max     = {s.interview_followup_max}")
        console.print(f"interview_practice_weight  = {s.interview_practice_weight}")
        console.print("\nSet the cap with: jobbot interview config --weight 0.08")
        return
    if not 0.0 <= weight <= 0.5:
        raise click.BadParameter("weight must be between 0.0 and 0.5")
    _write_env({"INTERVIEW_PRACTICE_WEIGHT": str(weight)})
    s.interview_practice_weight = weight
    console.print(f"[green]interview_practice_weight set to {weight}[/] (saved to .env)")


@cli.command(name="auto-apply")
@click.option("--confirm", is_flag=True, help="Actually send (otherwise dry-run + plan only)")
@click.option("--max", "max_applies", type=int, default=None, help="Cap for this run")
@click.option("--min-score", type=float, default=None, help="Match score floor (default from .env)")
@click.option("--min-callback", type=float, default=None, help="P(callback) floor (default from .env)")
@click.option("--watchlist-only", is_flag=True, help="Only apply to watchlist companies")
@click.option("--default-recipient", default=None, help="Email to use when posting has none")
@click.option("--require-score-outside-watchlist", type=float, default=None,
              help="Outside-watchlist score floor (default from .env, typ. 0.65)")
def auto_apply(confirm, max_applies, min_score, min_callback, watchlist_only, default_recipient,
               require_score_outside_watchlist):
    """Auto-apply to jobs that meet user thresholds AND model recommendation."""
    from .auto_apply import AutoApplySettings, run_auto_apply
    cfg = AutoApplySettings(
        confirm=confirm,
        max_applies=max_applies,
        min_match_score=min_score if min_score is not None else settings.auto_apply_min_score,
        min_callback_probability=min_callback if min_callback is not None else settings.auto_apply_min_callback_prob,
        require_watchlist_or_score=(require_score_outside_watchlist
                                    if require_score_outside_watchlist is not None
                                    else settings.auto_apply_require_watchlist_or_score),
        watchlist_only=watchlist_only or settings.auto_apply_watchlist_only,
        default_recipient=default_recipient if default_recipient is not None else settings.auto_apply_default_recipient,
        skip_stale_days=settings.auto_apply_skip_stale_days,
    )
    if not cfg.confirm:
        console.print("[yellow]DRY RUN — pass --confirm to actually send.[/]")
    report = run_auto_apply(cfg)
    table = Table(title=f"Auto-apply plan ({len(report['details'])} jobs)",
                  show_header=True, header_style="bold cyan", expand=True)
    table.add_column("#", width=5)
    table.add_column("Title", ratio=3, overflow="fold")
    table.add_column("Company", ratio=2, overflow="fold")
    table.add_column("P(callback)", width=11)
    table.add_column("Recipient", ratio=2, overflow="fold")
    table.add_column("Action", width=22, overflow="fold")
    for d in report["details"]:
        table.add_row(str(d["job_id"]), d["title"], d["company"],
                      f"{d['p_callback']:.1%}", d["recipient"], d["action"])
    console.print(table)
    console.print(f"\nQuota remaining (before/after): "
                  f"{report['quota_remaining_before']} -> {report.get('quota_remaining_after','-')}")
    if report["errors"]:
        console.print(f"[red]Errors:[/] {report['errors']}")


@cli.command(name="apply-queue")
@click.option("--build", "build_n", type=int, default=0,
              help="Build Apply Kits for the top N candidate jobs (fires AI calls)")
@click.option("--rebuild", is_flag=True, help="Rebuild kits even if one already exists")
@click.option("--min-score", type=float, default=None, help="Match score floor (default from .env)")
@click.option("--min-callback", type=float, default=None, help="P(callback) floor (default from .env)")
def apply_queue(build_n, rebuild, min_score, min_callback):
    """Show the apply queue; --build N prepares kits for the top N jobs."""
    from .auto_apply import AutoApplySettings, apply_queue_status, prepare_apply_queue

    if build_n > 0:
        cfg = AutoApplySettings(
            max_applies=max(build_n * 3, 10),
            min_match_score=min_score if min_score is not None else settings.auto_apply_min_score,
            min_callback_probability=min_callback if min_callback is not None else settings.auto_apply_min_callback_prob,
            require_watchlist_or_score=settings.auto_apply_require_watchlist_or_score,
            watchlist_only=settings.auto_apply_watchlist_only,
            skip_stale_days=settings.auto_apply_skip_stale_days,
        )
        console.print(f"[cyan]Building Apply Kits for up to {build_n} jobs "
                      f"(tailoring + form fetch + AI answers — ~30-60s each)…[/]")
        report = prepare_apply_queue(cfg, build_limit=build_n, rebuild=rebuild)
        for b in report["built"]:
            ready = "[green]READY[/]" if b["ready"] else f"[yellow]{b['blanks']} need you[/]"
            console.print(f"  built  #{b['job_id']:>4}  {b['title'][:50]:50} "
                          f"@ {b['company'][:24]:24} via {b['source']:10} {ready}")
        if report["skipped_existing"]:
            console.print(f"  [dim]{len(report['skipped_existing'])} already had kits "
                          f"(use --rebuild to refresh)[/]")
        for err in report["errors"]:
            console.print(f"  [red]error #{err['job_id']}: {err['error']}[/]")

    rows = apply_queue_status()
    if not rows:
        console.print("[yellow]Apply queue is empty — run `jobbot apply-queue --build 5`.[/]")
        return
    table = Table(title=f"Apply queue ({len(rows)} prepared)", show_header=True,
                  header_style="bold cyan", expand=True)
    table.add_column("Job", width=5)
    table.add_column("Title", ratio=3, overflow="fold")
    table.add_column("Company", ratio=2, overflow="fold")
    table.add_column("P(callback)", width=11)
    table.add_column("Form", width=10)
    table.add_column("Status", width=14)
    for r in rows:
        status = "[green]READY[/]" if r["ready"] else f"[yellow]{r['blanks']} need you[/]"
        table.add_row(str(r["job_id"]), r["title"], r["company"],
                      f"{r['p_callback']:.1%}", r["source"], status)
    console.print(table)


@cli.command(name="apply-run")
@click.argument("job_id", type=int)
@click.option("--headed/--no-headed", default=None,
              help="Show (or hide) the Chromium window. Default: config.")
def apply_run_cmd(job_id: int, headed):
    """Reliably auto-apply to a Greenhouse JOB_ID via Playwright (verify-then-1-click)."""
    run_id = apply_runner.run_apply(job_id, headed=headed)
    click.echo(f"apply-run complete — ApplyRun id {run_id}")


def _workday_login_target(job_id, url):
    if url:
        return url
    if job_id:
        with session() as db:
            j = db.get(Job, job_id)
            if j and j.url:
                return j.url
    return "https://www.myworkdayjobs.com/"


@cli.command(name="workday-login")
@click.argument("job_id", type=int, required=False)
@click.option("--url", default=None,
              help="Page to open for login (default: the job's URL, else Workday home).")
def workday_login_cmd(job_id, url):
    """Log into Workday ONCE; the session is saved and reused by apply-run."""
    from .browser_engine import BrowserEngine
    init_db()
    target = _workday_login_target(job_id, url)
    console.print(f"Opening [bold]{_cli_safe(target)}[/] in the Workday profile window.")
    console.print("[yellow]Log in and finish any MFA[/], then come back here.")
    with BrowserEngine(headed=True, user_data_dir=settings.workday_profile_dir) as eng:
        eng.goto(target)
        try:
            click.prompt("Press Enter AFTER you are fully logged in to save the session",
                         default="", show_default=False)
        except (EOFError, click.Abort):
            pass
    console.print("[green]Session saved.[/] Re-run `apply-run` — you'll already be signed in.")


@cli.group(name="workday-account")
def workday_account():
    """Per-tenant Workday sign-in credentials (fallback for a lapsed session)."""


@workday_account.command("add")
@click.argument("tenant")
@click.option("--email", prompt="Workday account email",
              help="The email address the Workday account is registered under.")
def workday_account_add(tenant, email):
    """Store credentials for TENANT (a host or a full job URL)."""
    from . import workday_accounts

    host = workday_accounts.tenant_for(tenant)
    if not host:
        raise click.BadParameter(
            f"{tenant!r} is not a Workday tenant. Pass a host like "
            f"acme.wd1.myworkdayjobs.com or a full job URL.")
    password = click.prompt("Workday password", hide_input=True,
                            confirmation_prompt=True)
    workday_accounts.remember(host, email, password)
    console.print(f"Stored credentials for [bold]{_cli_safe(host)}[/] "
                  f"({_cli_safe(email)}).")
    console.print(f"File: {_cli_safe(str(workday_accounts._path()))}")
    console.print("[yellow]That file holds the password in plain text[/] and is "
                  "gitignored. On Windows it carries no OS-level protection.")


@workday_account.command("list")
def workday_account_list():
    """Show stored tenants and their email addresses (never the password)."""
    from . import workday_accounts

    rows = workday_accounts.known_tenants()
    if not rows:
        console.print("No stored Workday accounts. "
                      "Add one with `jobbot workday-account add <tenant>`.")
        return
    for row in rows:
        console.print(f"  {_cli_safe(row['tenant'])}  {_cli_safe(row['email'])}  ***")


@workday_account.command("remove")
@click.argument("tenant")
def workday_account_remove(tenant):
    """Forget the stored credentials for TENANT."""
    from . import workday_accounts

    if workday_accounts.forget(tenant):
        console.print(f"Removed {_cli_safe(workday_accounts.tenant_for(tenant))}.")
    else:
        console.print("Nothing stored for that tenant.")


@cli.command()
@click.option("--send", is_flag=True,
              help="Email all due follow-ups that have a known contact address (asks to confirm).")
def followups(send):
    """List applications due for a follow-up email."""
    from .followup import followups_due, send_followup
    rows = followups_due()
    if not rows:
        console.print(f"Nothing due. Applications get a follow-up nudge after "
                      f"{settings.followup_after_days} days.")
        return
    table = Table(title=f"Follow-ups due ({len(rows)})", show_header=True,
                  header_style="bold cyan", expand=True)
    table.add_column("App", width=5)
    table.add_column("Title", ratio=3, overflow="fold")
    table.add_column("Company", ratio=2, overflow="fold")
    table.add_column("Days since sent", width=16)
    table.add_column("Follow-up #", width=11)
    table.add_column("Contact email", ratio=2, overflow="fold")
    for r in rows:
        table.add_row(str(r["application_id"]), r["title"], r["company"],
                      str(r["days_since_sent"]),
                      f"#{r['followup_count'] + 1} of {settings.followup_max}",
                      r["contact_email"] or "[no contact]")
    console.print(table)
    if not send:
        console.print("\n[dim]Pass --send to email the ones with a known contact, "
                      "or use the dashboard's Follow-ups page to review drafts.[/]")
        return
    sendable = [r for r in rows if r["contact_email"]]
    if not sendable:
        console.print("[yellow]No due applications have a known contact email — "
                      "run `jobbot contacts JOB_ID` first.[/]")
        return
    for r in sendable:
        console.print(f"  App #{r['application_id']}: {r['title']} @ {r['company']} "
                      f"-> {r['contact_email']}")
    if not click.confirm(f"Send {len(sendable)} follow-up email(s)?"):
        console.print("Aborted.")
        return
    ok = fail = 0
    for r in sendable:
        try:
            send_followup(r["application_id"], r["contact_email"])
            console.print(f"  sent #{r['application_id']} -> {r['contact_email']}")
            ok += 1
        except Exception as exc:  # noqa: BLE001
            console.print(f"  [red]error #{r['application_id']}: {exc}[/]")
            fail += 1
    console.print(f"Done. Sent={ok} Errors={fail}")


@cli.command()
@click.argument("job_id", type=int)
@click.option("--refresh", is_flag=True, help="Re-run the search even if contacts already exist")
def contacts(job_id, refresh):
    """Find recruiter / hiring-team contacts for a job + a networking note."""
    from .contacts import find_contacts, load_contacts, networking_note
    found = [] if refresh else load_contacts(job_id)
    if not found:
        console.print("[cyan]Searching for contacts (posting text + public web)...[/]")
        found = find_contacts(job_id, save=True)
    if not found:
        console.print(f"[yellow]No contacts found for job {job_id}.[/] "
                      "Tip: set GOOGLE_CSE_KEY/GOOGLE_CSE_CX in .env — free search "
                      "engines bot-block automated lookups.")
        return
    table = Table(title=f"Contacts for job {job_id}", show_header=True,
                  header_style="bold cyan", expand=True)
    table.add_column("Name", ratio=2)
    table.add_column("Role", ratio=2, overflow="fold")
    table.add_column("Email", ratio=2, overflow="fold")
    table.add_column("LinkedIn", ratio=3, overflow="fold")
    for c in found:
        table.add_row(c.get("name") or "-", c.get("role") or "-",
                      c.get("email") or "-", c.get("linkedin") or "-")
    console.print(table)
    console.print("\n[bold]Networking note (LinkedIn connect message):[/]")
    note = networking_note(job_id, found[0])
    console.print(f"  {note}  [dim]({len(note)} chars)[/]")


@cli.command()
@click.argument("job_id", type=int)
@click.option("--contact", "contact_idx", type=int, default=1,
              help="Which contact to write to (1-based, from `contacts JOB_ID`). Default 1.")
@click.option("--channel", type=click.Choice(["linkedin", "email", "both"]),
              default="both", help="Message type to draft.")
@click.option("--refresh", is_flag=True, help="Re-find contacts before drafting.")
def network(job_id, contact_idx, channel, refresh):
    """Draft an AI-tailored networking message to a contact for a job."""
    from .contacts import (find_contacts, load_contacts, networking_note,
                           networking_email, networking_targets)
    found = [] if refresh else load_contacts(job_id)
    if not found:
        console.print("[cyan]Finding contacts first (posting text + public web)...[/]")
        found = find_contacts(job_id, save=True)
    if not found:
        console.print(f"[yellow]No contacts auto-found for job {job_id}.[/] "
                      "Free search engines bot-block lookups; set "
                      "GOOGLE_CSE_KEY/GOOGLE_CSE_CX in .env for automated search.")
        targets = networking_targets(job_id)
        if targets:
            console.print("\n[bold]People-search links (open in your browser):[/]")
            for t in targets:
                console.print(f"  - {t['label']}\n    [blue]{t['url']}[/]")
        return
    if contact_idx < 1 or contact_idx > len(found):
        console.print(f"[red]--contact {contact_idx} out of range (1..{len(found)}).[/]")
        raise SystemExit(1)
    contact = found[contact_idx - 1]
    who = contact.get("name") or contact.get("email") or "contact"
    console.print(f"[bold]Drafting outreach to:[/] {who} "
                  f"[dim]({contact.get('role') or 'unknown role'})[/]")
    console.print("[dim]Tailoring with AI (Gemini first)...[/]")

    if channel in ("linkedin", "both"):
        note = networking_note(job_id, contact)
        console.print("\n[bold cyan]LinkedIn connection note[/] "
                      f"[dim]({len(note)} chars)[/]")
        console.print(_cli_safe(note))
    if channel in ("email", "both"):
        em = networking_email(job_id, contact)
        console.print("\n[bold cyan]Networking email[/]")
        console.print(f"[bold]Subject:[/] {_cli_safe(em.get('subject', ''))}")
        console.print(_cli_safe(em.get("body", "")))
    if contact.get("email"):
        console.print(f"\n[dim]Send to:[/] {contact['email']}")
    elif contact.get("linkedin"):
        console.print(f"\n[dim]Connect via:[/] {contact['linkedin']}")


@cli.command("network-research")
@click.argument("company_name")
@click.option("--contacts-json", "contacts_json", default=None,
              help="Path to a JSON file with a list of contact dicts. "
                   "If omitted, reads from stdin.")
@click.option("--output", "output_path", default=None,
              help="Output markdown file. Defaults to "
                   "output/networking/{company_slug}_research_packages.md")
def network_research(company_name, contacts_json, output_path):
    """Build research intelligence packages for a batch of networking contacts."""
    from .research_enricher import run_enrichment_cli
    console.print(f"[cyan]Building research packages for {company_name}...[/]")
    out = run_enrichment_cli(contacts_json, company_name, output_path)
    console.print(f"[green]Research packages written to:[/] {out}")


@cli.command("network-package")
@click.argument("name")
@click.option("--company", default="", help="Their organization / company / institution.")
@click.option("--role", default="", help="Their role / title (helps research).")
@click.option("--context", default="",
              help="How you connect (e.g. 'same university', 'met at retreat').")
@click.option("--goals", default="",
              help="What you want (e.g. 'advice, collaboration, job leads').")
@click.option("--format", "fmt",
              type=click.Choice(["call", "video", "email", "linkedin", "event", "coffee"]),
              default="call", help="The networking format you're preparing for.")
@click.option("--no-podcast", is_flag=True, help="Skip the audio podcast.")
@click.option("--podcast-style",
              type=click.Choice(["two_host", "narrator", "coach"]), default="two_host")
@click.option("--out-dir", default=None,
              help="Output directory (default: output/networking/<slug>/).")
def network_package(name, company, role, context, goals, fmt, no_podcast,
                    podcast_style, out_dir):
    """Build the ULTIMATE networking package for one person."""
    from . import networking_package as npkg
    console.print(f"[cyan]Building the ultimate networking package for "
                  f"{name}...[/] (research -> prep -> outreach -> one-pager"
                  f"{'' if no_podcast else ' -> podcast'})")
    try:
        res = npkg.build_package(
            name, company=company, role=role, context=context, goals=goals,
            fmt=fmt, do_podcast=not no_podcast, podcast_style=podcast_style,
            out_dir=out_dir)
    except Exception as exc:  # noqa: BLE001
        console.print(f"[red]Networking package failed:[/] {exc}")
        raise SystemExit(1)
    if not res.get("used_llm"):
        console.print("[yellow]Note: LLM unavailable — prep is a raw-evidence "
                      "template. Start Ollama for a full tailored brief.[/]")
    console.print(f"[green]Folder:[/] {res['dir']}")
    console.print(f"[green]Prep:[/] {res['prep_path']}")
    if res.get("outreach_path"):
        console.print(f"[green]Outreach:[/] {res['outreach_path']}")
    if res.get("onepager_path"):
        console.print(f"[green]One-pager:[/] {res['onepager_path']}")
    if res.get("podcast_path"):
        console.print(f"[green]Podcast:[/] {res['podcast_path']}")


@cli.command()
@click.argument("job_ids", nargs=-1, type=int)
@click.option("--status", help="Delete all jobs with this status (e.g. 'new', 'rejected').")
@click.option("--all", "delete_all", is_flag=True, help="Delete ALL jobs (requires --yes).")
@click.option("--yes", is_flag=True, help="Skip the confirmation prompt.")
def delete(job_ids, status, delete_all, yes):
    """Delete job(s) by id, by --status, or --all. Cascades to applications."""
    init_db()
    with session() as db:
        q = db.query(Job)
        if delete_all:
            targets = q.all()
            label = "ALL jobs"
        elif status:
            targets = q.filter(Job.status == status).all()
            label = f"jobs with status '{status}'"
        elif job_ids:
            targets = q.filter(Job.id.in_(job_ids)).all()
            label = f"job(s) {', '.join(map(str, job_ids))}"
        else:
            console.print("[yellow]Nothing to delete.[/] Pass job ids, --status, or --all.")
            return
        if not targets:
            console.print(f"[yellow]No matching {label}.[/]")
            return
        n = len(targets)
        if (delete_all or status or n > 1) and not yes:
            console.print(f"[red]About to delete {n} {label}[/] (and their "
                          "applications). Re-run with [bold]--yes[/] to confirm.")
            for j in targets[:10]:
                console.print(f"  - {j.id}: {_cli_safe(j.title)} @ {_cli_safe(j.company)}")
            if n > 10:
                console.print(f"  ...and {n - 10} more")
            return
        for j in targets:
            db.delete(j)
        db.commit()
    console.print(f"[green]Deleted {n} {label}.[/]")


@cli.command(name="prefetch-contacts")
@click.option("--limit", type=int, default=20, show_default=True,
              help="How many top-scoring jobs to fetch contacts for.")
@click.option("--min-score", type=float, default=0.0, show_default=True,
              help="Only jobs with match_score >= this.")
@click.option("--refresh", is_flag=True, help="Re-fetch even jobs that already have contacts.")
@click.option("--delay", type=float, default=2.0, show_default=True,
              help="Seconds to wait between jobs (DuckDuckGo rate-limits bursts).")
def prefetch_contacts_cmd(limit, min_score, refresh, delay):
    """Bulk pre-fetch contacts for the top-N matched jobs (keyless ddgs)."""
    from .contacts import prefetch_contacts, active_search_provider
    console.print(f"[cyan]Pre-fetching contacts for up to {limit} top jobs "
                  f"(provider: {active_search_provider()})...[/]")

    def _prog(done, total, title, n):
        console.print(f"  [{done}/{total}] {_cli_safe(title)[:60]} -> "
                      f"{n} contact(s)")

    stats = prefetch_contacts(limit=limit, min_score=min_score,
                              refresh=refresh, delay=delay, progress=_prog)
    if stats["selected"] == 0:
        console.print("[yellow]No jobs needed contacts.[/] "
                      "(All top jobs already have them — use --refresh to redo, "
                      "or lower --min-score.)")
        return
    console.print(f"[green]Done.[/] Processed {stats['processed']} job(s); "
                  f"{stats['jobs_with_contacts']} got contacts "
                  f"({stats['total_contacts']} total).")


@cli.command(name="cse-test")
def cse_test():
    """Verify the search backend works (needed for auto-populating contacts)."""
    from .contacts import search_selftest
    console.print("[cyan]Probing contact-search backend...[/]")
    res = search_selftest()
    if not res["configured"]:
        console.print("[red]Not configured.[/] Set [bold]SERPER_API_KEY[/] "
                      "(recommended — get a free key at https://serper.dev) or "
                      "GOOGLE_CSE_KEY + GOOGLE_CSE_CX in .env, then re-run "
                      "[bold]python -m jobbot cse-test[/].")
        raise SystemExit(1)
    provider = res.get("provider", "?")
    if res["ok"]:
        console.print(f"[green]Working![/] (provider: {provider}) "
                      f"Returned {res['count']} result(s). Contacts will now "
                      "auto-populate when you open a job.")
        for url in res["sample"]:
            console.print(f"  [blue]{url}[/]")
    else:
        console.print(f"[red]Search call failed[/] (provider: {provider}): "
                      f"{_cli_safe(res['error'])}")
        raise SystemExit(1)


@cli.command(name="ollama-search-test")
@click.argument("query", default="computational structural biology research scientist")
def ollama_search_test(query):
    """Verify Ollama hosted web search works (live results for research)."""
    from . import ollama_search
    console.print("[cyan]Probing Ollama web search...[/]")
    res = ollama_search.selftest(query)
    if not res["configured"]:
        console.print("[red]Not configured.[/] Get a free key at "
                      "[bold]https://ollama.com[/] (Settings -> API keys) and add "
                      "[bold]OLLAMA_API_KEY=...[/] to your .env, then re-run "
                      "[bold]python -m jobbot ollama-search-test[/].")
        raise SystemExit(1)
    if res["ok"]:
        console.print(f"[green]Working![/] Returned {res['count']} result(s). "
                      "Research/networking now use live Ollama web search.")
        for r in ollama_search.web_search(query, 3):
            console.print(f"  [blue]{r['url']}[/] — {r['title']}")
    else:
        console.print(f"[red]Search failed:[/] {_cli_safe(res['error'])}")
        raise SystemExit(1)


@cli.command(name="fill-plan")
@click.argument("job_id", type=int)
def fill_plan_cmd(job_id):
    """Turn a job's Apply Kit into ordered browser fill steps (JSON)."""
    import json as _json
    from .browser_apply import write_fill_plan
    try:
        path = write_fill_plan(job_id)
    except ValueError as e:
        console.print(f"[red]{e}[/]")
        raise SystemExit(1)
    plan = _json.loads(Path(path).read_text(encoding="utf-8"))
    steps = plan["steps"]
    console.print(f"[green]Fill plan written:[/] {path}")
    console.print(f"  {plan['title']} @ {plan['company']} — {len(steps)} steps, "
                  f"{len(plan['needs_user'])} need you")
    table = Table(show_header=True, header_style="bold cyan", expand=True)
    table.add_column("#", width=4)
    table.add_column("Action", width=13)
    table.add_column("Field / note", ratio=2, overflow="fold")
    table.add_column("Value", ratio=3, overflow="fold")
    for s in steps:
        table.add_row(str(s.get("step", "")), s.get("action", ""),
                      (s.get("label") or s.get("note") or "")[:70],
                      (str(s.get("value") or ""))[:80])
    console.print(table)
    if plan["needs_user"]:
        console.print("[yellow]Needs your input before submitting:[/]")
        for lbl in plan["needs_user"]:
            console.print(f"  - {lbl}")


@cli.command(name="predict-train")
def predict_train():
    """Show callback-model status and retrain from logged outcomes."""
    import json as _json
    from .predict import model_status, train_callback_model
    console.print("[bold cyan]Callback model status[/]")
    console.print(_json.dumps(model_status(), indent=2))
    result = train_callback_model()
    console.print("[bold cyan]Training result[/]")
    console.print(_json.dumps(result, indent=2))
    if result.get("trained"):
        console.print(f"[green]Model trained on {result['n']} outcomes — "
                      "predictions now blend learned + heuristic.[/]")
    else:
        console.print(f"[yellow]{result.get('reason', 'not trained')} — "
                      "predictions stay heuristic until you log more outcomes "
                      "(mark applications applied / add interviews).[/]")


def _print_apply_kit(pkg) -> None:
    """Shared pretty-printer for apply-kit / browser-apply."""
    src_label = {
        "greenhouse": "Greenhouse API (real form)",
        "lever": "Lever apply page (real form)",
        "ashby": "Ashby API (real form)",
        "html": "HTML form parse (real form)",
    }.get(pkg.questions_source, "common defaults (form not readable — likely login-walled)")
    console.print(f"[green]Apply kit ready.[/] {pkg.title} @ {pkg.company}")
    console.print(f"  Apply URL:   {pkg.url or '-'}")
    console.print(f"  Resume:      {pkg.resume_path or '-'}")
    console.print(f"  Cover:       {pkg.cover_letter_path or '-'}")
    console.print(f"  Questions:   [bold]{src_label}[/]")
    console.print(f"  Copy sheet:  {pkg.kit_markdown_path or '-'}")
    console.print(f"  Auto-submit: {'[red]ON[/]' if pkg.autosubmit else '[green]OFF (review first)[/]'}")

    table = Table(title=f"Form questions ({len(pkg.form_questions)})",
                  show_header=True, header_style="bold cyan", expand=True, show_lines=True)
    table.add_column("#", width=3)
    table.add_column("Question", ratio=3, overflow="fold")
    table.add_column("Answer", ratio=4, overflow="fold")
    table.add_column("Via", width=12)
    for i, q in enumerate(pkg.form_questions, 1):
        req = " [red]*[/]" if q.get("required") else ""
        if q.get("answer"):
            ans = q["answer"]
            if q.get("needs_review"):
                ans = f"[yellow]{ans}[/]  [dim](confirm)[/]"
        else:
            ans = "[yellow]NEEDS YOU[/]"
            if q.get("guess"):
                ans += f"  [dim](we had \"{q['guess']}\" - not a valid option)[/]"
        table.add_row(str(i), q.get("text", "") + req, ans, q.get("answer_source") or "-")
    console.print(table)
    if pkg.blanks:
        console.print(f"\n[yellow]{len(pkg.blanks)} question(s) need your input "
                      f"(marked above).[/]")
        console.print(f"  Answer them now:  [bold]jobbot apply-fill {pkg.job_id}[/]")
        console.print("  ...or in the dashboard's Apply Kit page "
                      "(tick 'remember' so future forms auto-fill).")


def _ask_gap(gap) -> "Reply":
    from .apply_fill import Reply

    q = gap.question
    req = " (required)" if q.required else ""
    console.print()
    if gap.error:
        console.print(f"  [red]{gap.error}[/]")
    if gap.reason == "confirm":
        console.print(f"[bold yellow]CONFIRM[/] {q.text}{req}")
        console.print(f"  Proposed ([dim]{q.answer_source or 'ai'}[/]): "
                      f"[cyan]{q.answer}[/]")
    else:
        console.print(f"[bold]{q.text}[/][red]{req}[/]")
    if q.options:
        for n, opt in enumerate(q.options, 1):
            console.print(f"    [dim]{n})[/] {opt}")
    if gap.reason == "blank" and gap.suggestion:
        console.print(f"  [dim]We had \"{gap.suggestion}\" but it does not match "
                      f"the options above.[/]")

    hint = ("Enter=accept" if gap.reason == "confirm"
            else ("number or text" if q.options else "your answer"))
    try:
        raw = click.prompt(f"  [{hint}, s=skip, q=quit]", default="",
                           show_default=False)
    except (EOFError, click.Abort):
        return Reply(quit=True)

    stripped = raw.strip().lower()
    if stripped == "q":
        return Reply(quit=True)
    if stripped == "s":
        return Reply(skip=True)
    if not raw.strip() and gap.reason == "confirm":
        return Reply(accept=True)
    return Reply(value=raw)


def _confirm_remember(gap, value: str) -> bool:
    try:
        return click.confirm(f"  Remember \"{value}\" for future forms?",
                             default=False)
    except (EOFError, click.Abort):
        return False


def _run_gap_fill(job_id: int) -> dict:
    from .apply_fill import fill_job_gaps

    report = fill_job_gaps(job_id, ask=_ask_gap,
                           confirm_remember=_confirm_remember)
    if report.get("error"):
        console.print(f"[yellow]{report['error']}[/]")
        return report
    console.print(f"\n[green]Job {job_id}:[/] {report['answered']} answered, "
                  f"{report['skipped']} skipped, "
                  f"{report.get('remaining', 0)} still open.")
    if report.get("remembered"):
        console.print(f"  [dim]{report['remembered']} saved to your answer bank "
                      f"- future forms will fill these automatically.[/]")
    return report


@cli.command(name="apply-fill")
@click.argument("job_id", type=int, required=False)
@click.option("--recheck", is_flag=True,
              help="First re-validate stored answers against each form's real "
                   "options, demoting any that were never valid.")
def apply_fill(job_id, recheck):
    """Answer the questions the bot could not fill on its own."""
    from .apply_fill import queue_gap_counts, recheck_job

    if job_id is not None:
        if recheck:
            n = recheck_job(job_id)
            console.print(f"[cyan]Rechecked job {job_id}: {n} stored answer(s) "
                          f"were not valid for the form and now need you.[/]")
        _run_gap_fill(job_id)
        return

    if recheck:
        init_db()
        with session() as db:
            ids = [a.job_id for a in db.query(Application)
                   .join(Job).filter(Application.questions_json != "",
                                     Application.status != "sent").all()]
        total = sum(recheck_job(i) for i in ids)
        console.print(f"[cyan]Rechecked {len(ids)} kit(s): {total} stored "
                      f"answer(s) were not valid for their form.[/]")

    rows = queue_gap_counts()
    if not rows:
        console.print("[green]Nothing needs you - every prepared kit is complete.[/]")
        return

    total = sum(r["total"] for r in rows)
    console.print(f"[cyan]{total} question(s) need you across {len(rows)} kit(s).[/]")
    for row in rows:
        console.print(f"\n[bold]=== {row['title']} @ {row['company']} "
                      f"(job {row['job_id']}) ===[/]")
        report = _run_gap_fill(row["job_id"])
        if report.get("quit"):
            console.print("[dim]Stopped. Re-run `jobbot apply-fill` to pick up "
                          "where you left off.[/]")
            break


@cli.command(name="apply-kit")
@click.argument("job_ids", metavar="JOB_ID...", type=int, nargs=-1, required=True)
@click.option("--question", "questions", multiple=True,
              help="Extra application question to pre-answer (repeatable).")
@click.option("--no-fetch", is_flag=True,
              help="Skip fetching the live form; use the common question set.")
@click.option("--interactive/--no-interactive", default=None,
              help="Prompt for questions the bot cannot answer.")
def apply_kit(job_ids, questions, no_fetch, interactive):
    """Build ready-to-paste apply kits for one or more jobs."""
    from .browser_apply import build_package

    if interactive is None:
        interactive = sys.stdin.isatty()
    batch = len(job_ids) > 1

    built = []
    failed = []

    for n, job_id in enumerate(job_ids, 1):
        prefix = f"[{n}/{len(job_ids)}] " if batch else ""
        console.print(f"[cyan]{prefix}Building apply kit for job {job_id}...[/]")
        try:
            pkg = build_package(job_id, extra_questions=list(questions),
                                fetch_form=not no_fetch)
        except Exception as e:  # noqa: BLE001
            console.print(f"[red]Job {job_id} failed: {e}[/]")
            failed.append((job_id, str(e)))
            continue

        _print_apply_kit(pkg)
        built.append(job_id)

        if interactive and pkg.blanks:
            console.print(f"\n[cyan]Let's fill the {len(pkg.blanks)} gap(s) now.[/]")
            _run_gap_fill(job_id)
            console.print(f"[dim]Updated copy sheet: {pkg.kit_markdown_path}[/]")

    if batch or failed:
        console.print(
            f"\n[bold]Apply kits: {len(built)} of {len(job_ids)} built[/]"
            + (f", {len(failed)} failed" if failed else ""))
        if built:
            console.print(f"[dim]  built:  {', '.join(str(j) for j in built)}[/]")
        for job_id, err in failed:
            console.print(f"[red]  failed: {job_id} - {err}[/]")

    if failed:
        raise SystemExit(1)


@cli.command(name="browser-apply")
@click.argument("job_id", type=int)
@click.option("--question", "questions", multiple=True,
              help="Extra application question to pre-answer (repeatable).")
def browser_apply(job_id, questions):
    """Prepare an assisted browser application package for a job."""
    from .browser_apply import build_package
    console.print(f"[cyan]Preparing application package for job {job_id}...[/]")
    pkg = build_package(job_id, extra_questions=list(questions))
    _print_apply_kit(pkg)


@cli.command("mark-applied")
@click.argument("job_id", type=int)
@click.option("--date", "date_str", default="",
              help="Date you actually applied (YYYY-MM-DD). Defaults to now.")
def mark_applied_cmd(job_id: int, date_str: str):
    """Record that you applied to JOB_ID by hand."""
    from datetime import datetime as _dt
    from .browser_apply import mark_applied

    when = None
    if date_str:
        try:
            when = _dt.strptime(date_str, "%Y-%m-%d")
        except ValueError:
            console.print(f"[red]Bad date {date_str!r} - use YYYY-MM-DD.[/red]")
            raise SystemExit(1)

    init_db()
    with session() as db:
        job = db.get(Job, job_id)
        if not job:
            console.print(f"[red]No job {job_id}.[/red]")
            raise SystemExit(1)
        if not job.applications:
            console.print(f"[red]Job {job_id} has no application - "
                          f"build a kit first (`jobbot apply-kit {job_id}`).[/red]")
            raise SystemExit(1)
        app_id = job.applications[0].id
        title, company = job.title, job.company

    mark_applied(app_id, method="manual", when=when)
    console.print(f"[green]Marked applied:[/green] {title} at {company}")


@cli.command()
@click.argument("job_id", type=int)
def predict(job_id):
    """Predict callback/interview/offer odds for a job."""
    from .predict import predict_job
    with session() as db:
        job = db.get(Job, job_id)
        if not job:
            console.print(f"[red]Job {job_id} not found[/]")
            return
        app = job.applications[0] if job.applications else None
        p = predict_job(job, app)
    console.print(f"\n[bold cyan]{job.title}[/] @ {job.company}")
    console.print(f"Match score: {job.match_score:.2f}  |  Confidence in prediction: {p.confidence}")
    console.print(f"\n  P(callback)   = [bold]{p.callback_probability:.1%}[/]")
    console.print(f"  P(interview)  = [bold]{p.interview_probability:.1%}[/]")
    console.print(f"  P(offer)      = [bold]{p.offer_probability:.1%}[/]")
    console.print("\n[bold]Time-to-response curve:[/]")
    for d, prob in p.p_by_day.items():
        console.print(f"  by day +{d}: {prob:.1%}")
    console.print("\n[bold]Logit contributions:[/]")
    for k, v in sorted(p.logit_contributions.items(), key=lambda kv: -abs(kv[1])):
        sign = "+" if v >= 0 else ""
        console.print(f"  {sign}{v:.2f}  {k}")
    if p.recommended_actions:
        console.print("\n[bold yellow]Recommended actions:[/]")
        for a in p.recommended_actions:
            console.print(f"  • {a}")


@cli.command()
@click.option("--start-date", required=True, help="Start date YYYY-MM-DD")
@click.option("--horizon-days", default=90)
@click.option("--apply-rate", default=5, help="Applications per week")
def forecast(start_date, horizon_days, apply_rate):
    """Project the search pipeline from a designated start date."""
    from datetime import datetime as _dt
    from .predict import forecast as _forecast
    sd = _dt.strptime(start_date, "%Y-%m-%d")
    f = _forecast(sd, horizon_days, apply_rate)
    table = Table(title=f"Forecast from {start_date} ({horizon_days} days, {apply_rate} apps/week)",
                  show_header=True, header_style="bold cyan")
    table.add_column("Metric"); table.add_column("Value")
    table.add_row("Horizon end", f["horizon_end"][:10])
    table.add_row("Projected applications", str(f["total_applications_projected"]))
    table.add_row("Expected callbacks", str(f["expected_callbacks"]))
    table.add_row("Expected 1st-round interviews", str(f["expected_first_round_interviews"]))
    table.add_row("Expected offers", str(f["expected_offers"]))
    table.add_row("P(at least one offer)", f"{f['p_at_least_one_offer']:.1%}")
    console.print(table)


@cli.command()
def doctor():
    """Diagnose configuration & connectivity."""
    import requests
    from . import ai_client
    ok = lambda b: "[green]OK[/]" if b else "[red]MISSING[/]"
    console.print(f"AI provider:       [bold]{settings.ai_provider}[/]  "
                  f"(active: [cyan]{ai_client.active_provider_name()}[/])")
    console.print(f"Gemini API key:    {ok(bool(settings.gemini_api_key))}  ({settings.gemini_model})")
    console.print(f"Claude API key:    {ok(bool(settings.anthropic_api_key))}  ({settings.claude_model})")
    console.print(f"Gemini fallback:   {settings.gemini_fallback_model or '(none)'}")
    console.print(f"SMTP creds:     {ok(bool(settings.smtp_user and settings.smtp_password))}")
    console.print(f"Digest target:  {ok(bool(settings.digest_email_to))}")
    console.print(f"Base resume:    {ok(Path(settings.base_resume_path).exists())} ({settings.base_resume_path})")
    console.print(f"Watchlist:      {ok(Path(settings.company_watchlist).exists())} ({settings.company_watchlist})")
    console.print(f"Keywords:       {len(settings.keywords)} configured")
    console.print(f"Locations:      {settings.locations}")
    console.print("\n[bold]Connectivity probes:[/]")
    probes = [
        ("Greenhouse (Stripe)",        "https://boards-api.greenhouse.io/v1/boards/stripe/jobs"),
        ("Greenhouse (GitHub)",        "https://boards-api.greenhouse.io/v1/boards/github/jobs"),
        ("Workday (Target)",           "https://target.wd5.myworkdayjobs.com/targetcareers"),
        ("Workday (Salesforce)",       "https://salesforce.wd1.myworkdayjobs.com/External_Career_Site"),
        ("Ashby (Ramp)",               "https://api.ashbyhq.com/posting-api/job-board/ramp"),
        ("RemoteOK",                   "https://remoteok.com"),
        ("Hacker News",                "https://news.ycombinator.com"),
        ("GitHub API",                 "https://api.github.com"),
    ]
    for name, url in probes:
        try:
            r = requests.get(url, timeout=10, headers={"User-Agent": "JobBot/1.0"})
            console.print(f"  {name}: HTTP {r.status_code} ({len(r.content)} bytes)")
        except Exception as e:
            console.print(f"  {name}: [red]FAIL[/] {e}")


@cli.command(name="batch-tailor")
@click.argument("job_ids", nargs=-1, type=int, required=True)
@click.option("--workers", default=2, show_default=True,
              help="Concurrent tailor jobs (each fires 3 Gemini calls, keep ≤3)")
def batch_tailor(job_ids, workers):
    """Tailor multiple jobs concurrently."""
    from .pipeline import batch_tailor_jobs
    console.print(f"[cyan]Tailoring {len(job_ids)} jobs with {workers} workers...[/]")
    results = batch_tailor_jobs(list(job_ids), max_workers=workers)
    for jid, r in zip(job_ids, results):
        if r.get("error"):
            console.print(f"[red]  Job {jid}: error — {r['error']}[/]")
        else:
            console.print(f"[green]  Job {jid}: OK[/] resume={r.get('resume', '')} cover={r.get('cover_letter', '')}")


@cli.command()
def export():
    """Export jobs + applications to CSV in OUTPUT_DIR."""
    from .exports import export_applications, export_jobs
    p1 = export_jobs()
    p2 = export_applications()
    console.print(f"[green]Exported:[/] {p1}\n          {p2}")


@cli.command()
def stats():
    """Show pipeline stats."""
    with session() as db:
        total = db.query(Job).count()
        by_status = {}
        for j in db.query(Job).all():
            by_status[j.status] = by_status.get(j.status, 0) + 1
        apps = db.query(Application).count()
        watch_count = db.query(Job).filter(Job.tags.contains("watchlist")).count()
    table = Table(show_header=True, header_style="bold cyan", title="JobBot stats")
    table.add_column("Metric"); table.add_column("Value")
    table.add_row("Total jobs", str(total))
    table.add_row("Watchlist hits", str(watch_count))
    table.add_row("Applications", str(apps))
    for k, v in sorted(by_status.items()):
        table.add_row(f"  status: {k}", str(v))
    console.print(table)


@cli.command()
def watchlist():
    """Show parsed company watchlist."""
    from .watchlist import load_companies
    companies = load_companies()
    console.print(f"[cyan]{len(companies)} companies on watchlist:[/]")
    for c in companies:
        console.print(f"  • {c}")


@cli.command()
@click.option("--focus", default="")
def advice(focus):
    """Get AI advice on the base resume (Claude or Gemini per AI_PROVIDER)."""
    from . import ai_client as gc
    from .resume import load_resume
    resume_text = load_resume(settings.base_resume_path)
    console.print(gc.advise_resume(resume_text, focus))


@cli.group()
def model():
    """Inspect and switch the active AI model (Ollama local/cloud, Gemini, Claude)."""


@model.command(name="list")
def model_list():
    """List every selectable provider+model and which one is active."""
    rows = model_registry.list_selectable()
    if not rows:
        console.print("No models available. Start Ollama or add an API key.")
        return
    group = None
    for r in rows:
        if r["group"] != group:
            group = r["group"]
            console.print(f"\n[bold cyan]{group}[/]")
        mark = "*" if r["current"] else " "
        status = "" if r["available"] else f"  [yellow](unavailable: {r['reason']})[/]"
        console.print(f"  {mark} {r['model']}{status}")


@model.command(name="set")
@click.argument("provider", type=click.Choice(["ollama", "gemini", "claude"]))
@click.argument("model_name")
def model_set(provider, model_name):
    """Set the active model, e.g. `jobbot model set ollama gpt-oss:120b-cloud`."""
    try:
        model_registry.set_selection(provider, model_name)
    except Exception as exc:  # noqa: BLE001
        console.print(f"[red]Could not set model:[/] {exc}")
        raise SystemExit(1)
    console.print(f"[green]Active model set to[/] {model_name} ({provider}).")


@model.command(name="pull")
@click.argument("model_name")
def model_pull(model_name):
    """Pull a local Ollama model, e.g. `jobbot model pull llama3.3:70b`."""
    console.print(f"Pulling {model_name} (this can take a while)...")
    ok = model_registry.pull_ollama_model(model_name, progress=lambda s: console.print(f"  {s}"))
    if not ok:
        console.print(f"[red]Pull failed for[/] {model_name}.")
        raise SystemExit(1)
    console.print(f"[green]Pulled[/] {model_name}.")


@cli.command()
@click.argument("source")
@click.option("-o", "--out", default=None,
              help="Output MP3 path (default: <source>_podcast.mp3).")
@click.option("--style", type=click.Choice(["two_host", "narrator", "coach"]),
              default="two_host",
              help="two_host (conversation), narrator (briefing), or coach (direct).")
@click.option("--title", default="", help="Episode title (defaults to filename).")
@click.option("--script-only", is_flag=True,
              help="Write the transcript only; skip audio synthesis.")
@click.option("--voice-a", default=None, help="Override host A / narrator voice.")
@click.option("--voice-b", default=None, help="Override host B voice.")
def podcast(source, out, style, title, script_only, voice_a, voice_b):
    """Turn a prep doc / markdown file (or text) into an audio podcast."""
    from . import podcast as pod
    voices = None
    if voice_a or voice_b:
        voices = {"A": voice_a or pod.VOICE_A, "B": voice_b or pod.VOICE_B,
                  "N": voice_a or pod.VOICE_NARRATOR}
    console.print(f"[cyan]Generating {style} script...[/]")
    try:
        res = pod.make_podcast(source, out_path=out, style=style, title=title,
                               audio=not script_only, voices=voices)
    except Exception as exc:  # noqa: BLE001
        console.print(f"[red]Podcast failed:[/] {exc}")
        raise SystemExit(1)
    console.print(f"[green]Script[/] ({res['n_lines']} lines): {res['script_path']}")
    if res.get("audio_path"):
        console.print(f"[green]Audio:[/] {res['audio_path']}")
    else:
        console.print("[yellow]Audio skipped (--script-only).[/]")


@cli.command(name="paper-podcast")
@click.argument("sources", nargs=-1, required=True)
@click.option("--mode", type=click.Choice(["explainer", "networking"]),
              default="explainer",
              help="explainer (teach the science) or networking (a contact's work).")
@click.option("--style", type=click.Choice(["two_host", "narrator", "coach"]),
              default="two_host", help="Voice format.")
@click.option("--contact", default="", help="Contact name (networking mode).")
@click.option("--title", default="", help="Episode title.")
@click.option("--script-only", is_flag=True, help="Write transcript only; skip audio.")
@click.option("-o", "--out", default=None, help="Output MP3 path.")
def paper_podcast_cmd(sources, mode, style, contact, title, script_only, out):
    """Turn scientific paper PDF(s) into an audio podcast."""
    console.print(f"[cyan]Building {mode} paper podcast...[/]")
    try:
        res = paper_podcast.make_paper_podcast(
            list(sources), out_path=out, mode=mode, style=style,
            contact=contact, title=title, audio=not script_only)
    except Exception as exc:  # noqa: BLE001
        console.print(f"[red]Paper podcast failed:[/] {exc}")
        raise SystemExit(1)
    console.print(f"[green]Papers:[/] {res['n_papers']}  "
                  f"[green]Script[/] ({res['n_lines']} lines): {res['script_path']}")
    if res.get("audio_path"):
        console.print(f"[green]Audio:[/] {res['audio_path']}")
    else:
        console.print("[yellow]Audio skipped (--script-only).[/]")


@cli.command()
@click.argument("application_id", type=int)
@click.argument("stage")
@click.option("--note", default="")
def outcome(application_id, stage, note):
    """Record an application outcome stage."""
    from jobbot import outcomes
    try:
        outcomes.record_outcome(application_id, stage, note=note)
        click.echo(f"Recorded '{stage}' for application {application_id}.")
    except ValueError as e:
        raise SystemExit(str(e))


@cli.command(name="outcomes-sweep")
def outcomes_sweep():
    """Auto-mark silent past-window applications as 'ghosted'."""
    from jobbot import outcomes
    n = outcomes.sweep_ghosted()
    click.echo(f"Ghosted {n} silent application(s).")


@cli.command(name="outcomes-insights")
def outcomes_insights():
    """Print learned response rates by source / format / archetype."""
    from jobbot import outcomes
    data = outcomes.insights()
    for dim, rows in data.items():
        click.echo(f"\n{dim}:")
        for r in rows[:8]:
            click.echo(f"  {r['value']:<24} {int(round(r['response_rate']*100)):>3}% "
                       f"({r['responses']}/{r['trials']})")


@cli.group()
def contact():
    """Manage your warm-network contacts."""


@contact.command(name="add")
@click.argument("name")
@click.option("--company", default="")
@click.option("--title", default="")
@click.option("--rel", "relationship", default="unknown")
@click.option("--pin", is_flag=True, default=False)
@click.option("--note", default="")
def contact_add_cli(name, company, title, relationship, pin, note):
    """Add a contact."""
    from jobbot.models import Contact
    init_db()
    with session() as db:
        db.add(Contact(name=name, company=company, title=title, relationship=relationship,
                       pinned=pin, notes=note, source="manual"))
        db.commit()
    click.echo(f"Added contact {name}.")


@contact.command(name="list")
def contact_list_cli():
    """List contacts (warmth desc)."""
    from jobbot import referrals
    for c in referrals.build_index():
        pin = "*" if c.pinned else " "
        click.echo(f" {pin} {c.name:<28} {c.company:<28} warmth={c.warmth:.2f}")


@cli.group()
def referrals():
    """Referral-first routing: discover contacts + list warm intros."""


@referrals.command(name="discover")
@click.option("--max", "max_companies", type=int, default=None)
@click.option("--force", is_flag=True, default=False)
@click.argument("company", required=False)
def referrals_discover_cli(max_companies, force, company):
    """Discover contacts for a company (or the top companies if omitted)."""
    from jobbot import referrals as rf
    if company:
        n = len(rf.discover_company(company))
        click.echo(f"Found {n} contact(s) at {company}.")
    else:
        res = rf.discover_top_companies(max_companies=max_companies, force=force)
        click.echo(f"Discovered across {len(res)} companies: "
                   f"{sum(res.values())} new contact(s).")


@referrals.command(name="seed-lead")
def referrals_seed_cli():
    """Seed a sample pinned contact lead."""
    from jobbot import referrals as rf
    c = rf.seed_sample_lead()
    click.echo("Seeded sample contact lead." if c else "Sample contact lead already present.")


@referrals.command(name="import-linkedin")
@click.argument("csv_path", type=click.Path(exists=True))
def referrals_import_linkedin_cli(csv_path):
    """Import a LinkedIn Connections.csv export into the contact store."""
    from jobbot import linkedin_import
    s = linkedin_import.import_connections(csv_path)
    click.echo(f"Imported {s['imported']}, updated {s['updated']}, "
               f"skipped {s['skipped']}; {s['at_target']} at target companies.")


@referrals.command(name="list")
def referrals_list_cli():
    """List in-range jobs that have a warm intro."""
    from jobbot import referrals as rf
    init_db()
    idx = rf.build_index()
    with session() as db:
        jobs = db.query(Job).filter(Job.status != "expired").all()
        for j in jobs:
            bonus, why = rf.referral_bonus(j, idx)
            if bonus > 0:
                click.echo(f"  {j.title[:40]:<40} @ {j.company:<24} {why}")


@cli.group()
def outreach():
    """Auto-draft outreach queue (approve-then-send)."""


@outreach.command("scan")
def outreach_scan_cmd():
    """Scan for warm-intro / follow-up / cold drafts."""
    from . import outreach as ox
    res = ox.scan()
    click.echo(f"Queued: {res}")


@outreach.command("list")
@click.option("--kind", default=None, help="warm_intro|follow_up|cold|reply")
def outreach_list_cmd(kind):
    """List pending + approved drafts."""
    from . import outreach as ox
    for d in ox.queue_list(kind=kind):
        click.echo(f"[{d.id}] {d.kind:10} {d.status:9} {d.recipient_email or '(no email)':28} {d.subject}")


@outreach.command("approve")
@click.argument("draft_id", type=int)
def outreach_approve_cmd(draft_id):
    """Approve a draft (required before send)."""
    from . import outreach as ox
    d = ox.approve(draft_id)
    click.echo(f"Approved draft {d.id}.")


@outreach.command("send")
@click.argument("draft_id", type=int)
def outreach_send_cmd(draft_id):
    """Send an approved draft by email."""
    from . import outreach as ox
    res = ox.send(draft_id)
    click.echo(f"Sent to {res['to']}.")


@outreach.command("dismiss")
@click.argument("draft_id", type=int)
def outreach_dismiss_cmd(draft_id):
    """Dismiss a draft."""
    from . import outreach as ox
    ox.dismiss(draft_id)
    click.echo(f"Dismissed draft {draft_id}.")


@outreach.command("reply")
@click.option("--file", "path", required=True, type=click.Path(exists=True),
              help="Path to a text file containing the received email.")
@click.option("--sender", default="", help="Sender email address.")
def outreach_reply_cmd(path, sender):
    """Draft a reply to a pasted/saved inbound email."""
    from . import outreach as ox
    with open(path, "r", encoding="utf-8") as fh:
        inbound = fh.read()
    d = ox.draft_reply(inbound, sender=sender)
    click.echo(f"Reply drafted as draft {d.id}.")


@cli.group()
def convo():
    """Email conversation threads (paste in, draft with full context)."""


@convo.command("start")
@click.option("--contact", "contact_id", type=int, default=None, help="Contact id.")
@click.option("--job", "job_id", type=int, default=None, help="Job id.")
@click.option("--application", "application_id", type=int, default=None, help="Application id.")
@click.option("--scenario", default="", help="The situation, in your own words.")
@click.option("--scenario-file", type=click.Path(exists=True), default=None,
              help="Read the scenario from a file instead.")
@click.option("--subject", default="", help="Thread subject line.")
def convo_start_cmd(contact_id, job_id, application_id, scenario, scenario_file, subject):
    """Open a new conversation thread."""
    from . import conversations as cv
    if scenario_file:
        with open(scenario_file, "r", encoding="utf-8") as fh:
            scenario = fh.read()
    t = cv.start(contact_id=contact_id, job_id=job_id,
                 application_id=application_id, scenario=scenario, subject=subject)
    click.echo(f"Created thread {t.id}.")


@convo.command("paste")
@click.argument("thread_id", type=int)
@click.option("--file", "path", required=True, type=click.Path(exists=True),
              help="Path to a text file containing the received email.")
@click.option("--sender", default="", help="Sender email address.")
def convo_paste_cmd(thread_id, path, sender):
    """Record a received email and queue a context-aware reply draft."""
    from . import conversations as cv
    with open(path, "r", encoding="utf-8") as fh:
        inbound = fh.read()
    d = cv.ingest(thread_id, inbound, sender=sender)
    if getattr(d, "is_new", True):
        click.echo(f"Queued reply as draft {d.id}. "
                   "Review: jobbot outreach list --kind reply")
    else:
        click.echo(f"Already recorded: this email matches draft {d.id} [{d.status}], "
                   "so no new draft was written.")
        click.echo("To redraft after changing the scenario, dismiss it first: "
                   f"jobbot outreach dismiss {d.id}")


@convo.command("sent")
@click.argument("thread_id", type=int)
@click.option("--file", "path", required=True, type=click.Path(exists=True),
              help="Path to a text file containing the email you sent.")
@click.option("--subject", default="", help="Subject line (defaults to the thread's).")
@click.option("--at", "when", default="",
              help="When it went out, e.g. '2026-07-24 16:09'. Defaults to now.")
def convo_sent_cmd(thread_id, path, subject, when):
    """Record an email you sent BY HAND so the thread knows about it."""
    from datetime import datetime as _dt
    from . import conversations as cv
    with open(path, "r", encoding="utf-8") as fh:
        body = fh.read()
    stamp = None
    if when:
        for fmt in ("%Y-%m-%d %H:%M", "%Y-%m-%d %H:%M:%S", "%Y-%m-%d"):
            try:
                stamp = _dt.strptime(when, fmt)
                break
            except ValueError:
                continue
        if stamp is None:
            raise click.BadParameter("use YYYY-MM-DD or 'YYYY-MM-DD HH:MM'", param_hint="--at")
    d = cv.record_sent(thread_id, body, subject=subject, sent_at=stamp)
    if getattr(d, "is_new", True):
        click.echo(f"Recorded on thread {thread_id} as message {d.id} "
                   f"({d.sent_at:%Y-%m-%d %H:%M}).")
    else:
        click.echo(f"Already recorded: this is message {d.id} on thread {thread_id}.")


@convo.command("nudge")
@click.argument("thread_id", type=int)
@click.option("--force", is_flag=True,
              help="Draft even inside the two-week quiet window.")
def convo_nudge_cmd(thread_id, force):
    """Draft a low-pressure follow-up for a message that got no reply."""
    from . import conversations as cv
    try:
        d = cv.draft_nudge(thread_id, force=force)
    except ValueError as exc:
        raise click.ClickException(str(exc))
    if getattr(d, "is_new", True):
        click.echo(f"Queued nudge as draft {d.id}. "
                   "Review: jobbot outreach list --kind follow_up")
    else:
        click.echo(f"A nudge for this message is already queued as draft {d.id} "
                   f"[{d.status}].")


@convo.command("show")
@click.argument("thread_id", type=int)
def convo_show_cmd(thread_id):
    """Print the scenario and full message history."""
    from . import conversations as cv
    with session() as db:
        t = cv._get(db, thread_id)
        click.echo(f"Thread {t.id} [{t.status}] {t.subject}")
        click.echo(f"With: {t.counterpart_name or '?'} <{t.counterpart_email or '?'}>")
        click.echo("")
        click.echo("Scenario:")
        click.echo(t.scenario or "  (none)")
    click.echo("")
    click.echo("History:")
    rows = cv.history(thread_id)
    if not rows:
        click.echo("  (no messages yet)")
    for r in rows:
        stamp = r["at"].strftime("%Y-%m-%d") if hasattr(r["at"], "strftime") else str(r["at"])
        arrow = "->" if r["direction"] == "sent" else "<-"
        click.echo(f"  {arrow} {stamp}  {r['text'][:200]}")


@convo.command("list")
def convo_list_cmd():
    """List open threads, most recent inbound first."""
    from . import conversations as cv
    threads = cv.list_open()
    if not threads:
        click.echo("No open threads.")
        return
    for t in threads:
        when = t.last_inbound_at.strftime("%Y-%m-%d") if t.last_inbound_at else "never"
        click.echo(f"  {t.id:<5} {(t.subject or '(no subject)')[:44]:<44} "
                   f"{(t.counterpart_email or '')[:28]:<28} last reply: {when}")


@convo.command("scenario")
@click.argument("thread_id", type=int)
@click.option("--set", "text", required=True, help="Replacement scenario text.")
def convo_scenario_cmd(thread_id, text):
    """Replace a thread's standing context."""
    from . import conversations as cv
    cv.set_scenario(thread_id, text)
    click.echo(f"Updated scenario on thread {thread_id}.")


@convo.command("close")
@click.argument("thread_id", type=int)
def convo_close_cmd(thread_id):
    """Close a thread (stops it appearing in `convo list`)."""
    from . import conversations as cv
    cv.close(thread_id)
    click.echo(f"Closed thread {thread_id}.")


@cli.group(name="warm-path")
def warm_path():
    """Warm-path engine: run the daily scrape -> rescore -> outreach scan."""


@warm_path.command(name="daily")
def warm_path_daily_cmd():
    """One-shot daily cycle: scrape new jobs, rescore, queue outreach drafts."""
    from jobbot import pipeline as _pl, ranking as _rk, outreach as _ox
    scrape = {}
    try:
        scrape = _pl.run_scrape_cycle(send_digest=False)
    except Exception as exc:  # noqa: BLE001
        click.echo(f"scrape failed (continuing): {exc}")
    try:
        rescore = _rk.rescore_jobs(_pl.load_profile())
    except Exception as exc:  # noqa: BLE001
        rescore = {}
        click.echo(f"rescore failed (continuing): {exc}")
    try:
        queued = _ox.scan()
    except Exception as exc:  # noqa: BLE001
        queued = {}
        click.echo(f"scan failed: {exc}")
    click.echo(f"warm-path daily: scrape={scrape} rescore={rescore} queued={queued}")
    click.echo("Review drafts: jobbot outreach list  (approve before send)")


@cli.group()
def answers():
    """Manage the cross-application answer bank."""


@answers.command("seed")
@click.option("--dry-run", is_flag=True, help="Show what would be written.")
def answers_seed(dry_run: bool):
    """Build answer-bank entries from your resume."""
    from .answer_seed import seed_from_resume

    report = seed_from_resume(dry_run=dry_run)
    for entry in report["active"]:
        console.print(f"  [green]active[/green]  {entry['answer']}")
    for entry in report["pending"]:
        console.print(f"  [yellow]pending[/yellow] {entry['answer']}")
    verb = "Would write" if dry_run else "Wrote"
    console.print(f"{verb} {len(report['active'])} active, "
                  f"{len(report['pending'])} pending "
                  f"(review with `jobbot answers review`).")


@answers.command("list")
@click.option("--pending", "only_pending", is_flag=True)
@click.option("--active", "only_active", is_flag=True)
def answers_list(only_pending: bool, only_active: bool):
    """Show stored bank rules."""
    from .apply_questions import load_answer_bank_raw

    for entry in load_answer_bank_raw():
        status = entry.get("status", "active")
        if only_pending and status != "pending":
            continue
        if only_active and status != "active":
            continue
        console.print(f"  [{status}] {entry['match']} -> {entry['answer']}")


@answers.command("review")
def answers_review():
    """Approve or reject staged (LLM-inferred) answers."""
    from .apply_questions import _answer_bank_path, load_answer_bank_raw
    import json as _json

    entries = load_answer_bank_raw()
    pending = [e for e in entries if e.get("status") == "pending"]
    if not pending:
        console.print("Nothing staged for review.")
        return

    keep = []
    for entry in entries:
        if entry.get("status") != "pending":
            keep.append(entry)
            continue
        console.print(f"\n[yellow]{entry['match']}[/yellow]")
        console.print(f"  answer: {entry['answer']}")
        choice = click.prompt("  [a]pprove / [r]eject / [s]kip", default="s")
        if choice.startswith("a"):
            entry["status"] = "active"
            keep.append(entry)
        elif choice.startswith("s"):
            keep.append(entry)

    path = _answer_bank_path()
    path.write_text(_json.dumps(keep, indent=2, ensure_ascii=False),
                    encoding="utf-8")
    console.print("\n[green]Review saved.[/green]")


@answers.command("calibrate")
@click.option("--pairs", "pairs_path",
              default="tests/fixtures/question_pairs.json",
              help="Labelled question pairs to score against.")
def answers_calibrate(pairs_path: str):
    """Suggest matching thresholds from labelled question pairs."""
    import json as _json
    from .answer_memory import calibrate

    p = Path(pairs_path)
    if not p.exists():
        console.print(f"[red]No pairs file at {pairs_path}.[/red]")
        raise SystemExit(1)

    report = calibrate(_json.loads(p.read_text(encoding="utf-8")))
    console.print(f"  equivalent pairs score at least: {report['equivalent_min']:.3f}")
    console.print(f"  opposite pairs score at most:    {report['opposite_max']:.3f}")
    if report["separable"]:
        console.print(f"\n[green]Separable.[/green] Suggested "
                      f"answer_match_high_threshold: {report['suggested_high']}")
    else:
        console.print("\n[red]Not separable.[/red] No threshold divides "
                      "equivalent from opposite pairs - keep the semantic tier "
                      "off and rely on the token tier.")


@cli.group()
def facts():
    """Structured resume facts used to fill application forms."""


@facts.command("extract")
@click.option("--from", "from_path", default="data/base_resume.md",
              show_default=True, help="Markdown resume to parse.")
def facts_extract(from_path: str):
    """Extract employment/education from the base resume. Never auto-approves."""
    from . import resume_facts as rf

    src = Path(from_path)
    if src.suffix.lower() not in {".md", ".markdown", ".txt"}:
        console.print(f"[red]Not a markdown resume: {src}[/]")
        console.print("This parser reads markdown. Pass one with --from, e.g. "
                      "jobbot facts extract --from data/base_resume.md")
        raise SystemExit(1)
    if not src.exists():
        console.print(f"[red]No such file: {src}[/]")
        raise SystemExit(1)

    try:
        text = src.read_text(encoding="utf-8")
    except (UnicodeDecodeError, OSError) as e:
        console.print(f"[red]Could not read {src} as text ({e}).[/]")
        console.print("This parser reads markdown. Pass one with --from, e.g. "
                      "jobbot facts extract --from data/base_resume.md")
        raise SystemExit(1)

    got = rf.extract_from_markdown(text)
    rf.save(got)
    console.print(f"Read {src}")
    console.print(f"Extracted {len(got.employment)} roles, "
                  f"{len(got.education)} education entries.")
    if not got.employment:
        console.print("[yellow]No roles found. That usually means the resume's "
                      "layout does not match what the parser expects, not that "
                      "there is no work history.[/]")
    console.print("These are NOT reviewed yet. Confirm with: jobbot facts review")


@facts.command("show")
def facts_show():
    """Print the current facts and whether they have been reviewed."""
    from . import resume_facts as rf

    got = rf.load()
    for e in got.employment:
        span = f"{e.start} - {'present' if e.current else e.end}"
        console.print(f"  {e.title} @ {e.employer} ({span})")
    for ed in got.education:
        degree = " ".join(p for p in (ed.degree, ed.field) if p)
        console.print(f"  {degree} @ {ed.school} ({ed.start} - {ed.end})")
    console.print("Status: reviewed" if got.is_reviewed() else "Status: NOT reviewed")


@facts.command("review")
@click.option("--yes", is_flag=True, help="Mark reviewed without prompting.")
def facts_review(yes: bool):
    """Confirm the extracted facts are correct. Required before form filling."""
    from . import resume_facts as rf

    got = rf.load()
    for e in got.employment:
        span = f"{e.start} - {'present' if e.current else e.end}"
        console.print(f"  {e.title} @ {e.employer} ({span})")
    if not yes and not click.confirm("Are these dates and employers correct?"):
        console.print("Left unreviewed. Edit data/resume_facts.json and re-run.")
        return
    got.reviewed = True
    rf.save(got)
    console.print("Marked reviewed.")


@cli.group()
def recipes():
    """Learned ATS field-resolution recipes (a perishable cache)."""


@recipes.command("list")
def recipes_list():
    """Show learned recipes."""
    console.print("[dim]Headless ATS bot recipes are decommissioned. Use JobBot In-Browser Copilot instead.[/]")


@recipes.command("clear")
@click.option("--yes", is_flag=True, help="Do not prompt.")
def recipes_clear(yes: bool):
    """Forget every learned recipe."""
    console.print("[dim]Cleared.[/]")


@cli.command(name="ats-replay")
@click.option("--tenant", default=None,
              help="Replay one tenant directory instead of all of them.")
def ats_replay_cmd(tenant):
    console.print("[dim]Headless ATS replay is decommissioned. Use JobBot In-Browser Copilot extension instead.[/]")
    return


@cli.command(name="ats-replay")
@click.option("--tenant", default=None,
              help="Replay one tenant directory instead of all of them.")
def ats_replay_cmd(tenant):
    """Replay committed tenant fixtures through the resolver and classifier."""
    from rich import box as _box
    from .ats import replay

    root = Path("tests/fixtures/tenants")
    if not root.exists():
        console.print("[yellow]No tenant fixtures yet.[/] "
                      "This harness only replays the committed JSON corpus "
                      "under tests/fixtures/tenants -- it does not capture.")
        return

    dirs = ([root / tenant] if tenant
            else sorted(p for p in root.iterdir() if p.is_dir()))
    pre_label_source = 0
    no_labels_captured = 0
    no_fields = 0
    for d in dirs:
        if not d.is_dir():
            console.print(f"[red]No such tenant fixture directory:[/] {d}")
            continue
        console.print(f"\n[bold]{d.name}[/]")
        results = replay.replay_tenant(d)
        summary = replay.tenant_summary(results)
        for step, data in results.items():
            if data["pre_label_source"]:
                pre_label_source += 1
            if data["state"] == "no_labels_captured":
                no_labels_captured += 1
            elif data["state"] == "no_fields":
                no_fields += 1

            state = data["state"]
            if state == "no_fields":
                console.print(f"  [dim]{step}: no fields captured[/]")
                continue
            rows = data["rows"]
            ok = sum(1 for r in rows
                     if r["status"] == "resolved" and r["kind"] not in ("", "unknown"))
            title = f"{step}  ({ok}/{len(rows)} resolved+classified)"
            if state == "no_labels_captured":
                title += (f"  [NO LABELS CAPTURED: {data['fields_total']} "
                          f"fields, 0 labelled -- NOT a pass]")
            table = Table(title=title, box=_box.ASCII)
            for col in ("label", "status", "kind", "candidates", "label_source"):
                table.add_column(col)
            for r in rows:
                table.add_row(r["label"][:48], r["status"], r["kind"],
                              str(r["candidates"]), r["label_source"])
            console.print(table)
        console.print(f"  [dim]tenant totals (state=ok steps only): "
                      f"resolved={summary['resolved']} "
                      f"ambiguous={summary['ambiguous']} "
                      f"not_found={summary['not_found']}[/]")
        if summary["no_labels_captured_steps"]:
            console.print(
                f"  [yellow]no_labels_captured steps (excluded above, NOT "
                f"a pass):[/] {', '.join(summary['no_labels_captured_steps'])}")

    console.print(f"\n[dim]Fixtures predating label_source tracking: "
                  f"{pre_label_source}. Steps with no_labels_captured: "
                  f"{no_labels_captured}. Steps with no fields at all: "
                  f"{no_fields}.[/]")
    console.print(f"\n[dim]{replay.SCOPE_NOTE}[/]")


@cli.group(name="clean")
def clean():
    """Database hygiene: dead postings, stale embeddings, duplicate rows."""


def _clean_backup() -> None:
    from .models import backup_db
    path = backup_db()
    if path:
        console.print(f"[dim]Backup: {_cli_safe(path)}[/]")


@clean.command(name="liveness")
@click.option("--limit", type=int, default=None,
              help="Probe at most N jobs this run (default: the whole due pool).")
@click.option("--workers", type=int, default=8, help="Concurrent probes (default 8).")
@click.option("--recheck-after-hours", type=int, default=24,
              help="Skip jobs probed more recently than this (default 24).")
@click.option("--apply", "do_apply", is_flag=True,
              help="Actually expire dead postings. Without it, nothing is written.")
def clean_liveness(limit, workers, recheck_after_hours, do_apply):
    """Probe active postings and soft-expire the dead ones (restorable)."""
    from . import liveness

    if not do_apply:
        from datetime import timedelta
        init_db()
        cutoff = datetime.utcnow() - timedelta(hours=recheck_after_hours)
        with session() as db:
            rows = (db.query(Job)
                    .filter(Job.status.in_(liveness._ACTIVE_STATUSES)).all())
            due = [j for j in rows
                   if j.last_checked_at is None or j.last_checked_at <= cutoff]
        n = len(due) if limit is None else min(limit, len(due))
        console.print(f"Would probe [bold]{n}[/] of {len(rows)} active jobs "
                      f"({len(rows) - len(due)} checked within "
                      f"{recheck_after_hours}h). Re-run with --apply.")
        return

    _clean_backup()
    console.print("Probing... this is network-bound and can take a while.")
    stats = liveness.sweep_all(limit=limit, workers=workers,
                               recheck_after_hours=recheck_after_hours)
    if stats["aborted"]:
        console.print("[red]ABORTED[/] - probes were failing, not postings. "
                      "Nothing was expired beyond what is reported below.")
    console.print(f"checked={stats['checked']} expired={stats['expired']} "
                  f"alive={stats['alive']} errors={stats['errors']} "
                  f"skipped={stats['skipped']}")


@clean.command(name="embeddings")
@click.option("--apply", "do_apply", is_flag=True,
              help="Actually clear them and VACUUM.")
def clean_embeddings(do_apply):
    """Drop stored embeddings from expired/applied jobs and reclaim the space."""
    from . import dbclean

    if do_apply:
        _clean_backup()
    rep = dbclean.clear_expired_embeddings(apply=do_apply)
    mb = lambda b: round(b / 1e6, 1)  # noqa: E731
    if not do_apply:
        console.print(f"Would clear embeddings on [bold]{rep['jobs']}[/] jobs "
                      f"(DB is {mb(rep['bytes_before'])} MB). Re-run with --apply.")
        return
    console.print(f"Cleared {rep['jobs']} embeddings. "
                  f"{mb(rep['bytes_before'])} MB -> {mb(rep['bytes_after'])} MB")


@clean.command(name="freelance")
@click.option("--apply", "do_apply", is_flag=True,
              help="Actually hide them (status -> skipped). Reversible.")
@click.option("--undo", is_flag=True,
              help="Put previously hidden freelance jobs back to 'new'.")
def clean_freelance(do_apply, undo):
    """Hide freelance / gig postings from the job pool."""
    from . import dbclean

    fn = dbclean.unhide_freelance_jobs if undo else dbclean.hide_freelance_jobs
    verb = "restore" if undo else "hide"
    rep = fn(apply=False)
    if not rep["jobs"]:
        console.print(f"Nothing to {verb}.")
        return
    for t in rep["titles"]:
        console.print(f"  {t[:96]}")
    if rep["jobs"] > len(rep["titles"]):
        console.print(f"  ... and {rep['jobs'] - len(rep['titles'])} more")
    if not do_apply:
        console.print(f"[bold]{rep['jobs']}[/] job(s) would {verb}. "
                      f"Re-run with --apply.")
        return
    _clean_backup()
    rep = fn(apply=True)
    console.print(f"{'Restored' if undo else 'Hid'} {rep['jobs']} job(s).")


@clean.command(name="duplicate-apps")
@click.option("--apply", "do_apply", is_flag=True,
              help="Actually merge. Outcome events are re-pointed, never deleted.")
def clean_duplicate_apps(do_apply):
    """Collapse a job's extra Application rows onto the one holding the kit."""
    from . import dbclean

    plan = dbclean.duplicate_app_plan()
    if not plan:
        console.print("No duplicate application rows.")
        return
    for p in plan:
        console.print(f"  job {p['job_id']}: keep app {p['survivor']}, "
                      f"remove {p['losers']}, "
                      f"re-point {p['events_to_repoint']} outcome event(s), "
                      f"status -> {p['merged_status']}")
    if not do_apply:
        console.print(f"[bold]{len(plan)}[/] job(s) would be merged. "
                      "Re-run with --apply.")
        return
    _clean_backup()
    rep = dbclean.merge_duplicate_apps(apply=True)
    console.print(f"Merged {rep['jobs']} job(s): removed {rep['rows_removed']} row(s), "
                  f"re-pointed {rep['events_repointed']} event(s) and "
                  f"{rep['runs_repointed']} run(s).")


def main():
    cli()


if __name__ == "__main__":
    sys.exit(main())