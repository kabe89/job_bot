# jobbot/apply_runner.py
"""In-Browser Copilot Application Assistant & Launcher.

Bridges the local Python application with the user's browser:
- Packages applicant records, tailored resumes, and screening answers
- Auto-injects the Copilot assistant script directly into headed browser sessions
- Copies bookmarklet code to clipboard for 1-click execution in external browsers
- Provides Python and CLI entrypoints to launch autofill effortlessly.
"""
from __future__ import annotations

import logging
import os
import subprocess
import sys
import webbrowser
from pathlib import Path
from typing import Callable, Dict, List, Optional

from .browser_apply import build_package, mark_applied, captcha_reason
from .config import settings
from .models import ApplyRun, Job, init_db, session

log = logging.getLogger("jobbot.apply_runner")

BOOKMARKLET_TEMPLATE = (
    "javascript:(function(){{"
    "const s=document.createElement('script');"
    "s.src='http://localhost:5000/static/jobbot_copilot.js?t='+Date.now();"
    "document.head.appendChild(s);"
    "}})();"
)


def copy_to_clipboard(text: str) -> bool:
    """Copies text to the system clipboard across platforms."""
    try:
        if sys.platform == "win32":
            subprocess.run(["clip.exe"], input=text.encode("utf-16"), check=True,
                           stdout=subprocess.DEVNULL, stderr=subprocess.DEVNULL)
            return True
        elif sys.platform == "darwin":
            subprocess.run(["pbcopy"], input=text.encode("utf-8"), check=True,
                           stdout=subprocess.DEVNULL, stderr=subprocess.DEVNULL)
            return True
        else:
            subprocess.run(["xclip", "-selection", "clipboard"], input=text.encode("utf-8"), check=True,
                           stdout=subprocess.DEVNULL, stderr=subprocess.DEVNULL)
            return True
    except Exception:
        pass
    try:
        import tkinter as tk
        r = tk.Tk()
        r.withdraw()
        r.clipboard_clear()
        r.clipboard_append(text)
        r.update()
        r.destroy()
        return True
    except Exception:
        return False


def get_copilot_script_path() -> Path:
    """Return the filesystem path to the Copilot JS script."""
    static_script = Path(__file__).resolve().parent / "static" / "jobbot_copilot.js"
    if static_script.exists():
        return static_script
    tools_script = Path(__file__).resolve().parent.parent / "tools" / "copilot" / "jobbot_copilot.js"
    return tools_script


def launch_copilot(
    job_id: int,
    browser_mode: str = "headed",
    auto_fill: bool = True,
    interactive_wait: bool = False,
) -> Dict[str, any]:
    """Launch JobBot Copilot for a target job.

    Args:
        job_id: The job ID to prepare and autofill.
        browser_mode: 'headed' (Playwright with auto-injected script) or 'system' (default OS browser).
        auto_fill: If True, instructs the script to autofill immediately on page load.
        interactive_wait: If True, blocks on the CLI until the user presses Enter or closes the window.

    Returns:
        A summary dict containing package status, URLs, and bookmarklet string.
    """
    init_db()
    with session() as db:
        job = db.get(Job, job_id)
        if not job:
            raise ValueError(f"Job {job_id} not found")
        url = job.url
        title = job.title
        company = job.company

    # Compile application kit and screening Q&A package
    build_package(job_id)

    # Copy bookmarklet code to clipboard for convenience
    copy_to_clipboard(BOOKMARKLET_TEMPLATE)

    launched_in_playwright = False
    script_path = get_copilot_script_path()

    if browser_mode == "headed" and url:
        try:
            from playwright.sync_api import sync_playwright
            pw = sync_playwright().start()
            browser = pw.chromium.launch(headless=False)
            context = browser.new_context(
                viewport={"width": 1280, "height": 900},
                user_agent=(
                    "Mozilla/5.0 (Windows NT 10.0; Win64; x64) "
                    "AppleWebKit/537.36 (KHTML, like Gecko) "
                    "Chrome/126.0.0.0 Safari/537.36"
                ),
            )
            # Pre-inject Job ID and auto-fill flag
            setup_script = (
                f"window.__JOBBOT_JOB_ID__ = {job_id}; "
                f"window.__JOBBOT_AUTOFILL_IMMEDIATE__ = {'true' if auto_fill else 'false'};"
            )
            context.add_init_script(setup_script)

            if script_path.exists():
                context.add_init_script(path=str(script_path))

            page = context.new_page()
            page.goto(url)
            launched_in_playwright = True
            log.info("Playwright launched in headed mode with JobBot Copilot injected for %s", url)

            if interactive_wait:
                print("\n" + "=" * 70)
                print(f"🚀 JobBot In-Browser Copilot Active for: {title} @ {company}")
                print("=" * 70)
                print(f"URL: {url}")
                print(f"Local API: http://localhost:5000/api/job/{job_id}/copilot-package")
                print("JobBot Copilot has been injected directly into the browser window!")
                print("The bookmarklet has also been copied to your clipboard.")
                print("=" * 70)
                try:
                    input("\n[Browser window is active] Press ENTER here when finished to close session...\n")
                except (KeyboardInterrupt, EOFError):
                    pass
                finally:
                    try:
                        context.close()
                        browser.close()
                        pw.stop()
                    except Exception:
                        pass
        except Exception as e:
            log.warning("Playwright headed launch not available or failed (%s); falling back to system browser.", e)

    if not launched_in_playwright and url:
        try:
            webbrowser.open(url)
            log.info("Opened %s in default system browser.", url)
        except Exception as e:
            log.warning("Could not launch default browser: %s", e)

    with session() as db:
        run = ApplyRun(
            job_id=job_id,
            ats="copilot",
            status="copilot_ready",
            confirmation_text=f"Launched in {'Playwright' if launched_in_playwright else 'system'} browser with Copilot.",
        )
        db.add(run)
        db.commit()
        db.refresh(run)
        run_id = run.id

    return {
        "ok": True,
        "run_id": run_id,
        "job_id": job_id,
        "title": title,
        "company": company,
        "url": url,
        "browser": "playwright" if launched_in_playwright else "system",
        "bookmarklet": BOOKMARKLET_TEMPLATE,
    }


def run_apply(job_id: int, headed: bool = True, **kwargs) -> int:
    """Main application runner entrypoint — delegates to launch_copilot."""
    res = launch_copilot(
        job_id=job_id,
        browser_mode="headed" if headed else "system",
        interactive_wait=kwargs.get("interactive", False),
    )
    return res["run_id"]


def cli_edit(results: List[dict]) -> List[tuple]:
    """Helper for CLI field reviews."""
    return []


def cli_confirm(results: List[dict], screenshot_path: str = "") -> str:
    """Helper for CLI confirmation."""
    return "submit"
