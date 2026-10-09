"""Thread-safe scrape-progress tracker (singleton)."""
from __future__ import annotations

import threading
from collections import deque
from datetime import datetime
from typing import Optional


class _Progress:
    def __init__(self):
        self._lock = threading.RLock()  # reentrant — start() calls reset() while holding the lock
        self.reset()

    def reset(self):
        with self._lock:
            self.state: str = "idle"           # idle | running | done | error
            self.started_at: Optional[datetime] = None
            self.finished_at: Optional[datetime] = None
            self.current: str = ""             # currently running scraper name
            self.done_count: int = 0
            self.total: int = 0
            self.jobs_found: int = 0           # raw count across scrapers so far
            self.per_scraper: dict[str, int] = {}
            self.log_lines: deque[str] = deque(maxlen=50)
            self.stats: dict = {}
            self.error: str = ""

    # ---------- mutators ----------
    def start(self, total: int):
        with self._lock:
            self.reset()
            self.state = "running"
            self.started_at = datetime.utcnow()
            self.total = total

    def scraper_start(self, name: str):
        with self._lock:
            self.current = name
            self.log_lines.append(f"> {name} starting...")

    def scraper_done(self, name: str, job_count: int):
        with self._lock:
            self.done_count += 1
            self.per_scraper[name] = job_count
            self.jobs_found += job_count
            self.log_lines.append(f"+ {name}: {job_count} jobs")
            self.current = ""

    def scraper_error(self, name: str, err: str):
        with self._lock:
            self.done_count += 1
            self.per_scraper[name] = 0
            self.log_lines.append(f"! {name} failed: {err}")
            self.current = ""

    def finish(self, stats: dict | None = None):
        with self._lock:
            self.state = "done"
            self.finished_at = datetime.utcnow()
            self.stats = stats or {}
            self.log_lines.append(f"-- pipeline complete: {self.stats.get('new', 0)} new jobs")

    def fail(self, err: str):
        with self._lock:
            self.state = "error"
            self.finished_at = datetime.utcnow()
            self.error = err
            self.log_lines.append(f"!! pipeline failed: {err}")

    # ---------- snapshot ----------
    def snapshot(self) -> dict:
        with self._lock:
            elapsed = None
            if self.started_at:
                end = self.finished_at or datetime.utcnow()
                elapsed = (end - self.started_at).total_seconds()
            pct = int(100 * self.done_count / self.total) if self.total else 0
            return {
                "state": self.state,
                "current": self.current,
                "done": self.done_count,
                "total": self.total,
                "percent": pct,
                "jobs_found": self.jobs_found,
                "per_scraper": dict(self.per_scraper),
                "log_lines": list(self.log_lines),
                "stats": dict(self.stats),
                "error": self.error,
                "elapsed_seconds": elapsed,
            }


progress = _Progress()
