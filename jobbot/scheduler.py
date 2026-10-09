"""APScheduler — runs the scrape pipeline on an interval."""
from __future__ import annotations

import logging
import time

from apscheduler.schedulers.background import BackgroundScheduler

from .config import settings
from .pipeline import run_scrape_cycle

log = logging.getLogger("jobbot.scheduler")


def start_scheduler(block: bool = True) -> BackgroundScheduler:
    sched = BackgroundScheduler(timezone="UTC")
    sched.add_job(
        run_scrape_cycle,
        "interval",
        hours=settings.schedule_interval_hours,
        id="scrape_cycle",
        next_run_time=None,  # First fire after the interval; call run() manually for immediate
    )
    sched.start()
    log.info("Scheduler started — every %d hours", settings.schedule_interval_hours)
    if block:
        try:
            while True:
                time.sleep(60)
        except (KeyboardInterrupt, SystemExit):
            sched.shutdown()
    return sched
