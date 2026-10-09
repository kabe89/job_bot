"""Centralized logging — rotating file handler + Rich console."""
from __future__ import annotations

import logging
from logging.handlers import RotatingFileHandler
from pathlib import Path

from rich.logging import RichHandler

from .config import settings

_configured = False


def setup_logging(level: int = logging.INFO, verbose: bool = False, debug: bool = False) -> logging.Logger:
    """Configure root logging.

    verbose: show INFO messages on the console (file always gets INFO+).
    debug:   show DEBUG everywhere and unmute noisy third-party loggers.
    """
    global _configured
    if _configured:
        # Allow re-tuning console verbosity on later calls.
        root = logging.getLogger()
        root.setLevel(level)
        for h in root.handlers:
            if isinstance(h, RichHandler):
                h.setLevel(level)
        return logging.getLogger("jobbot")

    Path(settings.log_path).parent.mkdir(parents=True, exist_ok=True)

    root = logging.getLogger()
    root.setLevel(logging.DEBUG if debug else logging.INFO)
    for h in list(root.handlers):
        root.removeHandler(h)

    fmt = logging.Formatter("%(asctime)s [%(levelname)s] %(name)s: %(message)s")

    fh = RotatingFileHandler(settings.log_path, maxBytes=2_000_000, backupCount=5, encoding="utf-8")
    fh.setFormatter(fmt)
    fh.setLevel(logging.DEBUG if debug else logging.INFO)
    root.addHandler(fh)

    ch = RichHandler(rich_tracebacks=True, show_path=False, show_time=verbose or debug,
                     markup=False)
    ch.setLevel(level)
    root.addHandler(ch)

    noisy_level = logging.INFO if debug else logging.WARNING
    logging.getLogger("apscheduler").setLevel(noisy_level)
    logging.getLogger("urllib3").setLevel(noisy_level)
    logging.getLogger("werkzeug").setLevel(noisy_level)

    _configured = True
    return logging.getLogger("jobbot")
