"""CSV export of jobs + applications."""
from __future__ import annotations

import csv
from pathlib import Path
from typing import Iterable

from .config import settings
from .models import Application, Job, session


JOB_FIELDS = ["id", "score", "title", "company", "location", "source", "status",
              "url", "posted_at", "discovered_at", "salary", "tags"]
APP_FIELDS = ["id", "job_title", "company", "status", "created_at", "sent_at",
              "resume_path", "cover_letter_path"]


def export_jobs(out_path: str | Path | None = None) -> Path:
    out_path = Path(out_path or Path(settings.output_dir) / "jobs.csv")
    out_path.parent.mkdir(parents=True, exist_ok=True)
    with session() as db, open(out_path, "w", newline="", encoding="utf-8") as f:
        w = csv.writer(f)
        w.writerow(JOB_FIELDS)
        for j in db.query(Job).order_by(Job.match_score.desc()).all():
            w.writerow([
                j.id, f"{j.match_score:.3f}", j.title, j.company, j.location,
                j.source, j.status, j.url,
                j.posted_at.isoformat() if j.posted_at else "",
                j.discovered_at.isoformat() if j.discovered_at else "",
                j.salary, j.tags,
            ])
    return out_path


def export_applications(out_path: str | Path | None = None) -> Path:
    out_path = Path(out_path or Path(settings.output_dir) / "applications.csv")
    out_path.parent.mkdir(parents=True, exist_ok=True)
    with session() as db, open(out_path, "w", newline="", encoding="utf-8") as f:
        w = csv.writer(f)
        w.writerow(APP_FIELDS)
        for a in db.query(Application).order_by(Application.created_at.desc()).all():
            w.writerow([
                a.id, a.job.title if a.job else "", a.job.company if a.job else "",
                a.status,
                a.created_at.isoformat() if a.created_at else "",
                a.sent_at.isoformat() if a.sent_at else "",
                a.tailored_resume_path, a.cover_letter_path,
            ])
    return out_path
