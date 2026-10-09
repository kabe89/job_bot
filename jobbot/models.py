"""SQLAlchemy ORM: Jobs, Applications, Interviews, Notes."""
from __future__ import annotations

import hashlib
from datetime import datetime
from typing import Optional

from sqlalchemy import String, Integer, Float, DateTime, Text, ForeignKey, create_engine
from sqlalchemy.orm import DeclarativeBase, Mapped, mapped_column, relationship, sessionmaker, Session

from .config import settings


class Base(DeclarativeBase):
    pass


class Job(Base):
    __tablename__ = "jobs"
    id: Mapped[int] = mapped_column(primary_key=True)
    hash: Mapped[str] = mapped_column(String(64), unique=True, index=True)
    source: Mapped[str] = mapped_column(String(64))
    title: Mapped[str] = mapped_column(String(512))
    company: Mapped[str] = mapped_column(String(256))
    location: Mapped[str] = mapped_column(String(256), default="")
    url: Mapped[str] = mapped_column(String(1024))
    description: Mapped[str] = mapped_column(Text, default="")
    salary: Mapped[str] = mapped_column(String(128), default="")
    tags: Mapped[str] = mapped_column(String(512), default="")
    posted_at: Mapped[Optional[datetime]] = mapped_column(DateTime, nullable=True)
    discovered_at: Mapped[datetime] = mapped_column(DateTime, default=datetime.utcnow)
    match_score: Mapped[float] = mapped_column(Float, default=0.0)
    status: Mapped[str] = mapped_column(String(32), default="new")
    notes: Mapped[str] = mapped_column(Text, default="")
    is_local: Mapped[bool] = mapped_column(default=True)
    is_remote: Mapped[bool] = mapped_column(default=False)
    starred: Mapped[bool] = mapped_column(default=False, index=True)
    # Recruiter / hiring-team contacts found for this job, as a JSON list of
    # {name, role, email, linkedin, source_url, found_at} (see contacts.py).
    contacts_json: Mapped[str] = mapped_column(Text, default="")
    interview_prep: Mapped[str] = mapped_column(Text, default="")
    interview_prep_generated_at: Mapped[Optional[datetime]] = mapped_column(DateTime, nullable=True)
    company_intel: Mapped[str] = mapped_column(Text, default="")
    # Machine-readable question bank (JSON) built by interview_coach.generate_pack.
    interview_question_bank: Mapped[str] = mapped_column(Text, default="")
    # AI advisor conversation history (JSON list of {role, content, timestamp})
    chat_history_json: Mapped[str] = mapped_column(Text, default="[]")
    # Auto-trigger flag: set by outcomes.record_outcome when an application
    # reaches the interview stage; cleared by generate_pack.
    interview_prep_pending: Mapped[bool] = mapped_column(default=False, index=True)
    # --- Profile-driven semantic matching (jobbot/embeddings, ranking) ---
    # Stage-1 job embedding as a JSON float list (lets a profile rebuild
    # re-score old jobs without re-embedding). Null until embedded.
    embedding: Mapped[Optional[str]] = mapped_column(Text, nullable=True)
    # Stage-2 LLM re-rank precision score (0..1); only top-K jobs get it.
    rerank_score: Mapped[Optional[float]] = mapped_column(Float, nullable=True)
    # One-line "why this job" from the re-rank judge.
    rerank_rationale: Mapped[Optional[str]] = mapped_column(Text, nullable=True)

    # --- Liveness / expiration (jobbot/liveness.py) ---
    # Set when a liveness probe finds the posting gone; status is flipped to
    # 'expired'. Restorable (status back to 'new', these cleared).
    expired_at: Mapped[Optional[datetime]] = mapped_column(DateTime, nullable=True)
    expiry_reason: Mapped[str] = mapped_column(String(256), default="")
    # When a liveness probe last LOOKED at this posting, whatever the verdict —
    # alive, dead, or a network error. `sweep_all` orders by this (NULLs first),
    # so a sweep resumes where the last one stopped instead of re-probing the
    # same oldest rows forever. Stamped on errors too, or an unreachable host
    # would pin the cursor and starve the rest of the pool.
    last_checked_at: Mapped[Optional[datetime]] = mapped_column(DateTime, nullable=True)

    applications: Mapped[list["Application"]] = relationship(back_populates="job", cascade="all, delete-orphan")

    @staticmethod
    def make_hash(source: str, url: str, title: str) -> str:
        return hashlib.sha256(f"{source}|{url}|{title}".encode()).hexdigest()


class Application(Base):
    __tablename__ = "applications"
    id: Mapped[int] = mapped_column(primary_key=True)
    job_id: Mapped[int] = mapped_column(ForeignKey("jobs.id"))
    tailored_resume_path: Mapped[str] = mapped_column(String(512), default="")
    cover_letter_path: Mapped[str] = mapped_column(String(512), default="")
    notes: Mapped[str] = mapped_column(Text, default="")
    sent_at: Mapped[Optional[datetime]] = mapped_column(DateTime, nullable=True)
    status: Mapped[str] = mapped_column(String(32), default="draft")
    ai_match_analysis: Mapped[str] = mapped_column(Text, default="")
    # Apply-kit: extracted form questions + answers as JSON
    # (see apply_questions.questions_to_json)
    questions_json: Mapped[str] = mapped_column(Text, default="")
    # Follow-up tracking (see followup.py)
    last_followup_at: Mapped[Optional[datetime]] = mapped_column(DateTime, nullable=True)
    followup_count: Mapped[int] = mapped_column(Integer, default=0)
    created_at: Mapped[datetime] = mapped_column(DateTime, default=datetime.utcnow)

    job: Mapped[Job] = relationship(back_populates="applications")
    interviews: Mapped[list["Interview"]] = relationship(back_populates="application", cascade="all, delete-orphan")
    outcome_events: Mapped[list["OutcomeEvent"]] = relationship(cascade="all, delete-orphan")


class Interview(Base):
    __tablename__ = "interviews"
    id: Mapped[int] = mapped_column(primary_key=True)
    application_id: Mapped[int] = mapped_column(ForeignKey("applications.id"))
    scheduled_at: Mapped[Optional[datetime]] = mapped_column(DateTime, nullable=True)
    round_name: Mapped[str] = mapped_column(String(64), default="phone screen")
    interviewer: Mapped[str] = mapped_column(String(256), default="")
    outcome: Mapped[str] = mapped_column(String(32), default="pending")  # pending|passed|failed|no-show
    notes: Mapped[str] = mapped_column(Text, default="")
    created_at: Mapped[datetime] = mapped_column(DateTime, default=datetime.utcnow)

    application: Mapped[Application] = relationship(back_populates="interviews")


class OutcomeEvent(Base):
    """Append-only funnel log: one row per stage transition of an application."""
    __tablename__ = "outcome_events"
    id: Mapped[int] = mapped_column(primary_key=True)
    application_id: Mapped[int] = mapped_column(ForeignKey("applications.id"), index=True)
    stage: Mapped[str] = mapped_column(String(32))
    occurred_at: Mapped[datetime] = mapped_column(DateTime, default=datetime.utcnow)
    note: Mapped[str] = mapped_column(Text, default="")


class Contact(Base):
    """The user's own network — people who could warm-intro / refer them.
    Distinct from Job.contacts_json (per-job recruiter blobs)."""
    __tablename__ = "contacts"
    id: Mapped[int] = mapped_column(primary_key=True)
    name: Mapped[str] = mapped_column(String(256))
    company: Mapped[str] = mapped_column(String(256), default="")
    # Semicolon/newline-separated substring patterns matched against a job's raw
    # company string, so a contact links to their org even when jobs list it
    # under a different name (e.g. Contact @ Subsidiary -> "parent company").
    company_aliases: Mapped[str] = mapped_column(Text, default="")
    title: Mapped[str] = mapped_column(String(256), default="")
    email: Mapped[str] = mapped_column(String(256), default="")
    linkedin: Mapped[str] = mapped_column(String(512), default="")
    source: Mapped[str] = mapped_column(String(32), default="manual")  # manual|discovered|import
    relationship: Mapped[str] = mapped_column(String(32), default="unknown")
    warmth_score: Mapped[float] = mapped_column(Float, default=0.0)
    pinned: Mapped[bool] = mapped_column(default=False, index=True)
    notes: Mapped[str] = mapped_column(Text, default="")
    discovered_at: Mapped[datetime] = mapped_column(DateTime, default=datetime.utcnow)
    last_contacted_at: Mapped[Optional[datetime]] = mapped_column(DateTime, nullable=True)


class OutreachDraft(Base):
    __tablename__ = "outreach_drafts"

    id: Mapped[int] = mapped_column(primary_key=True)
    kind: Mapped[str] = mapped_column(String(16))  # warm_intro|follow_up|cold|reply
    status: Mapped[str] = mapped_column(String(16), default="pending", index=True)
    job_id: Mapped[Optional[int]] = mapped_column(ForeignKey("jobs.id"), nullable=True)
    contact_id: Mapped[Optional[int]] = mapped_column(ForeignKey("contacts.id"), nullable=True)
    application_id: Mapped[Optional[int]] = mapped_column(ForeignKey("applications.id"), nullable=True)
    recipient_email: Mapped[str] = mapped_column(String(256), default="")
    subject: Mapped[str] = mapped_column(String(512), default="")
    body: Mapped[str] = mapped_column(Text, default="")
    inbound: Mapped[str] = mapped_column(Text, default="")
    rationale: Mapped[str] = mapped_column(String(512), default="")
    priority: Mapped[float] = mapped_column(Float, default=0.0)
    dedupe_key: Mapped[str] = mapped_column(String(128), default="", index=True)
    thread_id: Mapped[Optional[int]] = mapped_column(ForeignKey("outreach_threads.id"), nullable=True, index=True)
    message_id: Mapped[str] = mapped_column(String(256), default="")
    in_reply_to: Mapped[str] = mapped_column(String(256), default="")
    created_at: Mapped[datetime] = mapped_column(DateTime, default=datetime.utcnow)
    sent_at: Mapped[Optional[datetime]] = mapped_column(DateTime, nullable=True)


class OutreachThread(Base):
    """One ongoing email conversation with one person.

    Holds the standing situation (`scenario`); the messages themselves live in
    OutreachDraft rows pointing back via thread_id.
    """
    __tablename__ = "outreach_threads"

    id: Mapped[int] = mapped_column(primary_key=True)
    contact_id: Mapped[Optional[int]] = mapped_column(ForeignKey("contacts.id"), nullable=True)
    job_id: Mapped[Optional[int]] = mapped_column(ForeignKey("jobs.id"), nullable=True)
    application_id: Mapped[Optional[int]] = mapped_column(ForeignKey("applications.id"), nullable=True)
    subject: Mapped[str] = mapped_column(String(512), default="")
    scenario: Mapped[str] = mapped_column(Text, default="")
    status: Mapped[str] = mapped_column(String(16), default="open", index=True)
    counterpart_name: Mapped[str] = mapped_column(String(256), default="")
    counterpart_email: Mapped[str] = mapped_column(String(256), default="")
    created_at: Mapped[datetime] = mapped_column(DateTime, default=datetime.utcnow)
    last_inbound_at: Mapped[Optional[datetime]] = mapped_column(DateTime, nullable=True)
    last_outbound_at: Mapped[Optional[datetime]] = mapped_column(DateTime, nullable=True)


class InterviewSession(Base):
    """One mock-interview practice run. Open while ended_at IS NULL."""
    __tablename__ = "interview_sessions"

    id: Mapped[int] = mapped_column(primary_key=True)
    job_id: Mapped[int] = mapped_column(ForeignKey("jobs.id"), index=True)
    # Set only for mode="contact" — the real person the interviewer role-plays.
    contact_id: Mapped[Optional[int]] = mapped_column(ForeignKey("contacts.id"), nullable=True)
    mode: Mapped[str] = mapped_column(String(16), default="general")
    created_at: Mapped[datetime] = mapped_column(DateTime, default=datetime.utcnow)
    ended_at: Mapped[Optional[datetime]] = mapped_column(DateTime, nullable=True)
    overall_score: Mapped[Optional[float]] = mapped_column(Float, nullable=True)
    summary: Mapped[str] = mapped_column(Text, default="")


class InterviewTurn(Base):
    """One question (+ the answer, critique and rubric it earned)."""
    __tablename__ = "interview_turns"

    id: Mapped[int] = mapped_column(primary_key=True)
    session_id: Mapped[int] = mapped_column(ForeignKey("interview_sessions.id"), index=True)
    idx: Mapped[int] = mapped_column(Integer, default=0)
    # Set on adaptive follow-ups: the idx of the base question that spawned this.
    parent_idx: Mapped[Optional[int]] = mapped_column(Integer, nullable=True)
    question: Mapped[str] = mapped_column(Text, default="")
    question_kind: Mapped[str] = mapped_column(String(16), default="behavioral")
    # Provenance, e.g. "company_intel", "jd", "contact:7", "follow_up".
    grounded_in: Mapped[str] = mapped_column(String(64), default="")
    answer: Mapped[str] = mapped_column(Text, default="")
    critique: Mapped[str] = mapped_column(Text, default="")
    score: Mapped[Optional[float]] = mapped_column(Float, nullable=True)
    rubric_json: Mapped[str] = mapped_column(Text, default="")
    created_at: Mapped[datetime] = mapped_column(DateTime, default=datetime.utcnow)


class ApplyRun(Base):
    """One Playwright auto-apply attempt — the observability backbone."""
    __tablename__ = "apply_runs"
    id: Mapped[int] = mapped_column(primary_key=True)
    job_id: Mapped[int] = mapped_column(ForeignKey("jobs.id"), index=True)
    application_id: Mapped[Optional[int]] = mapped_column(
        ForeignKey("applications.id"), nullable=True)
    ats: Mapped[str] = mapped_column(String(32), default="greenhouse")
    # preparing|awaiting_review|submitted|checkpoint|failed|skipped
    status: Mapped[str] = mapped_column(String(32), default="preparing", index=True)
    started_at: Mapped[datetime] = mapped_column(DateTime, default=datetime.utcnow)
    finished_at: Mapped[Optional[datetime]] = mapped_column(DateTime, nullable=True)
    screenshot_path: Mapped[str] = mapped_column(String(512), default="")
    answers_json: Mapped[str] = mapped_column(Text, default="")          # intended + read-back actual
    reconciliation_json: Mapped[str] = mapped_column(Text, default="")   # matched/missing/extra
    checkpoint_reason: Mapped[str] = mapped_column(String(256), default="")
    confirmation_text: Mapped[str] = mapped_column(String(512), default="")
    error: Mapped[str] = mapped_column(Text, default="")


_engine = create_engine(f"sqlite:///{settings.db_path}", echo=False, future=True)
SessionLocal = sessionmaker(bind=_engine, autoflush=False, autocommit=False, future=True)


def init_db() -> None:
    Base.metadata.create_all(_engine)
    # Lightweight migration: ensure newly-added columns exist on pre-existing DBs.
    from sqlalchemy import inspect, text
    inspector = inspect(_engine)
    if "jobs" in inspector.get_table_names():
        existing = {c["name"] for c in inspector.get_columns("jobs")}
        with _engine.begin() as conn:
            for col, ddl in (
                ("interview_prep", "ALTER TABLE jobs ADD COLUMN interview_prep TEXT DEFAULT ''"),
                ("interview_prep_generated_at", "ALTER TABLE jobs ADD COLUMN interview_prep_generated_at DATETIME"),
                ("company_intel", "ALTER TABLE jobs ADD COLUMN company_intel TEXT DEFAULT ''"),
                ("notes", "ALTER TABLE jobs ADD COLUMN notes TEXT DEFAULT ''"),
                ("is_local", "ALTER TABLE jobs ADD COLUMN is_local BOOLEAN DEFAULT 1"),
                ("is_remote", "ALTER TABLE jobs ADD COLUMN is_remote BOOLEAN DEFAULT 0"),
                ("starred", "ALTER TABLE jobs ADD COLUMN starred BOOLEAN DEFAULT 0"),
                ("contacts_json", "ALTER TABLE jobs ADD COLUMN contacts_json TEXT DEFAULT ''"),
                ("embedding", "ALTER TABLE jobs ADD COLUMN embedding TEXT"),
                ("rerank_score", "ALTER TABLE jobs ADD COLUMN rerank_score FLOAT"),
                ("rerank_rationale", "ALTER TABLE jobs ADD COLUMN rerank_rationale TEXT"),
                ("expired_at", "ALTER TABLE jobs ADD COLUMN expired_at DATETIME"),
                ("expiry_reason", "ALTER TABLE jobs ADD COLUMN expiry_reason TEXT DEFAULT ''"),
                ("last_checked_at", "ALTER TABLE jobs ADD COLUMN last_checked_at DATETIME"),
                ("interview_question_bank", "ALTER TABLE jobs ADD COLUMN interview_question_bank TEXT DEFAULT ''"),
                ("chat_history_json", "ALTER TABLE jobs ADD COLUMN chat_history_json TEXT DEFAULT '[]'"),
                ("interview_prep_pending", "ALTER TABLE jobs ADD COLUMN interview_prep_pending BOOLEAN DEFAULT 0"),
            ):
                if col not in existing:
                    try:
                        conn.execute(text(ddl))
                    except Exception:
                        pass
    if "applications" in inspector.get_table_names():
        existing = {c["name"] for c in inspector.get_columns("applications")}
        with _engine.begin() as conn:
            for col, ddl in (
                ("questions_json", "ALTER TABLE applications ADD COLUMN questions_json TEXT DEFAULT ''"),
                ("last_followup_at", "ALTER TABLE applications ADD COLUMN last_followup_at DATETIME"),
                ("followup_count", "ALTER TABLE applications ADD COLUMN followup_count INTEGER DEFAULT 0"),
            ):
                if col not in existing:
                    try:
                        conn.execute(text(ddl))
                    except Exception:
                        pass
    if "contacts" in inspector.get_table_names():
        existing = {c["name"] for c in inspector.get_columns("contacts")}
        with _engine.begin() as conn:
            for col, ddl in (
                ("company_aliases", "ALTER TABLE contacts ADD COLUMN company_aliases TEXT DEFAULT ''"),
            ):
                if col not in existing:
                    try:
                        conn.execute(text(ddl))
                    except Exception:
                        pass
    if "outreach_drafts" in inspector.get_table_names():
        existing = {c["name"] for c in inspector.get_columns("outreach_drafts")}
        with _engine.begin() as conn:
            for col, ddl in (
                ("thread_id", "ALTER TABLE outreach_drafts ADD COLUMN thread_id INTEGER"),
                ("message_id", "ALTER TABLE outreach_drafts ADD COLUMN message_id TEXT DEFAULT ''"),
                ("in_reply_to", "ALTER TABLE outreach_drafts ADD COLUMN in_reply_to TEXT DEFAULT ''"),
            ):
                if col not in existing:
                    try:
                        conn.execute(text(ddl))
                    except Exception:
                        pass
    if "apply_runs" not in inspector.get_table_names():
        Base.metadata.tables["apply_runs"].create(_engine, checkfirst=True)


def session() -> Session:
    return SessionLocal()


def backup_db(keep: int = 10) -> Optional[str]:
    """Make a timestamped copy of the SQLite DB next to it, keeping the most
    recent `keep` backups. Returns the backup path, or None if there's no DB
    yet / nothing to back up. Safe to call often (skips if a same-second backup
    already exists). Used before risky ops (scrape cycles, serve startup).
    """
    import glob
    import shutil
    from datetime import datetime as _dt
    from pathlib import Path as _Path

    src = _Path(settings.db_path)
    if not src.exists() or src.stat().st_size == 0:
        return None
    dst = src.with_name(f"{src.name}.autobak-{_dt.now():%Y%m%d-%H%M%S}")
    if dst.exists():
        return str(dst)
    try:
        shutil.copy2(src, dst)
    except Exception:  # noqa: BLE001
        return None
    # Rotate: keep only the newest `keep` autobaks.
    baks = sorted(glob.glob(str(src.with_name(src.name + ".autobak-*"))))
    for old in baks[:-keep]:
        try:
            _Path(old).unlink()
        except Exception:  # noqa: BLE001
            pass
    return str(dst)
