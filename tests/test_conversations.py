from datetime import datetime, timedelta

import pytest

from jobbot.models import (Application, Contact, Job, OutreachDraft,
                           OutreachThread, init_db, session)


@pytest.fixture(autouse=True)
def _clean_db():
    def _wipe():
        init_db()
        with session() as db:
            db.query(OutreachDraft).delete()
            db.query(OutreachThread).delete()
            db.query(Contact).delete()
            db.query(Application).delete()
            db.query(Job).delete()
            db.commit()
    _wipe()
    yield
    _wipe()


def _job(db, **kw):
    d = dict(hash=kw.pop("hash", "j1"), source="greenhouse:test",
             title="Research Scientist", company="Acme Corporation",
             url="https://x", description="d", match_score=0.6)
    d.update(kw)
    j = Job(**d); db.add(j); db.commit(); db.refresh(j); return j


def test_thread_roundtrips_and_draft_links_to_it():
    with session() as db:
        j = _job(db)
        t = OutreachThread(job_id=j.id, subject="Research Scientist role",
                           scenario="Recruiter reached out on LinkedIn.",
                           counterpart_email="jane@acme.com")
        db.add(t); db.commit(); db.refresh(t)
        tid = t.id
        d = OutreachDraft(kind="reply", status="pending", thread_id=tid,
                          subject="Re: Research Scientist role", body="hi")
        db.add(d); db.commit(); db.refresh(d)
        assert d.thread_id == tid
        assert d.message_id == ""
        assert d.in_reply_to == ""
        assert db.get(OutreachThread, tid).status == "open"


def test_existing_drafts_have_null_thread_id():
    with session() as db:
        d = OutreachDraft(kind="cold", status="pending", subject="s", body="b")
        db.add(d); db.commit(); db.refresh(d)
        assert d.thread_id is None


def test_migration_ladder_adds_draft_columns(tmp_path, monkeypatch):
    """Verify ALTER TABLE migrations execute on pre-existing outreach_drafts tables.

    This test must create an OLD-SHAPE database (without the new columns) BEFORE
    the model is instantiated, so the new column definitions don't leak into the
    initial CREATE TABLE. We do this with a separate engine on a temp file, using
    raw SQL to avoid the model's definition. Then we point jobbot's module-level
    engine at that DB and call init_db() to exercise the ALTER ladder.
    """
    from sqlalchemy import create_engine, inspect, text
    import jobbot.models

    # Create a temp database with OLD-SHAPE outreach_drafts table (no new columns).
    tmp_db = tmp_path / "test_old_schema.db"
    old_engine = create_engine(f"sqlite:///{tmp_db}")

    # Create old-shape outreach_drafts table (only the columns that existed before).
    with old_engine.begin() as conn:
        conn.execute(text("""
            CREATE TABLE outreach_drafts (
                id INTEGER PRIMARY KEY,
                kind TEXT NOT NULL,
                status TEXT DEFAULT 'pending',
                job_id INTEGER,
                contact_id INTEGER,
                application_id INTEGER,
                recipient_email TEXT DEFAULT '',
                subject TEXT DEFAULT '',
                body TEXT DEFAULT '',
                inbound TEXT DEFAULT '',
                rationale TEXT DEFAULT '',
                priority FLOAT DEFAULT 0.0,
                dedupe_key TEXT DEFAULT '',
                created_at DATETIME DEFAULT CURRENT_TIMESTAMP,
                sent_at DATETIME
            )
        """))
        # Insert one row to verify data survives migration.
        conn.execute(text("""
            INSERT INTO outreach_drafts (id, kind, status, subject, body)
            VALUES (1, 'cold', 'pending', 'Test Subject', 'Test Body')
        """))

    # Monkeypatch jobbot.models._engine and SessionLocal to point at our temp DB.
    original_engine = jobbot.models._engine
    original_session_local = jobbot.models.SessionLocal
    monkeypatch.setattr(jobbot.models, "_engine", old_engine)
    monkeypatch.setattr(jobbot.models, "SessionLocal",
                        jobbot.models.sessionmaker(bind=old_engine, autoflush=False, autocommit=False, future=True))

    try:
        # Call init_db() — this must run the ALTER TABLE migrations.
        init_db()

        # Verify the three new columns exist.
        inspector = inspect(old_engine)
        cols = {c["name"] for c in inspector.get_columns("outreach_drafts")}
        assert {"thread_id", "message_id", "in_reply_to"} <= cols

        # Verify the old row still exists and its original values are unchanged.
        with old_engine.connect() as conn:
            result = conn.execute(text(
                "SELECT id, kind, status, subject, body FROM outreach_drafts WHERE id = 1"
            )).fetchone()
            assert result is not None
            assert result[0] == 1  # id
            assert result[1] == "cold"  # kind
            assert result[2] == "pending"  # status
            assert result[3] == "Test Subject"  # subject
            assert result[4] == "Test Body"  # body

            # Verify new columns are NULL for the old row.
            result = conn.execute(text(
                "SELECT thread_id, message_id, in_reply_to FROM outreach_drafts WHERE id = 1"
            )).fetchone()
            assert result[0] is None  # thread_id
            assert result[1] == ""  # message_id
            assert result[2] == ""  # in_reply_to
    finally:
        # Restore the original engine and SessionLocal.
        monkeypatch.setattr(jobbot.models, "_engine", original_engine)
        monkeypatch.setattr(jobbot.models, "SessionLocal", original_session_local)


def test_responded_application_still_trains_callback_model():
    from jobbot import predict
    with session() as db:
        j = _job(db, hash="j-train", status="interview")
        app = Application(job_id=j.id, status="responded",
                          sent_at=datetime.utcnow() - timedelta(days=60))
        db.add(app); db.commit()
    X, y, n_pos, n_neg = predict.collect_training_data()
    assert n_pos == 1, "a responded application with an interview must count as positive"
    assert y == [1]


def test_ghosted_application_counts_as_negative():
    from jobbot import predict
    with session() as db:
        j = _job(db, hash="j-ghost", status="rejected")
        app = Application(job_id=j.id, status="ghosted",
                          sent_at=datetime.utcnow() - timedelta(days=90))
        db.add(app); db.commit()
    X, y, n_pos, n_neg = predict.collect_training_data()
    assert n_neg == 1


def test_scrub_removes_em_dashes():
    from jobbot import outreach
    assert "—" not in outreach._scrub("I am free — let me know.")
    assert "–" not in outreach._scrub("Tue–Thu works.")
    assert "--" not in outreach._scrub("I am free -- let me know.")


def test_fallback_template_has_no_em_dash(monkeypatch):
    from jobbot import outreach
    monkeypatch.setattr(outreach.ollama_client, "is_available", lambda: False)
    _subject, body = outreach._reply_text("Are you available Tuesday?")
    assert "—" not in body and "--" not in body


def test_model_output_is_scrubbed(monkeypatch):
    from jobbot import outreach
    monkeypatch.setattr(outreach.ollama_client, "is_available", lambda: True)
    monkeypatch.setattr(outreach.ollama_client, "_generate",
                        lambda *a, **k: "Sounds great — Tuesday works for me.")
    _subject, body = outreach._reply_text("Are you available Tuesday?")
    assert "—" not in body
    assert "Tuesday works for me" in body


def _capture_prompt(monkeypatch, seen):
    """Stub Ollama as available and record the prompt it is handed."""
    from jobbot import outreach

    def _gen(prompt, **kwargs):
        seen["p"] = prompt
        return "ok"

    monkeypatch.setattr(outreach.ollama_client, "is_available", lambda: True)
    monkeypatch.setattr(outreach.ollama_client, "_generate", _gen)


def test_scenario_and_history_reach_the_prompt(monkeypatch):
    from jobbot import outreach
    seen = {}
    _capture_prompt(monkeypatch, seen)
    hist = [
        {"direction": "sent", "text": "Applied for the role.",
         "at": datetime(2026, 7, 28)},
        {"direction": "received", "text": "Thanks, reviewing now.",
         "at": datetime(2026, 8, 2)},
    ]
    outreach._reply_text("Can you start in September?",
                         scenario="Recruiter at Metro Agency. Asked about Python/SQL.",
                         history=hist, subject="Research Scientist role")
    p = seen["p"]
    assert "Recruiter at Metro Agency" in p
    assert "SENT (2026-07-28): Applied for the role." in p
    assert "RECEIVED (2026-08-02): Thanks, reviewing now." in p
    assert "Can you start in September?" in p


def test_prompt_omits_empty_blocks(monkeypatch):
    from jobbot import outreach
    seen = {}
    _capture_prompt(monkeypatch, seen)
    outreach._reply_text("Just this.")
    assert "--- Situation ---" not in seen["p"]
    assert "--- Conversation so far ---" not in seen["p"]


def test_history_is_trimmed_to_budget(monkeypatch):
    from jobbot import outreach
    seen = {}
    _capture_prompt(monkeypatch, seen)
    hist = [{"direction": "sent", "text": f"message number {i}",
             "at": datetime(2026, 7, 1)} for i in range(20)]
    outreach._reply_text("latest", history=hist)
    assert "message number 19" in seen["p"], "newest must be kept"
    assert "message number 0" not in seen["p"], "oldest must be dropped"


def test_subject_uses_thread_subject(monkeypatch):
    from jobbot import outreach
    # Stub the client off: this test is about subject derivation only, and an
    # unstubbed _reply_text would reach a real Ollama server.
    monkeypatch.setattr(outreach.ollama_client, "is_available", lambda: False)
    subject, _body = outreach._reply_text("x", subject="Research Scientist role")
    assert subject == "Re: Research Scientist role"
    subject2, _ = outreach._reply_text("x", subject="Re: Already prefixed")
    assert subject2 == "Re: Already prefixed"


def test_start_creates_open_thread_with_scenario():
    from jobbot import conversations as cv
    with session() as db:
        j = _job(db, hash="j-start")
        jid = j.id
    t = cv.start(job_id=jid, scenario="Recruiter reached out.", subject="RS role")
    assert t.id is not None
    assert t.status == "open"
    assert t.scenario == "Recruiter reached out."
    assert t.subject == "RS role"


def test_start_fills_counterpart_from_contact():
    from jobbot import conversations as cv
    with session() as db:
        c = Contact(name="Jane Doe", email="jane@acme.com", company="Acme")
        db.add(c); db.commit(); db.refresh(c)
        cid = c.id
    t = cv.start(contact_id=cid, scenario="s")
    assert t.counterpart_name == "Jane Doe"
    assert t.counterpart_email == "jane@acme.com"


def test_history_is_empty_for_new_thread():
    from jobbot import conversations as cv
    t = cv.start(scenario="s")
    assert cv.history(t.id) == []


def test_history_orders_by_entry_timestamp_not_draft_creation():
    """A draft created Monday but sent Friday must fall AFTER a reply
    received Wednesday."""
    from jobbot import conversations as cv
    t = cv.start(scenario="s")
    with session() as db:
        early = OutreachDraft(kind="reply", status="sent", thread_id=t.id,
                              body="sent late", created_at=datetime(2026, 8, 3),
                              sent_at=datetime(2026, 8, 7))
        later = OutreachDraft(kind="reply", status="pending", thread_id=t.id,
                              inbound="received midweek",
                              created_at=datetime(2026, 8, 5))
        db.add_all([early, later]); db.commit()
    rows = cv.history(t.id)
    assert [r["direction"] for r in rows] == ["received", "sent"]
    assert rows[0]["text"] == "received midweek"


def test_history_skips_pending_outbound():
    from jobbot import conversations as cv
    t = cv.start(scenario="s")
    with session() as db:
        db.add(OutreachDraft(kind="reply", status="pending", thread_id=t.id,
                             body="not sent yet"))
        db.commit()
    assert cv.history(t.id) == []


def test_history_seeds_opening_entry_from_application():
    from jobbot import conversations as cv
    with session() as db:
        j = _job(db, hash="j-seed")
        app = Application(job_id=j.id, status="sent",
                          sent_at=datetime(2026, 7, 1))
        db.add(app); db.commit(); db.refresh(app)
        aid, jid = app.id, j.id
    t = cv.start(job_id=jid, application_id=aid, scenario="s")
    rows = cv.history(t.id)
    assert len(rows) == 1
    assert rows[0]["direction"] == "sent"
    assert "Applied to" in rows[0]["text"]


def test_set_scenario_replaces_text():
    from jobbot import conversations as cv
    t = cv.start(scenario="old")
    updated = cv.set_scenario(t.id, "new situation")
    assert updated.scenario == "new situation"


def test_close_hides_thread_from_list():
    from jobbot import conversations as cv
    t = cv.start(scenario="s")
    assert t.id in [x.id for x in cv.list_open()]
    cv.close(t.id)
    assert t.id not in [x.id for x in cv.list_open()]


def test_unknown_thread_id_raises():
    from jobbot import conversations as cv
    with pytest.raises(ValueError):
        cv.history(99999)


@pytest.fixture
def _no_ollama(monkeypatch):
    from jobbot import outreach
    monkeypatch.setattr(outreach.ollama_client, "is_available", lambda: False)


def test_ingest_records_message_and_queues_draft(_no_ollama):
    from jobbot import conversations as cv
    t = cv.start(scenario="Recruiter at Acme.", subject="RS role")
    d = cv.ingest(t.id, "Are you free Tuesday?", sender="jane@acme.com")
    assert d.thread_id == t.id
    assert d.kind == "reply"
    assert d.status == "pending"
    assert d.inbound == "Are you free Tuesday?"
    assert d.recipient_email == "jane@acme.com"
    assert d.subject == "Re: RS role"


def test_ingest_stamps_last_inbound_at(_no_ollama):
    from jobbot import conversations as cv
    t = cv.start(scenario="s")
    cv.ingest(t.id, "hello")
    with session() as db:
        assert db.get(OutreachThread, t.id).last_inbound_at is not None


def test_same_paste_twice_dedupes(_no_ollama):
    from jobbot import conversations as cv
    t = cv.start(scenario="s")
    d1 = cv.ingest(t.id, "Are you free Tuesday?")
    d2 = cv.ingest(t.id, "  are you   FREE tuesday?  ")   # normalizes identically
    assert d1.id == d2.id
    with session() as db:
        n = db.query(OutreachDraft).filter_by(thread_id=t.id).count()
        assert n == 1


def test_different_pastes_create_separate_drafts(_no_ollama):
    from jobbot import conversations as cv
    t = cv.start(scenario="s")
    d1 = cv.ingest(t.id, "Are you free Tuesday?")
    d2 = cv.ingest(t.id, "Actually, can we do Thursday?")
    assert d1.id != d2.id


def test_ingest_passes_scenario_and_history_to_drafter(monkeypatch, _no_ollama):
    from jobbot import conversations as cv, outreach
    seen = {}

    def _fake(inbound, sender="", role="", scenario="", history=(), subject=""):
        seen["scenario"] = scenario
        seen["history"] = list(history)
        return "Re: x", "body"

    monkeypatch.setattr(outreach, "_reply_text", _fake)
    t = cv.start(scenario="Recruiter at Metro Agency.", subject="x")
    cv.ingest(t.id, "First question?")
    cv.ingest(t.id, "Second question?")
    assert seen["scenario"] == "Recruiter at Metro Agency."
    assert any("First question?" in h["text"] for h in seen["history"]), \
        "the second ingest must see the first inbound message"


def test_ingest_marks_application_responded(_no_ollama):
    from jobbot import conversations as cv
    with session() as db:
        j = _job(db, hash="j-resp")
        app = Application(job_id=j.id, status="sent",
                          sent_at=datetime.utcnow() - timedelta(days=30))
        db.add(app); db.commit(); db.refresh(app)
        aid = app.id
    t = cv.start(application_id=aid, scenario="s")
    cv.ingest(t.id, "Thanks for applying, let's talk.")
    with session() as db:
        assert db.get(Application, aid).status == "responded"


def test_ingest_suppresses_the_followup_nudge(_no_ollama):
    """The claim this whole feature rests on. Builds a REAL application that
    genuinely qualifies rather than stubbing followups_due."""
    from jobbot import conversations as cv, followup
    from jobbot.config import settings
    with session() as db:
        j = _job(db, hash="j-nudge", status="new")
        app = Application(
            job_id=j.id, status="sent", followup_count=0,
            sent_at=datetime.utcnow() - timedelta(days=settings.followup_after_days + 5))
        db.add(app); db.commit(); db.refresh(app)
        aid = app.id
    due_before = [r["application_id"] for r in followup.followups_due()]
    assert aid in due_before, "precondition: the app must actually be due"

    t = cv.start(application_id=aid, scenario="s")
    cv.ingest(t.id, "Hi, we would like to schedule a call.")

    due_after = [r["application_id"] for r in followup.followups_due()]
    assert aid not in due_after, "a replied-to application must stop being nudged"


def test_ingest_without_application_is_a_noop_not_an_error(_no_ollama):
    from jobbot import conversations as cv
    t = cv.start(scenario="s")
    d = cv.ingest(t.id, "hello")
    assert d is not None


def test_record_outcome_failure_does_not_lose_the_message(monkeypatch, _no_ollama):
    from jobbot import conversations as cv
    def _boom(*a, **k):
        raise RuntimeError("outcome store down")
    monkeypatch.setattr(cv.outcomes, "record_outcome", _boom)
    with session() as db:
        j = _job(db, hash="j-boom")
        app = Application(job_id=j.id, status="sent", sent_at=datetime.utcnow())
        db.add(app); db.commit(); db.refresh(app)
        aid = app.id
    t = cv.start(application_id=aid, scenario="s")
    d = cv.ingest(t.id, "important message")
    assert d.inbound == "important message"


def test_ingest_falls_back_to_template_when_ollama_down(_no_ollama):
    from jobbot import conversations as cv
    t = cv.start(scenario="s")
    d = cv.ingest(t.id, "hello")
    assert d.body.strip() != ""
    assert "ungenerated" in d.rationale.lower()


def test_ingest_rejects_empty_paste(_no_ollama):
    from jobbot import conversations as cv
    t = cv.start(scenario="s")
    with pytest.raises(ValueError):
        cv.ingest(t.id, "   \n  ")


def test_ingest_rejects_closed_thread(_no_ollama):
    from jobbot import conversations as cv
    t = cv.start(scenario="s")
    cv.close(t.id)
    with pytest.raises(ValueError):
        cv.ingest(t.id, "hello")


def test_emailer_sets_and_returns_message_id(monkeypatch):
    from jobbot import emailer
    captured = {}

    class _FakeSMTP:
        def __enter__(self): return self
        def __exit__(self, *a): return False
        def send_message(self, msg): captured["msg"] = msg

    monkeypatch.setattr(emailer, "_smtp", lambda: _FakeSMTP())
    mid = emailer.send(to="a@b.com", subject="s", body_html="<p>x</p>", body_text="x")
    assert mid
    assert mid.startswith("<") and mid.endswith(">")
    assert captured["msg"]["Message-ID"] == mid


def test_send_stores_message_id_on_draft(monkeypatch):
    from jobbot import outreach
    monkeypatch.setattr(outreach.emailer, "send",
                        lambda **kw: "<abc123@example.com>")
    d = outreach._queue(kind="reply", subject="s", body="b",
                        recipient_email="x@y.com", dedupe_key="mid-test-1")
    outreach.approve(d.id)
    outreach.send(d.id)
    with session() as db:
        assert db.get(OutreachDraft, d.id).message_id == "<abc123@example.com>"


def test_send_tolerates_emailer_returning_none(monkeypatch):
    """tests/test_outreach.py stubs emailer.send as `lambda **kw: None`.
    An unconditional store would break the existing suite."""
    from jobbot import outreach
    monkeypatch.setattr(outreach.emailer, "send", lambda **kw: None)
    d = outreach._queue(kind="reply", subject="s", body="b",
                        recipient_email="x@y.com", dedupe_key="mid-test-2")
    outreach.approve(d.id)
    res = outreach.send(d.id)
    assert res["status"] == "sent"
    with session() as db:
        assert db.get(OutreachDraft, d.id).message_id == ""


def test_cli_convo_start_paste_and_show(tmp_path, monkeypatch):
    from click.testing import CliRunner
    from jobbot import outreach
    from jobbot.cli import cli
    monkeypatch.setattr(outreach.ollama_client, "is_available", lambda: False)
    runner = CliRunner()

    res = runner.invoke(cli, ["convo", "start", "--scenario", "Recruiter at Acme.",
                              "--subject", "RS role"])
    assert res.exit_code == 0, res.output
    tid = int(res.output.strip().split()[-1].rstrip("."))

    email_file = tmp_path / "inbound.txt"
    email_file.write_text("Are you free Tuesday?", encoding="utf-8")
    res = runner.invoke(cli, ["convo", "paste", str(tid), "--file", str(email_file)])
    assert res.exit_code == 0, res.output

    res = runner.invoke(cli, ["convo", "show", str(tid)])
    assert res.exit_code == 0
    assert "Recruiter at Acme." in res.output
    assert "Are you free Tuesday?" in res.output

    res = runner.invoke(cli, ["convo", "list"])
    assert str(tid) in res.output

    res = runner.invoke(cli, ["convo", "close", str(tid)])
    assert res.exit_code == 0
    res = runner.invoke(cli, ["convo", "list"])
    assert str(tid) not in res.output


def test_thread_page_renders(monkeypatch):
    from jobbot import conversations as cv, outreach
    from jobbot.web import create_app
    monkeypatch.setattr(outreach.ollama_client, "is_available", lambda: False)
    t = cv.start(scenario="Recruiter at Acme.", subject="RS role")
    cv.ingest(t.id, "Are you free Tuesday?")
    app = create_app()
    app.config.update(TESTING=True)
    cl = app.test_client()
    r = cl.get(f"/outreach/thread/{t.id}")
    assert r.status_code == 200
    html = r.get_data(as_text=True)
    assert "Recruiter at Acme." in html
    assert "Are you free Tuesday?" in html


def test_thread_page_404s_on_unknown_id():
    from jobbot.web import create_app
    app = create_app()
    app.config.update(TESTING=True)
    assert app.test_client().get("/outreach/thread/99999").status_code == 404


# --- Whole-branch review fixes -------------------------------------------

def test_scrub_preserves_double_hyphen_inside_a_token():
    """A bare `--` inside a token is not an em dash. Rewriting it silently
    breaks any URL, DOI or path the model cites in a reply."""
    from jobbot import outreach
    assert outreach._scrub("See https://x.com/a--b now.") == "See https://x.com/a--b now."
    assert "--" not in outreach._scrub("I am free -- let me know.")
    assert outreach._scrub("a --- b") == "a, b"


def test_dedupe_hit_is_flagged_and_still_stamps_last_inbound(_no_ollama):
    from jobbot import conversations as cv
    t = cv.start(scenario="s")
    d1 = cv.ingest(t.id, "Are you free Tuesday?")
    assert d1.is_new is True
    with session() as db:                      # clear the stamp the first call set
        db.get(OutreachThread, t.id).last_inbound_at = None
        db.commit()
    d2 = cv.ingest(t.id, "  are you   FREE tuesday?  ")
    assert d2.id == d1.id
    assert d2.is_new is False, "a dedupe hit must be distinguishable from a fresh draft"
    with session() as db:
        assert db.get(OutreachThread, t.id).last_inbound_at is not None, \
            "a repeated paste is still an inbound message and must move the clock"


def test_dedupe_hit_still_records_the_outcome(monkeypatch, _no_ollama):
    """Two different emails whose visible text is identical ('Thanks!') collide.
    The second must still suppress the follow-up nudge."""
    from jobbot import conversations as cv
    with session() as db:
        j = _job(db, hash="j-dedupe-outcome")
        app = Application(job_id=j.id, status="sent", sent_at=datetime.utcnow())
        db.add(app); db.commit(); db.refresh(app)
        aid = app.id
    t = cv.start(application_id=aid, scenario="s")
    cv.ingest(t.id, "Thanks!")
    calls = []
    monkeypatch.setattr(cv.outcomes, "record_outcome",
                        lambda *a, **k: calls.append(a))
    cv.ingest(t.id, "Thanks!")
    assert calls, "a deduped paste must still re-assert the responded outcome"


def test_rationale_marks_ungenerated_when_generation_raises(monkeypatch):
    """is_available() reports server reachability, not that generation happened.
    _reply_text catches its own failures and falls back to the template."""
    from jobbot import conversations as cv, outreach

    def _boom(*a, **k):
        raise RuntimeError("model fell over")

    monkeypatch.setattr(outreach.ollama_client, "is_available", lambda: True)
    monkeypatch.setattr(outreach.ollama_client, "_generate", _boom)
    t = cv.start(scenario="s")
    d = cv.ingest(t.id, "hello")
    assert "ungenerated" in d.rationale.lower()


def test_daily_quota_counts_applications_that_got_a_reply():
    """The 'responded' flip must not buy back apply-rate-limit headroom."""
    from jobbot import pipeline
    from jobbot.config import settings
    with session() as db:
        j = _job(db, hash="j-quota")
        db.add(Application(job_id=j.id, status="responded",
                           sent_at=datetime.utcnow()))
        db.commit()
    assert pipeline.daily_apply_quota_remaining() == settings.apply_rate_limit_per_day - 1


def test_scrub_catches_dashes_not_followed_by_a_word():
    """The dash form must not survive just because punctuation or end-of-string
    follows it. Only a token-internal `--` (no whitespace on either side) is
    preserved, because that is the URL/slug shape."""
    from jobbot import outreach
    for text in ('Talk soon --', 'word -- ', 'She said no -- "yes" was my answer',
                 'wrap up -- .', '-- leading dash', 'a --- b'):
        assert "--" not in outreach._scrub(text), text
    # Token-internal stays intact.
    assert outreach._scrub("See https://x.com/a--b now.") == "See https://x.com/a--b now."
    assert outreach._scrub("word--word") == "word--word"


# ---------------------------------------------------------------- convo sent


def _thread(scenario="Networking intro.", subject="Intro", email="alex@x.com"):
    from jobbot import conversations as cv
    with session() as db:
        c = Contact(name="Alex", email=email); db.add(c); db.commit(); db.refresh(c)
        cid = c.id
    return cv.start(contact_id=cid, scenario=scenario, subject=subject)


def test_record_sent_lands_in_history_at_its_own_timestamp():
    """The whole point: an email sent by hand, outside the bot, becomes history.

    Without this the thread renders empty and every later draft is written with
    no idea what was already said.
    """
    from jobbot import conversations as cv
    t = _thread()
    when = datetime(2026, 7, 24, 16, 9)
    cv.record_sent(t.id, "Thank you for meeting with me.", subject="Thank you",
                   sent_at=when)
    rows = cv.history(t.id)
    assert len(rows) == 1
    assert rows[0]["direction"] == "sent"
    assert rows[0]["at"] == when
    assert "Thank you for meeting" in rows[0]["text"]


def test_record_sent_moves_last_outbound_at():
    from jobbot import conversations as cv
    t = _thread()
    when = datetime(2026, 7, 24, 16, 9)
    cv.record_sent(t.id, "body", sent_at=when)
    with session() as db:
        assert db.get(OutreachThread, t.id).last_outbound_at == when


def test_record_sent_is_not_queued_for_sending():
    """It already went out. It must never be approvable or re-sendable."""
    from jobbot import conversations as cv
    t = _thread()
    d = cv.record_sent(t.id, "body")
    with session() as db:
        row = db.get(OutreachDraft, d.id)
        assert row.status == "sent"
        assert row.sent_at is not None


def test_record_sent_dedupes_on_content():
    from jobbot import conversations as cv
    t = _thread()
    a = cv.record_sent(t.id, "Thanks again for the time.")
    b = cv.record_sent(t.id, "Thanks   again for the time.\n")
    assert b.id == a.id
    assert getattr(b, "is_new", True) is False
    with session() as db:
        assert db.query(OutreachDraft).filter(
            OutreachDraft.thread_id == t.id).count() == 1


def test_record_sent_rejects_empty_body():
    from jobbot import conversations as cv
    t = _thread()
    with pytest.raises(ValueError):
        cv.record_sent(t.id, "   ")


# --------------------------------------------------------------- convo nudge


def test_nudge_requires_a_prior_outbound():
    """Nothing has been said yet, so there is nothing to nudge about."""
    from jobbot import conversations as cv
    t = _thread()
    with pytest.raises(ValueError):
        cv.draft_nudge(t.id)


def test_nudge_queues_a_pending_draft_on_the_thread(monkeypatch):
    from jobbot import conversations as cv, outreach
    monkeypatch.setattr(outreach.ollama_client, "is_available", lambda: False)
    t = _thread()
    cv.record_sent(t.id, "CV attached.", sent_at=datetime.utcnow() - timedelta(days=18))
    d = cv.draft_nudge(t.id)
    assert d.thread_id == t.id
    assert d.status == "pending"
    assert d.recipient_email == "alex@x.com"
    assert d.subject.startswith("Re: ")
    assert "--" not in d.body and "—" not in d.body


def test_nudge_prompt_carries_days_elapsed_scenario_and_history(monkeypatch):
    from jobbot import conversations as cv
    seen = {}
    _capture_prompt(monkeypatch, seen)
    t = _thread(scenario="He offered to reach out to recruiters.")
    cv.record_sent(t.id, "CV attached for forwarding.",
                   sent_at=datetime.utcnow() - timedelta(days=18))
    cv.draft_nudge(t.id)
    p = seen["p"]
    assert "18 days" in p
    assert "He offered to reach out to recruiters." in p
    assert "CV attached for forwarding." in p


def test_nudge_is_deduped_until_dismissed(monkeypatch):
    from jobbot import conversations as cv, outreach
    monkeypatch.setattr(outreach.ollama_client, "is_available", lambda: False)
    t = _thread()
    cv.record_sent(t.id, "CV attached.", sent_at=datetime.utcnow() - timedelta(days=18))
    a = cv.draft_nudge(t.id)
    b = cv.draft_nudge(t.id)
    assert b.id == a.id
    assert getattr(b, "is_new", True) is False


def test_nudge_refuses_a_closed_thread(monkeypatch):
    from jobbot import conversations as cv, outreach
    monkeypatch.setattr(outreach.ollama_client, "is_available", lambda: False)
    t = _thread()
    cv.record_sent(t.id, "CV attached.", sent_at=datetime.utcnow() - timedelta(days=18))
    cv.close(t.id)
    with pytest.raises(ValueError):
        cv.draft_nudge(t.id)


def test_nudge_holds_off_inside_the_quiet_window(monkeypatch):
    """Two days after sending is not a nudge, it is pestering.

    The playbook the applicant works from says give a warm contact two weeks before
    following up, so the default guard matches that rather than the 8-day
    application follow-up window.
    """
    from jobbot import conversations as cv, outreach
    monkeypatch.setattr(outreach.ollama_client, "is_available", lambda: False)
    t = _thread()
    cv.record_sent(t.id, "CV attached.", sent_at=datetime.utcnow() - timedelta(days=2))
    with pytest.raises(ValueError):
        cv.draft_nudge(t.id)
    assert cv.draft_nudge(t.id, force=True) is not None


def test_nudge_stands_down_once_they_have_replied(monkeypatch):
    """A reply already arrived, so the follow-up draft is the reply, not a nudge."""
    from jobbot import conversations as cv, outreach
    monkeypatch.setattr(outreach.ollama_client, "is_available", lambda: False)
    t = _thread()
    cv.record_sent(t.id, "CV attached.", sent_at=datetime.utcnow() - timedelta(days=18))
    cv.ingest(t.id, "Sounds good, let me ask around.")
    with pytest.raises(ValueError):
        cv.draft_nudge(t.id)


def test_a_recorded_sent_email_can_never_be_approved():
    """Approving it would queue an email that already went out for a second send.

    The row is only there as history, so the gate belongs at approve() where
    every path to send() has to pass through.
    """
    from jobbot import conversations as cv, outreach
    t = _thread()
    d = cv.record_sent(t.id, "Already went out on 7/24.")
    with pytest.raises(ValueError):
        outreach.approve(d.id)
    with session() as db:
        assert db.get(OutreachDraft, d.id).status == "sent"


def test_recorded_sent_email_stays_out_of_the_queue():
    from jobbot import conversations as cv, outreach
    t = _thread()
    cv.record_sent(t.id, "Already went out.")
    assert outreach.queue_list() == []
