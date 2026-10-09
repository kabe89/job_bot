from datetime import datetime

import pytest

from jobbot.config import settings
from jobbot.models import (Application, Contact, Job, OutreachDraft,
                           init_db, session)


@pytest.fixture(autouse=True)
def _clean_db():
    def _wipe():
        init_db()
        with session() as db:
            db.query(OutreachDraft).delete()
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


def test_config_defaults_present():
    assert settings.outreach_enabled is True
    assert settings.outreach_cold_max == 10
    assert settings.outreach_max_per_scan == 25


def test_outreach_draft_roundtrips():
    init_db()
    with session() as db:
        d = OutreachDraft(kind="warm_intro", status="pending",
                          recipient_email="a@b.com", subject="Hi",
                          body="Body", rationale="warm", priority=0.5,
                          dedupe_key="warm_intro:1:1")
        db.add(d); db.commit(); db.refresh(d)
        got = db.query(OutreachDraft).filter_by(dedupe_key="warm_intro:1:1").one()
        assert got.kind == "warm_intro" and got.status == "pending"
        assert got.sent_at is None and got.inbound == ""


def test_draft_reply_queues_reply_from_pasted_text(monkeypatch):
    from jobbot import outreach
    # Force the template path (Ollama unavailable) for determinism.
    monkeypatch.setattr(outreach.ollama_client, "is_available", lambda: False)
    d = outreach.draft_reply(
        "Hi Jane, thanks for applying. Are you available for a call Tuesday?",
        sender="Jane Recruiter <jane@acme.com>")
    assert d.kind == "reply"
    assert d.status == "pending"
    assert "call Tuesday" in d.inbound
    assert d.body.strip()          # some reply text was produced
    assert d.recipient_email == "jane@acme.com"


def test_queue_dedupes_live_drafts():
    from jobbot import outreach
    from jobbot.models import session, OutreachDraft
    a = outreach._queue(kind="warm_intro", subject="s", body="b",
                        dedupe_key="warm_intro:5:5")
    b = outreach._queue(kind="warm_intro", subject="s2", body="b2",
                        dedupe_key="warm_intro:5:5")
    assert a is not None and b is None
    with session() as db:
        assert db.query(OutreachDraft).filter_by(dedupe_key="warm_intro:5:5").count() == 1


def test_draft_reply_falls_back_when_generate_raises(monkeypatch):
    from jobbot import outreach
    monkeypatch.setattr(outreach.ollama_client, "is_available", lambda: True)
    def _boom(*a, **k):
        raise RuntimeError("ollama exploded")
    monkeypatch.setattr(outreach.ollama_client, "_generate", _boom)
    d = outreach.draft_reply("Hi, are you free Thursday?", sender="pat@co.com")
    assert d.kind == "reply"
    assert d.status == "pending"
    assert d.body.strip()            # template body still produced
    assert d.recipient_email == "pat@co.com"


def test_scan_warm_intro_queues_draft():
    from jobbot import outreach
    from jobbot.models import session, OutreachDraft
    with session() as db:
        _job(db, hash="wi", company="Acme Corporation", match_score=0.5)
        db.add(Contact(name="Ada Warm", company="Acme", email="ada@acme.com",
                       pinned=True, source="manual"))
        db.commit()
    res = outreach.scan()
    assert res["warm_intro"] >= 1
    with session() as db:
        d = db.query(OutreachDraft).filter_by(kind="warm_intro").first()
        assert d is not None and d.recipient_email == "ada@acme.com"


def test_scan_followup_queues_draft(monkeypatch):
    from jobbot import outreach
    from jobbot.models import session, OutreachDraft
    # Mock followups_due + draft_followup so no real timing/contacts needed.
    monkeypatch.setattr(outreach.followup, "followups_due",
                        lambda: [{"application_id": 7, "job_id": 1,
                                  "contact_email": "hm@acme.com",
                                  "contact_name": "HM", "days_since_touch": 9}])
    monkeypatch.setattr(outreach.followup, "draft_followup",
                        lambda app_id: {"to": "hm@acme.com",
                                        "subject": "Following up",
                                        "body": "Any update?"})
    res = outreach.scan()
    assert res["follow_up"] >= 1
    with session() as db:
        d = db.query(OutreachDraft).filter_by(kind="follow_up").first()
        assert d is not None and d.application_id == 7


def test_scan_cold_queues_when_search_configured(monkeypatch):
    from jobbot import outreach
    from jobbot.models import session, OutreachDraft
    with session() as db:
        _job(db, hash="cold", company="Nowhere Labs", match_score=0.8)
        db.commit()
    monkeypatch.setattr(outreach._contacts_mod, "search_configured", lambda: True)
    monkeypatch.setattr(outreach._contacts_mod, "find_contacts",
                        lambda job_id, **kw: [{"name": "Rex Hiring",
                                               "email": "rex@nowhere.com",
                                               "title": "Hiring Manager"}])
    monkeypatch.setattr(outreach._contacts_mod, "networking_email",
                        lambda job_id, contact, **kw: {"subject": "Hello",
                                                       "body": "Interested in your team."})
    monkeypatch.setattr(outreach._people_finder, "find_shared_thread_contacts",
                        lambda job_id: [])
    res = outreach.scan()
    assert res["cold"] >= 1
    with session() as db:
        d = db.query(OutreachDraft).filter_by(kind="cold").first()
        assert d is not None and d.recipient_email == "rex@nowhere.com"


def test_scan_cold_noop_without_search(monkeypatch):
    from jobbot import outreach
    from jobbot.models import session, OutreachDraft
    with session() as db:
        _job(db, hash="cold2", company="Nowhere Labs", match_score=0.8)
        db.commit()
    monkeypatch.setattr(outreach._contacts_mod, "search_configured", lambda: False)
    res = outreach.scan()
    assert res["cold"] == 0
    with session() as db:
        assert db.query(OutreachDraft).filter_by(kind="cold").count() == 0


def test_scan_is_idempotent():
    from jobbot import outreach
    from jobbot.models import session, OutreachDraft
    with session() as db:
        _job(db, hash="wi2", company="Acme Corporation", match_score=0.5)
        db.add(Contact(name="Ada", company="Acme", email="ada@acme.com",
                       source="manual"))
        db.commit()
    outreach.scan()
    with session() as db:
        first = db.query(OutreachDraft).count()
    outreach.scan()
    with session() as db:
        assert db.query(OutreachDraft).count() == first


def test_send_refuses_unapproved():
    from jobbot import outreach
    d = outreach._queue(kind="cold", subject="s", body="b",
                        recipient_email="x@y.com", dedupe_key="cold:99")
    with pytest.raises(ValueError):
        outreach.send(d.id)


def test_send_refuses_blank_recipient():
    from jobbot import outreach
    d = outreach._queue(kind="warm_intro", subject="s", body="b",
                        recipient_email="", dedupe_key="warm_intro:1:2")
    outreach.approve(d.id)
    with pytest.raises(ValueError):
        outreach.send(d.id)


def test_approve_then_send_marks_sent(monkeypatch):
    from jobbot import outreach
    from jobbot.models import session, OutreachDraft
    sent = {}
    monkeypatch.setattr(outreach.emailer, "send",
                        lambda **kw: sent.update(kw))
    d = outreach._queue(kind="cold", subject="Hi", body="Body\n\nLine2",
                        recipient_email="x@y.com", dedupe_key="cold:1")
    outreach.approve(d.id)
    out = outreach.send(d.id)
    assert sent["to"] == "x@y.com" and sent["subject"] == "Hi"
    with session() as db:
        got = db.get(OutreachDraft, d.id)
        assert got.status == "sent" and got.sent_at is not None


def test_dismiss_and_edit():
    from jobbot import outreach
    from jobbot.models import session, OutreachDraft
    d = outreach._queue(kind="cold", subject="s", body="b",
                        recipient_email="x@y.com", dedupe_key="cold:2")
    outreach.edit(d.id, "New subject", "New body")
    outreach.dismiss(d.id)
    with session() as db:
        got = db.get(OutreachDraft, d.id)
        assert got.subject == "New subject" and got.status == "dismissed"


def test_outreach_page_renders_and_roundtrips(monkeypatch):
    from jobbot.web import create_app
    from jobbot import outreach
    from jobbot.models import session, OutreachDraft
    cl = create_app().test_client()
    # Empty queue renders.
    assert cl.get("/outreach").status_code == 200
    # Reply route queues a draft (Ollama off for determinism).
    monkeypatch.setattr(outreach.ollama_client, "is_available", lambda: False)
    r = cl.post("/outreach/reply", data={"inbound": "Can you meet Tuesday?",
                                         "sender": "jane@acme.com"},
                follow_redirects=True)
    assert r.status_code == 200
    with session() as db:
        d = db.query(OutreachDraft).filter_by(kind="reply").first()
        assert d is not None
        did = d.id
    # Approve then dismiss round-trip.
    assert cl.post(f"/outreach/{did}/approve", follow_redirects=True).status_code == 200
    assert cl.post(f"/outreach/{did}/dismiss", follow_redirects=True).status_code == 200
    with session() as db:
        assert db.get(OutreachDraft, did).status == "dismissed"


def test_second_followup_not_blocked_by_sent_first(monkeypatch):
    from jobbot import outreach
    from jobbot.models import session, OutreachDraft
    # First follow-up (count 0) already sent for app 42.
    d0 = outreach._queue(kind="follow_up", subject="FU1", body="b",
                         recipient_email="hm@acme.com", application_id=42,
                         dedupe_key="follow_up:42:0")
    outreach.approve(d0.id)
    import jobbot.emailer as _em
    monkeypatch.setattr(_em, "send", lambda **kw: None)
    outreach.send(d0.id)   # d0 now status=sent, key follow_up:42:0 stays live
    # Now app 42 is due for follow-up #2 (count 1).
    monkeypatch.setattr(outreach.followup, "followups_due",
                        lambda: [{"application_id": 42, "job_id": 1,
                                  "contact_email": "hm@acme.com", "contact_name": "HM",
                                  "days_since_touch": 20, "followup_count": 1}])
    monkeypatch.setattr(outreach.followup, "draft_followup",
                        lambda app_id: {"to": "hm@acme.com", "subject": "FU2", "body": "second"})
    res = outreach.scan()
    assert res["follow_up"] >= 1
    with session() as db:
        keys = {d.dedupe_key for d in db.query(OutreachDraft).filter_by(kind="follow_up")}
        assert "follow_up:42:1" in keys   # the second follow-up WAS queued


def test_cold_scan_prefers_thread_contact(monkeypatch):
    from jobbot import outreach as ox
    from jobbot.models import Job, init_db, session

    init_db()
    with session() as db:
        db.query(Job).delete(); db.commit()
        db.add(Job(hash="cj", source="s", title="Scientist", company="Acme",
                   url="u", description="d", match_score=0.9, status="new"))
        db.commit()

    monkeypatch.setattr(ox.settings, "outreach_enabled", True, raising=False)
    monkeypatch.setattr(ox.settings, "outreach_cold_max", 5, raising=False)
    monkeypatch.setattr(ox.settings, "outreach_max_per_scan", 10, raising=False)
    monkeypatch.setattr(ox._contacts_mod, "search_configured", lambda: True)
    monkeypatch.setattr(ox.referrals, "build_index", lambda: [])
    # No warm contacts; a thread contact WITH an email exists.
    monkeypatch.setattr(ox._people_finder, "find_shared_thread_contacts",
                        lambda job_id: [{"name": "Taylor NA", "email": "b@x.org",
                                         "role": "PI", "thread": "coauthor"}])
    monkeypatch.setattr(ox._contacts_mod, "networking_email",
                        lambda job_id, c: {"subject": "s", "body": "b"})

    added = {"warm_intro": 0, "follow_up": 0, "cold": 0, "reply": 0, "_total": 0}
    ox._scan_cold(added)
    assert added["cold"] == 1
    d = ox.queue_list(kind="cold")[0]
    assert d.recipient_email == "b@x.org"
    assert "coauthor" in d.rationale.lower()


def test_cli_outreach_scan_and_list():
    from click.testing import CliRunner
    from jobbot.cli import cli
    from jobbot.models import session, OutreachDraft
    with session() as db:
        _job(db, hash="cli", company="Acme Corporation", match_score=0.5)
        db.add(Contact(name="Ada", company="Acme", email="ada@acme.com",
                       source="manual"))
        db.commit()
    r = CliRunner().invoke(cli, ["outreach", "scan"])
    assert r.exit_code == 0
    r2 = CliRunner().invoke(cli, ["outreach", "list"])
    assert r2.exit_code == 0
    with session() as db:
        assert db.query(OutreachDraft).count() >= 1
