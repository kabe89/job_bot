"""Dismissed jobs must not come back in the job list.

`status='skipped'` is what the dashboard's skip button and `jobbot clean
freelance` both write. The listing query hid `expired` but not `skipped`, so 333
already-dismissed rows (317 of them gig postings from company 'agency') were
still being rendered. Dismissing a job has to make it go away, or the pool reads
as full of junk the user already said no to.
"""
from jobbot.web import create_app
from jobbot import models


def _client(tmp_path, monkeypatch):
    monkeypatch.setattr(models.settings, "db_path", str(tmp_path / "w.db"))
    from sqlalchemy import create_engine
    from sqlalchemy.orm import sessionmaker
    eng = create_engine(f"sqlite:///{models.settings.db_path}", future=True)
    monkeypatch.setattr(models, "_engine", eng)
    monkeypatch.setattr(models, "SessionLocal", sessionmaker(bind=eng, future=True))
    models.init_db()
    with models.session() as db:
        db.add(models.Job(hash="live", source="s", title="Computational Biologist",
                          company="C", url="http://x", status="new",
                          is_local=True, match_score=0.9))
        db.add(models.Job(hash="gig", source="s",
                          title="Chemistry Specialist - Freelance AI Trainer Project",
                          company="agency", url="http://y", status="skipped",
                          tags="gig-hidden", is_local=True, match_score=0.9))
        db.commit()
    app = create_app()
    app.config.update(TESTING=True)
    return app.test_client()


def test_skipped_hidden_by_default(tmp_path, monkeypatch):
    c = _client(tmp_path, monkeypatch)
    html = c.get("/?scope=all").get_data(as_text=True)
    assert "Computational Biologist" in html
    assert "Freelance AI Trainer" not in html


def test_skipped_still_reachable_when_asked_for(tmp_path, monkeypatch):
    """Hidden, not deleted -- an explicit status filter must still show them so
    the user can review and restore."""
    c = _client(tmp_path, monkeypatch)
    html = c.get("/?scope=all&status=skipped").get_data(as_text=True)
    assert "Freelance AI Trainer" in html


def test_expired_scope_is_unaffected(tmp_path, monkeypatch):
    """The skipped filter must not interfere with the expired tab."""
    from datetime import datetime
    c = _client(tmp_path, monkeypatch)
    with models.session() as db:
        db.add(models.Job(hash="exp", source="s", title="Gone Role", company="C",
                          url="http://z", status="expired",
                          expired_at=datetime.utcnow(), expiry_reason="404",
                          is_local=True, match_score=0.9))
        db.commit()
    html = c.get("/?scope=expired").get_data(as_text=True)
    assert "Gone Role" in html
    assert "Freelance AI Trainer" not in html
