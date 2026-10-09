from datetime import datetime
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
        db.add(models.Job(hash="live", source="s", title="Live Role",
                          company="C", url="http://x", status="new",
                          is_local=True, match_score=0.9))
        db.add(models.Job(hash="exp", source="s", title="Expired Role",
                          company="C", url="http://y", status="expired",
                          expired_at=datetime.utcnow(), expiry_reason="404",
                          is_local=True, match_score=0.9))
        db.commit()
    app = create_app()
    app.config.update(TESTING=True)
    return app.test_client()


def test_expired_hidden_by_default(tmp_path, monkeypatch):
    c = _client(tmp_path, monkeypatch)
    html = c.get("/?scope=all").get_data(as_text=True)
    assert "Live Role" in html
    assert "Expired Role" not in html


def test_expired_scope_shows_them(tmp_path, monkeypatch):
    c = _client(tmp_path, monkeypatch)
    html = c.get("/?scope=expired").get_data(as_text=True)
    assert "Expired Role" in html


def test_restore_route_reactivates(tmp_path, monkeypatch):
    c = _client(tmp_path, monkeypatch)
    with models.session() as db:
        jid = db.query(models.Job).filter_by(hash="exp").first().id
    c.post(f"/job/{jid}/restore", follow_redirects=True)
    with models.session() as db:
        assert db.get(models.Job, jid).status == "new"


def test_applications_page_renders_with_data(tmp_path, monkeypatch):
    # Regression: the template reads app.job.* after the session closes; the
    # route must eager-load .job or this 500s with DetachedInstanceError.
    c = _client(tmp_path, monkeypatch)
    with models.session() as db:
        jid = db.query(models.Job).filter_by(hash="live").first().id
        db.add(models.Application(job_id=jid, status="draft"))
        db.commit()
    r = c.get("/applications")
    assert r.status_code == 200
    assert "Live Role" in r.get_data(as_text=True)
