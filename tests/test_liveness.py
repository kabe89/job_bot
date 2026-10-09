from datetime import datetime
from jobbot import models
from jobbot import liveness


class _Resp:
    def __init__(self, status_code=200, text="", url="http://x"):
        self.status_code = status_code
        self.text = text
        self.url = url


def test_check_alive_404_is_dead():
    alive, reason = liveness.check_job_alive(
        "http://x/job", fetch=lambda url, **k: _Resp(404))
    assert alive is False and "404" in reason


def test_check_alive_dead_phrase_is_dead():
    body = "This position is no longer accepting applications."
    alive, reason = liveness.check_job_alive(
        "http://x/job", fetch=lambda url, **k: _Resp(200, body))
    assert alive is False and "no longer" in reason.lower()


def test_check_alive_ok():
    body = "Apply now. Responsibilities. Requirements."
    alive, reason = liveness.check_job_alive(
        "http://x/job", fetch=lambda url, **k: _Resp(200, body))
    assert alive is True


def test_sweep_expires_only_dead(tmp_path, monkeypatch):
    _use_temp_db(monkeypatch, tmp_path)  # helper defined below
    with liveness.session() as db:
        db.add(liveness.Job(hash="a", source="s", title="t", company="c",
                            url="http://dead", status="new"))
        db.add(liveness.Job(hash="b", source="s", title="t", company="c",
                            url="http://live", status="new"))
        db.commit()

    def fake_fetch(url, **k):
        return _Resp(404) if "dead" in url else _Resp(200, "Apply now requirements")

    stats = liveness.sweep_expired(limit=10, fetch=fake_fetch)
    assert stats["expired"] == 1 and stats["alive"] == 1
    with liveness.session() as db:
        dead = db.query(liveness.Job).filter_by(hash="a").first()
        live = db.query(liveness.Job).filter_by(hash="b").first()
        assert dead.status == "expired" and dead.expiry_reason
        assert live.status == "new"


def _use_temp_db(monkeypatch, tmp_path):
    from sqlalchemy import create_engine
    from sqlalchemy.orm import sessionmaker
    from jobbot import models
    monkeypatch.setattr(models.settings, "db_path", str(tmp_path / "l.db"))
    eng = create_engine(f"sqlite:///{models.settings.db_path}", future=True)
    monkeypatch.setattr(models, "_engine", eng)
    monkeypatch.setattr(models, "SessionLocal", sessionmaker(bind=eng, future=True))
    models.init_db()
    monkeypatch.setattr(liveness, "session", models.session)
    monkeypatch.setattr(liveness, "Job", models.Job)


def test_job_has_expiration_columns(tmp_path, monkeypatch):
    monkeypatch.setattr(models.settings, "db_path", str(tmp_path / "t.db"))
    # Rebuild engine against the temp DB.
    from sqlalchemy import create_engine
    from sqlalchemy.orm import sessionmaker
    eng = create_engine(f"sqlite:///{models.settings.db_path}", future=True)
    monkeypatch.setattr(models, "_engine", eng)
    monkeypatch.setattr(models, "SessionLocal",
                        sessionmaker(bind=eng, future=True))
    models.init_db()
    with models.session() as db:
        j = models.Job(hash="h1", source="s", title="t", company="c",
                       url="http://x", expired_at=datetime.utcnow(),
                       expiry_reason="404")
        db.add(j); db.commit()
        got = db.query(models.Job).first()
        assert got.expiry_reason == "404"


def test_liveness_progress_lifecycle():
    p = liveness._LivenessProgress()
    p.reset()
    assert p.snapshot()["state"] == "idle"

    p.start(total=10, skipped=2)
    snap = p.snapshot()
    assert snap["state"] == "running"
    assert snap["total"] == 10
    assert snap["skipped"] == 2
    assert snap["checked"] == 0

    p.update({"checked": 5, "expired": 2, "alive": 3, "errors": 0})
    snap = p.snapshot()
    assert snap["checked"] == 5
    assert snap["percent"] == 50
    assert snap["expired"] == 2

    p.finish({"checked": 10, "expired": 3, "alive": 7, "errors": 0})
    snap = p.snapshot()
    assert snap["state"] == "done"
    assert snap["percent"] == 100
    assert snap["expired"] == 3


def test_liveness_web_routes(tmp_path, monkeypatch):
    from jobbot.web import create_app
    _use_temp_db(monkeypatch, tmp_path)

    app = create_app()
    app.config["TESTING"] = True
    client = app.test_client()

    # Reset progress to idle
    liveness.progress.reset()

    # Check status endpoint
    res = client.get("/liveness/status")
    assert res.status_code == 200
    data = res.get_json()
    assert data["state"] == "idle"

    # Mock sweep_all to prevent long network operations
    monkeypatch.setattr(liveness, "sweep_all", lambda **k: {"checked": 1, "expired": 0, "alive": 1, "errors": 0})

    # Trigger sweep endpoint
    post_res = client.post("/liveness/sweep")
    assert post_res.status_code == 302

