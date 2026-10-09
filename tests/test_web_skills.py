from jobbot.web import create_app


def _client(tmp_path, monkeypatch):
    from jobbot import web
    monkeypatch.setattr(web.settings, "skills_path", str(tmp_path / "SK.md"))
    app = create_app()
    app.config.update(TESTING=True)
    return app.test_client()


def test_skills_get_renders(tmp_path, monkeypatch):
    c = _client(tmp_path, monkeypatch)
    r = c.get("/skills")
    assert r.status_code == 200


def test_skills_post_saves_and_parses(tmp_path, monkeypatch):
    c = _client(tmp_path, monkeypatch)
    body = ("====\nA. TEST\n====\n- Python: Proficient - NumPy pandas\n")
    r = c.post("/skills", data={"content": body}, follow_redirects=True)
    assert r.status_code == 200
    from jobbot import web
    assert "Python" in web.Path(web.settings.skills_path).read_text(encoding="utf-8")
