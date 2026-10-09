from jobbot import web as web_mod


def _client(monkeypatch):
    rows = [
        {"group": "Ollama — local", "provider": "ollama", "model": "qwen3.5:latest",
         "available": True, "reason": "", "current": True},
        {"group": "Gemini", "provider": "gemini", "model": "gemini-2.5-flash",
         "available": False, "reason": "set GEMINI_API_KEY", "current": False},
    ]
    monkeypatch.setattr(web_mod, "_md_to_html", lambda s: s)
    import jobbot.model_registry as mr
    monkeypatch.setattr(mr, "list_selectable", lambda: rows)
    monkeypatch.setattr(mr, "current_selection", lambda: ("ollama", "qwen3.5:latest"))
    app = web_mod.create_app()
    app.config.update(TESTING=True)
    return app.test_client(), mr


def test_models_page_renders(monkeypatch):
    client, _ = _client(monkeypatch)
    resp = client.get("/models")
    assert resp.status_code == 200
    assert b"qwen3.5:latest" in resp.data


def test_models_set_posts_to_registry(monkeypatch):
    client, mr = _client(monkeypatch)
    seen = {}
    monkeypatch.setattr(mr, "set_selection", lambda p, m: seen.update(p=p, m=m))
    resp = client.post("/models/set", data={"selection": "gemini::gemini-2.5-flash"},
                       follow_redirects=False)
    assert resp.status_code in (302, 303)
    assert seen == {"p": "gemini", "m": "gemini-2.5-flash"}
