from jobbot.web import create_app
from jobbot import web


def _client(monkeypatch, fake):
    monkeypatch.setattr(web, "web_apply", fake)
    app = create_app()
    app.config.update(TESTING=True)
    return app.test_client()


class _FakeWebApply:
    def __init__(self):
        self.started = None
        self.decided = None

    def start_web_apply(self, job_id):
        self.started = job_id
        return 7

    def get_state(self, token):
        return {"token": token, "phase": "awaiting_review", "job_id": 3,
                "run_id": None, "results": [], "screenshot": "", "error": "",
                "confirmation": ""}

    def submit_decision(self, token, choice):
        self.decided = (token, choice)
        return True


def test_start_redirects_to_view(monkeypatch):
    fake = _FakeWebApply()
    c = _client(monkeypatch, fake)
    r = c.post("/apply-run/3")
    assert r.status_code in (301, 302)
    assert "/apply-run/view/7" in r.headers["Location"]
    assert fake.started == 3


def test_status_returns_json(monkeypatch):
    fake = _FakeWebApply()
    c = _client(monkeypatch, fake)
    j = c.get("/apply-run/status/7").get_json()
    assert j["phase"] == "awaiting_review"


def test_decision_calls_submit(monkeypatch):
    fake = _FakeWebApply()
    c = _client(monkeypatch, fake)
    c.post("/apply-run/decision/7", data={"choice": "submit"})
    assert fake.decided == (7, "submit")
