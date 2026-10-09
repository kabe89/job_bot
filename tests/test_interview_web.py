"""Web practice routes. They must be THIN: the engine holds the logic, routes
serialize dicts. All fail-open -- coach trouble never 500s."""
import inspect
import json

import pytest

from jobbot import interview_coach as ic
from jobbot import web as web_mod
from jobbot.models import Job, init_db, session

BANK = {
    "behavioral": [{"q": "B1", "why": "", "answer_hook": "", "grounded_in": "jd"},
                   {"q": "B2", "why": "", "answer_hook": "", "grounded_in": "jd"}],
    "technical": [], "research": [], "contact": [],
}


@pytest.fixture(autouse=True)
def _mock_enrich(monkeypatch):
    monkeypatch.setattr(ic.research_enricher, "enrich_contact", lambda c, profile=None: c)


@pytest.fixture
def client():
    app = web_mod.create_app()
    app.config["TESTING"] = True
    with app.test_client() as c:
        yield c


def _job(db, **kw):
    d = dict(hash=kw.pop("hash", "w1"), source="greenhouse:test",
             title="Research Scientist", company="Acme Bio", url="https://x/1",
             description="comp bio", match_score=0.7, status="new",
             interview_question_bank=json.dumps(BANK),
             interview_prep="prep", company_intel="intel")
    d.update(kw)
    j = Job(**d); db.add(j); db.commit(); db.refresh(j); return j


class FakeAI:
    def score_answer(self, question, question_kind, answer, job_title, company, resume):
        return {"relevance": 4, "specificity": 4, "structure": 4, "fit": 4,
                "critique": "good", "follow_up": ""}

    def interview_session_summary(self, job_title, company, transcript):
        return "nice work"


def test_practice_start_returns_session_and_first_question(client, monkeypatch):
    monkeypatch.setattr(ic, "gc", FakeAI())
    init_db()
    with session() as db:
        jid = _job(db, hash="w_start").id
    r = client.post(f"/interview/{jid}/practice", data={"mode": "behavioral"})
    assert r.status_code == 200
    body = r.get_json()
    assert body["error"] == "" and body["session_id"] > 0
    assert body["question"]["question"] in ("B1", "B2")


def test_practice_start_reports_a_bad_mode_without_500ing(client):
    init_db()
    with session() as db:
        jid = _job(db, hash="w_badmode").id
    r = client.post(f"/interview/{jid}/practice", data={"mode": "nonsense"})
    assert r.status_code == 200
    assert r.get_json()["error"]


def test_answer_returns_score_and_the_next_question(client, monkeypatch):
    monkeypatch.setattr(ic, "gc", FakeAI())
    init_db()
    with session() as db:
        jid = _job(db, hash="w_answer").id
    start = client.post(f"/interview/{jid}/practice",
                        data={"mode": "behavioral"}).get_json()
    sid = start["session_id"]
    r = client.post(f"/practice/{sid}/answer",
                    json={"turn_id": start["question"]["turn_id"],
                          "answer": "I did X and got Y."})
    assert r.status_code == 200
    body = r.get_json()
    assert body["score"] == 4.0 and body["critique"] == "good"
    assert body["follow_up"] is None
    assert body["next"]["question"] not in (None, start["question"]["question"])


def test_answer_returns_next_null_when_the_bank_is_exhausted(client, monkeypatch):
    monkeypatch.setattr(ic, "gc", FakeAI())
    init_db()
    with session() as db:
        jid = _job(db, hash="w_exhaust").id
    start = client.post(f"/interview/{jid}/practice",
                        data={"mode": "behavioral"}).get_json()
    sid = start["session_id"]
    first = client.post(f"/practice/{sid}/answer",
                        json={"turn_id": start["question"]["turn_id"],
                              "answer": "a"}).get_json()
    last = client.post(f"/practice/{sid}/answer",
                       json={"turn_id": first["next"]["turn_id"],
                             "answer": "b"}).get_json()
    assert last["next"] is None


def test_end_returns_the_summary_and_overall_score(client, monkeypatch):
    monkeypatch.setattr(ic, "gc", FakeAI())
    init_db()
    with session() as db:
        jid = _job(db, hash="w_end").id
    start = client.post(f"/interview/{jid}/practice",
                        data={"mode": "behavioral"}).get_json()
    sid = start["session_id"]
    client.post(f"/practice/{sid}/answer",
                json={"turn_id": start["question"]["turn_id"], "answer": "a"})
    body = client.post(f"/practice/{sid}/end").get_json()
    assert body["overall_score"] == 4.0 and body["summary"] == "nice work"


def test_practice_page_renders_the_transcript(client, monkeypatch):
    monkeypatch.setattr(ic, "gc", FakeAI())
    init_db()
    with session() as db:
        jid = _job(db, hash="w_page").id
    start = client.post(f"/interview/{jid}/practice",
                        data={"mode": "behavioral"}).get_json()
    r = client.get(f"/practice/{start['session_id']}")
    assert r.status_code == 200
    assert b"Acme Bio" in r.data


def test_practice_page_404s_on_an_unknown_session(client):
    init_db()
    assert client.get("/practice/999999").status_code == 404


def test_answer_route_does_not_500_when_the_engine_raises(client, monkeypatch):
    monkeypatch.setattr(ic, "gc", FakeAI())
    init_db()
    with session() as db:
        jid = _job(db, hash="w_boom").id
    start = client.post(f"/interview/{jid}/practice",
                        data={"mode": "behavioral"}).get_json()

    def boom(db, turn_id, text):
        raise RuntimeError("engine exploded")

    monkeypatch.setattr(ic, "submit_answer", boom)
    r = client.post(f"/practice/{start['session_id']}/answer",
                    json={"turn_id": start["question"]["turn_id"], "answer": "a"})
    assert r.status_code == 200
    assert r.get_json()["error"]


@pytest.mark.parametrize("route_fn", ["practice_start", "practice_answer",
                                      "practice_end"])
def test_routes_are_thin(route_fn):
    """Guard the architecture: business logic lives in interview_coach, not here.
    A route body over ~25 statements means logic leaked in."""
    app = web_mod.create_app()
    fn = app.view_functions[route_fn]
    src = inspect.getsource(fn)
    assert len(src.splitlines()) <= 28, f"{route_fn} is too fat - move logic to the engine"
