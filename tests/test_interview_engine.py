"""Practice-engine state machine. One engine, two front-ends -- these tests are
the contract both the web routes and the CLI loop depend on."""
import json

import pytest

from jobbot import interview_coach as ic
from jobbot.models import Contact, InterviewSession, InterviewTurn, Job, init_db, session

BANK = {
    "behavioral": [{"q": "B1", "why": "", "answer_hook": "", "grounded_in": "jd"},
                   {"q": "B2", "why": "", "answer_hook": "", "grounded_in": "jd"}],
    "technical": [{"q": "T1", "why": "", "answer_hook": "", "grounded_in": "jd"}],
    "research": [{"q": "R1", "why": "", "answer_hook": "",
                  "grounded_in": "company_intel"}],
    "contact": [{"q": "C1", "why": "", "answer_hook": "", "grounded_in": "contact:1"}],
}


def _job(db, bank=BANK, **kw):
    d = dict(hash=kw.pop("hash", "e1"), source="greenhouse:test",
             title="Research Scientist", company="Acme Bio", url="https://x/1",
             description="comp bio", match_score=0.7, status="new",
             interview_question_bank=json.dumps(bank) if bank is not None else "")
    d.update(kw)
    j = Job(**d); db.add(j); db.commit(); db.refresh(j); return j


def test_start_creates_an_open_session_and_the_first_turn():
    init_db()
    with session() as db:
        jid = _job(db, hash="e_start").id
        out = ic.start(db, jid, mode="behavioral")
        assert out["error"] == ""
        s = db.get(InterviewSession, out["session_id"])
        assert s.ended_at is None and s.mode == "behavioral" and s.job_id == jid
        q = out["question"]
        assert q["question"] in ("B1", "B2") and q["question_kind"] == "behavioral"
        assert q["idx"] == 0 and q["parent_idx"] is None
        turn = db.get(InterviewTurn, q["turn_id"])
        assert turn.session_id == s.id and turn.answer == ""


def test_mode_filters_the_bank():
    init_db()
    with session() as db:
        jid = _job(db, hash="e_mode").id
        out = ic.start(db, jid, mode="research")
        assert out["question"]["question"] == "R1"
        assert out["question"]["grounded_in"] == "company_intel"


def test_general_mode_mixes_every_non_contact_kind():
    init_db()
    with session() as db:
        jid = _job(db, hash="e_general").id
        out = ic.start(db, jid, mode="general")
        sid = out["session_id"]
        kinds = {out["question"]["question_kind"]}
        while (q := ic.next_question(db, sid)) is not None:
            kinds.add(q["question_kind"])
        assert kinds == {"behavioral", "technical", "research"}


def test_next_question_never_repeats_and_returns_none_when_exhausted():
    init_db()
    with session() as db:
        jid = _job(db, hash="e_exhaust").id
        out = ic.start(db, jid, mode="behavioral")
        sid = out["session_id"]
        asked = [out["question"]["question"]]
        while (q := ic.next_question(db, sid)) is not None:
            asked.append(q["question"])
        assert sorted(asked) == ["B1", "B2"]
        assert ic.next_question(db, sid) is None


def test_contact_mode_refuses_to_start_without_a_matched_contact():
    init_db()
    with session() as db:
        jid = _job(db, hash="e_nocontact", company="Nobody Labs").id
        out = ic.start(db, jid, mode="contact")
    assert out["session_id"] == 0
    assert "no contact" in out["error"].lower()


def test_contact_mode_starts_and_records_contact_id_when_one_matches(monkeypatch):
    monkeypatch.setattr(ic.research_enricher, "enrich_contact", lambda c, profile=None: c)
    init_db()
    # Unique company so the warmest match is unambiguously this contact even when
    # the session-scoped test DB carries contacts seeded by other test files.
    company = "Zylophon Unique Bioworks"
    with session() as db:
        c = Contact(name="Dr. Ada Reed", company=company, title="PI",
                    relationship="pi", source="manual", pinned=True)
        db.add(c); db.commit(); db.refresh(c)
        cid = c.id
        jid = _job(db, hash="e_contact", company=company).id
        out = ic.start(db, jid, mode="contact")
        assert out["error"] == ""
        assert out["question"]["question"] == "C1"
        assert db.get(InterviewSession, out["session_id"]).contact_id == cid


def test_start_rejects_an_unknown_mode():
    init_db()
    with session() as db:
        jid = _job(db, hash="e_badmode").id
        out = ic.start(db, jid, mode="interrogation")
    assert out["session_id"] == 0 and "mode" in out["error"].lower()


def test_start_on_an_empty_bank_reports_it_rather_than_crashing():
    init_db()
    with session() as db:
        jid = _job(db, hash="e_emptybank", bank=None).id
        out = ic.start(db, jid, mode="general")
    assert out["session_id"] == 0 and "no questions" in out["error"].lower()


class FakeScorer:
    """ai_client stand-in whose score/follow_up are set per test."""

    def __init__(self, axes=(4, 4, 4, 4), critique="solid", follow_up="", boom=False):
        self.axes = axes
        self.critique = critique
        self.follow_up = follow_up
        self.boom = boom
        self.calls = []

    def score_answer(self, question, question_kind, answer, job_title, company, resume):
        self.calls.append(question)
        if self.boom:
            raise RuntimeError("ollama down")
        r, s, st, f = self.axes
        return {"relevance": r, "specificity": s, "structure": st, "fit": f,
                "critique": self.critique, "follow_up": self.follow_up}


@pytest.mark.parametrize("axes,expected", [
    ((5, 5, 5, 5), 5.0),
    ((0, 0, 0, 0), 0.0),
    ((4, 4, 4, 4), 4.0),
    ((4, 3, 3, 3), 3.0),      # mean 3.25 -> 3.0
    ((4, 4, 3, 3), 3.5),      # mean 3.5  -> 3.5
    ((5, 4, 4, 4), 4.0),      # mean 4.25 -> 4.0
    ((5, 5, 4, 4), 4.5),      # mean 4.5  -> 4.5
])
def test_rubric_score_is_the_mean_rounded_to_a_half_point(axes, expected):
    r, s, st, f = axes
    assert ic.rubric_score({"relevance": r, "specificity": s,
                            "structure": st, "fit": f}) == expected


def test_submit_answer_fills_the_turn_and_persists_the_rubric(monkeypatch):
    monkeypatch.setattr(ic, "gc", FakeScorer(axes=(5, 4, 4, 3), critique="good"))
    init_db()
    with session() as db:
        jid = _job(db, hash="e_answer").id
        st = ic.start(db, jid, mode="behavioral")
        out = ic.submit_answer(db, st["question"]["turn_id"], "I did X, got Y.")
        assert out["error"] == ""
        assert out["score"] == 4.0       # mean 4.0
        assert out["critique"] == "good"
        assert out["follow_up"] is None
        t = db.get(InterviewTurn, st["question"]["turn_id"])
        assert t.answer == "I did X, got Y." and t.score == 4.0
        assert json.loads(t.rubric_json)["relevance"] == 5.0


def test_weak_answer_with_a_gap_spawns_one_follow_up(monkeypatch):
    monkeypatch.setattr(ic, "gc", FakeScorer(axes=(2, 2, 2, 2),
                                             follow_up="Which assay, exactly?"))
    init_db()
    with session() as db:
        jid = _job(db, hash="e_followup").id
        st = ic.start(db, jid, mode="behavioral")
        base_turn_id = st["question"]["turn_id"]
        base_idx = st["question"]["idx"]
        out = ic.submit_answer(db, base_turn_id, "we did some stuff")
        fu = out["follow_up"]
        assert fu is not None
        assert fu["question"] == "Which assay, exactly?"
        assert fu["parent_idx"] == base_idx
        assert fu["question_kind"] == "behavioral"
        assert fu["grounded_in"] == "follow_up"


def test_strong_answer_never_follows_up_even_with_a_suggestion(monkeypatch):
    monkeypatch.setattr(ic, "gc", FakeScorer(axes=(5, 5, 5, 5),
                                             follow_up="a question we should skip"))
    init_db()
    with session() as db:
        jid = _job(db, hash="e_nofollow").id
        st = ic.start(db, jid, mode="behavioral")
        out = ic.submit_answer(db, st["question"]["turn_id"], "detailed answer")
        assert out["score"] == 5.0 and out["follow_up"] is None


def test_weak_answer_with_no_gap_named_does_not_follow_up(monkeypatch):
    monkeypatch.setattr(ic, "gc", FakeScorer(axes=(2, 2, 2, 2), follow_up=""))
    init_db()
    with session() as db:
        jid = _job(db, hash="e_nogap").id
        st = ic.start(db, jid, mode="behavioral")
        out = ic.submit_answer(db, st["question"]["turn_id"], "weak")
        assert out["follow_up"] is None


def test_follow_up_cap_holds_at_one_per_base_question(monkeypatch):
    monkeypatch.setattr(ic, "gc", FakeScorer(axes=(1, 1, 1, 1), follow_up="again?"))
    init_db()
    with session() as db:
        jid = _job(db, hash="e_cap").id
        st = ic.start(db, jid, mode="behavioral")
        first = ic.submit_answer(db, st["question"]["turn_id"], "weak")
        # answering the follow-up itself weakly must NOT spawn another
        second = ic.submit_answer(db, first["follow_up"]["turn_id"], "still weak")
        assert second["follow_up"] is None
        turns = db.query(InterviewTurn).filter_by(session_id=st["session_id"]).all()
        assert sum(1 for t in turns if t.parent_idx is not None) == 1


def test_follow_up_cap_is_configurable(monkeypatch):
    monkeypatch.setattr(ic, "gc", FakeScorer(axes=(1, 1, 1, 1), follow_up="again?"))
    monkeypatch.setattr(ic.settings, "interview_followup_max", 0)
    init_db()
    with session() as db:
        jid = _job(db, hash="e_cap0").id
        st = ic.start(db, jid, mode="behavioral")
        out = ic.submit_answer(db, st["question"]["turn_id"], "weak")
        assert out["follow_up"] is None


def test_submit_answer_is_fail_open_when_the_llm_dies(monkeypatch):
    monkeypatch.setattr(ic, "gc", FakeScorer(boom=True))
    init_db()
    with session() as db:
        jid = _job(db, hash="e_scoreboom").id
        st = ic.start(db, jid, mode="behavioral")
        out = ic.submit_answer(db, st["question"]["turn_id"], "my answer")
        # practice must not dead-end: neutral score, answer still saved
        assert out["error"] == ""
        assert out["score"] is None
        assert out["follow_up"] is None
        assert "could not be scored" in out["critique"].lower()
        t = db.get(InterviewTurn, st["question"]["turn_id"])
        assert t.answer == "my answer" and t.score is None


def test_submit_answer_rejects_an_empty_answer(monkeypatch):
    monkeypatch.setattr(ic, "gc", FakeScorer())
    init_db()
    with session() as db:
        jid = _job(db, hash="e_empty").id
        st = ic.start(db, jid, mode="behavioral")
        out = ic.submit_answer(db, st["question"]["turn_id"], "   ")
    assert out["error"] and out["score"] is None


def test_submit_answer_on_an_unknown_turn_reports_an_error(monkeypatch):
    monkeypatch.setattr(ic, "gc", FakeScorer())
    init_db()
    with session() as db:
        out = ic.submit_answer(db, 999999, "hello")
    assert "not found" in out["error"].lower()


class FakeCoachAI(FakeScorer):
    def __init__(self, summary="You were strong on X.", **kw):
        super().__init__(**kw)
        self.summary = summary
        self.summary_calls = []

    def score_answer(self, question, question_kind, answer, job_title, company, resume):
        # `boom` here scopes to the summary only: these end-of-session tests need
        # scoring to succeed so there is an overall score to aggregate.
        self.calls.append(question)
        r, s, st, f = self.axes
        return {"relevance": r, "specificity": s, "structure": st, "fit": f,
                "critique": self.critique, "follow_up": self.follow_up}

    def interview_session_summary(self, job_title, company, transcript):
        self.summary_calls.append(transcript)
        if self.boom:
            raise RuntimeError("ollama down")
        return self.summary


def test_end_averages_scored_turns_and_stores_the_summary(monkeypatch):
    fake = FakeCoachAI(axes=(4, 4, 4, 4))
    monkeypatch.setattr(ic, "gc", fake)
    init_db()
    with session() as db:
        jid = _job(db, hash="e_end").id
        st = ic.start(db, jid, mode="behavioral")
        sid = st["session_id"]
        ic.submit_answer(db, st["question"]["turn_id"], "answer one")
        q2 = ic.next_question(db, sid)
        fake.axes = (2, 2, 2, 2)
        ic.submit_answer(db, q2["turn_id"], "answer two")
        out = ic.end(db, sid)
        assert out["error"] == ""
        assert out["overall_score"] == 3.0        # mean(4.0, 2.0)
        assert out["summary"] == "You were strong on X."
        assert out["turns"] == 2
        s = db.get(InterviewSession, sid)
        assert s.ended_at is not None and s.overall_score == 3.0
    # the summary prompt must see the real transcript, not an empty string
    assert "answer one" in fake.summary_calls[0]


def test_end_ignores_unanswered_turns_in_the_average(monkeypatch):
    fake = FakeCoachAI(axes=(5, 5, 5, 5))
    monkeypatch.setattr(ic, "gc", fake)
    init_db()
    with session() as db:
        jid = _job(db, hash="e_end_partial").id
        st = ic.start(db, jid, mode="behavioral")
        sid = st["session_id"]
        ic.submit_answer(db, st["question"]["turn_id"], "answered")
        ic.next_question(db, sid)                  # drawn but never answered
        out = ic.end(db, sid)
    assert out["overall_score"] == 5.0
    assert out["turns"] == 1


def test_end_on_a_session_with_no_answers_scores_none(monkeypatch):
    monkeypatch.setattr(ic, "gc", FakeCoachAI())
    init_db()
    with session() as db:
        jid = _job(db, hash="e_end_empty").id
        st = ic.start(db, jid, mode="behavioral")
        out = ic.end(db, st["session_id"])
    assert out["overall_score"] is None and out["turns"] == 0


def test_end_is_fail_open_when_the_summary_llm_dies(monkeypatch):
    monkeypatch.setattr(ic, "gc", FakeCoachAI(axes=(4, 4, 4, 4), boom=True))
    init_db()
    with session() as db:
        jid = _job(db, hash="e_end_boom").id
        st = ic.start(db, jid, mode="behavioral")
        sid = st["session_id"]
        ic.submit_answer(db, st["question"]["turn_id"], "answered")
        out = ic.end(db, sid)
        # session still closes and still scores; only the prose is missing
        assert out["error"] == ""
        assert out["overall_score"] == 4.0
        assert out["summary"] == ""
        assert db.get(InterviewSession, sid).ended_at is not None


def test_end_is_idempotent(monkeypatch):
    monkeypatch.setattr(ic, "gc", FakeCoachAI(axes=(4, 4, 4, 4)))
    init_db()
    with session() as db:
        jid = _job(db, hash="e_end_twice").id
        st = ic.start(db, jid, mode="behavioral")
        sid = st["session_id"]
        ic.submit_answer(db, st["question"]["turn_id"], "answered")
        first = ic.end(db, sid)
        ended_at = db.get(InterviewSession, sid).ended_at
        second = ic.end(db, sid)
        assert second["overall_score"] == first["overall_score"]
        assert db.get(InterviewSession, sid).ended_at == ended_at


def test_transcript_is_ordered_and_carries_the_rubric(monkeypatch):
    monkeypatch.setattr(ic, "gc", FakeCoachAI(axes=(5, 4, 4, 3)))
    init_db()
    with session() as db:
        jid = _job(db, hash="e_transcript").id
        st = ic.start(db, jid, mode="behavioral")
        ic.submit_answer(db, st["question"]["turn_id"], "my answer")
        rows = ic.transcript(db, st["session_id"])
    assert [r["idx"] for r in rows] == sorted(r["idx"] for r in rows)
    assert rows[0]["answer"] == "my answer"
    assert rows[0]["rubric"]["relevance"] == 5.0
    assert rows[0]["score"] == 4.0
