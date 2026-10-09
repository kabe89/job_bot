"""Interview coach: prep pack, question bank, practice engine, readiness."""
from jobbot.config import settings


def test_interview_settings_have_documented_defaults():
    assert settings.interview_coach_enabled is True
    assert settings.interview_followup_max == 1
    assert settings.interview_practice_weight == 0.05


from jobbot.models import InterviewSession, InterviewTurn, Job, init_db, session


def _job(db, **kw):
    d = dict(hash=kw.pop("hash", "ic1"), source="greenhouse:test",
             title="Research Scientist", company="Acme Bio",
             url="https://x/1", description="comp bio, protein design",
             match_score=0.72, status="new")
    d.update(kw)
    j = Job(**d); db.add(j); db.commit(); db.refresh(j); return j


def test_job_has_bank_and_pending_columns_defaulting_empty():
    init_db()
    with session() as db:
        j = _job(db, hash="ic_cols")
        assert j.interview_question_bank == ""
        assert j.interview_prep_pending is False


def test_interview_session_and_turn_tables_round_trip():
    init_db()
    with session() as db:
        j = _job(db, hash="ic_tables")
        s = InterviewSession(job_id=j.id, mode="research")
        db.add(s); db.commit(); db.refresh(s)
        assert s.ended_at is None and s.overall_score is None and s.summary == ""
        t = InterviewTurn(session_id=s.id, idx=0, question="Why us?",
                          question_kind="research", grounded_in="company_intel")
        db.add(t); db.commit(); db.refresh(t)
        assert t.parent_idx is None and t.answer == "" and t.score is None
        assert t.rubric_json == "" and t.critique == ""


import json

from jobbot import interview_coach as ic


class FakeAI:
    """Deterministic stand-in for ai_client. Records calls; never touches Ollama."""

    def __init__(self, bank=None, boom=False):
        self.calls = []
        self.boom = boom
        self._bank = bank if bank is not None else {
            "behavioral": [{"q": "Tell me about a hard project.", "why": "w",
                            "answer_hook": "h", "grounded_in": ""}],
            "technical": [{"q": "How do you validate a docking pose?", "why": "w",
                           "answer_hook": "h", "grounded_in": ""}],
            "research": [{"q": "Our AKT program stalled at selectivity - why?",
                          "why": "w", "answer_hook": "h",
                          "grounded_in": "company_intel"}],
            "contact": [],
        }

    def interview_prep(self, *a, **k):
        self.calls.append("interview_prep")
        if self.boom:
            raise RuntimeError("ollama down")
        return "## Role decoded\nprep text"

    def company_intel(self, *a, **k):
        self.calls.append("company_intel")
        if self.boom:
            raise RuntimeError("ollama down")
        return "## What they do\nintel text"

    def interview_question_bank(self, *a, **k):
        self.calls.append("interview_question_bank")
        if self.boom:
            raise RuntimeError("ollama down")
        return self._bank


def test_generate_pack_stores_a_valid_four_key_bank(monkeypatch, tmp_path):
    monkeypatch.setattr(ic, "gc", FakeAI())
    init_db()
    with session() as db:
        jid = _job(db, hash="ic_pack").id
    out = ic.generate_pack(jid)
    assert out["error"] == ""
    with session() as db:
        j = db.get(Job, jid)
        bank = json.loads(j.interview_question_bank)
        assert set(bank) == {"behavioral", "technical", "research", "contact"}
        assert bank["research"][0]["grounded_in"] == "company_intel"
        assert j.interview_prep and j.company_intel
        assert j.interview_prep_generated_at is not None


def test_generate_pack_clears_the_pending_flag(monkeypatch):
    monkeypatch.setattr(ic, "gc", FakeAI())
    init_db()
    with session() as db:
        jid = _job(db, hash="ic_pending", interview_prep_pending=True).id
    ic.generate_pack(jid)
    with session() as db:
        assert db.get(Job, jid).interview_prep_pending is False


def test_generate_pack_is_fail_open_and_keeps_existing_prep(monkeypatch):
    monkeypatch.setattr(ic, "gc", FakeAI(boom=True))
    init_db()
    with session() as db:
        jid = _job(db, hash="ic_failopen",
                   interview_prep="OLD PREP", interview_prep_pending=True).id
    out = ic.generate_pack(jid)          # must NOT raise
    assert out["error"]
    with session() as db:
        j = db.get(Job, jid)
        assert j.interview_prep == "OLD PREP"      # preserved, not blanked
        assert j.interview_prep_pending is False   # flag still cleared - no retry storm
        assert ic.load_bank(j) == {"behavioral": [], "technical": [],
                                   "research": [], "contact": []}


def test_load_bank_fail_open_on_corrupt_json():
    init_db()
    with session() as db:
        j = _job(db, hash="ic_corrupt", interview_question_bank="{not json")
        assert ic.load_bank(j) == {"behavioral": [], "technical": [],
                                   "research": [], "contact": []}


def test_generate_pack_noop_when_coach_disabled(monkeypatch):
    fake = FakeAI()
    monkeypatch.setattr(ic, "gc", fake)
    monkeypatch.setattr(ic.settings, "interview_coach_enabled", False)
    init_db()
    with session() as db:
        jid = _job(db, hash="ic_disabled", interview_prep_pending=True).id
    out = ic.generate_pack(jid)
    assert out["error"] == "interview coach disabled"
    assert fake.calls == []
    with session() as db:
        assert db.get(Job, jid).interview_prep_pending is False


from jobbot.models import Contact


def _contact(db, **kw):
    d = dict(name="Dr. Ada Reed", company="Acme Bio", title="Principal Investigator",
             relationship="pi", source="manual", pinned=True)
    d.update(kw)
    c = Contact(**d); db.add(c); db.commit(); db.refresh(c); return c


def test_matched_contact_returns_none_without_a_contact():
    init_db()
    with session() as db:
        j = _job(db, hash="ic_nocontact", company="Nobody Labs")
        assert ic.matched_contact(j) is None


def test_matched_contact_carries_research_into_the_payload(monkeypatch):
    monkeypatch.setattr(ic.research_enricher, "enrich_contact",
                        lambda c, profile=None: {**c, "research_package": {
                            "hook": "allosteric AKT inhibitors",
                            "key_papers": ["Reed 2024, Nature"],
                            "skill_connection": "MD + GNN scoring",
                            "health_impact": "oncology"}})
    init_db()
    with session() as db:
        _contact(db, name="Dr. Ada Reed", company="Acme Bio")
        j = _job(db, hash="ic_contact", company="Acme Bio", match_score=0.72)
        got = ic.matched_contact(j)
    assert got["name"] == "Dr. Ada Reed"
    assert got["title"] == "Principal Investigator"
    assert "allosteric AKT inhibitors" in got["research"]
    assert "Reed 2024, Nature" in got["research"]


def test_matched_contact_is_fail_open_when_enrichment_dies(monkeypatch):
    def boom(c, profile=None):
        raise RuntimeError("search down")
    monkeypatch.setattr(ic.research_enricher, "enrich_contact", boom)
    init_db()
    with session() as db:
        _contact(db, name="Dr. Ada Reed", company="Acme Bio")
        j = _job(db, hash="ic_contact_boom", company="Acme Bio", match_score=0.72)
        got = ic.matched_contact(j)
    assert got is not None and got["name"] == "Dr. Ada Reed"
    assert got["research"] == ""          # degraded, still usable


def test_generate_pack_populates_contact_questions_when_a_contact_matches(monkeypatch):
    bank = {"behavioral": [], "technical": [], "research": [],
            "contact": [{"q": "How would you rescore my AKT poses?", "why": "w",
                         "answer_hook": "h", "grounded_in": ""}]}
    monkeypatch.setattr(ic, "gc", FakeAI(bank=bank))
    monkeypatch.setattr(ic.research_enricher, "enrich_contact",
                        lambda c, profile=None: {**c, "research_package": {
                            "hook": "allosteric AKT inhibitors", "key_papers": [],
                            "skill_connection": "", "health_impact": ""}})
    init_db()
    with session() as db:
        _contact(db, name="Dr. Ada Reed", company="Acme Bio")
        jid = _job(db, hash="ic_cq", company="Acme Bio", match_score=0.72).id
    out = ic.generate_pack(jid)
    assert out["bank"]["contact"], "contact questions should be generated"
    # grounded_in provenance is stamped by generate_pack, not trusted from the LLM
    assert out["bank"]["contact"][0]["grounded_in"].startswith("contact:")


def test_generate_pack_leaves_contact_section_empty_without_a_contact(monkeypatch):
    bank = {"behavioral": [], "technical": [], "research": [],
            "contact": [{"q": "leaked contact question", "why": "", "answer_hook": ""}]}
    monkeypatch.setattr(ic, "gc", FakeAI(bank=bank))
    init_db()
    with session() as db:
        jid = _job(db, hash="ic_nocq", company="Nobody Labs").id
    out = ic.generate_pack(jid)
    assert out["bank"]["contact"] == [], "no contact -> no contact questions"


from jobbot import outcomes
from jobbot.models import Application


def _app(db, job_id, status="applied"):
    a = Application(job_id=job_id, status=status)
    db.add(a); db.commit(); db.refresh(a); return a


def test_record_outcome_interview_sets_pending_and_calls_no_llm(monkeypatch):
    fake = FakeAI()
    monkeypatch.setattr(ic, "gc", fake)
    init_db()
    with session() as db:
        jid = _job(db, hash="ic_trigger").id
        aid = _app(db, jid).id
    outcomes.record_outcome(aid, "interview")
    with session() as db:
        assert db.get(Job, jid).interview_prep_pending is True
    assert fake.calls == [], "record_outcome must stay pure - no LLM inside it"


def test_record_outcome_non_interview_stage_leaves_pending_false():
    init_db()
    with session() as db:
        jid = _job(db, hash="ic_trigger_screen").id
        aid = _app(db, jid, status="applied").id
    outcomes.record_outcome(aid, "screen")
    with session() as db:
        assert db.get(Job, jid).interview_prep_pending is False


def test_record_outcome_does_not_re_flag_a_job_that_already_has_a_bank():
    init_db()
    with session() as db:
        jid = _job(db, hash="ic_trigger_hasbank",
                   interview_question_bank='{"behavioral": [{"q": "x"}]}').id
        aid = _app(db, jid).id
    outcomes.record_outcome(aid, "interview")
    with session() as db:
        assert db.get(Job, jid).interview_prep_pending is False


def test_generate_pending_packs_flagged_jobs_only(monkeypatch):
    monkeypatch.setattr(ic, "gc", FakeAI())
    monkeypatch.setattr(ic.research_enricher, "enrich_contact", lambda c, profile=None: c)
    init_db()
    with session() as db:
        flagged = _job(db, hash="ic_pend_yes", interview_prep_pending=True).id
        quiet = _job(db, hash="ic_pend_no", interview_prep_pending=False).id
    done = ic.generate_pending(limit=50)
    # Isolation-robust: the session-scoped test DB may carry pending jobs from
    # earlier tests, so assert the behaviour (flagged packed, unflagged skipped)
    # rather than exact-equality on the whole batch.
    assert flagged in done
    assert quiet not in done
    with session() as db:
        assert db.get(Job, flagged).interview_question_bank
        assert db.get(Job, quiet).interview_question_bank == ""


def test_generate_pending_respects_the_limit(monkeypatch):
    monkeypatch.setattr(ic, "gc", FakeAI())
    monkeypatch.setattr(ic.research_enricher, "enrich_contact", lambda c, profile=None: c)
    init_db()
    with session() as db:
        for n in range(3):
            _job(db, hash=f"ic_pend_lim{n}", interview_prep_pending=True)
    assert len(ic.generate_pending(limit=2)) == 2
