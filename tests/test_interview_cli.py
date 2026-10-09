"""CLI front-end for the interview coach. Drives the SAME engine as the web."""
from click.testing import CliRunner

from jobbot import cli as cli_mod
from jobbot import interview_coach as ic
from jobbot.models import Job, init_db, session


def _job(db, **kw):
    d = dict(hash=kw.pop("hash", "cli1"), source="greenhouse:test",
             title="Research Scientist", company="Acme Bio", url="https://x/1",
             description="comp bio", match_score=0.7, status="new")
    d.update(kw)
    j = Job(**d); db.add(j); db.commit(); db.refresh(j); return j


def test_interview_prep_subcommand_generates_a_pack(monkeypatch):
    init_db()
    with session() as db:
        jid = _job(db, hash="cli_prep").id
    seen = {}

    def fake_pack(job_id):
        seen["job_id"] = job_id
        return {"job_id": job_id, "prep": "p", "intel": "i",
                "bank": {"behavioral": [{"q": "a"}], "technical": [],
                         "research": [], "contact": []},
                "path": "output/x.md", "error": ""}

    monkeypatch.setattr(ic, "generate_pack", fake_pack)
    res = CliRunner().invoke(cli_mod.cli, ["interview", "prep", str(jid)])
    assert res.exit_code == 0, res.output
    assert seen["job_id"] == jid
    assert "output/x.md" in res.output
    assert res.output.isascii(), "Windows console: ASCII output only"


def test_interview_prep_reports_a_failure_without_crashing(monkeypatch):
    init_db()
    with session() as db:
        jid = _job(db, hash="cli_prep_err").id
    monkeypatch.setattr(ic, "generate_pack", lambda job_id: {
        "job_id": job_id, "prep": "", "intel": "", "bank": {}, "path": "",
        "error": "ollama down"})
    res = CliRunner().invoke(cli_mod.cli, ["interview", "prep", str(jid)])
    assert res.exit_code == 0
    assert "ollama down" in res.output


import json

from jobbot.models import InterviewSession


BANK = {"behavioral": [{"q": "B1", "why": "", "answer_hook": "", "grounded_in": "jd"},
                       {"q": "B2", "why": "", "answer_hook": "", "grounded_in": "jd"}],
        "technical": [], "research": [], "contact": []}


class FakeAI:
    def score_answer(self, question, question_kind, answer, job_title, company, resume):
        return {"relevance": 4, "specificity": 4, "structure": 4, "fit": 4,
                "critique": "concrete and clear", "follow_up": ""}

    def interview_session_summary(self, job_title, company, transcript):
        return "Strong on methods, thin on impact."


def test_practice_drives_a_full_loop_over_piped_answers(monkeypatch):
    monkeypatch.setattr(ic, "gc", FakeAI())
    init_db()
    with session() as db:
        jid = _job(db, hash="cli_practice", interview_question_bank=json.dumps(BANK)).id
    res = CliRunner().invoke(cli_mod.cli,
                             ["interview", "practice", str(jid), "--mode", "behavioral"],
                             input="my first answer\nmy second answer\n")
    assert res.exit_code == 0, res.output
    assert "B1" in res.output and "B2" in res.output
    assert "4.0/5" in res.output
    assert "concrete and clear" in res.output
    assert "Strong on methods" in res.output
    assert res.output.isascii(), "Windows console: ASCII output only"
    with session() as db:
        s = db.query(InterviewSession).filter_by(job_id=jid).one()
        assert s.ended_at is not None and s.overall_score == 4.0


def test_practice_quit_ends_the_session_early(monkeypatch):
    monkeypatch.setattr(ic, "gc", FakeAI())
    init_db()
    with session() as db:
        jid = _job(db, hash="cli_quit", interview_question_bank=json.dumps(BANK)).id
    res = CliRunner().invoke(cli_mod.cli,
                             ["interview", "practice", str(jid), "--mode", "behavioral"],
                             input="an answer\nquit\n")
    assert res.exit_code == 0
    with session() as db:
        assert db.query(InterviewSession).filter_by(job_id=jid).one().ended_at is not None


def test_practice_reports_a_start_error_without_a_traceback(monkeypatch):
    monkeypatch.setattr(ic, "gc", FakeAI())
    init_db()
    with session() as db:
        jid = _job(db, hash="cli_nobank", interview_question_bank="").id
    res = CliRunner().invoke(cli_mod.cli, ["interview", "practice", str(jid)])
    assert res.exit_code == 0
    assert "no questions" in res.output.lower()
    assert "Traceback" not in res.output


def test_practice_contact_mode_refuses_without_a_contact(monkeypatch):
    monkeypatch.setattr(ic, "gc", FakeAI())
    init_db()
    with session() as db:
        jid = _job(db, hash="cli_nocontact", company="Nobody Labs",
                   interview_question_bank=json.dumps(BANK)).id
    res = CliRunner().invoke(cli_mod.cli,
                             ["interview", "practice", str(jid), "--mode", "contact"])
    assert res.exit_code == 0
    assert "no contact" in res.output.lower()


from pathlib import Path

import pytest


@pytest.fixture(autouse=True)
def _restore_weight():
    from jobbot.config import settings
    before = settings.interview_practice_weight
    yield
    settings.interview_practice_weight = before


def test_config_prints_the_current_weight():
    res = CliRunner().invoke(cli_mod.cli, ["interview", "config"])
    assert res.exit_code == 0, res.output
    assert "interview_practice_weight" in res.output
    assert res.output.isascii()


def test_config_sets_the_weight_and_persists_it(monkeypatch, tmp_path):
    env = tmp_path / ".env"
    env.write_text("AI_PROVIDER=ollama\n", encoding="utf-8")
    monkeypatch.chdir(tmp_path)
    res = CliRunner().invoke(cli_mod.cli, ["interview", "config", "--weight", "0.12"])
    assert res.exit_code == 0, res.output
    assert "INTERVIEW_PRACTICE_WEIGHT=0.12" in env.read_text(encoding="utf-8")
    # live settings updated too - the running dashboard must agree immediately
    from jobbot.config import settings
    assert settings.interview_practice_weight == 0.12


def test_config_rejects_an_out_of_range_weight(monkeypatch, tmp_path):
    monkeypatch.chdir(tmp_path)
    res = CliRunner().invoke(cli_mod.cli, ["interview", "config", "--weight", "2.0"])
    assert res.exit_code != 0
    assert "between 0.0 and 0.5" in res.output
