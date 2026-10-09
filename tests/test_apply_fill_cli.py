"""`jobbot apply-fill` — the terminal side of asking the user."""
from __future__ import annotations

from uuid import uuid4

import pytest
from click.testing import CliRunner

from jobbot.apply_questions import FormQuestion, questions_to_json
from jobbot.cli import cli
from jobbot.models import Application, Job, init_db, session


@pytest.fixture()
def kit_job():
    """A job with a stored kit containing one blank and one held AI guess."""
    init_db()
    with session() as db:
        # Unique per test: the suite shares one session-scoped DB, so a fixed
        # URL collides on jobs.hash across tests in this file.
        url = f"https://example.test/j/{uuid4()}"
        job = Job(hash=Job.make_hash("test", url, "Scientist"),
                  title="Scientist", company="Zzyzx Labs",
                  url=url, source="test")
        db.add(job)
        db.flush()
        qs = [
            FormQuestion(text="Are you willing to relocate?", qtype="select",
                         options=["Yes", "No"], required=True, needs_user=True,
                         guess="Springfield, IL"),
            FormQuestion(text="How did you hear about us?", qtype="text",
                         required=True, answer="LinkedIn", answer_source="ai",
                         needs_review=True),
            FormQuestion(text="Email", answer="a@b.c", answer_source="profile"),
        ]
        app_row = Application(job_id=job.id, status="prepared",
                              questions_json=questions_to_json(qs, "greenhouse"))
        db.add(app_row)
        db.commit()
        return job.id


def _stored(job_id):
    from jobbot.apply_questions import questions_from_json
    with session() as db:
        job = db.get(Job, job_id)
        return questions_from_json(job.applications[0].questions_json)[0]


def test_apply_fill_prompts_and_persists_answers(kit_job):
    runner = CliRunner()
    # relocate -> "1" (Yes), don't remember; then Enter to accept the AI guess.
    result = runner.invoke(cli, ["apply-fill", str(kit_job)], input="1\nn\n\n")

    assert result.exit_code == 0, result.output
    qs = _stored(kit_job)
    assert qs[0].answer == "Yes"
    assert qs[0].answer_source == "user"
    assert qs[0].needs_user is False
    assert qs[1].answer == "LinkedIn"       # accepted as-is
    assert qs[1].needs_review is False
    assert qs[1].answer_source == "user"


def test_apply_fill_shows_the_real_options_and_the_rejected_guess(kit_job):
    runner = CliRunner()
    result = runner.invoke(cli, ["apply-fill", str(kit_job)], input="q\n")

    assert "Are you willing to relocate?" in result.output
    assert "Yes" in result.output and "No" in result.output
    assert "Springfield, IL" in result.output          # the value we couldn't use
    assert result.output.isascii()                # cp1252-safe console


def test_apply_fill_quit_leaves_everything_open(kit_job):
    runner = CliRunner()
    runner.invoke(cli, ["apply-fill", str(kit_job)], input="q\n")

    qs = _stored(kit_job)
    assert qs[0].needs_user is True
    assert qs[1].needs_review is True


def test_apply_fill_rejects_an_answer_that_is_not_an_option(kit_job):
    runner = CliRunner()
    # "chartreuse" is invalid -> re-asked -> "no" accepted, then quit.
    result = runner.invoke(cli, ["apply-fill", str(kit_job)],
                           input="chartreuse\nno\nn\nq\n")

    assert "Not one of the options" in result.output
    qs = _stored(kit_job)
    assert qs[0].answer == "No"


def test_apply_fill_can_remember_an_answer(kit_job, tmp_path, monkeypatch):
    from jobbot.answer_memory import AnswerMemory
    from jobbot.apply_questions import FormQuestion, settings

    mem_path = tmp_path / "mem.json"
    monkeypatch.setattr(settings, "answer_memory_path", str(mem_path))

    runner = CliRunner()
    runner.invoke(cli, ["apply-fill", str(kit_job)], input="1\ny\nq\n")

    hit = AnswerMemory(str(mem_path)).lookup(
        FormQuestion(text="Are you willing to relocate?"))
    assert hit is not None and hit.answer == "Yes"


def test_apply_fill_on_a_job_with_no_kit_is_a_clean_message():
    init_db()
    with session() as db:
        url = f"https://example.test/j/{uuid4()}"
        job = Job(hash=Job.make_hash("test", url, "No Kit"),
                  title="No Kit", company="Nowhere",
                  url=url, source="test")
        db.add(job)
        db.commit()
        job_id = job.id

    result = CliRunner().invoke(cli, ["apply-fill", str(job_id)])
    assert result.exit_code == 0
    assert "no application kit" in result.output.lower()


def test_apply_fill_with_no_job_id_walks_the_whole_queue(kit_job):
    from jobbot.apply_fill import queue_gap_counts

    # The queue-wide view must account for this kit's two open questions.
    assert any(r["job_id"] == kit_job and r["total"] == 2
               for r in queue_gap_counts())

    result = CliRunner().invoke(cli, ["apply-fill"], input="q\n")
    assert result.exit_code == 0
    assert "need you" in result.output.lower()
