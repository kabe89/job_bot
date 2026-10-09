import json

from click.testing import CliRunner

from jobbot import apply_questions
from jobbot.cli import cli


def _bank(tmp_path, monkeypatch, entries):
    path = tmp_path / "bank.json"
    path.write_text(json.dumps(entries), encoding="utf-8")
    monkeypatch.setattr(apply_questions.settings, "answer_bank_path", str(path),
                        raising=False)
    return path


def test_seed_dry_run_writes_nothing(tmp_path, monkeypatch):
    path = _bank(tmp_path, monkeypatch, [{"match": "age", "answer": "Yes"}])
    before = path.read_text(encoding="utf-8")
    result = CliRunner().invoke(cli, ["answers", "seed", "--dry-run"])
    assert result.exit_code == 0, result.output
    assert path.read_text(encoding="utf-8") == before


def test_seed_survives_an_unreachable_model(tmp_path, monkeypatch):
    # Seeding must not require Ollama to be up. Today this test would pass
    # vacuously -- conftest never seeds CANDIDATE_PROFILE_PATH, so skills is []
    # and the model is never consulted. Make the dependency explicit: give it a
    # real skill AND a model that always fails, and require a clean exit with
    # the deterministic half still produced.
    _bank(tmp_path, monkeypatch, [])
    profile = tmp_path / "candidate_profile.json"
    profile.write_text(json.dumps({"skills": ["Python"]}), encoding="utf-8")
    monkeypatch.setattr(apply_questions.settings, "candidate_profile_path",
                        str(profile), raising=False)

    def _boom(*a, **kw):
        raise RuntimeError("ollama is down")

    monkeypatch.setattr("jobbot.ollama_client._generate", _boom)
    result = CliRunner().invoke(cli, ["answers", "seed", "--dry-run"])
    assert result.exit_code == 0, result.output


def test_answers_list_shows_pending_separately(tmp_path, monkeypatch):
    _bank(tmp_path, monkeypatch, [
        {"match": "age", "answer": "Yes"},
        {"match": "years python", "answer": "6", "source": "resume-ai",
         "status": "pending"},
    ])
    result = CliRunner().invoke(cli, ["answers", "list", "--pending"])
    assert result.exit_code == 0, result.output
    assert "years python" in result.output
    assert "age" not in result.output


def test_review_approve_activates_a_pending_entry(tmp_path, monkeypatch):
    path = _bank(tmp_path, monkeypatch, [
        {"match": "years python", "answer": "6", "source": "resume-ai",
         "status": "pending"},
    ])
    result = CliRunner().invoke(cli, ["answers", "review"], input="a\n")
    assert result.exit_code == 0, result.output
    saved = json.loads(path.read_text(encoding="utf-8"))
    assert saved[0]["status"] == "active"


def test_review_reject_deletes_a_pending_entry(tmp_path, monkeypatch):
    path = _bank(tmp_path, monkeypatch, [
        {"match": "years python", "answer": "6", "source": "resume-ai",
         "status": "pending"},
    ])
    result = CliRunner().invoke(cli, ["answers", "review"], input="r\n")
    assert result.exit_code == 0, result.output
    assert json.loads(path.read_text(encoding="utf-8")) == []
