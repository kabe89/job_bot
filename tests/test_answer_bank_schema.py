import json

from jobbot import apply_questions
from jobbot.apply_questions import load_answer_bank, write_bank_entries


def _bank(tmp_path, monkeypatch, entries):
    path = tmp_path / "bank.json"
    path.write_text(json.dumps(entries), encoding="utf-8")
    monkeypatch.setattr(apply_questions.settings, "answer_bank_path", str(path),
                        raising=False)
    return path


def test_untagged_entries_still_load(tmp_path, monkeypatch):
    # All 15 existing entries have neither key. They must keep working.
    _bank(tmp_path, monkeypatch, [{"match": "age", "answer": "Yes"}])
    assert len(load_answer_bank()) == 1


def test_pending_entries_are_not_loaded(tmp_path, monkeypatch):
    _bank(tmp_path, monkeypatch, [
        {"match": "age", "answer": "Yes"},
        {"match": "years", "answer": "6", "source": "resume-ai",
         "status": "pending"},
    ])
    assert [e["match"] for e in load_answer_bank()] == ["age"]


def test_write_replaces_only_its_own_source(tmp_path, monkeypatch):
    # Re-seeding must never clobber a hand-written rule.
    path = _bank(tmp_path, monkeypatch, [
        {"match": "age", "answer": "Yes"},
        {"match": "degree", "answer": "BS", "source": "resume"},
    ])
    write_bank_entries([{"match": "degree", "answer": "PhD"}], source="resume")
    saved = json.loads(path.read_text(encoding="utf-8"))
    assert {e["match"]: e["answer"] for e in saved} == {"age": "Yes",
                                                        "degree": "PhD"}
    assert [e for e in saved if e["match"] == "age"][0].get("source") is None
