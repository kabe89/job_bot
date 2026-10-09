import json
from pathlib import Path

from jobbot.answer_memory import AnswerMemory
from jobbot.apply_questions import FormQuestion


def test_remember_then_lookup_roundtrips(tmp_path):
    mem = AnswerMemory(str(tmp_path / "mem.json"))
    q = FormQuestion(text="Are you authorized to work in the US?", kind="screening")
    mem.remember(q, "Yes", source="config")
    got = mem.lookup(FormQuestion(text="Are you authorized to work in the US?",
                                  kind="screening"))
    assert got is not None
    assert got.answer == "Yes"
    assert got.kind == "static"


def test_lookup_matches_across_company_phrasing(tmp_path):
    mem = AnswerMemory(str(tmp_path / "mem.json"))
    mem.remember(FormQuestion(text="Why do you want to work at Acme?",
                              qtype="textarea"), "Because mission.")
    got = mem.lookup(FormQuestion(text="Why do you want to work at Genentech?",
                                  qtype="textarea"))
    assert got is not None
    assert got.kind == "tailored"
    assert got.answer == "Because mission."


def test_store_file_is_written(tmp_path):
    p = tmp_path / "mem.json"
    mem = AnswerMemory(str(p))
    mem.remember(FormQuestion(text="First Name", kind="identity"), "Jane")
    assert p.exists()
    data = json.loads(p.read_text(encoding="utf-8"))
    assert any(e["answer"] == "Jane" for e in data)
