import json

from jobbot import apply_questions
from jobbot.apply_questions import FormQuestion, answer_questions


def _q(text: str, **kw) -> FormQuestion:
    return FormQuestion(text=text, **kw)


def _answer(questions, monkeypatch, bank=None, memory=None, tmp_path=None):
    """Run answer_questions with the bank and memory stubbed, AI off."""
    monkeypatch.setattr(apply_questions, "load_answer_bank", lambda: bank or [])
    monkeypatch.setattr(apply_questions, "personal_info_bank_entries_safe",
                        lambda: [])
    if memory is not None:
        path = tmp_path / "mem.json"
        path.write_text(json.dumps(memory), encoding="utf-8")
        monkeypatch.setattr(apply_questions.settings, "answer_memory_path",
                            str(path))
    return answer_questions(questions, resume="", job_title="", company="",
                            job_description="", use_ai=False)


def test_memory_answers_a_question_the_bank_does_not(monkeypatch, tmp_path):
    mem = [{"key": "favourite colour", "kind": "static", "answer": "Blue",
            "source": "apply"}]
    out = _answer([_q("Favourite colour?")], monkeypatch, memory=mem,
                  tmp_path=tmp_path)
    assert out[0].answer == "Blue"
    assert out[0].answer_source == "memory"


def test_bank_wins_over_memory(monkeypatch, tmp_path):
    # The bank is the small hand-edited override layer. If memory won, editing
    # answer_bank.json to correct a bad learned answer would do nothing.
    bank = [{"match": r"favourite colour", "answer": "Green"}]
    mem = [{"key": "favourite colour", "kind": "static", "answer": "Blue",
            "source": "apply"}]
    out = _answer([_q("Favourite colour?")], monkeypatch, bank=bank, memory=mem,
                  tmp_path=tmp_path)
    assert out[0].answer == "Green"


def test_a_memory_answer_that_fits_no_option_asks_instead(monkeypatch, tmp_path):
    mem = [{"key": "shirt size", "kind": "static", "answer": "Enormous",
            "source": "apply"}]
    out = _answer([_q("Shirt size?", options=["S", "M", "L"])], monkeypatch,
                  memory=mem, tmp_path=tmp_path)
    assert out[0].answer == ""
    assert out[0].needs_user is True
    assert out[0].guess == "Enormous"


def test_a_tailored_memory_is_not_reused_on_another_job(monkeypatch, tmp_path):
    # A tailored answer was written about ONE company. Replaying it into another
    # company's kit would submit a cover-letter-grade lie -- and "memory" is in
    # VERIFIED_SOURCES, so it would go out unreviewed. apply_runner.py:157
    # already applies this same kind guard to its pre-fill.
    #
    # The stored key MUST be the canonical key of the asked question
    # (canonical_key("Why do you want to work here?") == "why work here"), or the
    # exact lookup misses and this test would pass on a lookup miss instead of on
    # the kind guard -- i.e. it would still pass with the guard deleted. With the
    # keys aligned, lookup() hits and only `hit.kind == "static"` stops the reuse.
    mem = [{"key": "why work here", "kind": "tailored",
            "answer": "I admire Acme's cryo-EM platform.", "source": "apply"}]
    out = _answer([_q("Why do you want to work here?")], monkeypatch,
                  memory=mem, tmp_path=tmp_path)
    assert out[0].answer == ""
    assert out[0].answer_source != "memory"


def test_memory_source_is_trusted_enough_to_submit():
    assert "memory" in apply_questions.VERIFIED_SOURCES


def test_remember_answer_writes_to_memory(tmp_path, monkeypatch):
    from jobbot.answer_memory import AnswerMemory
    from jobbot.apply_questions import FormQuestion, remember_answer

    path = tmp_path / "mem.json"
    monkeypatch.setattr(apply_questions.settings, "answer_memory_path", str(path))
    remember_answer("Are you 18 or older?", "Yes")

    hit = AnswerMemory(str(path)).lookup(FormQuestion(text="Are you 18 or older?"))
    assert hit is not None and hit.answer == "Yes"


def test_the_bank_writer_is_gone():
    # The bank is a hand-edited rules file now. Leaving a writer in place would
    # quietly recreate the two-store split this work exists to close.
    assert not hasattr(apply_questions, "save_to_answer_bank")


def test_a_confident_fuzzy_match_answers(monkeypatch, tmp_path):
    # A pure rewording -- same tokens, different order -- scores 1.0 and so
    # clears the shipped 0.93 default without the test tuning it.
    mem = [{"key": "are authorized work united states", "kind": "static",
            "answer": "Yes", "source": "user"}]
    out = _answer([_q("In the United States, are you authorized to work?",
                      kind="screening")],
                  monkeypatch, memory=mem, tmp_path=tmp_path)
    assert out[0].answer == "Yes"


def test_a_borderline_fuzzy_match_is_held_for_review(monkeypatch, tmp_path):
    # A below-confidence fuzzy match is held for the user, never submitted. It is
    # held as a BLANK answer with the value on q.guess (-> needs_user), NOT as a
    # populated answer flagged needs_review: browser_apply's autosubmit planner
    # pauses on a blank answer but does not pause on needs_review alone, so a
    # value left in q.answer here would be submitted unreviewed.
    mem = [{"key": "authorized work united states", "kind": "static",
            "answer": "Yes", "source": "user"}]
    monkeypatch.setattr(apply_questions.settings,
                        "answer_match_high_threshold", 0.99)
    monkeypatch.setattr(apply_questions.settings,
                        "answer_match_review_threshold", 0.1)
    out = _answer([_q("Do you have work authorization in the US?",
                      kind="screening")],
                  monkeypatch, memory=mem, tmp_path=tmp_path)
    assert out[0].answer == ""
    assert out[0].needs_user is True
    assert out[0].guess == "Yes"


def test_a_tailored_entry_is_not_reachable_by_the_fuzzy_tier(monkeypatch, tmp_path):
    # Companion to the exact-tier guard test already in this file. Neither tier
    # may replay a company-specific essay into another company's application.
    mem = [{"key": "why do you want to work here", "kind": "tailored",
            "answer": "I admire Acme's platform.", "source": "apply"}]
    monkeypatch.setattr(apply_questions.settings,
                        "answer_match_high_threshold", 0.1)
    monkeypatch.setattr(apply_questions.settings,
                        "answer_match_review_threshold", 0.05)
    out = _answer([_q("Why would you like to work here?", kind="essay")],
                  monkeypatch, memory=mem, tmp_path=tmp_path)
    assert out[0].answer == ""
    assert out[0].guess == ""


def test_confirming_a_fuzzy_match_makes_it_exact_next_time(monkeypatch, tmp_path):
    # The self-sharpening property. A confirmed fuzzy match is stored under the
    # NEW question's exact canonical key, so the same phrasing resolves at the
    # free exact tier on every later run and never pays the fuzzy cost again.
    from jobbot.answer_memory import AnswerMemory
    from jobbot.apply_questions import FormQuestion, remember_answer

    path = tmp_path / "mem.json"
    monkeypatch.setattr(apply_questions.settings, "answer_memory_path", str(path))
    asked = "Do you have work authorization in the US?"

    remember_answer(asked, "Yes")

    mem = AnswerMemory(str(path))
    assert mem.lookup(FormQuestion(text=asked)).answer == "Yes"


def test_lookup_survives_an_unreadable_memory(monkeypatch, tmp_path):
    path = tmp_path / "mem.json"
    path.write_text("{not json", encoding="utf-8")
    monkeypatch.setattr(apply_questions.settings, "answer_memory_path", str(path))
    out = _answer([_q("Anything?")], monkeypatch)
    assert out[0].answer == ""  # fell through cleanly, did not raise
