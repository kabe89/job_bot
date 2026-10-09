from jobbot.answer_memory import (AnswerMemory, polarity, token_similarity)
from jobbot.apply_questions import FormQuestion


def _mem(tmp_path, entries):
    mem = AnswerMemory(str(tmp_path / "m.json"))
    for text, answer, kind in entries:
        mem.remember(FormQuestion(text=text, kind=kind), answer)
    return mem


def test_token_similarity_is_1_for_identical_keys():
    assert token_similarity("years of python", "years of python") == 1.0


def test_token_similarity_survives_rewording():
    score = token_similarity("are you legally authorized to work",
                             "are you authorized to work legally")
    assert score > 0.9


def test_polarity_detects_sponsorship_as_an_inversion():
    assert polarity("Do you require sponsorship to work in the US?")
    assert not polarity("Are you legally authorized to work in the US?")


def test_a_reworded_question_matches_confidently(tmp_path):
    mem = _mem(tmp_path, [("Are you legally authorized to work in the US?",
                           "Yes", "screening")])
    hit = mem.fuzzy_lookup(
        FormQuestion(text="Are you authorized legally to work in the US?",
                     kind="screening"), high=0.85, review=0.6)
    assert hit is not None
    entry, confident = hit
    assert entry.answer == "Yes" and confident is True


def test_the_sponsorship_question_never_auto_answers(tmp_path):
    # THE headline regression test. These two embed and tokenise almost
    # identically but take OPPOSITE answers. Auto-answering "Yes" to the
    # sponsorship question would torpedo the application.
    #
    # NOTE ON THE THRESHOLDS: these two only score 0.167 on token overlap, so
    # the plan's original high=0.5 made this test pass VACUOUSLY -- the score
    # gate forced confident=False and the polarity guard was never reached. high
    # is set BELOW the real score here deliberately, so the score gate passes
    # and the polarity guard is the only thing that can produce False. If the
    # guard regresses, this test fails.
    mem = _mem(tmp_path, [("Are you legally authorized to work in the US?",
                           "Yes", "screening")])
    q = FormQuestion(text="Do you require sponsorship to work in the US?",
                     kind="screening")
    assert token_similarity(q.text, "are legally authorized work") >= 0.15, (
        "threshold below is meant to sit under the real score; retune it")
    hit = mem.fuzzy_lookup(q, high=0.15, review=0.1)
    assert hit is not None, "expected a match so the guard is actually exercised"
    assert hit[1] is False, "must never answer confidently across an inversion"


def test_an_inverted_question_is_demoted_even_at_a_high_score(tmp_path):
    # "authorized" vs "not authorized" overlap 0.75, well above high=0.5, so the
    # score gate passes and ONLY the polarity guard can stop a confident "Yes"
    # being submitted to the exact opposite question.
    mem = _mem(tmp_path, [("Are you authorized to work in the US?",
                           "Yes", "screening")])
    hit = mem.fuzzy_lookup(
        FormQuestion(text="Are you not authorized to work in the US?",
                     kind="screening"), high=0.5, review=0.1)
    assert hit is not None and hit[1] is False


def test_willing_and_unwilling_do_not_auto_answer_each_other(tmp_path):
    # Same shape as above with a different marker, so the guard is not just
    # special-cased around one word.
    mem = _mem(tmp_path, [("Are you willing to relocate?", "Yes", "screening")])
    hit = mem.fuzzy_lookup(
        FormQuestion(text="Are you unwilling to relocate?", kind="screening"),
        high=0.5, review=0.1)
    assert hit is not None and hit[1] is False


def test_a_tailored_question_never_fuzzy_matches(tmp_path):
    mem = _mem(tmp_path, [("Why do you want to work here?", "Because.", "essay")])
    hit = mem.fuzzy_lookup(
        FormQuestion(text="Why would you like to work here?", kind="essay"),
        high=0.5, review=0.1)
    assert hit is None


def test_no_polarity_marker_is_swallowed_by_canonicalisation():
    # The negation guard compares polarity across two canonical keys. If a
    # marker word were ever added to _STOPWORDS, canonical_key would strip it,
    # both sides would agree, and the guard would silently stop firing -- and
    # the sponsorship question would start auto-answering "Yes". Fail loudly
    # here instead.
    from jobbot.answer_memory import canonical_key
    for word in ("not", "no", "never", "without", "unable", "decline",
                 "sponsorship", "unwilling", "except"):
        assert word in canonical_key(f"do you {word} qualify"), word
