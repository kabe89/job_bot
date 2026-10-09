import json
from pathlib import Path

from jobbot.answer_memory import calibrate

PAIRS = json.loads(
    (Path(__file__).parent / "fixtures" / "question_pairs.json")
    .read_text(encoding="utf-8"))


def test_calibrate_separates_equivalent_from_opposite_pairs():
    report = calibrate(PAIRS)
    assert report["equivalent_min"] > report["opposite_max"], report
    assert report["separable"] is True


def test_calibrate_suggests_a_threshold_inside_the_gap():
    report = calibrate(PAIRS)
    assert report["opposite_max"] < report["suggested_high"] <= report["equivalent_min"]


def test_calibrate_reports_unseparable_rather_than_guessing():
    # If no threshold separates the two sets, the honest answer is to say so --
    # per the spec, that is the signal to ship with the semantic tier disabled.
    bad = [{"a": "Same question", "b": "Same question", "equivalent": False},
           {"a": "Same question", "b": "Same question", "equivalent": True}]
    report = calibrate(bad)
    assert report["separable"] is False


def test_calibrate_zeroes_an_inverted_pair():
    # The opposite pairs in the fixture only stay below the equivalent pairs
    # because calibrate applies the same polarity guard fuzzy_lookup does: a
    # sponsorship-vs-authorization pair shares most of its words and would score
    # ABOVE the equivalent rewordings on raw overlap. If that guard were dropped
    # from calibrate, this inverted pair would report a high score and the fixture
    # would stop being separable -- so pin it directly.
    from jobbot.answer_memory import token_similarity
    a = "Are you legally authorized to work in the US?"
    b = "Do you require sponsorship to work in the US?"
    raw = token_similarity(a, b)
    report = calibrate([{"a": a, "b": b, "equivalent": False}])
    assert raw > 0.0, "raw overlap must be non-zero or this test proves nothing"
    assert report["opposite_max"] == 0.0
