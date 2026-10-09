import json

import pytest

from jobbot import resume_facts as rf


def test_round_trip_preserves_employment_order(tmp_path, monkeypatch):
    monkeypatch.setattr(rf, "_path", lambda: tmp_path / "facts.json")
    facts = rf.ResumeFacts(
        employment=[
            rf.Employment(employer="Acme Research", title="Research Assistant",
                          location="Springfield, ST", start="2022-06", end="",
                          current=True, bullets=["Optimized buffer conditions"]),
            rf.Employment(employer="State University", title="Lab Tech", location="Springfield, ST",
                          start="2020-01", end="2022-05", current=False, bullets=[]),
        ],
        education=[rf.Education(school="State University", degree="BS", field="Biochemistry",
                                location="Springfield, ST", start="2016-08", end="2020-05")],
        identity={"name": "Jane Doe", "email": "k@example.com"},
        reviewed=False)
    rf.save(facts)
    got = rf.load()
    assert [e.employer for e in got.employment] == ["Acme Research", "State University"]
    assert got.employment[0].current is True
    assert got.education[0].degree == "BS"
    assert got.identity["name"] == "Jane Doe"


def test_missing_file_returns_empty_facts(tmp_path, monkeypatch):
    monkeypatch.setattr(rf, "_path", lambda: tmp_path / "nope.json")
    got = rf.load()
    assert got.employment == []
    assert got.is_reviewed() is False


def test_is_reviewed_gates_on_the_flag(tmp_path, monkeypatch):
    monkeypatch.setattr(rf, "_path", lambda: tmp_path / "facts.json")
    facts = rf.ResumeFacts(employment=[], education=[], identity={}, reviewed=True)
    rf.save(facts)
    assert rf.load().is_reviewed() is True


def test_dates_must_be_yyyy_mm_or_empty(tmp_path, monkeypatch):
    monkeypatch.setattr(rf, "_path", lambda: tmp_path / "facts.json")
    with pytest.raises(ValueError, match="YYYY-MM"):
        rf.save(rf.ResumeFacts(
            employment=[rf.Employment(employer="X", title="Y", location="",
                                      start="June 2022", end="", current=True,
                                      bullets=[])],
            education=[], identity={}, reviewed=False))


def test_current_role_must_not_have_an_end_date(tmp_path, monkeypatch):
    monkeypatch.setattr(rf, "_path", lambda: tmp_path / "facts.json")
    with pytest.raises(ValueError, match="current"):
        rf.save(rf.ResumeFacts(
            employment=[rf.Employment(employer="X", title="Y", location="",
                                      start="2022-06", end="2023-01", current=True,
                                      bullets=[])],
            education=[], identity={}, reviewed=False))


def test_reviewed_string_false_is_not_treated_as_true(tmp_path, monkeypatch):
    # A hand-edited facts file with the JSON string "false" (not the boolean)
    # must not flip the human trust gate to reviewed=True. bool("false") is
    # True in Python, which is exactly the trap this guards against.
    p = tmp_path / "facts.json"
    p.write_text(json.dumps({"employment": [], "education": [], "identity": {},
                              "reviewed": "false"}), encoding="utf-8")
    monkeypatch.setattr(rf, "_path", lambda: p)
    assert rf.load().is_reviewed() is False


def test_trailing_newline_in_date_is_rejected(tmp_path, monkeypatch):
    monkeypatch.setattr(rf, "_path", lambda: tmp_path / "facts.json")
    with pytest.raises(ValueError, match="YYYY-MM"):
        rf.save(rf.ResumeFacts(
            employment=[rf.Employment(employer="X", title="Y", location="",
                                      start="2022-01\n", end="", current=False,
                                      bullets=[])],
            education=[], identity={}, reviewed=False))


def test_missing_file_still_returns_empty_facts_not_an_error(tmp_path, monkeypatch):
    # First-run state: no file on disk at all is legitimate and must not raise.
    monkeypatch.setattr(rf, "_path", lambda: tmp_path / "nope.json")
    got = rf.load()
    assert got.employment == []
    assert got.education == []


def test_corrupt_existing_file_raises_instead_of_returning_empty(tmp_path, monkeypatch):
    # A file that EXISTS but fails to parse (e.g. truncated mid-write) is an
    # error state, not "no employment history" - a later task fills Workday's
    # repeating employment section from this store and must not silently
    # submit a blank work history because the store was broken.
    p = tmp_path / "facts.json"
    p.write_text('{"employment": [', encoding="utf-8")  # truncated JSON
    monkeypatch.setattr(rf, "_path", lambda: p)
    with pytest.raises(ValueError):
        rf.load()


@pytest.mark.parametrize("body", ["null", "[]", '"x"', "3"])
def test_valid_json_that_is_not_an_object_raises_valueerror(body, tmp_path,
                                                            monkeypatch):
    # These parse cleanly, so they slip past the JSONDecodeError guard and used
    # to reach raw.get() and die with a bare AttributeError naming neither this
    # module nor the file. A caller catching (ValueError, OSError) around
    # load() - this module's own convention - would not have caught it.
    p = tmp_path / "facts.json"
    p.write_text(body, encoding="utf-8")
    monkeypatch.setattr(rf, "_path", lambda: p)
    with pytest.raises(ValueError, match="not a JSON object"):
        rf.load()
