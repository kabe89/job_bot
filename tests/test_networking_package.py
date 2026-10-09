"""Tests for the ultimate networking package builder (no network / LLM)."""
import pytest

from jobbot import networking_package as npkg


FAKE_RESEARCH = {
    "name": "Jane Researcher", "title": "Principal Scientist",
    "company": "Acme Bio", "pubmed": "- Cool paper. Nature. 2024.",
    "lab_page": "Jane studies RNA things.", "general": "- News snippet.",
}


def _stub_gen(prompt: str) -> str:
    # Echo which artifact is being generated so we can assert routing.
    if "ULTIMATE\npreparation brief" in prompt or "preparation brief" in prompt:
        return "# Networking Prep — Jane Researcher\nGreat brief body."
    if "outreach\nmessages" in prompt.lower() or "outreach messages" in prompt.lower():
        return "# Outreach Messages\nHi Jane."
    if "one-page research summary" in prompt.lower() or "one-pager" in prompt.lower():
        return "# One-Pager\nAbout the user."
    return "Generic content for the section."


def test_slug_combines_name_and_company():
    assert npkg._slug("Jane Researcher", "Acme Biolabs") == \
        "jane_researcher_acme_biolabs"
    assert npkg._slug("Jane Doe") == "jane_doe"


def test_requires_name():
    with pytest.raises(ValueError):
        npkg.build_package("   ")


def test_build_writes_all_artifacts(tmp_path):
    res = npkg.build_package(
        "Jane Researcher", company="Acme Bio", role="Principal Scientist",
        context="same field", goals="advice", fmt="video",
        do_podcast=False, out_dir=str(tmp_path),
        generate=_stub_gen, research=FAKE_RESEARCH)
    assert res["used_llm"] is True
    assert res["dir"] == str(tmp_path)
    for key in ("prep_path", "outreach_path", "onepager_path"):
        assert res[key], f"missing {key}"
        assert open(res[key], encoding="utf-8").read().strip()
    assert res["podcast_path"] is None  # podcast skipped


def test_fallback_when_llm_unavailable(tmp_path):
    res = npkg.build_package(
        "Jane Researcher", company="Acme Bio", fmt="call",
        do_podcast=False, out_dir=str(tmp_path),
        generate=lambda p: "", research=FAKE_RESEARCH)
    assert res["used_llm"] is False
    body = open(res["prep_path"], encoding="utf-8").read()
    # Fallback embeds the raw evidence so the user still has something usable.
    assert "Jane Researcher" in body
    assert "Cool paper" in body


def test_invalid_format_defaults_to_call(tmp_path):
    res = npkg.build_package(
        "Jane Researcher", fmt="telepathy", do_podcast=False,
        out_dir=str(tmp_path), generate=_stub_gen, research=FAKE_RESEARCH)
    # No crash; prep written.
    assert open(res["prep_path"], encoding="utf-8").read().strip()


def test_evidence_block_handles_missing_fields():
    block = npkg._evidence_block({"name": "X"})
    assert "X" in block
    assert "(none found)" in block
