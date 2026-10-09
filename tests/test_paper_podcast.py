from pathlib import Path

import jobbot.paper_podcast as pp


def test_segment_sections_detects_and_drops_references():
    text = (
        "A Cool Paper Title\n"
        "ABSTRACT\nWe did a thing and it worked.\n"
        "INTRODUCTION\nBackground here.\n"
        "METHODS\nWe used a method.\n"
        "RESULTS\nThe result was positive.\n"
        "DISCUSSION\nIt matters because reasons.\n"
        "REFERENCES\n1. Someone et al. 2020.\n2. Another 2019.\n"
    )
    secs = pp.segment_sections(text)
    assert "abstract" in secs and "results" in secs and "discussion" in secs
    assert "references" not in secs
    assert "Someone et al" not in " ".join(secs.values())


def test_segment_sections_drops_singular_acknowledgement():
    text = (
        "Title Line Here\n"
        "ABSTRACT\nWe did a thing.\n"
        "RESULTS\nIt worked well.\n"
        "ACKNOWLEDGMENT\nWe thank the funders and Jane Doe for help.\n"
    )
    secs = pp.segment_sections(text)
    assert "results" in secs
    assert "acknowledgement" not in secs and "acknowledgements" not in secs
    assert "Jane Doe" not in " ".join(secs.values())


def test_segment_sections_no_headers_returns_preamble():
    secs = pp.segment_sections("Just some unstructured text with no headers.")
    assert "preamble" in secs
    assert "unstructured text" in secs["preamble"]


def test_extract_paper_uses_loader(monkeypatch):
    monkeypatch.setattr(pp, "load_resume",
                        lambda p: "My Title\nABSTRACT\nHello.\nRESULTS\nGood.\n")
    doc = pp.extract_paper("whatever.pdf")
    assert doc.title == "My Title"
    assert "abstract" in doc.sections and "results" in doc.sections


def _doc():
    return pp.PaperDoc(
        title="Ribosome Paper",
        sections={"abstract": "We bound a drug to the 30S subunit.",
                  "results": "Binding was tight. The site was clear. Affinity was high.",
                  "discussion": "This explains resistance."},
        raw="raw text",
    )


def test_summarize_paper_extractive_when_no_llm(monkeypatch):
    monkeypatch.setattr(pp, "_llm_available", lambda: False)
    brief = pp.summarize_paper(_doc(), mode="explainer")
    assert "30S subunit" in brief or "Binding was tight" in brief
    assert brief.strip()


def test_summarize_paper_uses_llm_when_available(monkeypatch):
    monkeypatch.setattr(pp, "_llm_available", lambda: True)
    monkeypatch.setattr(pp, "_llm_generate", lambda prompt, system: "LLM BRIEF OUTPUT")
    brief = pp.summarize_paper(_doc(), mode="networking")
    assert brief == "LLM BRIEF OUTPUT"


def test_generate_paper_script_explainer_vs_networking(monkeypatch):
    captured = {}
    monkeypatch.setattr(pp, "_llm_available", lambda: True)

    def fake_gen(prompt, system):
        captured["system"] = system
        return "A: Welcome to the show.\nB: Today we cover a paper.\nA: Let's dig in."
    monkeypatch.setattr(pp, "_llm_generate", fake_gen)

    lines = pp.generate_paper_script("brief", style="two_host", mode="explainer")
    assert [l.speaker for l in lines] == ["A", "B", "A"]
    assert pp._SYS_EXPLAINER == captured["system"]

    pp.generate_paper_script("brief", style="two_host", mode="networking")
    assert pp._SYS_NETWORKING == captured["system"]


def test_generate_paper_script_falls_back_without_llm(monkeypatch):
    monkeypatch.setattr(pp, "_llm_available", lambda: False)
    lines = pp.generate_paper_script("# T\n- one\n- two\n- three",
                                     style="two_host", mode="explainer", title="T")
    assert len(lines) >= 3


def test_journal_club_fallback_concatenates(monkeypatch):
    monkeypatch.setattr(pp, "_llm_available", lambda: False)
    out = pp.synthesize_journal_club(["brief one", "brief two"], mode="explainer")
    assert "brief one" in out and "brief two" in out


def test_make_paper_podcast_single_script_only(tmp_path, monkeypatch):
    paper = tmp_path / "paper1.txt"
    paper.write_text("Great Title\nABSTRACT\nWe found X.\nRESULTS\nX works.\n", encoding="utf-8")
    monkeypatch.setattr(pp, "_llm_available", lambda: False)  # deterministic
    out = tmp_path / "ep.mp3"
    res = pp.make_paper_podcast(str(paper), out_path=str(out), mode="explainer",
                                audio=False)
    assert res["n_papers"] == 1
    assert res["audio_path"] is None
    assert Path(res["script_path"]).exists()


def test_make_paper_podcast_multi_uses_journal_club(tmp_path, monkeypatch):
    for i in (1, 2):
        (tmp_path / f"p{i}.txt").write_text(
            f"Title {i}\nABSTRACT\nPaper {i} did a thing.\nRESULTS\nIt worked {i}.\n",
            encoding="utf-8")
    monkeypatch.setattr(pp, "_llm_available", lambda: False)
    seen = {}
    real = pp.synthesize_journal_club
    monkeypatch.setattr(pp, "synthesize_journal_club",
                        lambda briefs, mode="explainer", title="": seen.update(n=len(briefs)) or real(briefs, mode, title))
    res = pp.make_paper_podcast(str(tmp_path), audio=False, out_path=str(tmp_path / "jc.mp3"))
    assert res["n_papers"] == 2
    assert seen["n"] == 2
