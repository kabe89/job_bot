"""Tests for the podcast script logic (pure functions — no TTS / network)."""
from jobbot import podcast
from jobbot.podcast import Line


def test_parse_two_host_basic():
    raw = "A: Hello there.\nB: Hi, great to be here.\nA: Let's dig in."
    lines = podcast._parse_script(raw, "two_host")
    assert [l.speaker for l in lines] == ["A", "B", "A"]
    assert lines[0].text == "Hello there."


def test_parse_two_host_handles_markdown_and_wraps():
    raw = "**A:** First **point** here\nthat wraps to a second line.\nB: Reacting."
    lines = podcast._parse_script(raw, "two_host")
    assert lines[0].speaker == "A"
    # wrapped continuation is appended; markdown stars stripped
    assert "wraps to a second line" in lines[0].text
    assert "**" not in lines[0].text
    assert lines[1].speaker == "B"


def test_clean_spoken_strips_links_and_urls():
    out = podcast._clean_spoken("See [the paper](http://x.com) at https://y.com now")
    assert "http" not in out
    assert "the paper" in out
    assert "[" not in out and "]" not in out


def test_fallback_two_host_alternates_and_wraps():
    text = "# Title\n- point one\n- point two\n- point three"
    lines = podcast._fallback_script(text, "two_host", "My Prep")
    speakers = [l.speaker for l in lines]
    assert speakers[0] == "A" and speakers[1] == "B"   # intro pair
    assert "point one" in " ".join(l.text for l in lines)
    assert speakers.count("A") and speakers.count("B")  # both voices used


def test_fallback_narrator_single_voice():
    text = "## Heading\n- alpha\n- beta"
    lines = podcast._fallback_script(text, "narrator", "Brief")
    assert all(l.speaker == "N" for l in lines)
    assert any("alpha" in l.text for l in lines)


def test_generate_script_empty_returns_empty():
    assert podcast.generate_script("", style="two_host") == []


def test_render_script_uses_host_names():
    out = podcast.render_script([Line("A", "hi"), Line("B", "yo"), Line("N", "end")])
    assert podcast.HOST_A_NAME in out and podcast.HOST_B_NAME in out
    assert "Narrator:" in out


def test_parse_narrator_splits_paragraphs():
    raw = "First paragraph here.\n\nSecond paragraph here."
    lines = podcast._parse_script(raw, "narrator")
    assert len(lines) == 2
    assert all(l.speaker == "N" for l in lines)
