from click.testing import CliRunner

from jobbot import resume_facts as rf
from jobbot.cli import cli


def test_facts_show_reports_unreviewed(tmp_path, monkeypatch):
    monkeypatch.setattr(rf, "_path", lambda: tmp_path / "facts.json")
    rf.save(rf.ResumeFacts(
        employment=[rf.Employment(employer="Meridian Research Institute", title="RA",
                                  start="2022-06", current=True)],
        education=[], identity={}, reviewed=False))
    out = CliRunner().invoke(cli, ["facts", "show"])
    assert out.exit_code == 0
    assert "Meridian Research Institute" in out.output
    assert "NOT reviewed" in out.output


def test_facts_review_marks_reviewed(tmp_path, monkeypatch):
    monkeypatch.setattr(rf, "_path", lambda: tmp_path / "facts.json")
    rf.save(rf.ResumeFacts(employment=[], education=[], identity={}, reviewed=False))
    out = CliRunner().invoke(cli, ["facts", "review", "--yes"])
    assert out.exit_code == 0
    assert rf.load().is_reviewed() is True


RESUME = """# Alex Rivera
Springfield, ST · alex.rivera@example.com

## Research & Professional Experience

### Research Assistant — Example Lab, Example University, Springfield, ST · 1/2021 – Present
- Did the work
"""


def test_facts_extract_reads_the_named_markdown_file(tmp_path, monkeypatch):
    monkeypatch.setattr(rf, "_path", lambda: tmp_path / "facts.json")
    src = tmp_path / "resume.md"
    src.write_text(RESUME, encoding="utf-8")
    out = CliRunner().invoke(cli, ["facts", "extract", "--from", str(src)])
    assert out.exit_code == 0, out.output
    assert str(src) in out.output          # which file was read must be visible
    assert "1 roles" in out.output
    assert rf.load().is_reviewed() is False  # extraction never confirms


def test_facts_extract_refuses_a_pdf_instead_of_parsing_binary(tmp_path, monkeypatch):
    """settings.base_resume_path is a PDF on a real install.

    Feeding it to a markdown parser either raises deep in the stack or yields
    zero rows that read as "this person has never worked". Refuse up front.
    """
    monkeypatch.setattr(rf, "_path", lambda: tmp_path / "facts.json")
    pdf = tmp_path / "base_resume.pdf"
    pdf.write_bytes(b"%PDF-1.4\x93\xff binary")
    out = CliRunner().invoke(cli, ["facts", "extract", "--from", str(pdf)])
    assert out.exit_code != 0
    assert "markdown" in out.output.lower()
    assert not (tmp_path / "facts.json").exists()  # no garbage store written


def test_facts_extract_refuses_binary_wearing_a_md_suffix(tmp_path, monkeypatch):
    """The suffix check is a label, not a guarantee."""
    monkeypatch.setattr(rf, "_path", lambda: tmp_path / "facts.json")
    fake = tmp_path / "resume.md"
    fake.write_bytes(b"%PDF-1.4\x93\xff\xfe not text at all")
    out = CliRunner().invoke(cli, ["facts", "extract", "--from", str(fake)])
    assert out.exit_code != 0
    assert "could not read" in out.output.lower()
    assert not (tmp_path / "facts.json").exists()


def test_facts_extract_says_so_when_it_finds_nothing(tmp_path, monkeypatch):
    monkeypatch.setattr(rf, "_path", lambda: tmp_path / "facts.json")
    src = tmp_path / "resume.md"
    src.write_text("# Alex Rivera\n\n## Experience\n\nnothing parseable\n",
                   encoding="utf-8")
    out = CliRunner().invoke(cli, ["facts", "extract", "--from", str(src)])
    assert out.exit_code == 0
    assert "0 roles" in out.output
    assert "layout" in out.output.lower()
