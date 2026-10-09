"""Scientific-paper podcast -- turn PDF papers into explainer/networking audio.

Reuses the `podcast` audio pipeline (Line, _parse_script, _fallback_script,
synthesize) and adds paper-specific preprocessing: PDF extraction (via
resume.load_resume), section-aware segmentation (drop references), map/reduce
summarization, and mode-aware (explainer / networking) script prompts.
"""
from __future__ import annotations

import glob as _glob
import logging
import re
from dataclasses import dataclass
from pathlib import Path
from typing import List, Optional

from .resume import load_resume

log = logging.getLogger("jobbot.paper_podcast")

MODES = ("explainer", "networking")

_SECTION_RE = re.compile(
    r"^\s*(?:\d+\.?\s+)?"
    r"(ABSTRACT|INTRODUCTION|BACKGROUND|MATERIALS\s+AND\s+METHODS|METHODS?|"
    r"RESULTS\s+AND\s+DISCUSSION|RESULTS|DISCUSSION|CONCLUSIONS?|"
    r"REFERENCES|BIBLIOGRAPHY|ACKNOWLEDG(?:E)?MENTS?|SUPPLEMENTARY)\b.*$",
    re.IGNORECASE,
)
_DROP_FROM = {"references", "acknowledgements", "supplementary"}
_CANON = {
    "abstract": "abstract", "introduction": "introduction", "background": "introduction",
    "materials": "methods", "method": "methods", "methods": "methods",
    "results": "results", "discussion": "discussion",
    "conclusion": "conclusions", "conclusions": "conclusions",
    "references": "references", "bibliography": "references",
    "acknowledgements": "acknowledgements", "acknowledgments": "acknowledgements",
    "acknowledgement": "acknowledgements", "acknowledgment": "acknowledgements",
    "supplementary": "supplementary",
}


@dataclass
class PaperDoc:
    title: str
    sections: dict
    raw: str


def _canon(header_word: str) -> str:
    first = header_word.strip().lower().split()[0]
    return _CANON.get(first, first)


def segment_sections(text: str) -> dict:
    """Split paper text into canonical sections; drop references onward."""
    out: dict = {}
    cur = "preamble"
    buf: List[str] = []
    broke = False
    for ln in (text or "").splitlines():
        m = _SECTION_RE.match(ln.strip())
        if m:
            out[cur] = "\n".join(buf).strip()
            buf = []
            canon = _canon(m.group(1))
            if canon in _DROP_FROM:
                broke = True
                break
            cur = canon
        else:
            buf.append(ln)
    if not broke:
        out[cur] = "\n".join(buf).strip()
    return {k: v for k, v in out.items() if v and k not in _DROP_FROM}


def _guess_title(raw: str, path) -> str:
    for ln in (raw or "").splitlines():
        s = ln.strip()
        if len(s) >= 8 and not _SECTION_RE.match(s):
            return s[:160]
    return Path(str(path)).stem.replace("_", " ")


def extract_paper(path) -> PaperDoc:
    raw = load_resume(path)
    if not (raw or "").strip():
        raise RuntimeError(f"Could not extract any text from {path} "
                           "(scanned image PDF or unreadable file?).")
    return PaperDoc(title=_guess_title(raw, path),
                    sections=segment_sections(raw), raw=raw)


# Reuse the podcast module's Ollama bridge so paper generation honors the
# model selected in the dashboard (local or cloud).
from .podcast import _llm_available, _llm_generate  # noqa: E402

_PRIORITY = ["abstract", "results", "discussion", "conclusions",
             "introduction", "methods", "preamble"]

_SYS_BRIEF = (
    "You are a meticulous science communicator. You summarize a research paper "
    "accurately and never invent findings, numbers, or claims beyond the source."
)


def _prioritized_text(paper: PaperDoc, limit: int = 9000) -> str:
    parts = []
    for k in _PRIORITY:
        if paper.sections.get(k):
            parts.append(f"## {k.title()}\n{paper.sections[k]}")
    text = "\n\n".join(parts).strip()
    return (text or paper.raw)[:limit]


def _first_sentences(text: str, n: int = 3) -> str:
    sents = re.split(r"(?<=[.!?])\s+", (text or "").strip())
    return " ".join(s for s in sents[:n] if s)


def _extractive_brief(paper: PaperDoc) -> str:
    bits = [f"Title: {paper.title}."]
    if paper.sections.get("abstract"):
        bits.append("Abstract: " + paper.sections["abstract"])
    for k in ("results", "discussion", "conclusions"):
        if paper.sections.get(k):
            bits.append(f"{k.title()}: " + _first_sentences(paper.sections[k], 3))
    return "\n".join(bits).strip()


def _brief_prompt(title: str, src: str, mode: str) -> str:
    angle = ("Frame it for a listener preparing to talk with the author about the "
             "work (what it is, why it's notable, smart questions to ask)."
             if mode == "networking" else
             "Frame it as a neutral explainer that teaches the listener the science.")
    return (
        f'Summarize the research paper "{title}" into a tight briefing.\n{angle}\n\n'
        "Cover: the central question, the approach, the key findings, why it "
        "matters, and the main limitations. Use ONLY the source below - no "
        "invented numbers or claims.\n\nSOURCE:\n" + src
    )


def summarize_paper(paper: PaperDoc, mode: str = "explainer") -> str:
    if mode not in MODES:
        mode = "explainer"
    src = _prioritized_text(paper)
    if _llm_available():
        try:
            out = _llm_generate(_brief_prompt(paper.title, src, mode), _SYS_BRIEF)
            if out and out.strip():
                return out.strip()
            log.warning("Empty LLM brief for %s - using extractive fallback.", paper.title)
        except Exception as exc:  # noqa: BLE001
            log.warning("LLM brief failed (%s) - using extractive fallback.", exc)
    return _extractive_brief(paper)


from . import podcast as _pod  # noqa: E402

_SYS_EXPLAINER = (
    "You are a scriptwriter for a lively two-host science podcast. The hosts are "
    "neutral, curious explainers who teach the listener accurately and never "
    "invent facts beyond the briefing."
)
_SYS_NETWORKING = (
    "You are a scriptwriter coaching a listener before they reach out to a "
    "researcher about their work. The hosts speak in the third person about the "
    "researcher, stay honest, and never fabricate shared history or findings."
)


def _script_prompt(brief: str, style: str, mode: str, title: str, contact: str) -> str:
    head = f'Episode topic: "{title}".\n' if title else ""
    who = (f"The listener wants to connect with {contact}. " if (mode == "networking" and contact)
           else "")
    if style == "two_host":
        fmt = ("OUTPUT FORMAT - strict. One line per turn, prefixed exactly:\n"
               "A: <host A>\nB: <host B>\nNo stage directions, no markdown. 12-28 turns.\n\n")
        ask = ("Two hosts break the paper down for the listener in an engaging "
               "~3-5 minute episode. Cover the question, approach, key findings, "
               "why it matters, and limitations. End with a crisp takeaway.\n\n")
    else:
        fmt = ("Write 6-14 short spoken paragraphs. No headings, no bullet "
               "characters, no meta commentary.\n\n")
        ask = ("Rewrite the briefing as a single-voice spoken audio briefing.\n\n")
    return head + who + ask + fmt + "BRIEFING:\n" + brief[:9000]


def generate_paper_script(brief: str, style: str = "two_host", mode: str = "explainer",
                          title: str = "", contact: str = "") -> List["_pod.Line"]:
    if style not in _pod.STYLES:
        style = "two_host"
    if mode not in MODES:
        mode = "explainer"
    if _llm_available():
        try:
            sys = _SYS_EXPLAINER if mode == "explainer" else _SYS_NETWORKING
            raw = _llm_generate(_script_prompt(brief, style, mode, title, contact), sys)
            lines = _pod._parse_script(raw, style)
            if len(lines) >= 3:
                return lines
            log.warning("Paper script too short - using fallback.")
        except Exception as exc:  # noqa: BLE001
            log.warning("Paper script generation failed (%s) - using fallback.", exc)
    return _pod._fallback_script(brief, style, title)


def synthesize_journal_club(briefs: List[str], mode: str = "explainer",
                            title: str = "") -> str:
    joined = "\n\n---\n\n".join(b for b in briefs if b)
    if _llm_available():
        try:
            prompt = (
                f'Synthesize these {len(briefs)} paper briefings into one connected '
                f'"journal club" narrative{(" titled " + title) if title else ""}. '
                "Find the common thread and contrasts; do not invent anything beyond "
                "the briefings.\n\n" + joined[:9000]
            )
            out = _llm_generate(prompt, _SYS_BRIEF)
            if out and out.strip():
                return out.strip()
        except Exception as exc:  # noqa: BLE001
            log.warning("Journal-club synthesis failed (%s) - concatenating.", exc)
    return joined


_PAPER_EXTS = (".pdf", ".txt", ".md", ".docx")


def _resolve_sources(sources) -> List[Path]:
    if isinstance(sources, (str, Path)):
        sources = [sources]
    paths: List[Path] = []
    for s in sources:
        s = str(s)
        p = Path(s)
        if p.is_dir():
            for ext in _PAPER_EXTS:
                paths += sorted(p.glob(f"*{ext}"))
        elif any(ch in s for ch in "*?["):
            paths += sorted(Path(x) for x in _glob.glob(s))
        elif p.exists():
            paths.append(p)
    # de-dup, keep order
    seen, out = set(), []
    for p in paths:
        if p not in seen:
            seen.add(p)
            out.append(p)
    return out


def make_paper_podcast(sources, out_path: Optional[str] = None, mode: str = "explainer",
                       style: str = "two_host", contact: str = "", title: str = "",
                       audio: bool = True, voices: Optional[dict] = None) -> dict:
    if mode not in MODES:
        mode = "explainer"
    paths = _resolve_sources(sources)
    if not paths:
        raise RuntimeError("No paper files found for the given source(s).")

    papers = [extract_paper(p) for p in paths]
    briefs = [summarize_paper(p, mode) for p in papers]

    if len(briefs) == 1:
        brief = briefs[0]
        if not title:
            title = papers[0].title
    else:
        brief = synthesize_journal_club(briefs, mode, title)
        if not title:
            title = "Journal club: " + "; ".join(p.title for p in papers)[:100]

    lines = generate_paper_script(brief, style=style, mode=mode, title=title, contact=contact)
    if not lines:
        raise RuntimeError("Could not build a script from the paper(s).")

    if out_path:
        out = Path(out_path)
    else:
        stem = re.sub(r"[^A-Za-z0-9]+", "_", (papers[0].title or "paper"))[:60].strip("_")
        out = paths[0].with_name(f"{stem}_paper_podcast.mp3")
    out.parent.mkdir(parents=True, exist_ok=True)

    script_path = out.with_suffix(".script.txt")
    script_path.write_text(_pod.render_script(lines), encoding="utf-8")

    result = {"mode": mode, "style": style, "title": title, "n_papers": len(papers),
              "n_lines": len(lines), "script_path": str(script_path), "audio_path": None}
    if audio:
        result["audio_path"] = str(_pod.synthesize(lines, out, voices=voices))
    return result
