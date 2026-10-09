"""Podcast generator — turn any prep doc / markdown into an audio podcast.

Two stages, each with a graceful fallback so the command always produces *some*
output:

  1. SCRIPT  — a local LLM (Ollama) rewrites the source text into a spoken
     script in one of three styles: a two-host conversation (NotebookLM-style),
     a single-narrator briefing, or a direct second-person coach. If Ollama is
     unavailable, a deterministic extractor builds a serviceable script from the
     document's headings and bullets.

  2. AUDIO   — each line is synthesized to speech with `edge-tts` (free Microsoft
     neural voices, no API key), normalized to a uniform format, and stitched
     into one MP3 with ffmpeg. On a machine with no internet / no edge-tts, it
     falls back to offline Windows SAPI voices.

Public API
----------
generate_script(text, style="two_host", title="") -> list[Line]
render_script(lines)                              -> str   (human-readable)
synthesize(lines, out_path, voices=None)          -> Path
make_podcast(source, out_path=None, style=...)    -> dict

`source` may be a path to a markdown/text file or a raw string.
"""
from __future__ import annotations

import asyncio
import logging
import os
import re
import shutil
import subprocess
import tempfile
from dataclasses import dataclass
from pathlib import Path
from typing import List, Optional

log = logging.getLogger("jobbot.podcast")

# --- Voices ---------------------------------------------------------------
# edge-tts neural voices. Andrew/Ava are the natural "conversational" pair;
# Brian narrates. Override via synthesize(voices=...).
VOICE_A = "en-US-AndrewMultilingualNeural"   # host A — warm, male
VOICE_B = "en-US-AvaMultilingualNeural"      # host B — bright, female
VOICE_NARRATOR = "en-US-BrianMultilingualNeural"

HOST_A_NAME = "Alex"
HOST_B_NAME = "Maya"

STYLES = ("two_host", "narrator", "coach")

# Offline SAPI fallback voice name hints (Windows ships David + Zira).
_SAPI_VOICE = {"A": "David", "B": "Zira", "N": "David"}


@dataclass
class Line:
    """One spoken line. speaker is 'A', 'B', or 'N' (narrator)."""
    speaker: str
    text: str


# ===========================================================================
# Stage 1 — script generation
# ===========================================================================

def _llm_available() -> bool:
    try:
        from . import ollama_client
        return ollama_client.is_available()
    except Exception:  # noqa: BLE001
        return False


def _llm_generate(prompt: str, system: str, temperature: float = 0.6) -> str:
    from . import ollama_client
    return ollama_client._generate(prompt, system=system, temperature=temperature)


_SYS_TWO_HOST = (
    "You are a scriptwriter for a short, lively two-host podcast in the style of "
    "an audio explainer. The hosts are warm, smart, and conversational — they "
    "react to each other, ask natural questions, and avoid corporate filler. "
    "They never invent facts beyond the source material."
)
_SYS_SOLO = (
    "You are a scriptwriter for a tight, engaging single-voice audio briefing. "
    "Clear, warm, and spoken-word — no headings, no bullet symbols read aloud, "
    "no invented facts."
)


def generate_script(text: str, style: str = "two_host", title: str = "") -> List[Line]:
    """Convert source *text* into a list of spoken Lines in the given *style*.

    Uses the local Ollama model when reachable; otherwise falls back to a
    deterministic extractor so output is always produced.
    """
    if style not in STYLES:
        style = "two_host"
    text = (text or "").strip()
    if not text:
        return []

    if _llm_available():
        try:
            raw = _llm_generate(_script_prompt(text, style, title),
                                _SYS_TWO_HOST if style == "two_host" else _SYS_SOLO)
            lines = _parse_script(raw, style)
            if len(lines) >= 3:
                return lines
            log.warning("LLM script too short (%d lines) — using fallback.", len(lines))
        except Exception as exc:  # noqa: BLE001
            log.warning("LLM script generation failed (%s) — using fallback.", exc)

    return _fallback_script(text, style, title)


def _script_prompt(text: str, style: str, title: str) -> str:
    head = f'The episode is about: "{title}".\n\n' if title else ""
    src = text[:9000]
    if style == "two_host":
        return (
            head +
            f"Two podcast hosts, A ({HOST_A_NAME}) and B ({HOST_B_NAME}), break "
            "down the briefing below for the listener in an engaging ~3-5 minute "
            "episode. \n\n"
            "CRITICAL FRAMING:\n"
            "- The hosts are NEUTRAL commentators. They are NOT the people in the "
            "document and must NOT role-play or act out any meeting.\n"
            "- The briefing prepares a person for a real conversation. Refer to "
            "that person and the other party in the THIRD PERSON (e.g. 'he should "
            "open by...'). Use names/roles from the source if given.\n"
            "- Do NOT invent what the other party will say or do, and do not "
            "promise outcomes. Coach the strategy; don't script the other side.\n"
            "- Cover the key points: who the contact is, why it matters, the game "
            "plan, smart questions, and pitfalls. End with a crisp takeaway.\n\n"
            "OUTPUT FORMAT — strict. One line per speaker turn, prefixed exactly:\n"
            "A: <what host A says>\n"
            "B: <what host B says>\n"
            "No stage directions, no markdown, no other prefixes. 12-30 turns.\n\n"
            f"SOURCE MATERIAL:\n{src}"
        )
    if style == "coach":
        return (
            head +
            "Rewrite the material below as a direct, motivating spoken briefing "
            "addressed to 'you' (second person), like a coach prepping someone. "
            "Warm, confident, concrete. 6-14 short paragraphs. Plain spoken "
            "sentences only — no headings, no bullet characters.\n\n"
            f"SOURCE MATERIAL:\n{src}"
        )
    return (
        head +
        "Rewrite the material below as a tight single-narrator audio briefing. "
        "Engaging, clear, spoken-word. 6-14 short paragraphs. No headings, no "
        "bullet characters, no meta commentary.\n\n"
        f"SOURCE MATERIAL:\n{src}"
    )


_TURN_RE = re.compile(r"^\s*(?:\*\*)?\s*([ABab])\s*(?:\*\*)?\s*[:\-]\s*(.+)$")


def _parse_script(raw: str, style: str) -> List[Line]:
    """Parse LLM output into Lines. Two-host expects 'A:'/'B:' prefixes; solo
    styles split into paragraphs read by the narrator."""
    raw = (raw or "").strip()
    if not raw:
        return []
    if style == "two_host":
        lines: List[Line] = []
        for row in raw.splitlines():
            m = _TURN_RE.match(row)
            if not m:
                # Continuation of the previous turn (wrapped line).
                if lines and row.strip():
                    lines[-1].text += " " + row.strip()
                continue
            speaker = m.group(1).upper()
            spoken = _clean_spoken(m.group(2))
            if spoken:
                lines.append(Line(speaker, spoken))
        return lines
    # Solo styles: paragraphs.
    paras = [p.strip() for p in re.split(r"\n\s*\n", raw) if p.strip()]
    return [Line("N", _clean_spoken(p)) for p in paras if _clean_spoken(p)]


def _clean_spoken(s: str) -> str:
    """Strip markdown / symbols that shouldn't be read aloud."""
    s = re.sub(r"[*_`#>|]", " ", s)            # md emphasis / headers / tables
    s = re.sub(r"\[(.*?)\]\(.*?\)", r"\1", s)   # links -> link text
    s = re.sub(r"https?://\S+", "", s)          # bare urls
    s = re.sub(r"\s+", " ", s).strip()
    return s


def _fallback_script(text: str, style: str, title: str) -> List[Line]:
    """Deterministic script from headings + bullets (no LLM needed)."""
    chunks: List[str] = []
    for row in text.splitlines():
        r = row.strip()
        if not r:
            continue
        r = re.sub(r"^#{1,6}\s*", "", r)        # headings
        r = re.sub(r"^[-*+]\s+", "", r)         # bullets
        r = re.sub(r"^\d+\.\s+", "", r)         # numbered
        r = _clean_spoken(r)
        if len(r) >= 3 and not set(r) <= set("-=_ "):
            chunks.append(r)
    chunks = chunks[:60]
    if not chunks:
        return []

    if style == "two_host":
        lines = [Line("A", f"Welcome back. Today we're breaking down "
                            f"{title or 'your prep'}."),
                 Line("B", "Love it — let's get into the highlights.")]
        for i, c in enumerate(chunks):
            lines.append(Line("A" if i % 2 == 0 else "B", c))
        lines.append(Line("A", "That's the rundown — go in prepared and confident."))
        lines.append(Line("B", "You've got this. Talk soon."))
        return lines

    intro = f"Here's your briefing on {title}." if title else "Here's your briefing."
    return [Line("N", intro)] + [Line("N", c) for c in chunks] + \
           [Line("N", "That's the rundown. Go in prepared and confident.")]


def render_script(lines: List[Line]) -> str:
    """Human-readable transcript (for saving alongside the audio)."""
    name = {"A": HOST_A_NAME, "B": HOST_B_NAME, "N": "Narrator"}
    return "\n\n".join(f"{name.get(l.speaker, l.speaker)}: {l.text}" for l in lines)


# ===========================================================================
# Stage 2 — synthesis
# ===========================================================================

def _ffmpeg() -> Optional[str]:
    return shutil.which("ffmpeg")


def _edge_available() -> bool:
    import importlib.util
    return importlib.util.find_spec("edge_tts") is not None


async def _edge_save(text: str, voice: str, path: Path) -> None:
    import edge_tts
    await edge_tts.Communicate(text, voice).save(str(path))


def _render_edge(text: str, voice: str, path: Path) -> bool:
    try:
        asyncio.run(_edge_save(text, voice, path))
        return path.exists() and path.stat().st_size > 0
    except Exception as exc:  # noqa: BLE001
        log.warning("edge-tts failed for voice %s (%s).", voice, exc)
        return False


def _render_sapi(text: str, speaker: str, path: Path) -> bool:
    """Offline Windows SAPI fallback. Renders a WAV via System.Speech."""
    if os.name != "nt":
        return False
    voice = _SAPI_VOICE.get(speaker, "David")
    txt_file = path.with_suffix(".txt")
    try:
        txt_file.write_text(text, encoding="utf-8")
        ps = (
            "Add-Type -AssemblyName System.Speech; "
            "$s = New-Object System.Speech.Synthesis.SpeechSynthesizer; "
            f"try {{ $s.SelectVoiceByHints([System.Speech.Synthesis.VoiceGender]::"
            f"{'Female' if speaker == 'B' else 'Male'}) }} catch {{}}; "
            f"$t = Get-Content -Raw -Encoding UTF8 '{txt_file}'; "
            f"$s.SetOutputToWaveFile('{path}'); $s.Speak($t); $s.Dispose()"
        )
        subprocess.run(["powershell", "-NoProfile", "-NonInteractive", "-Command", ps],
                       check=True, timeout=120,
                       stdout=subprocess.DEVNULL, stderr=subprocess.DEVNULL)
        return path.exists() and path.stat().st_size > 0
    except Exception as exc:  # noqa: BLE001
        log.warning("SAPI render failed (%s).", exc)
        return False
    finally:
        txt_file.unlink(missing_ok=True)


def _normalize(seg: Path, out_wav: Path) -> bool:
    """Decode any segment to a uniform 24 kHz mono WAV so segments concat cleanly."""
    ff = _ffmpeg()
    if not ff:
        return False
    try:
        subprocess.run([ff, "-y", "-i", str(seg), "-ar", "24000", "-ac", "1",
                        str(out_wav)], check=True,
                       stdout=subprocess.DEVNULL, stderr=subprocess.DEVNULL)
        return out_wav.exists() and out_wav.stat().st_size > 0
    except Exception as exc:  # noqa: BLE001
        log.warning("ffmpeg normalize failed (%s).", exc)
        return False


def _silence(path: Path, seconds: float = 0.35) -> bool:
    ff = _ffmpeg()
    if not ff:
        return False
    try:
        subprocess.run([ff, "-y", "-f", "lavfi", "-i",
                        "anullsrc=r=24000:cl=mono", "-t", str(seconds), str(path)],
                       check=True, stdout=subprocess.DEVNULL, stderr=subprocess.DEVNULL)
        return path.exists()
    except Exception:  # noqa: BLE001
        return False


def _concat_to_mp3(wavs: List[Path], out_path: Path) -> None:
    ff = _ffmpeg()
    if not ff:
        raise RuntimeError("ffmpeg not found on PATH — cannot assemble audio.")
    listfile = out_path.parent / "_concat_list.txt"
    listfile.write_text("".join(f"file '{w.as_posix()}'\n" for w in wavs),
                        encoding="utf-8")
    try:
        subprocess.run([ff, "-y", "-f", "concat", "-safe", "0", "-i", str(listfile),
                        "-c:a", "libmp3lame", "-q:a", "4", str(out_path)],
                       check=True, stdout=subprocess.DEVNULL, stderr=subprocess.DEVNULL)
    finally:
        listfile.unlink(missing_ok=True)


def synthesize(lines: List[Line], out_path: Path,
               voices: Optional[dict] = None) -> Path:
    """Render *lines* to a single MP3 at *out_path*. edge-tts first, SAPI fallback.

    Raises RuntimeError if no audio could be produced.
    """
    if not lines:
        raise RuntimeError("No script lines to synthesize.")
    out_path = Path(out_path)
    out_path.parent.mkdir(parents=True, exist_ok=True)
    voices = voices or {"A": VOICE_A, "B": VOICE_B, "N": VOICE_NARRATOR}
    use_edge = _edge_available()

    tmp = Path(tempfile.mkdtemp(prefix="jobbot_pod_"))
    try:
        pause = tmp / "pause.wav"
        have_pause = _silence(pause)
        norm_wavs: List[Path] = []
        for i, ln in enumerate(lines):
            seg = tmp / f"seg_{i:04d}"
            rendered: Optional[Path] = None
            if use_edge:
                mp3 = seg.with_suffix(".mp3")
                if _render_edge(ln.text, voices.get(ln.speaker, VOICE_A), mp3):
                    rendered = mp3
            if rendered is None:
                wav = seg.with_suffix(".sapi.wav")
                if _render_sapi(ln.text, ln.speaker, wav):
                    rendered = wav
            if rendered is None:
                log.warning("Skipping line %d — no engine produced audio.", i)
                continue
            norm = seg.with_suffix(".norm.wav")
            if _normalize(rendered, norm):
                norm_wavs.append(norm)
                if have_pause:
                    norm_wavs.append(pause)

        if not norm_wavs:
            raise RuntimeError(
                "No audio segments produced. Install edge-tts (`pip install "
                "edge-tts`) and ensure internet access, or run on Windows for "
                "offline SAPI voices.")
        _concat_to_mp3(norm_wavs, out_path)
        return out_path
    finally:
        shutil.rmtree(tmp, ignore_errors=True)


# ===========================================================================
# Orchestrator
# ===========================================================================

def make_podcast(source: str, out_path: Optional[str] = None,
                 style: str = "two_host", title: str = "",
                 audio: bool = True, voices: Optional[dict] = None) -> dict:
    """End-to-end: read *source* (file path or raw text) -> script -> MP3.

    Returns {style, title, n_lines, script_path, audio_path}. When *audio* is
    False, only the transcript is written (no TTS).
    """
    src_path = Path(source) if isinstance(source, str) else None
    if src_path and src_path.exists():
        text = src_path.read_text(encoding="utf-8")
        if not title:
            title = src_path.stem.replace("_", " ")
        default_stem = src_path.with_suffix("")
    else:
        text = str(source)
        default_stem = Path(out_path).with_suffix("") if out_path else Path("podcast")

    out = Path(out_path) if out_path else default_stem.with_name(
        default_stem.name + "_podcast.mp3")

    lines = generate_script(text, style=style, title=title)
    if not lines:
        raise RuntimeError("Could not build a script from the source text.")

    script_path = out.with_suffix(".script.txt")
    script_path.parent.mkdir(parents=True, exist_ok=True)
    script_path.write_text(render_script(lines), encoding="utf-8")

    result = {"style": style, "title": title, "n_lines": len(lines),
              "script_path": str(script_path), "audio_path": None}
    if audio:
        result["audio_path"] = str(synthesize(lines, out, voices=voices))
    return result
