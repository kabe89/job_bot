"""Single cached source of the "applicant context" block injected into every AI
tailoring prompt: the freeform profile markdown (settings.profile_path) plus the
verified-skills inventory parsed from settings.skills_path. All three AI provider
clients delegate here so skills reach Gemini, Claude, and Ollama identically."""
from __future__ import annotations

import logging
from functools import lru_cache
from pathlib import Path

from .config import settings
from .skills import load_skills, inventory_markdown
from . import resume_facts

log = logging.getLogger("jobbot.profile_context")


@lru_cache(maxsize=1)
def applicant_context() -> str:
    parts = []
    p = Path(settings.profile_path)
    if p.exists():
        try:
            parts.append(p.read_text(encoding="utf-8")[:8000])
        except Exception as e:  # noqa: BLE001
            log.info("profile.md read failed (%s).", e)
    inv = inventory_markdown(load_skills())
    if inv:
        parts.append(inv)
    facts = resume_facts.load()
    if facts.is_reviewed() and facts.employment:
        lines = ["VERIFIED EMPLOYMENT HISTORY (never contradict these):"]
        for e in facts.employment:
            span = f"{e.start} to {'present' if e.current else e.end}"
            loc = f", {e.location}" if e.location else ""
            lines.append(f"- {e.title}, {e.employer}{loc} ({span})")
            for b in e.bullets[:4]:
                lines.append(f"    * {b}")
        for ed in facts.education:
            lines.append(f"- {ed.degree} {ed.field}, {ed.school} "
                         f"({ed.start} to {ed.end})")
        parts.append("\n".join(lines))
    return "\n\n".join(parts).strip()


def refresh() -> None:
    applicant_context.cache_clear()
