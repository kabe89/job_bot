"""Parse the hand-authored master skills file into structured, level-ranked
skills and expose them for (a) semantic matching and (b) job-aware tailoring.

The file (settings.skills_path) is loosely formatted: category banners like
"A. COMPUTATIONAL CHEMISTRY / CADD" between "====" rules, then bullet lines
"- Skill name: <level/notes>". Levels may be explicit (Expert/Proficient/Some/
Learning/None) or informal ("Yep", "Nope", "a ton"). We extract the strongest
declared level and keep the freeform notes.
"""
from __future__ import annotations

import hashlib
import logging
import re
from dataclasses import dataclass
from pathlib import Path
from typing import List, Optional

from .config import settings

log = logging.getLogger("jobbot.skills")

LEVEL_RANK = {"Expert": 4, "Proficient": 3, "Some": 2, "Learning": 1, "None": 0}

# Informal phrases mapped to a canonical level (checked case-insensitively).
_INFORMAL = [
    (("a ton", "a lot", "alot", "yep", "yes", "all of my", "over 6 years",
      "specialty", "proficient"), "Proficient"),
    (("some", "a bit", "briefly", "basic", "learning", "still learning"), "Some"),
    (("nope", "none", "no ", "not ", "never"), "None"),
]
_CANONICAL = ("Expert", "Proficient", "Some", "Learning", "None")
_CATEGORY_RE = re.compile(r"^[A-Z]\.\s+\S")


@dataclass
class Skill:
    name: str
    level: str
    notes: str
    category: str

    @property
    def rank(self) -> int:
        return LEVEL_RANK.get(self.level, 0)


def _detect_level(rest: str) -> str:
    """Return the strongest level declared anywhere in `rest` ('' if unknown)."""
    low = rest.lower()
    best = ""
    best_rank = -1
    # Explicit canonical words (word-boundary) take priority by rank.
    for lvl in _CANONICAL:
        if re.search(rf"\b{lvl.lower()}\b", low):
            if LEVEL_RANK[lvl] > best_rank:
                best, best_rank = lvl, LEVEL_RANK[lvl]
    # Informal phrases only fill in when no canonical word was found.
    if not best:
        for phrases, lvl in _INFORMAL:
            if any(p in low for p in phrases):
                if LEVEL_RANK[lvl] > best_rank:
                    best, best_rank = lvl, LEVEL_RANK[lvl]
    return best


def _clean_notes(rest: str) -> str:
    """Strip parenthetical prompts, leading level word, and '...' scaffolding."""
    txt = re.sub(r"\([^)]*\)", " ", rest)          # drop "(which tools?)" hints
    txt = txt.replace("—", " ").replace("...", " ")
    for lvl in _CANONICAL:
        txt = re.sub(rf"\b{lvl}\b", " ", txt, flags=re.I)
    txt = re.sub(r"\bPRE-FILLED\b|\bCONFIRM\b", " ", txt, flags=re.I)
    return re.sub(r"\s+", " ", txt).strip(" .,;:-")


def parse_skills(text: str) -> List[Skill]:
    out: List[Skill] = []
    category = ""
    for raw in (text or "").splitlines():
        line = raw.rstrip()
        stripped = line.strip()
        if _CATEGORY_RE.match(stripped):
            category = stripped
            continue
        if not stripped.startswith("- "):
            continue
        body = stripped[2:]
        name, sep, rest = body.partition(":")
        if not sep:
            continue
        name = name.strip()
        if not name:
            continue
        level = _detect_level(rest) or ""
        out.append(Skill(name=name, level=level, notes=_clean_notes(rest),
                         category=category))
    return out


def load_skills(path: Optional[str] = None) -> List[Skill]:
    p = Path(path or settings.skills_path)
    if not p.exists():
        return []
    try:
        return parse_skills(p.read_text(encoding="utf-8", errors="ignore"))
    except Exception as e:  # noqa: BLE001
        log.warning("Could not read skills file %s: %s", p, e)
        return []


def skills_fingerprint(path: Optional[str] = None) -> str:
    p = Path(path or settings.skills_path)
    try:
        return hashlib.sha256(p.read_bytes()).hexdigest() if p.exists() else ""
    except Exception:  # noqa: BLE001
        return ""


def known_skills(skills: List[Skill], min_rank: int = 1) -> List[Skill]:
    return [s for s in skills if s.rank >= min_rank]


def skill_names(skills: List[Skill], min_rank: int = 2) -> List[str]:
    seen, out = set(), []
    for s in sorted(known_skills(skills, min_rank), key=lambda s: -s.rank):
        if s.name.lower() not in seen:
            seen.add(s.name.lower())
            out.append(s.name)
    return out


def inventory_markdown(skills: List[Skill]) -> str:
    """Full verified-skills inventory grouped by level (for prompt grounding)."""
    known = known_skills(skills, min_rank=1)
    if not known:
        return ""
    lines = ["VERIFIED SKILLS (self-reported ground truth — do not claim beyond this list):"]
    for lvl in ("Expert", "Proficient", "Some", "Learning"):
        names = [s.name for s in known if s.level == lvl]
        if names:
            lines.append(f"- {lvl}: " + ", ".join(names))
    return "\n".join(lines)


_SKILL_EMBED_CACHE_PATH = Path("data/skill_embeddings_cache.json")
_skill_embed_cache: dict[str, List[float]] = {}
_cache_loaded: bool = False


def _ensure_cache_loaded() -> None:
    global _skill_embed_cache, _cache_loaded
    if _cache_loaded:
        return
    _cache_loaded = True
    if _SKILL_EMBED_CACHE_PATH.exists():
        try:
            import json as _json
            _skill_embed_cache = _json.loads(_SKILL_EMBED_CACHE_PATH.read_text(encoding="utf-8"))
        except Exception:
            _skill_embed_cache = {}


def _save_cache() -> None:
    try:
        import json as _json
        _SKILL_EMBED_CACHE_PATH.parent.mkdir(parents=True, exist_ok=True)
        _SKILL_EMBED_CACHE_PATH.write_text(_json.dumps(_skill_embed_cache), encoding="utf-8")
    except Exception:
        pass


def _safe_embed(text: str) -> Optional[List[float]]:
    _ensure_cache_loaded()
    key = hashlib.sha256(text.encode("utf-8")).hexdigest()
    if key in _skill_embed_cache:
        return _skill_embed_cache[key]
    from . import embeddings
    try:
        vec = embeddings.embed(text)
        if vec:
            _skill_embed_cache[key] = vec
            _save_cache()
        return vec
    except Exception as e:  # noqa: BLE001
        log.info("skill embed failed (%s) — keyword fallback.", e)
        return None


def _cosine(a: List[float], b: List[float]) -> float:
    if not a or not b or len(a) != len(b):
        return 0.0
    num = sum(x * y for x, y in zip(a, b))
    da = sum(x * x for x in a) ** 0.5
    db = sum(y * y for y in b) ** 0.5
    return (num / (da * db)) if da and db else 0.0


def _keyword_score(skill: Skill, job_low: str) -> float:
    """Fraction of the skill's significant tokens present in the job text."""
    toks = [t for t in re.split(r"[^a-z0-9]+", skill.name.lower()) if len(t) > 2]
    if not toks:
        return 0.0
    return sum(1 for t in toks if t in job_low) / len(toks)


def relevant_skills(job_text: str, job_embedding: Optional[List[float]] = None,
                    top_n: int = 8, min_rank: int = 2) -> List[Skill]:
    """Rank verified skills by relevance to one job. Uses cosine similarity to
    the job embedding when available, else keyword overlap. Ties break on level."""
    pool = known_skills(load_skills(), min_rank=min_rank)
    if not pool:
        return []
    job_low = (job_text or "").lower()
    scored: List[tuple] = []
    if job_embedding:
        for s in pool:
            emb = _safe_embed(f"{s.name}. {s.notes}")
            sim = _cosine(emb, job_embedding) if emb else _keyword_score(s, job_low)
            scored.append((sim, s.rank, s))
    else:
        for s in pool:
            scored.append((_keyword_score(s, job_low), s.rank, s))
    scored.sort(key=lambda t: (t[0], t[1]), reverse=True)
    return [s for _sim, _rank, s in scored[:top_n]]


def relevant_markdown(job_text: str, job_embedding: Optional[List[float]] = None,
                      top_n: int = 8) -> str:
    rel = relevant_skills(job_text, job_embedding, top_n=top_n)
    if not rel:
        return ""
    parts = [f"{s.name} ({s.level})" for s in rel]
    return ("MOST RELEVANT VERIFIED SKILLS FOR THIS ROLE (emphasize these, "
            "truthfully): " + "; ".join(parts))
