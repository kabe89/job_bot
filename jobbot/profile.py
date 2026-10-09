# jobbot/profile.py
"""Auto-derived structured candidate profile — the single source of truth for
semantic ranking and query expansion.

`build_profile` extracts a `CandidateProfile` from the base resume via Ollama;
`load_profile` caches it (keyed on a resume hash) at
`settings.candidate_profile_path`, attaches its embedding, and rebuilds only
when the resume changes. Fail-open: if Ollama is down, a minimal profile is
synthesized from the configured keywords/tags so the rest of the system runs.
"""
from __future__ import annotations

import hashlib
import json
import logging
import re
from dataclasses import asdict, dataclass, field
from datetime import datetime
from pathlib import Path
from typing import List, Optional

from . import gemini_client as _g          # reuse SYSTEM string + JSON parser
from . import ollama_client as oc
from .config import settings
from .skills import load_skills, skill_names, skills_fingerprint

log = logging.getLogger("jobbot.profile")

_PROFILE_PROMPT = """Extract a structured candidate profile from this resume.
Return STRICT JSON only (no prose outside JSON), with this exact shape:
{{
  "skills": ["concrete skills, tools, techniques"],
  "role_titles": ["canonical target job titles the candidate fits"],
  "role_archetypes": ["broader synonyms / adjacent titles for the same work"],
  "domains": ["scientific / industry domains"],
  "seniority": "junior|mid|senior|staff|unknown",
  "dealbreakers": ["role types the candidate should avoid"],
  "summary": "2-3 sentence narrative of who this candidate is and the roles they fit"
}}
Base everything ONLY on the resume. Do not invent.

`dealbreakers` means KINDS OF WORK ONLY - the function of the job. Never
geography, remote/onsite, salary, company size, or seniority: those are scored
separately, and a dealbreaker here hard-caps the match. "Positions outside the
Seattle area" is WRONG. "Roles whose primary function is synthetic medicinal
chemistry" is right. A stated location preference in the resume is a preference,
NOT a dealbreaker.

RESUME:
{resume}"""


# A dealbreaker hard-caps the fit judge at 0.1, so anything scored ELSEWHERE
# must never appear as one - it would double-count and silently bury good jobs.
# Location is already handled by user_locations.txt and Job.is_local; salary and
# seniority have their own paths. The model does not reliably respect that from
# the prompt alone (it once emitted "Positions outside the Pacific Northwest",
# which would have capped Cambridge, San Diego and San Francisco), so the rule is
# enforced here in code.
_NON_ROLE_DEALBREAKER = re.compile(
    r"""(
        \b(remote|onsite|on-site|hybrid|relocat\w*|commut\w*|geograph\w*)\b
      | \b(?:outside|beyond|away\s+from|not\s+(?:in|near|within))\b[^.]{0,40}
        \b(region|area|state|city|county|country|metro|capital)\b
      | \b(?:located|location)\b[^.]{0,30}\b(outside|not)\b
      | \bwithin\s+\w+\s+(?:miles|minutes|hours)\b
      # salary, in either word order: "salary below X" and "less than market salary"
      | \b(salary|compensation|pay|wage|paying|stipend)\b[^.]{0,40}
        \b(below|less\s+than|under|market|minimum)\b
      | \b(below|less\s+than|under|beneath)\b[^.]{0,40}
        \b(salary|compensation|pay|wage|market\s+rate|stipend)\b
    )""",
    re.I | re.X,
)


def _drop_non_role_dealbreakers(entries) -> List[str]:
    """Keep only dealbreakers that describe a KIND OF WORK.

    Returns a new list; entries that are blank, or that describe location,
    salary or similar separately-scored axes, are dropped with a log line so a
    silent filter never looks like a model that simply behaved.
    """
    kept: List[str] = []
    for raw in entries or []:
        s = str(raw or "").strip()
        if not s:
            continue
        if _NON_ROLE_DEALBREAKER.search(s):
            log.warning("Dropping non-role dealbreaker (scored elsewhere): %r", s)
            continue
        kept.append(s)
    return kept


@dataclass
class CandidateProfile:
    skills: List[str]
    role_titles: List[str]
    role_archetypes: List[str]
    domains: List[str]
    seniority: str
    dealbreakers: List[str]
    summary: str
    source_resume_hash: str
    built_at: str
    embedding: Optional[List[float]] = field(default=None)

    def seed_text(self) -> str:
        """Text embedded to represent the candidate."""
        parts = [self.summary] + self.skills + self.role_titles + self.role_archetypes + self.domains
        return " ; ".join(p for p in parts if p)

    def to_dict(self) -> dict:
        return asdict(self)

    @classmethod
    def from_dict(cls, d: dict) -> "CandidateProfile":
        return cls(
            skills=_strlist(d.get("skills")),
            role_titles=_strlist(d.get("role_titles")),
            role_archetypes=_strlist(d.get("role_archetypes")),
            domains=_strlist(d.get("domains")),
            seniority=str(d.get("seniority") or "unknown"),
            dealbreakers=_drop_non_role_dealbreakers(_strlist(d.get("dealbreakers"))),
            summary=str(d.get("summary") or ""),
            source_resume_hash=str(d.get("source_resume_hash") or ""),
            built_at=str(d.get("built_at") or ""),
            embedding=(list(d["embedding"]) if d.get("embedding") else None),
        )


def _strlist(v) -> List[str]:
    if not v:
        return []
    if isinstance(v, str):
        return [v.strip()] if v.strip() else []
    return [str(x).strip() for x in v if str(x).strip()]


def resume_fingerprint(resume_text: str) -> str:
    base = (resume_text or "").encode("utf-8")
    return hashlib.sha256(base + skills_fingerprint().encode("utf-8")).hexdigest()


def _read_resume() -> str:
    from .resume import load_resume
    try:
        return load_resume(settings.base_resume_path)
    except Exception as e:  # noqa: BLE001
        # Fully fail-open: a missing/unreadable/undecodable resume must not crash
        # the scrape cycle (load_profile falls back to a minimal profile).
        log.warning("Could not read base resume at %s (%s) — profile will be minimal.",
                    settings.base_resume_path, e)
        return ""


def _safe_embed(text: str) -> Optional[List[float]]:
    from . import embeddings
    try:
        return embeddings.embed(text)
    except Exception as e:  # noqa: BLE001
        log.warning("profile embedding failed (%s) — semantic ranking degraded.", e)
        return None


def build_profile(resume_text: str) -> CandidateProfile:
    """Extract a structured profile from the resume via Ollama. Raises on a hard
    failure (server unreachable) so `load_profile` can fall back."""
    out = oc._generate(
        _PROFILE_PROMPT.format(resume=(resume_text or "")[:10000]),
        system=_g.SYSTEM_RESUME_TAILOR, temperature=0.2, include_profile=False)
    data = _g._parse_json_block(out) or {}
    prof = CandidateProfile.from_dict(data)
    prof.source_resume_hash = resume_fingerprint(resume_text)
    prof.built_at = datetime.utcnow().isoformat()
    return prof


def _minimal_profile(resume_text: str) -> CandidateProfile:
    """Fallback when Ollama is unavailable: synthesize from configured keywords."""
    from .user_prefs import load_tags
    try:
        tags = load_tags()
    except Exception:  # noqa: BLE001
        tags = []
    skills = list(dict.fromkeys((settings.keywords or []) + (tags or []))) or ["biochemistry"]
    return CandidateProfile(
        skills=skills, role_titles=[], role_archetypes=[], domains=[],
        seniority="unknown", dealbreakers=list(settings.excludes or []),
        summary=" ".join(skills),
        source_resume_hash=resume_fingerprint(resume_text),
        built_at=datetime.utcnow().isoformat())


def _save(prof: CandidateProfile, path: Path) -> None:
    try:
        path.parent.mkdir(parents=True, exist_ok=True)
        path.write_text(json.dumps(prof.to_dict(), indent=2), encoding="utf-8")
    except Exception as e:  # noqa: BLE001
        log.warning("Could not write candidate profile to %s: %s", path, e)


def _merge_verified_skills(prof: "CandidateProfile") -> None:
    """Fold hand-authored verified skill names into the profile skill set so they
    influence the semantic embedding used for matching. Idempotent."""
    try:
        names = skill_names(load_skills(), min_rank=2)
    except Exception as e:  # noqa: BLE001
        log.warning("Could not merge verified skills (%s).", e)
        return
    have = {s.lower() for s in prof.skills}
    prof.skills = prof.skills + [n for n in names if n.lower() not in have]


def load_profile(force: bool = False) -> CandidateProfile:
    """Return the cached profile, rebuilding when the resume changed or `force`.
    Always returns a usable profile (minimal one if Ollama is down)."""
    resume_text = _read_resume()
    fp = resume_fingerprint(resume_text)
    path = Path(settings.candidate_profile_path)

    if not force and path.exists():
        try:
            cached = json.loads(path.read_text(encoding="utf-8"))
            if cached.get("source_resume_hash") == fp:
                prof = CandidateProfile.from_dict(cached)
                if prof.embedding:
                    return prof
                _merge_verified_skills(prof)
                prof.embedding = _safe_embed(prof.seed_text())
                _save(prof, path)
                return prof
        except Exception as e:  # noqa: BLE001
            log.warning("Could not read cached profile (%s) — rebuilding.", e)

    try:
        prof = build_profile(resume_text)
    except Exception as e:  # noqa: BLE001
        log.warning("Profile build failed (%s) — using minimal profile.", e)
        prof = _minimal_profile(resume_text)
    _merge_verified_skills(prof)
    prof.embedding = _safe_embed(prof.seed_text())
    _save(prof, path)
    return prof
