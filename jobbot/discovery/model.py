# jobbot/discovery/model.py
"""The canonical value object every discoverer produces."""
from __future__ import annotations

from dataclasses import asdict, dataclass, field
from typing import List, Optional, Tuple


@dataclass
class DiscoveredTarget:
    provider: str            # greenhouse | lever | ashby | workday
    key: str                 # slug (gh/lever/ashby) or "tenant/board" (workday)
    display_name: str
    source: str              # harvest | websearch | llm
    wd_host: Optional[str] = None   # workday only, e.g. "wd1"
    valid: bool = False
    fit_score: float = 0.0
    fit_reason: str = ""
    sample_titles: List[str] = field(default_factory=list)
    job_count: int = 0
    first_seen: str = ""
    status: str = "pending"  # pending | approved | rejected

    def coord(self) -> Tuple[str, str]:
        return (self.provider, self.key)

    def to_dict(self) -> dict:
        return asdict(self)

    @classmethod
    def from_dict(cls, d: dict) -> "DiscoveredTarget":
        return cls(
            provider=str(d.get("provider", "")),
            key=str(d.get("key", "")),
            display_name=str(d.get("display_name", "")),
            source=str(d.get("source", "")),
            wd_host=d.get("wd_host"),
            valid=bool(d.get("valid", False)),
            fit_score=float(d.get("fit_score", 0.0) or 0.0),
            fit_reason=str(d.get("fit_reason", "")),
            sample_titles=list(d.get("sample_titles") or []),
            job_count=int(d.get("job_count", 0) or 0),
            first_seen=str(d.get("first_seen", "")),
            status=str(d.get("status", "pending")),
        )
