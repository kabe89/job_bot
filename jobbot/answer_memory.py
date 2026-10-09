"""Cross-job answer memory.

Remembers answers between applications so the boilerplate questions
(work auth, sponsorship, identity, EEO) auto-fill verbatim and only the
genuinely company-specific essays are re-generated. A superset of the
legacy data/answer_bank.json.

Two kinds of answer:
  * static   — identity / screening / EEO. Stored once, reused verbatim.
  * tailored — company/role essays. The stored answer is a starting point;
               the live apply re-generates these per company via the AI
               answerer, then remembers the user's edits as the new template.
"""
from __future__ import annotations

import json
import logging
import re
from dataclasses import dataclass, asdict
from datetime import datetime, timezone
from pathlib import Path
from typing import List, Optional

from .apply_questions import FormQuestion, classify

log = logging.getLogger("jobbot.answer_memory")

# Company-name-ish tokens get stripped so "...work at Acme" collapses with
# "...work at Genentech". Keep the stable interrogative skeleton.
_STOPWORDS = {"the", "a", "an", "at", "for", "to", "do", "you", "your", "our",
              "us", "this", "that", "want", "would", "like", "in", "of", "and"}

# Prepositions that introduce a company/product name ("...at Acme", "...for Google").
_COMPANY_INTRO = {"at", "for", "with", "by", "from"}


def canonical_key(question_text: str) -> str:
    """Normalize a question to a stable semantic slot key.

    Strips proper-noun tokens (company/product names) so that
    "work at Acme Corp" and "work at Genentech" collapse to the same slot.
    Only a Title-Case token immediately following a company-introducing
    preposition ("at"/"for"/...) is treated as a proper noun and dropped — so
    "First Name"/"Last Name"/"Email Address" keep their second word and
    don't collide with "First Language"/"Last Employer", and a plain stopword
    ("a"/"the") before a Title-Case word ("a Masters Degree") keeps it.
    """
    raw_words = re.findall(r"[A-Za-z]+", question_text)
    filtered = []
    dropped_prev = False  # was the previous token dropped as a proper noun?
    for i, w in enumerate(raw_words):
        if i > 0 and w[0].isupper():
            prev = raw_words[i - 1].lower()
            # Drop a Title-Case token that opens a company phrase ("at Acme")
            # and any consecutive Title-Case tokens continuing it ("Acme Corp").
            if prev in _COMPANY_INTRO or dropped_prev:
                dropped_prev = True
                continue
        dropped_prev = False
        filtered.append(w.lower())
    kept = [w for w in filtered if w not in _STOPWORDS]
    return " ".join(kept)


# Markers that flip a question's meaning. "Are you authorized to work" and "do
# you require sponsorship to work" are near-identical as text and take OPPOSITE
# answers, so similarity alone must never be allowed to answer across them.
_POLARITY = (
    r"\bnot\b", r"\bno\b", r"\bnever\b", r"\bwithout\b", r"\bunable\b",
    r"\bdecline\b", r"\brequire\s+sponsorship\b", r"\bneed\s+sponsorship\b",
    r"\bsponsorship\b", r"\bunwilling\b", r"\bexcept\b",
)


def polarity(text: str) -> frozenset:
    """The inversion markers present in `text`."""
    low = (text or "").lower()
    return frozenset(p for p in _POLARITY if re.search(p, low))


def token_similarity(a: str, b: str) -> float:
    """Jaccard overlap of the two canonical keys' word sets.

    Deterministic and free -- it catches most rewordings with no Ollama call, so
    the embedding tier is only reached when this is inconclusive.
    """
    ta = set(canonical_key(a).split())
    tb = set(canonical_key(b).split())
    if not ta or not tb:
        return 0.0
    return len(ta & tb) / len(ta | tb)


# Kinds whose answer is a fact about the CANDIDATE, so it reads the same on
# every form and can be replayed verbatim.
#
# "file" is deliberately NOT here. A file answer is a PATH to a document
# tailored for one job, which makes it the most job-specific answer there is,
# not the least. It was in this set once, and the result was a live Workday kit
# (job 3311, Employer) whose Resume/CV pointed at
# `5118_acme-corp_..._resume.pdf`: the remembered path won the lookup
# in `answer_questions` and the resume build_package had just generated for
# that job was never applied. Every application after the first would have
# carried the first one's resume.
_STATIC_KINDS = {"identity", "screening", "eeo"}


def classify_kind(q: FormQuestion) -> str:
    """Return 'static' (reuse verbatim) or 'tailored' (regenerate per job)."""
    kind = q.kind or classify(q)
    return "static" if kind in _STATIC_KINDS else "tailored"


@dataclass
class MemoryEntry:
    key: str
    kind: str
    answer: str
    source: str = "user"
    last_used: str = ""


class AnswerMemory:
    def __init__(self, path: str):
        self.path = Path(path)
        self._entries: List[MemoryEntry] = self._load()

    def _load(self) -> List[MemoryEntry]:
        if not self.path.exists():
            return []
        try:
            raw = json.loads(self.path.read_text(encoding="utf-8"))
        except Exception as e:  # noqa: BLE001
            log.warning("Could not read answer memory %s: %s", self.path, e)
            return []
        out: List[MemoryEntry] = []
        for d in raw:
            if isinstance(d, dict) and d.get("key"):
                out.append(MemoryEntry(**{k: d.get(k, "") for k in
                                          ("key", "kind", "answer", "source", "last_used")}))
        return out

    def _save(self) -> None:
        self.path.parent.mkdir(parents=True, exist_ok=True)
        self.path.write_text(
            json.dumps([asdict(e) for e in self._entries], indent=2, ensure_ascii=False),
            encoding="utf-8")

    def lookup(self, question: FormQuestion) -> Optional[MemoryEntry]:
        key = canonical_key(question.text)
        for e in self._entries:
            if e.key == key:
                return e
        return None

    def static_answer_for(self, question: FormQuestion) -> Optional[str]:
        """The exact remembered answer for `question`, if it is safe to replay.

        Two guards, both here so no call site can apply only one:
          - the question in hand must not be a FILE upload, and
          - the stored entry must claim static.

        The file guard reads the question, not the entry, and that is the
        point: rows written before "file" left `_STATIC_KINDS` still say
        `kind: "static"` on disk, so trusting the row alone would keep
        replaying one job's resume path into every later kit. It is scoped to
        file uploads on purpose. A deliberately remembered answer to a question
        the classifier does not recognise ("Favourite colour?") is still the
        user's own curated answer and must keep working.
        """
        if (question.qtype or "") == "file" or (question.kind or "") == "file":
            return None
        hit = self.lookup(question)
        return hit.answer if hit and hit.kind == "static" else None

    def remember(self, question: FormQuestion, answer: str, source: str = "user") -> None:
        if not answer.strip():
            return
        key = canonical_key(question.text)
        kind = classify_kind(question)
        now = datetime.now(timezone.utc).replace(tzinfo=None).isoformat()
        for e in self._entries:
            if e.key == key:
                e.answer, e.kind, e.source, e.last_used = answer, kind, source, now
                break
        else:
            self._entries.append(MemoryEntry(key=key, kind=kind, answer=answer,
                                             source=source, last_used=now))
        self._save()

    def fuzzy_lookup(self, question: FormQuestion, high: float, review: float):
        """Best non-exact match for `question`, or None.

        Returns `(entry, confident)`. `confident` False means the caller must
        hold the answer for confirmation rather than submit it.

        Never matches a tailored question (those regenerate per company) and
        never matches across a kind boundary.
        """
        kind = classify_kind(question)
        if kind != "static":
            return None

        q_polarity = polarity(canonical_key(question.text))
        best, best_score = None, 0.0
        for e in self._entries:
            if e.kind != kind:
                continue
            score = token_similarity(question.text, e.key)
            if score > best_score:
                best, best_score = e, score

        if best is None or best_score < review:
            return None

        # An inversion marker on one side and not the other means these are
        # opposite questions however similar the words are. Demote, never drop:
        # the user still gets it as a one-click confirmation.
        inverted = q_polarity != polarity(best.key)
        confident = best_score >= high and not inverted
        return best, confident


def calibrate(pairs: List[dict]) -> dict:
    """Score labelled question pairs and report whether a threshold separates them.

    `pairs` are {"a": str, "b": str, "equivalent": bool}. Returns the lowest
    equivalent-pair score, the highest opposite-pair score, whether a gap exists,
    and a suggested threshold sitting inside it.

    An unseparable result is a real answer, not a failure: per the design, it is
    the signal to ship with the semantic tier off and the free token tier only.
    """
    equivalent, opposite = [], []
    for p in pairs:
        score = token_similarity(p["a"], p["b"])
        # An inversion marker on one side only makes the pair opposite however
        # similar the words are -- the same rule fuzzy_lookup applies.
        if polarity(p["a"]) != polarity(p["b"]):
            score = 0.0
        (equivalent if p.get("equivalent") else opposite).append(score)

    equivalent_min = min(equivalent) if equivalent else 0.0
    opposite_max = max(opposite) if opposite else 0.0
    separable = equivalent_min > opposite_max
    suggested = round((equivalent_min + opposite_max) / 2, 2) if separable else 0.0
    if separable and suggested <= opposite_max:
        suggested = round(opposite_max + 0.01, 2)
    return {"equivalent_min": equivalent_min, "opposite_max": opposite_max,
            "separable": separable, "suggested_high": suggested}
