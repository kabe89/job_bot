"""Reconcile the planned (API ground-truth) questions against the fields
actually rendered on the live application page. Pure function — no I/O.

This is the 'self-healing' core: planned answers are matched to real fields,
fields planned-but-absent are flagged missing_on_page, and fields present-
but-unplanned are flagged extra_on_page so the live flow surfaces them.
"""
from __future__ import annotations

import re
from dataclasses import dataclass
from typing import List

from .apply_questions import FormQuestion


@dataclass
class FieldRecon:
    text: str
    status: str          # matched | missing_on_page | extra_on_page
    value: str
    live_label: str


def _norm(s: str) -> str:
    return re.sub(r"[^a-z0-9]+", "", (s or "").lower())


def reconcile(planned: List[FormQuestion], live_labels: List[str]) -> List[FieldRecon]:
    live_norm = {_norm(lbl): lbl for lbl in live_labels if lbl.strip()}
    out: List[FieldRecon] = []
    used: set[str] = set()

    for q in planned:
        key = _norm(q.text)
        match = None
        if key in live_norm and key not in used:
            match = live_norm[key]
        else:
            # containment fallback: planned label inside a live label or vice versa
            for nk, lbl in live_norm.items():
                if nk in used:
                    continue
                if key and (key in nk or nk in key):
                    match = lbl
                    break
        if match is not None:
            used.add(_norm(match))
            out.append(FieldRecon(text=q.text, status="matched",
                                  value=q.answer, live_label=match))
        else:
            out.append(FieldRecon(text=q.text, status="missing_on_page",
                                  value=q.answer, live_label=""))

    for nk, lbl in live_norm.items():
        if nk not in used:
            out.append(FieldRecon(text="", status="extra_on_page",
                                  value="", live_label=lbl))
    return out
