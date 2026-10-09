# jobbot/embeddings.py
"""Local text embeddings via Ollama (`/api/embeddings`) + cosine similarity.

Small, dependency-light: reuses the existing `ollama_client` to start the local
server, then calls the embeddings endpoint with `settings.ollama_embed_model`
(default `nomic-embed-text`). `embed` raises on any failure so callers can fall
back to the legacy bag-of-words score.
"""
from __future__ import annotations

import logging
from typing import List

import numpy as np
import requests

from . import ollama_client as oc
from .config import settings

log = logging.getLogger("jobbot.embeddings")


def cosine(a: List[float], b: List[float]) -> float:
    """Cosine similarity. Returns 0.0 if either vector is empty or zero-norm."""
    if not a or not b:
        return 0.0
    va = np.asarray(a, dtype=float)
    vb = np.asarray(b, dtype=float)
    na = float(np.linalg.norm(va))
    nb = float(np.linalg.norm(vb))
    if na == 0.0 or nb == 0.0:
        return 0.0
    return float(np.dot(va, vb) / (na * nb))


def embed(text: str) -> List[float]:
    """Return one embedding vector for `text`. Raises RuntimeError on failure.

    Starts the local Ollama server if needed (pull=False: do NOT block pulling
    the chat model — the embed model is pulled separately via `ollama pull`).
    """
    oc.ensure_ready(pull=False)
    host = oc._host()
    model = settings.ollama_embed_model
    payload = {"model": model, "prompt": (text or "")[:8000]}
    resp = requests.post(f"{host}/api/embeddings", json=payload, timeout=oc._timeout())
    # A 404 here almost always means the embed model was never pulled (Ollama
    # replies `model "<name>" not found, try pulling it first`). Turn that into
    # an actionable message instead of a bare HTTPError so semantic ranking's
    # degradation is self-explanatory.
    if resp.status_code == 404 and "not found" in (resp.text or "").lower():
        raise RuntimeError(
            f"Ollama embed model {model!r} is not installed — run "
            f"`ollama pull {model}` (semantic ranking falls back to keywords "
            f"until then)."
        )
    resp.raise_for_status()
    vec = (resp.json() or {}).get("embedding") or []
    if not vec:
        raise RuntimeError(
            f"Ollama returned no embedding (is model {model!r} pulled? "
            f"run `ollama pull {model}`)."
        )
    return [float(x) for x in vec]


def embed_batch(texts: List[str]) -> List[List[float]]:
    """Embed several texts (one Ollama call each)."""
    return [embed(t) for t in texts]
