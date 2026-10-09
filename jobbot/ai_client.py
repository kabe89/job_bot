"""AI provider dispatcher.

Routes every AI call (resume tailoring, cover letters, match analysis,
interview prep, etc.) to the configured backend — Gemini or Claude — and
transparently falls back to the other provider on error.

Provider selection (settings.ai_provider, also AI_PROVIDER in .env):
  "gemini" -> [gemini, claude]   (Gemini first; Claude as a safety net)
  "claude" -> [claude, gemini]   (Claude first; Gemini as a safety net)
  "auto"   -> prefer Claude if its key is set, else Gemini; the other backs up

Any provider whose API key is unset is skipped, so a single configured key is
always enough. If every provider fails, the last exception propagates — the
pipeline already guards these calls with `gemini_skip_on_error`.

Both backends expose the same surface; this module re-exports it 1:1 so callers
can simply `from . import ai_client as gc` and keep using `gc.tailor_resume(...)`.
"""
from __future__ import annotations

import logging
from typing import Callable, List

from .config import settings
from . import gemini_client, claude_client, ollama_client

log = logging.getLogger("jobbot.ai")

# Functions that return text/dict and may raise -> eligible for failover.
_DISPATCHED = (
    "analyze_match",
    "tailor_resume",
    "generate_cover_letter",
    "interview_prep",
    "company_intel",
    "advise_resume",
    "answer_application_questions",
    "networking_message",
)


def _ordered_providers() -> List:
    """Return providers in preference order, skipping any that aren't usable.

    Ollama (local, free) is always appended as a last-resort backend so AI
    features keep working even when cloud keys are missing or out of quota.
    """
    p = (settings.ai_provider or "auto").strip().lower()
    if p == "gemini":
        order = [gemini_client, claude_client, ollama_client]
    elif p == "claude":
        order = [claude_client, gemini_client, ollama_client]
    elif p == "ollama":
        order = [ollama_client, gemini_client, claude_client]
    else:  # auto
        if claude_client.is_available():
            order = [claude_client, gemini_client, ollama_client]
        else:
            order = [gemini_client, claude_client, ollama_client]
    avail = [prov for prov in order if prov.is_available()]
    # If nothing is configured, fall back to the requested order anyway so the
    # caller gets a clear "key not set" error instead of a silent no-op.
    return avail or order


def _provider_label(prov) -> str:
    if prov is claude_client:
        return "claude"
    if prov is ollama_client:
        return "ollama"
    return "gemini"


def active_provider_name() -> str:
    provs = _ordered_providers()
    return _provider_label(provs[0]) if provs else "gemini"


def _dispatch(fn_name: str, *args, **kwargs):
    providers = _ordered_providers()
    last_err: Exception | None = None
    for prov in providers:
        fn: Callable = getattr(prov, fn_name)
        try:
            return fn(*args, **kwargs)
        except Exception as e:  # noqa: BLE001
            last_err = e
            log.warning("AI provider '%s' failed on %s (%s); trying next.",
                        _provider_label(prov), fn_name, e)
            continue
    if last_err:
        raise last_err
    raise RuntimeError(f"No AI provider available for {fn_name}")


def _make(fn_name: str):
    def _fn(*args, **kwargs):
        return _dispatch(fn_name, *args, **kwargs)
    _fn.__name__ = fn_name
    return _fn


# Bind dispatched functions onto the module namespace.
analyze_match = _make("analyze_match")
tailor_resume = _make("tailor_resume")
generate_cover_letter = _make("generate_cover_letter")
interview_prep = _make("interview_prep")
company_intel = _make("company_intel")
advise_resume = _make("advise_resume")
answer_application_questions = _make("answer_application_questions")
networking_message = _make("networking_message")
interview_question_bank = _make("interview_question_bank")
score_answer = _make("score_answer")
interview_session_summary = _make("interview_session_summary")


# ----- Non-dispatched helpers (operate across both providers) ---------------

def is_available() -> bool:
    """True if at least one provider is usable (cloud key or local Ollama)."""
    return (gemini_client.is_available() or claude_client.is_available()
            or ollama_client.is_available())


def probe(timeout_seconds: float = 5.0) -> bool:
    """Probe providers in preference order; True if any responds."""
    for prov in _ordered_providers():
        try:
            if prov.probe(timeout_seconds):
                return True
        except Exception:  # noqa: BLE001
            continue
    return False


def refresh_profile() -> None:
    gemini_client.refresh_profile()
    claude_client.refresh_profile()
    ollama_client.refresh_profile()


def reset_fallback_state() -> None:
    gemini_client.reset_fallback_state()
    claude_client.reset_fallback_state()
    ollama_client.reset_fallback_state()
