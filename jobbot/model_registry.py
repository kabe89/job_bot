"""AI model selector — the write-side of the settings ai_client already reads.

Lists every selectable provider+model (local Ollama, Ollama cloud, Gemini,
Claude), reports current selection and availability, persists a new selection
to .env + the live settings object, and pulls local Ollama models.
"""
from __future__ import annotations

import logging
import re
import subprocess
from pathlib import Path
from typing import Callable, Optional

from .config import settings
from . import ollama_client

log = logging.getLogger("jobbot.models")

# Curated cloud model IDs. Free-text custom entries are still allowed via the
# CLI/web "custom" field; this is just the convenient menu.
CLOUD_CATALOG: dict[str, list[str]] = {
    "ollama-cloud": [
        "gpt-oss:20b-cloud", "gpt-oss:120b-cloud",
        "deepseek-v3.1:671b-cloud", "qwen3-coder:480b-cloud",
    ],
    "gemini": [
        "gemini-3-flash-preview", "gemini-2.5-flash",
        "gemini-3.1-flash-lite-preview",
    ],
    "claude": ["claude-opus-4-8", "claude-sonnet-4-6"],
}

# provider -> (env var, settings attribute)
_MODEL_ENV = {"ollama": "OLLAMA_MODEL", "gemini": "GEMINI_MODEL", "claude": "CLAUDE_MODEL"}
_MODEL_ATTR = {"ollama": "ollama_model", "gemini": "gemini_model", "claude": "claude_model"}


def list_ollama_models() -> list[str]:
    """Local installed models (from `/api/tags`)."""
    return [m for m in ollama_client._installed_models() if m]


def current_selection() -> tuple[str, str]:
    provider = (getattr(settings, "ai_provider", "") or "auto").strip().lower()
    attr = _MODEL_ATTR.get(provider)
    model = getattr(settings, attr, "") if attr else ""
    return provider, model


def _row(group: str, provider: str, model: str, available: bool, reason: str,
         cur: tuple[str, str]) -> dict:
    return {"group": group, "provider": provider, "model": model,
            "available": available, "reason": reason,
            "current": (provider == cur[0] and model == cur[1])}


def list_selectable() -> list[dict]:
    cur = current_selection()
    rows: list[dict] = []

    local_ok = ollama_client._server_reachable() or ollama_client._can_autostart()
    for m in list_ollama_models():
        rows.append(_row("Ollama — local", "ollama", m, local_ok,
                         "" if local_ok else "Ollama server unreachable", cur))

    if bool(getattr(settings, "ollama_api_key", "")):
        for m in CLOUD_CATALOG["ollama-cloud"]:
            rows.append(_row("Ollama — cloud", "ollama", m, True, "", cur))

    g_ok = bool(getattr(settings, "gemini_api_key", ""))
    for m in CLOUD_CATALOG["gemini"]:
        rows.append(_row("Gemini", "gemini", m, g_ok,
                         "" if g_ok else "set GEMINI_API_KEY", cur))

    c_ok = bool(getattr(settings, "anthropic_api_key", ""))
    for m in CLOUD_CATALOG["claude"]:
        rows.append(_row("Claude", "claude", m, c_ok,
                         "" if c_ok else "set ANTHROPIC_API_KEY", cur))
    return rows


def _write_env(updates: dict[str, str], env_path: Optional[Path] = None) -> None:
    """Replace each KEY=... line in .env (or append under a header)."""
    env_path = env_path or Path(".env")
    current = env_path.read_text(encoding="utf-8") if env_path.exists() else ""
    appended: list[str] = []
    for key, val in updates.items():
        line = f"{key}={val}"
        if re.search(rf"^{key}=.*$", current, re.MULTILINE):
            current = re.sub(rf"^{key}=.*$", line, current, flags=re.MULTILINE)
        else:
            appended.append(line)
    if appended:
        current = current.rstrip() + "\n\n# Set via model selector\n" + "\n".join(appended) + "\n"
    env_path.parent.mkdir(parents=True, exist_ok=True)
    env_path.write_text(current, encoding="utf-8")


def set_selection(provider: str, model: str, env_path: Optional[Path] = None) -> None:
    provider = (provider or "").strip().lower()
    model = (model or "").strip()
    if provider not in _MODEL_ENV:
        raise ValueError(f"Unknown provider: {provider!r} (use ollama|gemini|claude).")
    if not model:
        raise ValueError("A model name is required.")
    _write_env({"AI_PROVIDER": provider, _MODEL_ENV[provider]: model}, env_path)
    settings.ai_provider = provider
    setattr(settings, _MODEL_ATTR[provider], model)
    try:
        ollama_client.reset_fallback_state()
    except Exception:  # noqa: BLE001
        pass
    log.info("Active AI model set to %s (%s).", model, provider)


def pull_ollama_model(name: str, progress: Optional[Callable[[str], None]] = None) -> bool:
    """`ollama pull <name>`, streaming status lines to `progress`. Local only."""
    def _say(msg: str) -> None:
        if progress:
            try:
                progress(msg)
            except Exception:  # noqa: BLE001
                pass

    name = (name or "").strip()
    if not name:
        _say("No model name given.")
        return False
    binp = ollama_client._ollama_bin()
    if not binp:
        _say("`ollama` binary not found on PATH — cannot pull.")
        return False
    _say(f"Pulling {name} ...")
    try:
        proc = subprocess.Popen([binp, "pull", name], stdout=subprocess.PIPE,
                                stderr=subprocess.STDOUT, text=True, bufsize=1)
        assert proc.stdout is not None
        for raw in proc.stdout:
            line = raw.rstrip()
            if line:
                _say(line)
        proc.wait()
        ok = proc.returncode == 0
        _say("Pull complete." if ok else f"Pull failed (exit {proc.returncode}).")
        return ok
    except Exception as e:  # noqa: BLE001
        _say(f"Pull error: {e}")
        return False
