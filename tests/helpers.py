"""Test helpers — skip-on-Gemini-denied decorator."""
import os
import unittest


def gemini_available() -> bool:
    """Cheap availability check — no network call.
    Skips tests when no key is set OR JOBBOT_SKIP_GEMINI_TESTS=1."""
    if os.environ.get("JOBBOT_SKIP_GEMINI_TESTS") == "1":
        return False
    return bool(os.environ.get("GEMINI_API_KEY"))


def gemini_probe_ok() -> bool:
    """Stronger check — actually pings Gemini. Cached per-process."""
    if not gemini_available():
        return False
    try:
        from jobbot import gemini_client as gc
        return gc.probe()
    except Exception:
        return False


require_gemini = unittest.skipUnless(
    gemini_available(),
    "Gemini unavailable (no key or JOBBOT_SKIP_GEMINI_TESTS=1) — skipping",
)


require_gemini_live = unittest.skipUnless(
    gemini_probe_ok() if gemini_available() else False,
    "Gemini API denied or unreachable — skipping live test",
)
