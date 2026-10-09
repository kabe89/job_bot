"""Pytest bootstrap — GUARANTEES tests never touch the real database.

pytest imports conftest.py before collecting any test module, so setting the
DB/path env vars here (before `jobbot` is ever imported) forces the SQLAlchemy
engine to bind to a throwaway temp DB no matter how the suite is invoked. A
session-scoped guard then hard-fails if the engine somehow points at the
production data/ directory, so a misconfigured run aborts instead of wiping
1000+ real jobs.
"""
from __future__ import annotations

import os
import tempfile
from pathlib import Path

import pytest

# --- Set isolation env vars BEFORE anything imports jobbot.config ------------
# CRITICAL: jobbot.config.settings reads these env vars ONCE, when the module is
# first imported. Whichever test file imports jobbot first freezes the singleton,
# so a per-file module-level override (e.g. in test_jobbot.py) arrives TOO LATE
# in a full-suite run and the paths silently fall back to the real data/ files.
# We MUST set every data-file path here — not just the DB — or tests read (and,
# via the /profile save route, OVERWRITE) the user's real profile/companies.
# Under pytest-xdist each worker is a separate process that INHERITS the
# controller's environment — including the DB_PATH the controller already set
# here. `setdefault` would therefore leave every worker bound to the SAME
# SQLite file and they would clobber each other's rows. So give each worker its
# own temp dir and force the override there. (Run parallel with
# `-n auto --dist loadfile`: loadfile keeps a whole file on one worker, which
# the order-dependent tests in this suite rely on.)
_WORKER = os.environ.get("PYTEST_XDIST_WORKER", "")
_TMP = Path(tempfile.mkdtemp(
    prefix=f"jobbot_conftest_{_WORKER}_" if _WORKER else "jobbot_conftest_"))


def _isolate(key: str, value: str) -> None:
    """Point `key` at the throwaway temp dir.

    Serial runs use setdefault so a caller can still pin a path deliberately.
    In an xdist worker that inherited value is the controller's shared path, not
    a deliberate choice, so isolation has to win.
    """
    if _WORKER:
        os.environ[key] = value
    else:
        os.environ.setdefault(key, value)


_isolate("DB_PATH", str(_TMP / "test.db"))
_isolate("OUTPUT_DIR", str(_TMP / "out"))
_isolate("LOG_PATH", str(_TMP / "logs" / "j.log"))
_isolate("BASE_RESUME_PATH", str(_TMP / "base_resume.md"))
_isolate("COMPANY_WATCHLIST", str(_TMP / "companies.md"))
_isolate("PROFILE_PATH", str(_TMP / "profile.md"))
_isolate("CANDIDATE_PROFILE_PATH", str(_TMP / "candidate_profile.json"))
# The two answer stores. Neither file is tracked by git, so a test that wrote to
# the real ones would destroy hand-curated rules and every learned answer with no
# way to recover them. The seeding/CLI code paths write to both by design.
_isolate("ANSWER_BANK_PATH", str(_TMP / "answer_bank.json"))
_isolate("ANSWER_MEMORY_PATH", str(_TMP / "answer_memory.json"))
# user_prefs writes these via the /locations and /tags save routes.
_isolate("USER_LOCATIONS_FILE", str(_TMP / "user_locations.txt"))
_isolate("USER_TAGS_FILE", str(_TMP / "user_tags.txt"))
# Structured resume facts (employment/education/identity) for Workday autofill.
_isolate("RESUME_FACTS_PATH", str(_TMP / "resume_facts.json"))
# Learned ATS field-resolution recipes (perishable cache of someone else's DOM).
_isolate("ATS_RECIPES_PATH", str(_TMP / "ats_recipes.json"))
# Per-tenant Workday credentials. Holds a PLAINTEXT password and is gitignored,
# so a test that wrote to the real file would both destroy the user's stored
# account and put a real password in reach of any test that dumps values.
_isolate("WORKDAY_ACCOUNTS_PATH", str(_TMP / "workday_accounts.json"))
# Never let a test auto-start a local Ollama server or hit a cloud key.
os.environ.setdefault("OLLAMA_AUTOSTART", "false")
os.environ.setdefault("GEMINI_API_KEY", "")
os.environ["CONTACTS_AUTOFIND"] = "false"
os.environ["GOOGLE_CSE_API_KEY"] = ""
os.environ["GOOGLE_CSE_CX"] = ""
os.environ["RERANK_ENABLED"] = "false"
os.environ["LIVENESS_CHECK_PER_CYCLE"] = "0"
os.environ["AUTO_DISCOVER_HARVEST"] = "false"

# Seed deterministic fixture files at the temp paths BEFORE jobbot.config is
# imported, so any test that reads the resume / watchlist / profile gets test
# data (and never the real data/ files). Only write if not already present so a
# caller-provided override wins.
def _seed(env_key: str, content: str) -> None:
    p = Path(os.environ[env_key])
    p.parent.mkdir(parents=True, exist_ok=True)
    if not p.exists():
        p.write_text(content, encoding="utf-8")


_seed("BASE_RESUME_PATH",
      "# Test Resume\n\n## Skills\nPython, machine learning, data engineering, "
      "SQL, cloud computing, Docker, Kubernetes, system design\n")
_seed("COMPANY_WATCHLIST",
      "## Companies\n- Acme Corp\n- Pinnacle Systems\n- Apex Innovations\n")
_seed("PROFILE_PATH",
      "# Profile\nSenior Engineer focused on machine learning and platform infrastructure.\n")

# Real production paths we must never write to during tests.
_REPO_ROOT = Path(__file__).resolve().parent.parent
_DATA_DIR = (_REPO_ROOT / "data").resolve()
_FORBIDDEN = {
    (_REPO_ROOT / "data" / "jobbot.db").resolve(),
}


@pytest.fixture(scope="session", autouse=True)
def _guard_against_real_data():
    """Abort the whole session if the engine or any data-file path is bound to
    the production data/ directory — isolation failure would corrupt real data
    (the /profile save route writes to settings.profile_path)."""
    from jobbot.models import _engine
    from jobbot.config import settings

    db_file = Path(_engine.url.database or "").resolve()
    if db_file in _FORBIDDEN or db_file.parent == _DATA_DIR:
        pytest.exit(
            f"REFUSING TO RUN: test DB engine is bound to a production path "
            f"({db_file}). Isolation failed — aborting to protect real data.",
            returncode=2,
        )
    for attr in ("profile_path", "company_watchlist", "base_resume_path",
                 "candidate_profile_path", "answer_bank_path",
                 "answer_memory_path", "resume_facts_path",
                 "ats_recipes_path", "workday_accounts_path"):
        p = Path(getattr(settings, attr)).resolve()
        if p.parent == _DATA_DIR:
            pytest.exit(
                f"REFUSING TO RUN: settings.{attr} points at the production "
                f"data/ dir ({p}). A test could overwrite real user data - "
                f"aborting. (conftest must set the env var before jobbot import.)",
                returncode=2,
            )
    from jobbot import user_prefs
    for name in ("USER_LOCATIONS_FILE", "USER_TAGS_FILE"):
        p = Path(getattr(user_prefs, name)).resolve()
        if p.parent == _DATA_DIR:
            pytest.exit(
                f"REFUSING TO RUN: user_prefs.{name} points at the production "
                f"data/ dir ({p}). The /locations & /tags save routes would "
                f"overwrite real user prefs — aborting.",
                returncode=2,
            )
    yield


@pytest.fixture(autouse=True)
def _reset_escalation_state():
    """Clear ats.base's module-level escalation state between tests.

    `apply_runner` calls `set_escalation_context(tenant, "single")` and nothing
    ever clears the tenant again. It is module state, so once any test has run
    an apply, every later test in the same process has escalation ARMED: a
    field the deterministic path could not fill escalates, Ollama is not
    running under test, and the precise reason is replaced by
    "escalation_failed".

    That is how three tests in test_ats_base_bound_fill.py passed alone and
    failed in the full suite -- they assert the exact reason ("not_found",
    "ambiguous_label", "verify_failed") and got the escalation one instead.
    The tests were right and the state was dirty.

    Imported inside the fixture so collecting a test that never touches the ATS
    layer does not pay for importing Playwright's dependents.
    """
    try:
        from jobbot.ats import base as ats_base
    except (ImportError, AttributeError):
        yield
        return

    def clear():
        ats_base._escalation_ctx["tenant"] = ""
        ats_base._escalation_ctx["step_id"] = ""
        ats_base._escalation_used["n"] = 0
        ats_base._escalation_dead["llm_unreachable"] = False

    clear()
    yield
    clear()
