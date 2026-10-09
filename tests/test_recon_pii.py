"""The gate that makes committing captured Workday HTML safe.

This walks the SAME identity list the sanitizer uses and greps every committed
fixture for every real token. A hit fails the suite. A denylist sanitizer leaks
quietly; this is the part that leaks LOUDLY.

It must read the OWNER's REAL settings: conftest isolates every data path to
seeded fixtures, so an in-process check would compare placeholder tokens
against placeholder fixtures and assert nothing at all.

It gets them in a SUBPROCESS rather than by reloading jobbot.config. Two
reasons, both fatal in-process:

  1. conftest does not merely redirect the data paths, it ABORTS the suite if
     any path resolves under the real data/ directory. Reloading config to pick
     up the real .env is precisely the condition that guard exists to kill.
  2. Every module does `from .config import settings`, binding the object by
     VALUE. importlib.reload builds a new settings object that already-imported
     modules never see, and the mutation outlives this test and lands on
     whatever runs next.

A fresh interpreter has the real .env, no conftest, and no shared state.

This is the SECOND gate. The first is scripts/make_fixtures.py, whose build()
refuses to publish a capture still holding a token. Neither is redundant: the
generator gate stops a leak from being written, this one stops a leak that was
somehow committed anyway. Do not delete one believing the other covers it.

Tokens are never printed. A failure names the file and the token's LENGTH.

The subprocess must NOT inherit this pytest process's environment verbatim.
tests/conftest.py's `_isolate()` helper sets a batch of data-file-path env
vars via raw `os.environ[key] = value` (or `setdefault`) at import time, with
no teardown - they stay set for the life of the whole pytest process. Two of
those, RESUME_FACTS_PATH and PROFILE_PATH, point resume_facts.load() and the
profile reader at conftest's seeded test fixtures instead of the owner's real
files. Left inherited, the subprocess's identity_tokens() silently drops every
real employer/school token (and any extra emails embedded in the real
profile) while the applicant_name/email/phone/location/linkedin/github
tokens - which conftest does NOT isolate - keep coming through from the real
.env. That made the gate look like it worked while never checking two of the
categories recon_sanitize.py's own docstring names by number. See task-18
fix-round-1 in the SDD report for the reproduction.
"""
import os
import subprocess
import sys
from pathlib import Path

import pytest

REPO_ROOT = Path(__file__).resolve().parents[1]
FIXTURE_DIR = REPO_ROOT / "tests" / "fixtures" / "tenants"

# Every key tests/conftest.py's `_isolate()` helper sets to redirect a
# data-file path at import time (see conftest.py, the `_isolate(...)` calls
# between DB_PATH and WORKDAY_ACCOUNTS_PATH). They have no teardown, so they
# stay set in os.environ for the rest of the pytest process. conftest does not
# expose this list programmatically, so it is duplicated here deliberately -
# if conftest starts isolating another data-file env var, add it here too, or
# this gate goes back to silently trusting the isolated (fake) path for it.
_CONFTEST_ISOLATION_ENV_KEYS = frozenset({
    "DB_PATH",
    "OUTPUT_DIR",
    "LOG_PATH",
    "BASE_RESUME_PATH",
    "COMPANY_WATCHLIST",
    "PROFILE_PATH",
    "CANDIDATE_PROFILE_PATH",
    "ANSWER_BANK_PATH",
    "ANSWER_MEMORY_PATH",
    "USER_LOCATIONS_FILE",
    "USER_TAGS_FILE",
    "RESUME_FACTS_PATH",
    "ATS_RECIPES_PATH",
    "WORKDAY_ACCOUNTS_PATH",
})

_CHECK = r"""
import sys
from pathlib import Path
from jobbot import recon_sanitize
from jobbot import resume_facts

root = Path("tests/fixtures/tenants")
tokens = [t for t, _ in recon_sanitize.identity_tokens()]
if not tokens:
    print("IDENTITY LIST EMPTY - the check would assert nothing")
    sys.exit(2)

# The gate must not pass vacuously: if the owner's real resume facts contain
# an employer or school, at least one of those real names must have made it
# into the identity list, or redaction (and this check) is silently skipping
# two whole PII categories. Never print the name itself - only whether
# coverage exists.
try:
    facts = resume_facts.load()
except Exception:
    facts = None

employer_school_names = []
if facts is not None:
    employer_school_names += [
        j.employer for j in (getattr(facts, "employment", None) or [])
        if getattr(j, "employer", "") and len(j.employer) > 2
    ]
    employer_school_names += [
        e.school for e in (getattr(facts, "education", None) or [])
        if getattr(e, "school", "") and len(e.school) > 2
    ]

if not employer_school_names:
    print("NO EMPLOYER/SCHOOL FACTS AVAILABLE - cannot prove the identity "
          "list covers those categories")
    sys.exit(3)
if not any(name in tokens for name in employer_school_names):
    print("IDENTITY LIST HAS NO EMPLOYER/SCHOOL TOKEN - the gate would pass "
          "vacuously on that category (%d employer/school fact(s) checked)"
          % len(employer_school_names))
    sys.exit(3)

import json

def leaves(o):
    '''Every string in a nested structure, keys included.

    A .json fixture must be searched DECODED, not as file text. On disk a
    newline inside a captured string is the two characters backslash-n, and a
    token sitting right after one is invisible to a word-boundary search of
    the raw bytes - a probe written the obvious way called a file clean while
    a street address was in it.
    '''
    if isinstance(o, str):
        yield o
    elif isinstance(o, dict):
        for k, v in o.items():
            yield k
            for leaf in leaves(v):
                yield leaf
    elif isinstance(o, list):
        for v in o:
            for leaf in leaves(v):
                yield leaf

paths = sorted(root.rglob("*.html")) + sorted(root.rglob("*.json"))
if not paths:
    print("NO FIXTURES FOUND - the gate would assert nothing")
    sys.exit(4)

leaks = []
# Shape-based, deliberately NOT token-based. The loop below can only catch an
# address the identity list already knows, and identity tokens are NAMES: a
# test account's address need not contain one. Measured before this was added,
# "testuser@example.com" passed both redaction and this gate untouched.
residual_emails = []
for path in paths:
    if path.suffix.lower() == ".json":
        try:
            texts = list(leaves(json.loads(path.read_text(encoding="utf-8"))))
        except Exception as e:
            leaks.append("%s (unreadable: %s)" % (path, type(e).__name__))
            continue
    else:
        texts = [path.read_text(encoding="utf-8", errors="ignore")]
    for text in texts:
        low = text.lower()
        for token in tokens:
            if token.lower() in low:
                leaks.append("%s (token of length %d)" % (path, len(token)))
        for found in recon_sanitize._EMAIL_RE.findall(text):
            if found.lower() != recon_sanitize._EMAIL.lower():
                # Never print the address itself - only where and how long.
                residual_emails.append(
                    "%s (email-shaped leaf, length %d)" % (path, len(found)))

if residual_emails:
    print("EMAIL-SHAPED STRINGS SURVIVED INTO FIXTURES:")
    for line in sorted(set(residual_emails)):
        print("  " + line)
    sys.exit(5)

if leaks:
    print("IDENTITY TOKENS LEAKED INTO FIXTURES:")
    for line in sorted(set(leaks)):
        print("  " + line)
    sys.exit(1)
print("clean: %d fixture(s) checked against %d token(s), no residual emails"
      % (len(paths), len(tokens)))
"""


@pytest.mark.skipif(not FIXTURE_DIR.exists() or not any(FIXTURE_DIR.iterdir()) if FIXTURE_DIR.exists() else True,
                    reason="tenant fixtures purged")
def test_no_identity_token_survives_into_a_committed_fixture():
    # Strip conftest's isolated data-file paths from the child's environment
    # so resume_facts.load() / the profile reader fall back to the owner's
    # real .env-configured paths instead of conftest's seeded fixtures.
    child_env = {k: v for k, v in os.environ.items()
                if k not in _CONFTEST_ISOLATION_ENV_KEYS}
    proc = subprocess.run([sys.executable, "-c", _CHECK], cwd=REPO_ROOT,
                          capture_output=True, text=True, env=child_env)
    assert proc.returncode == 0, proc.stdout + proc.stderr
