#!/usr/bin/env python
"""Run the test suite in parallel — the fast path for the inner dev loop.

The serial suite takes ~21 minutes, which is too slow to run between tasks, so
in practice it got skipped and regressions surfaced late. This runs the same
tests across every core.

    python scripts/fasttest.py              # whole suite, parallel
    python scripts/fasttest.py tests/test_apply_marking.py
    python scripts/fasttest.py -k localdt

Why the flags are what they are:

* ``--dist loadfile`` keeps every test in a file on ONE worker. The suite shares
  a session-scoped SQLite DB per process, and several files have tests that
  depend on rows an earlier test in the same file created. Plain ``--dist load``
  scatters those across workers and they fail for reasons that have nothing to
  do with the code under test.
* ``tests/`` is always the target, never a bare ``pytest``. A stray root-level
  ``test_followup.py`` calls ``sys.exit(0)`` and kills collection.
* Each worker gets its own temp DB — see the ``_isolate`` helper in
  ``tests/conftest.py``, which exists specifically because workers inherit the
  controller's env and would otherwise share one SQLite file.

This is not a replacement for the serial run at a phase boundary: if a result
here looks surprising, confirm it serially before believing it.
"""
from __future__ import annotations

import os
import subprocess
import sys


def main() -> int:
    args = sys.argv[1:]
    # Anything that looks like a path/-k/-m selection replaces the default
    # target; bare flags (-x, -q, --lf) are additive.
    has_target = any(not a.startswith("-") for a in args)
    target = [] if has_target else ["tests/"]

    cmd = [sys.executable, "-m", "pytest", *target,
           "-n", "auto", "--dist", "loadfile", "-q", *args]
    print("+ " + " ".join(cmd), flush=True)
    return subprocess.call(cmd, env=os.environ.copy())


if __name__ == "__main__":
    raise SystemExit(main())
