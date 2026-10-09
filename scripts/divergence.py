#!/usr/bin/env python
"""What is the same at every tenant, and what only looks the same.

    python scripts/divergence.py [--step my-information]

A selector that works at three Workday tenants is a durable rule. One that
works at a single tenant is a per-tenant recipe wearing a rule's clothes, and
the difference is not visible from one capture - which is the whole reason the
corpus spans `wd1`, `wd3` and `wd5` rather than being three captures of
the same tenant.

The table joins on STEP NAME, so a typo in a step name shows up here as a step
that exists at one tenant only. That is deliberate: a silent mis-join would
report a shared field as tenant-unique and send someone off to write a special
case for a field that is in fact standard.

Nothing here reads a value. The fixtures do not contain any.
"""
from __future__ import annotations

import argparse
import json
from collections import defaultdict
from pathlib import Path
from typing import Dict, List

FIXTURES = Path("tests/fixtures/tenants")


def load() -> Dict[str, Dict[str, dict]]:
    """{step: {tenant: fixture}}"""
    out: Dict[str, Dict[str, dict]] = defaultdict(dict)
    for path in sorted(FIXTURES.rglob("*.json")):
        try:
            data = json.loads(path.read_text(encoding="utf-8"))
        except Exception:  # noqa: BLE001
            continue
        out[data.get("step") or path.stem][data.get("tenant") or path.parent.name] = data
    return out


def _fields(fixture: dict) -> Dict[str, dict]:
    """Field handles, with the per-block number stripped.

    workExperience-7--jobTitle at one tenant and workExperience-396--jobTitle at
    another are the SAME field: the number is a server-assigned row id that
    differs per draft, so comparing raw handles would report every repeating
    field as tenant-unique.
    """
    out: Dict[str, dict] = {}
    for f in fixture.get("fields") or []:
        if not f.get("handle"):
            continue
        key = f["handle"]
        if f.get("section") and f.get("block"):
            key = "%s-N--%s" % (f["section"], f["suffix"])
        out.setdefault(key, f)
    return out


def report(steps: Dict[str, Dict[str, dict]], only: str | None) -> int:
    tenants = sorted({t for per in steps.values() for t in per})
    print("tenants in the corpus: %s" % ", ".join(tenants))
    print()

    shared_total = unique_total = 0
    for step in sorted(steps):
        if only and step != only:
            continue
        per = steps[step]
        if len(per) < 2:
            print("%-26s only at %s - nothing to compare"
                  % (step, ", ".join(sorted(per))))
            continue

        print("=" * 72)
        print("%s   (%s)" % (step, ", ".join(sorted(per))))
        print("=" * 72)

        by_tenant = {t: _fields(fx) for t, fx in per.items()}
        every = sorted({k for f in by_tenant.values() for k in f})
        shared = [k for k in every if all(k in f for f in by_tenant.values())]
        unique = [k for k in every if k not in shared]
        shared_total += len(shared)
        unique_total += len(unique)

        print("  %d field(s) at every tenant, %d at only some"
              % (len(shared), len(unique)))

        # A field present everywhere but REQUIRED in only some places is the
        # dangerous case: the selector is durable and the fill plan is not.
        for key in shared:
            flags = {t: bool(by_tenant[t][key].get("required")) for t in by_tenant}
            if len(set(flags.values())) > 1:
                yes = sorted(t for t, v in flags.items() if v)
                no = sorted(t for t, v in flags.items() if not v)
                print("    REQUIRED DIFFERS  %-42s required at %s, optional at %s"
                      % (key, ",".join(yes), ",".join(no)))

        for key in unique:
            where = sorted(t for t in by_tenant if key in by_tenant[t])
            print("    ONLY AT %-12s %s" % (",".join(where), key))

        # Labelled dropdowns are matched by their human label, so a label that
        # differs by a word is a rule that silently misses a tenant.
        labels = {t: {b["label"] for b in (fx.get("labelled_buttons") or [])}
                  for t, fx in per.items()}
        common = set.intersection(*labels.values()) if labels else set()
        for t, ls in labels.items():
            odd = sorted(ls - common)
            if odd:
                print("    LABELS ONLY AT %-12s %s"
                      % (t, "; ".join(x[:52] for x in odd[:6])))
        print()

    print("-" * 72)
    print("%d shared field(s), %d tenant-specific across the compared steps."
          % (shared_total, unique_total))
    return 0


def main() -> int:
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--step", default=None)
    args = ap.parse_args()
    if not FIXTURES.is_dir():
        raise SystemExit("no fixtures yet: %s" % FIXTURES)
    return report(load(), args.step)


if __name__ == "__main__":
    raise SystemExit(main())
