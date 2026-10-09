"""One-off repair: backfill real JD bodies onto Workday jobs scraped before
`fetch_description` existed, then re-score them against the real text.

Every Workday job stored a ~200-char listing-card stub as its description, so
every Workday match_score was computed against scraper noise. This fetches the
real body per job and recomputes the semantic score.

Safe to re-run: it only touches rows that still carry a stub, and any job whose
fetch fails is left exactly as-is.

    python scripts/backfill_workday_descriptions.py [--limit N] [--dry-run]
"""
from __future__ import annotations

import argparse
import json
import sys
import time

from jobbot.liveness import _ACTIVE_STATUSES, expire_job
from jobbot.models import Job, init_db, session
from jobbot.scrapers.workday import fetch_posting


def is_stub(desc: str) -> bool:
    """The old card-metadata description, not a real JD."""
    d = (desc or "").strip()
    return (not d) or (d.startswith("Job ID:") and len(d) < 600)


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--limit", type=int, default=0, help="0 = all")
    ap.add_argument("--dry-run", action="store_true")
    ap.add_argument("--delay", type=float, default=0.2)
    args = ap.parse_args()

    init_db()
    with session() as db:
        # ACTIVE jobs only. Without the status filter this re-fetches every job a
        # previous run already expired -- they keep their stub description, so
        # they stay "targets" forever. A re-run then burned ~846 fetches (~15
        # min) re-confirming known-dead postings, reporting "ok=0 gone=799",
        # which reads exactly like a WAF mass-expiry event and is not one.
        # Nothing here can revive an expired job, so probing them buys nothing.
        rows = (db.query(Job)
                .filter(Job.source.like("workday:%"),
                        Job.status.in_(_ACTIVE_STATUSES))
                .all())
        targets = [j.id for j in rows if is_stub(j.description)]
    if args.limit:
        targets = targets[: args.limit]

    print(f"stub-description ACTIVE Workday jobs: {len(targets)}")
    if args.dry_run:
        print("(dry run - nothing written)")
        return 0

    # Re-score against the real text once it lands.
    profile = None
    try:
        from jobbot import ranking
        from jobbot.profile import load_profile
        profile = load_profile()
        if not (profile and profile.embedding):
            print("WARNING: no profile embedding - backfilling text only, no re-score")
            profile = None
    except Exception as e:  # noqa: BLE001
        print(f"WARNING: ranking unavailable ({e}) - text only, no re-score")

    ok = gone = errored = rescored = 0
    for n, jid in enumerate(targets, 1):
        with session() as db:
            job = db.get(Job, jid)
            if job is None or not is_stub(job.description):
                continue
            url = job.url
        try:
            body, state = fetch_posting(url)
        except Exception as e:  # noqa: BLE001
            body, state = "", "error"
            print(f"  [{jid}] fetch raised: {e}")

        if state == "gone":
            # A pulled posting (403/S22). This is a liveness fact, not a fetch
            # failure -- retire the job instead of leaving it to be re-scored
            # forever against a stub. Reversible: `restore_job` un-expires it.
            expire_job(jid, "workday detail 403/S22 - posting pulled")
            gone += 1
        elif state == "ok":
            with session() as db:
                job = db.get(Job, jid)
                job.description = body
                if profile is not None:
                    try:
                        score, emb = ranking.semantic_score(
                            f"{job.title}\n{job.company}\n{body}", profile.embedding)
                        job.match_score = round(score, 4)
                        job.embedding = json.dumps(emb)
                        rescored += 1
                    except Exception as e:  # noqa: BLE001
                        print(f"  [{jid}] re-score failed: {e}")
                db.commit()
            ok += 1
        else:
            errored += 1        # transient - left untouched, safe to re-run

        if n % 25 == 0:
            print(f"  {n}/{len(targets)}  ok={ok} gone={gone} err={errored} "
                  f"rescored={rescored}", flush=True)
        if args.delay:
            time.sleep(args.delay)

    print(f"\nDONE  backfilled={ok}  expired(pulled)={gone}  "
          f"transient-errors={errored}  rescored={rescored}")
    if errored:
        print("Transient errors were left untouched - re-run to retry them.")
    return 0


if __name__ == "__main__":
    sys.exit(main())
