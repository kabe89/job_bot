"""Full liveness sweep: probe EVERY active job and expire the dead ones.

`liveness.sweep_expired` is built for the per-cycle trickle (settings.
liveness_check_per_cycle, default 40) -- it has no concurrency, no progress and
no politeness delay, so it is impractical for a full ~4k-job pass. This runner
does the same work, concurrently, with progress and a per-host rate limit.

Uses liveness.check_job_alive, so it inherits the Workday CXS probe: a pulled
Workday posting answers its public URL with HTTP 200 + a JS shell and would
otherwise be reported alive.

Fail-open, like the sweep it wraps: any error keeps the job. Safe to re-run.
Expiry is reversible (`liveness.restore_job`).

    python scripts/full_liveness_sweep.py [--limit N] [--workers 8] [--dry-run]
"""
from __future__ import annotations

import argparse
import collections
import sys
import threading
import time
from concurrent.futures import ThreadPoolExecutor, as_completed
from urllib.parse import urlparse

try:
    import truststore
    truststore.inject_into_ssl()
except Exception:
    pass

from jobbot.liveness import _ACTIVE_STATUSES, check_job_alive, expire_job, _stamp_checked
from jobbot.models import Job, init_db, session

# Never hammer one ATS host, however many workers are running.
_MIN_HOST_INTERVAL = 0.35
_host_lock = threading.Lock()
_host_last: dict[str, float] = {}


def _throttle(url: str) -> None:
    host = (urlparse(url).hostname or "").lower()
    while True:
        with _host_lock:
            now = time.monotonic()
            wait = _host_last.get(host, 0.0) + _MIN_HOST_INTERVAL - now
            if wait <= 0:
                _host_last[host] = now
                return
        time.sleep(min(wait, 1.0))


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--limit", type=int, default=0, help="0 = every active job")
    ap.add_argument("--workers", type=int, default=8)
    ap.add_argument("--dry-run", action="store_true",
                    help="report what WOULD be expired; write nothing")
    args = ap.parse_args()

    init_db()
    with session() as db:
        q = (db.query(Job)
             .filter(Job.status.in_(_ACTIVE_STATUSES))
             .order_by(Job.discovered_at.asc()))
        if args.limit:
            q = q.limit(args.limit)
        rows = [(j.id, j.url, j.source, j.title) for j in q.all()]

    print(f"active jobs to probe: {len(rows)}  (workers={args.workers}"
          f"{', DRY RUN' if args.dry_run else ''})", flush=True)

    stats = collections.Counter()
    by_source = collections.Counter()
    done = 0
    lock = threading.Lock()

    def probe(row):
        jid, url, source, title = row
        _throttle(url)
        try:
            alive, reason = check_job_alive(url)
        except Exception as e:  # noqa: BLE001 - fail-open, keep the job
            return jid, source, title, True, f"error:{e}"[:120]
        return jid, source, title, alive, reason

    with ThreadPoolExecutor(max_workers=max(1, args.workers)) as pool:
        futures = [pool.submit(probe, r) for r in rows]
        for fut in as_completed(futures):
            jid, source, title, alive, reason = fut.result()
            with lock:
                done += 1
                stats["checked"] += 1
                if not alive:
                    stats["expired"] += 1
                    by_source[source] += 1
                    if not args.dry_run:
                        expire_job(jid, reason)
                elif reason.startswith("error:"):
                    stats["errors"] += 1
                else:
                    stats["alive"] += 1
                if not args.dry_run:
                    _stamp_checked(jid)
                if done % 200 == 0:
                    print(f"  {done}/{len(rows)}  alive={stats['alive']} "
                          f"dead={stats['expired']} err={stats['errors']}", flush=True)

    print(f"\nDONE  checked={stats['checked']}  alive={stats['alive']}  "
          f"{'would-expire' if args.dry_run else 'expired'}={stats['expired']}  "
          f"errors={stats['errors']}")
    if by_source:
        print("\ndead by source:")
        for src, n in by_source.most_common(15):
            print(f"  {n:5d}  {src}")
    if stats["errors"]:
        print(f"\n{stats['errors']} transient errors were KEPT (fail-open). "
              f"Re-run to retry them.")
    return 0


if __name__ == "__main__":
    sys.exit(main())
