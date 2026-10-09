# tests/test_apply_run_model.py
from datetime import datetime

from jobbot.models import ApplyRun, Job, init_db, session


def test_apply_run_persists_and_reads_back():
    init_db()
    with session() as db:
        job = Job(hash=Job.make_hash("test", "http://x/1", "Sci"),
                  source="test", title="Sci", company="Acme", url="http://x/1")
        db.add(job)
        db.commit()
        run = ApplyRun(job_id=job.id, ats="greenhouse", status="preparing",
                       started_at=datetime.utcnow())
        db.add(run)
        db.commit()
        run_id = run.id

    with session() as db:
        got = db.get(ApplyRun, run_id)
        assert got.ats == "greenhouse"
        assert got.status == "preparing"
        assert got.screenshot_path == ""   # default
        assert got.answers_json == ""       # default
