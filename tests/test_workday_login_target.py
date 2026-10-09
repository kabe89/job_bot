"""`jobbot workday-login` target resolution: explicit --url wins, else the
job's own URL (redirects to that tenant's sign-in), else the Workday home."""
from itertools import count

from jobbot.cli import _workday_login_target
from jobbot.models import Job, init_db, session

# jobs.hash is UNIQUE and the test DB persists across tests in this file, so a
# fixed hash makes the second _job() call raise IntegrityError.
_seq = count()


def _job(url):
    init_db()
    with session() as db:
        j = Job(hash=f"wl{next(_seq)}", source="workday:innotech", title="Sci",
                company="Innotech", url=url, description="d", status="new")
        db.add(j); db.commit(); db.refresh(j)
        return j.id


def test_explicit_url_wins_over_job():
    jid = _job("https://innotech.wd1.myworkdayjobs.com/en-US/Careers/job/x")
    assert _workday_login_target(jid, "https://example.com/login") == "https://example.com/login"


def test_falls_back_to_job_url():
    url = "https://innotech.wd1.myworkdayjobs.com/en-US/Careers/job/x"
    jid = _job(url)
    assert _workday_login_target(jid, None) == url


def test_generic_home_when_nothing_given():
    assert _workday_login_target(None, None) == "https://www.myworkdayjobs.com/"
