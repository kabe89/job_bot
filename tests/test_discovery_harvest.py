import pytest

from jobbot.discovery import discoverers as D
from jobbot.models import Job, init_db, session


@pytest.fixture(autouse=True)
def _clean():
    init_db()
    with session() as db:
        db.query(Job).delete(); db.commit()
    yield


def test_coord_from_url_all_providers():
    assert D._coord_from_url("https://boards.greenhouse.io/nexustech/jobs/123") == ("greenhouse", "nexustech", None)
    assert D._coord_from_url("https://jobs.lever.co/acme/0d6f1b2c-1111-2222-3333-444455556666") == ("lever", "acme", None)
    assert D._coord_from_url("https://jobs.ashbyhq.com/techflow/0d6f1b2c-1111-2222-3333-444455556666") == ("ashby", "techflow", None)
    assert D._coord_from_url(
        "https://pinnacle.wd5.myworkdayjobs.com/en-US/Pinnacle-Careers/job/Remote/Engineer_R-1"
    ) == ("workday", "pinnacle/Pinnacle-Careers", "wd5")
    assert D._coord_from_url("https://example.com/not-an-ats") is None


def test_coord_from_url_board_root_fallbacks():
    # Web/LLM search returns careers landing / board-root URLs, not deep postings.
    assert D._coord_from_url("https://boards.greenhouse.io/nexustech") == ("greenhouse", "nexustech", None)
    assert D._coord_from_url("https://jobs.lever.co/acme") == ("lever", "acme", None)
    assert D._coord_from_url("https://jobs.ashbyhq.com/techflow") == ("ashby", "techflow", None)
    assert D._coord_from_url(
        "https://pinnacle.wd5.myworkdayjobs.com/en-US/Pinnacle-Careers"
    ) == ("workday", "pinnacle/Pinnacle-Careers", "wd5")
    # Non-ATS roots still return None.
    assert D._coord_from_url("https://greenhouse.io/customers") is None


def test_harvest_dedupes_and_names():
    with session() as db:
        db.add(Job(hash="h1", source="jobspy", title="Engineer", company="Nexus Tech",
                   url="https://boards.greenhouse.io/nexustech/jobs/1"))
        db.add(Job(hash="h2", source="jobspy", title="Developer", company="Nexus Tech",
                   url="https://boards.greenhouse.io/nexustech/jobs/2"))  # same board
        db.add(Job(hash="h3", source="li", title="X", company="Acme",
                   url="https://jobs.lever.co/acme/0d6f1b2c-1111-2222-3333-444455556666"))
        db.commit()
    out = D.harvest_from_jobs()
    coords = {t.coord() for t in out}
    assert coords == {("greenhouse", "nexustech"), ("lever", "acme")}   # deduped
    nexus = next(t for t in out if t.key == "nexustech")
    assert nexus.display_name == "Nexus Tech" and nexus.source == "harvest"
