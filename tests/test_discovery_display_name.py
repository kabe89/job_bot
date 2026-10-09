"""Web/LLM-discovered boards had no company name available, so _candidates_from_urls
set `display_name = key`. For Workday the key IS the coordinate ("tenant/Careers"),
so that slug got written into data/workday_targets.txt as the display name and the
scraper then stored it as Job.company -- rows reading "tenant/Careers",
plus a "Dear tenant/Careers" hazard in generated letters and broken watchlist /
referral company matching.

A display name must never be a raw coordinate.
"""
import pytest

from jobbot.discovery import discoverers as d

_FAKE_TARGETS = [
    ("pinnacle", "Careers", "Pinnacle Systems", "wd1"),
    ("acme", "Careers", "Acme Corporation", "wd5"),
    ("vertextech", "jobs", "Vertex Technologies", "wd501"),
]


@pytest.fixture(autouse=True)
def _pin_targets(monkeypatch):
    monkeypatch.setattr("jobbot.scrapers.workday._load_targets",
                        lambda: list(_FAKE_TARGETS))


def test_pretty_name_never_returns_a_coordinate():
    assert d._pretty_name("workday", "innotech/Careers") == "Innotech"
    assert "/" not in d._pretty_name("workday", "innotech/Careers")


@pytest.mark.parametrize("provider,key,expected", [
    # unknown tenants fall back to a humanized slug
    ("workday", "innotech/Careers", "Innotech"),
    ("workday", "globalcorp/Careers", "Globalcorp"),
    # known tenants win via the curated config (better than titlecasing)
    ("workday", "pinnacle/Careers", "Pinnacle Systems"),
    ("workday", "vertextech/jobs", "Vertex Technologies"),
    # non-workday providers have a flat slug, not a coordinate
    ("greenhouse", "acme-corp", "Acme Corp"),
    ("lever", "quantum-systems", "Quantum Systems"),
    ("ashby", "techflow", "Techflow"),
])
def test_pretty_name_humanizes_a_slug(provider, key, expected):
    assert d._pretty_name(provider, key) == expected


def test_pretty_name_uses_the_curated_name_when_there_is_one():
    """A curated config name beats titlecasing the tenant slug."""
    assert d._pretty_name("workday", "acme/Careers") == "Acme Corporation"
    assert d._pretty_name("workday", "vertextech/jobs") == "Vertex Technologies"


def test_pretty_name_ignores_a_curated_name_that_is_itself_a_coordinate(monkeypatch):
    """Guard against the original bug re-entering via the config file."""
    monkeypatch.setattr("jobbot.scrapers.workday._load_targets",
                        lambda: [("innotech", "Careers", "innotech/Careers", "wd1")])
    assert d._pretty_name("workday", "innotech/Careers") == "Innotech"


def test_pretty_name_survives_an_unreadable_config(monkeypatch):
    def boom():
        raise OSError("config gone")
    monkeypatch.setattr("jobbot.scrapers.workday._load_targets", boom)
    assert d._pretty_name("workday", "innotech/Careers") == "Innotech"


def test_pretty_name_handles_separators_in_a_slug():
    assert d._pretty_name("greenhouse", "acme-software-labs") == "Acme Software Labs"
    assert d._pretty_name("greenhouse", "acme_tech") == "Acme Tech"


def test_pretty_name_is_fail_open_on_junk():
    assert d._pretty_name("workday", "") == ""
    assert d._pretty_name("workday", "/") == ""


def test_candidates_from_urls_no_longer_names_a_target_after_its_coordinate():
    out = d._candidates_from_urls(
        ["https://innotech.wd1.myworkdayjobs.com/en-US/Careers/job/City/X_R1"],
        source="websearch")
    assert len(out) == 1
    t = out[0]
    assert t.key == "innotech/Careers"          # coordinate still correct
    assert t.display_name == "Innotech"         # but the NAME is human
    assert "/" not in t.display_name


def test_harvest_still_prefers_the_real_company_name_over_a_derived_one(monkeypatch):
    """Harvest reads Job.company, which is a real name -- don't regress it."""
    from jobbot.models import Job, init_db, session
    init_db()
    with session() as db:
        db.add(Job(hash="disc_h1", source="workday:innotech", title="Eng",
                   company="InnoTech", status="new", description="d",
                   url="https://innotech.wd1.myworkdayjobs.com/en-US/Careers/job/C/X_R2"))
        db.commit()
    out = [t for t in d.harvest_from_jobs() if t.key == "innotech/Careers"]
    assert out and out[0].display_name == "InnoTech"


def test_harvest_falls_back_to_a_pretty_name_not_the_coordinate():
    """A harvested row whose company is itself already a bad slug must not
    propagate that slug back out as a display name."""
    from jobbot.models import Job, init_db, session
    init_db()
    with session() as db:
        db.add(Job(hash="disc_h2", source="workday:apextech", title="Eng",
                   company="apextech/ApexCareers", status="new", description="d",
                   url="https://apextech.wd12.myworkdayjobs.com/en-US/ApexCareers/job/C/X_R3"))
        db.commit()
    out = [t for t in d.harvest_from_jobs() if t.key == "apextech/ApexCareers"]
    assert out
    assert out[0].display_name == "Apextech"
    assert "/" not in out[0].display_name
