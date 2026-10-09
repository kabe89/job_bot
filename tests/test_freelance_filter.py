"""Freelance / gig work is filtered on the TITLE only, never the description.

The constraint this file exists to protect: 665 real staff postings in the live
DB mention "contract" somewhere in the body (contract research organization,
contract negotiation, contracting strategy). A description-wide exclude would
delete all of them. Only 316 postings carry a freelance signal in the TITLE,
and that is the signal we act on.

Every "survives" case below is a real title taken from the job pool.
"""
from __future__ import annotations

import pytest

from jobbot import dbclean, matcher, models
from jobbot.config import Settings

# Real titles that MUST survive the filter. These are staff roles whose subject
# matter happens to be contracts, or whose employer is a consultancy.
SURVIVES = [
    "Senior Manager, Contract Strategy",
    "Senior Manager, IT Sourcing & Contract Management",
    "Senior Director, Global Clinical Contracts, Pricing & Payments",
    "Corporate Counsel/Senior Corporate Counsel, Commercial Contracts",
    "Director, Data Governance and Contract Compliance",
    "Investigator Contracts Lead, Sr. Manager",
    "Vice President, US Vaccines Contracting Strategy",
    "Lead Consultant - Data Engineering",
    "Senior Consultant - Workday Integration",
    "Lead Consultant - Knowledge graph expert",
    "Computational Drug Designer, London",
    "Senior Computational Biologist - Early Discovery Oncology",
    "Temp Sr Associate Scientist",
    "Laboratory Operation Assistant (Temp to Perm)",
]

# Real titles that MUST be dropped.
DROPPED = [
    "Drug Discovery Specialist - Freelance AI Trainer Project",
    "Synthetic Biology Specialist - Freelance AI Trainer Project",
    "Analytical Chemistry Specialist - Freelance AI Trainer Project",
    "Bash Coding Specialist - Freelance AI Trainer Project",
    "Contractor Neuropsychologist",
    "Compound Management Inventory Specialist (Contractor)",
    "Recruiting Coordinator - Contractor",
    "Telehealth Endocrinologist (1099) | Flexible Schedule",
    "Community Support Contractor, Virtual Cell Challenge",
]


@pytest.fixture
def title_excludes():
    return Settings().excludes_title


def test_default_list_is_not_empty(title_excludes):
    assert title_excludes, "shipping an empty default would silently no-op"


@pytest.mark.parametrize("title", SURVIVES)
def test_real_staff_titles_survive(title, title_excludes):
    assert not matcher.excluded(title, title_excludes), (
        f"{title!r} is a staff role and must not be filtered as freelance"
    )


@pytest.mark.parametrize("title", DROPPED)
def test_real_freelance_titles_are_dropped(title, title_excludes):
    assert matcher.excluded(title, title_excludes), (
        f"{title!r} is gig/contract work and should have been filtered"
    )


def test_freelancer_needs_its_own_entry(title_excludes):
    r"""\bfreelance\b does NOT match "freelancer" - the trailing boundary falls
    between two word characters. Both spellings must be listed."""
    assert not matcher.excluded("freelancer", ["freelance"])
    assert matcher.excluded("Freelancer Science Writer", title_excludes)


def test_contract_in_a_description_never_filters_the_job(title_excludes):
    """The false-positive trap: the title channel must only read the title."""
    body = ("We partner with a contract research organization and manage "
            "contract negotiation across contracting vendors.")
    assert not matcher.excluded("Computational Drug Designer", title_excludes)
    # The body is never passed to the title channel, but assert the intent:
    # even if it were, "contract" alone is not in the default list.
    assert not matcher.excluded(body, ["freelance", "freelancer", "1099"])


def test_bare_contract_is_not_in_the_code_default():
    """Guard against a future edit adding 'contract', which would delete 665
    legitimate postings, or 'consultant', which would delete 17 staff roles.

    Reads the CODE default rather than `Settings().excludes_title`, which would
    pick up whatever SEARCH_EXCLUDE_TITLE happens to be in the developer's .env
    and make this guard pass without checking anything.
    """
    default = Settings.model_fields["search_exclude_title"].default
    terms = [t.strip().lower() for t in default.split(",")]
    assert "contract" not in terms
    assert "consultant" not in terms
    assert "temp" not in terms
    assert "freelance" in terms and "freelancer" in terms


# --- the backfill -----------------------------------------------------------

@pytest.fixture
def db(tmp_path, monkeypatch):
    from sqlalchemy import create_engine
    from sqlalchemy.orm import sessionmaker

    monkeypatch.setattr(models.settings, "db_path", str(tmp_path / "fl.db"))
    eng = create_engine(f"sqlite:///{models.settings.db_path}", future=True)
    monkeypatch.setattr(models, "_engine", eng)
    monkeypatch.setattr(models, "SessionLocal", sessionmaker(bind=eng, future=True))
    models.init_db()
    return models


def _seed(db):
    with db.session() as s:
        for i, t in enumerate(DROPPED[:3] + SURVIVES[:3]):
            s.add(db.Job(source="greenhouse", title=t, company="agency",
                         url=f"http://x/{i}", status="new",
                         hash=f"h{i}", description="contract research organization"))
        # An already-applied freelance row must never be touched.
        s.add(db.Job(source="greenhouse", title=DROPPED[0], company="agency",
                     url="http://x/applied", status="applied", hash="happ"))
        s.commit()


def test_backfill_is_dry_run_by_default(db):
    _seed(db)
    rep = dbclean.hide_freelance_jobs(apply=False)
    assert rep["jobs"] == 3, rep
    with db.session() as s:
        assert s.query(db.Job).filter_by(status="skipped").count() == 0


def test_backfill_hides_only_freelance_active_rows(db):
    _seed(db)
    rep = dbclean.hide_freelance_jobs(apply=True)
    assert rep["jobs"] == 3, rep
    with db.session() as s:
        hidden = {j.title for j in s.query(db.Job).filter_by(status="skipped")}
        assert hidden == set(DROPPED[:3])
        # staff roles untouched
        assert s.query(db.Job).filter_by(status="new").count() == 3
        # an applied row keeps its status - history is never rewritten
        assert s.query(db.Job).filter_by(status="applied").count() == 1


def test_backfill_uses_skipped_not_expired(db):
    """status='skipped' is the channel, because liveness._ACTIVE_STATUSES does
    not include it - so a re-probe can never resurrect a hidden gig posting.
    Marking them 'expired' would be silently undone by `restore_job`.
    """
    from jobbot import liveness

    assert "skipped" not in liveness._ACTIVE_STATUSES
    _seed(db)
    dbclean.hide_freelance_jobs(apply=True)
    with db.session() as s:
        for j in s.query(db.Job).filter_by(status="skipped"):
            assert j.expired_at is None, "must not overload the liveness channel"


def test_backfill_is_idempotent(db):
    _seed(db)
    dbclean.hide_freelance_jobs(apply=True)
    again = dbclean.hide_freelance_jobs(apply=True)
    assert again["jobs"] == 0


def test_backfill_is_reversible(db):
    _seed(db)
    dbclean.hide_freelance_jobs(apply=True)
    rep = dbclean.unhide_freelance_jobs(apply=True)
    assert rep["jobs"] == 3
    with db.session() as s:
        assert s.query(db.Job).filter_by(status="new").count() == 6
        # the marker tag is cleaned up, not left behind
        assert not [j for j in s.query(db.Job) if "gig-hidden" in (j.tags or "")]


def test_undo_leaves_hand_skipped_gig_jobs_alone(db):
    """The real footgun: 316 rows in the live DB were already status='skipped'
    with freelance titles. An undo scoped only by title would resurrect all of
    them. It must restore only what `hide_freelance_jobs` itself hid.
    """
    with db.session() as s:
        s.add(db.Job(source="greenhouse", title=DROPPED[0], company="agency",
                     url="http://x/hand", status="skipped", hash="hhand"))
        s.add(db.Job(source="greenhouse", title=DROPPED[1], company="agency",
                     url="http://x/active", status="new", hash="hactive"))
        s.commit()

    assert dbclean.hide_freelance_jobs(apply=True)["jobs"] == 1
    assert dbclean.unhide_freelance_jobs(apply=True)["jobs"] == 1
    with db.session() as s:
        hand = s.query(db.Job).filter_by(hash="hhand").one()
        assert hand.status == "skipped", "a hand-skipped row must stay skipped"
        assert s.query(db.Job).filter_by(hash="hactive").one().status == "new"


def test_hide_preserves_existing_tags(db):
    with db.session() as s:
        s.add(db.Job(source="greenhouse", title=DROPPED[0], company="agency",
                     url="http://x/w", status="new", hash="hw", tags="watchlist"))
        s.commit()
    dbclean.hide_freelance_jobs(apply=True)
    with db.session() as s:
        j = s.query(db.Job).filter_by(hash="hw").one()
        assert "watchlist" in j.tags, "the watchlist flag drives ranking, keep it"
        assert "gig-hidden" in j.tags
