from datetime import datetime

import pytest

from jobbot.config import settings
from jobbot.models import Application, Contact, Job, init_db, session


@pytest.fixture(autouse=True)
def _clean_db():
    def _wipe():
        init_db()
        with session() as db:
            db.query(Contact).delete()
            db.query(Application).delete()
            db.query(Job).delete()
            db.commit()
    _wipe()
    yield
    _wipe()


def _job(db, **kw):
    d = dict(hash=kw.pop("hash", "j1"), source="greenhouse:test", title="Research Scientist",
             company="Acme Corporation", url="https://x", description="d", match_score=0.6)
    d.update(kw)
    j = Job(**d); db.add(j); db.commit(); db.refresh(j); return j


def test_config_defaults_present():
    assert settings.referral_bonus_cap == 0.15
    assert settings.referral_discover_max == 15
    assert settings.contacts_max_per_company == 5
    assert settings.affiliation_terms == ""


def test_contact_table_roundtrips():
    init_db()
    with session() as db:
        c = Contact(name="Dr. Alex Taylor", company="Acme Research Institute",
                    title="Principal Investigator", relationship="pi",
                    source="manual", pinned=True, notes="insider")
        db.add(c); db.commit(); db.refresh(c)
        got = db.query(Contact).filter_by(name="Dr. Alex Taylor").one()
        assert got.pinned is True and got.company.startswith("Acme")


from jobbot import referrals


class _CSnap:
    def __init__(self, name="X", company="Acme Corporation", title="",
                 relationship="unknown", pinned=False, notes=""):
        self.name = name; self.company = company; self.title = title
        self.relationship = relationship; self.pinned = pinned; self.notes = notes


def test_warmth_pinned_beats_recruiter():
    terms = {"rna", "biochemistry"}
    hi, _ = referrals.warmth_score(_CSnap(pinned=True, title="Principal Investigator"), terms)
    lo, _ = referrals.warmth_score(_CSnap(title="Technical Recruiter"), terms)
    assert hi > lo
    assert 0.0 <= hi <= 1.0 and 0.0 <= lo <= 1.0


def test_warmth_shared_field_and_failopen():
    terms = {"rna aptamer", "molecular docking"}
    s, why = referrals.warmth_score(_CSnap(title="RNA aptamer scientist"), terms)
    assert s > 0 and why
    # fail-open on a broken contact (no attributes)
    assert referrals.warmth_score(object(), terms) == (0.0, "")


def test_contacts_for_job_matches_company_tokens():
    init_db()
    with session() as db:
        j = _job(db, company="Acme Corporation")
        db.add(Contact(name="A", company="Acme", pinned=True, source="manual"))
        db.add(Contact(name="B", company="GlobalCorp", source="discovered"))
        db.commit()
        jj = db.query(Job).get(j.id)
        idx = referrals.build_index()
        matched = referrals.contacts_for_job(jj, idx)
    names = [c.name for c in matched]
    assert "A" in names and "B" not in names


def test_center_alone_is_not_a_warm_match():
    """Regression: a lone generic token ('center') must not link a contact to an
    unrelated employer. Taylor @ 'Acme Research Center' must NOT match a
    'Metro Medical Center' job (they only share the non-distinctive 'center')."""
    init_db()
    with session() as db:
        j = _job(db, company="Metro Medical Center")
        db.add(Contact(name="Dr. Alex Taylor", company="Acme Research Institute",
                       pinned=True, relationship="pi", source="manual"))
        db.commit()
        jj = db.query(Job).get(j.id)
        idx = referrals.build_index()
        matched = referrals.contacts_for_job(jj, idx)
        bonus, _ = referrals.referral_bonus(jj, idx)
    assert matched == []
    assert bonus == 0.0


def test_alias_pattern_links_reworded_company():
    """A contact's alias substring patterns link them to their org even when the
    job lists the company under a different name. Taylor's 'department of health'
    alias must match a 'Health, Department of' job (which tokenizes to nothing)
    while still NOT matching 'Department of Taxation and Finance'."""
    init_db()
    with session() as db:
        j_doh = _job(db, hash="doh", company="Health, Department of", match_score=0.6)
        j_tax = _job(db, hash="tax", company="Department of Taxation and Finance",
                     match_score=0.6)
        db.add(Contact(name="Dr. Alex Taylor", company="Acme Research Institute",
                       company_aliases="acme; department of health; health, department of",
                       pinned=True, relationship="pi", source="manual"))
        db.commit()
        idx = referrals.build_index()
        doh_hit = [c.name for c in referrals.contacts_for_job(db.query(Job).get(j_doh.id), idx)]
        tax_hit = [c.name for c in referrals.contacts_for_job(db.query(Job).get(j_tax.id), idx)]
        bonus_doh, why = referrals.referral_bonus(db.query(Job).get(j_doh.id), idx)
    assert "Dr. Alex Taylor" in doh_hit
    assert "Dr. Alex Taylor" not in tax_hit
    assert bonus_doh > 0 and why


def test_field_gate_suppresses_low_match_warm_intro():
    """An org match on an off-field (low match_score) job yields no boost/badge."""
    init_db()
    with session() as db:
        low = _job(db, hash="low", company="Acme Corporation",
                   match_score=settings.warm_intro_min_score - 0.05)
        db.add(Contact(name="A", company="Acme", pinned=True, source="manual"))
        db.commit()
        idx = referrals.build_index()
        bonus, why = referrals.referral_bonus(db.query(Job).get(low.id), idx)
    assert bonus == 0.0 and why == ""


def test_seed_sample_lead_sets_doh_aliases():
    init_db()
    c = referrals.seed_sample_lead()
    assert c is not None
    assert "department of health" in (c.company_aliases or "").lower()


def test_referral_bonus_bounded_and_zero_without_match():
    init_db()
    with session() as db:
        j = _job(db, company="Acme Corporation")
        db.add(Contact(name="A", company="Acme", pinned=True, source="manual"))
        db.commit()
        jj = db.query(Job).get(j.id)
        idx = referrals.build_index()
        bonus, why = referrals.referral_bonus(jj, idx)
        # a company with no contact
        j2 = _job(db, hash="j2", company="Nowhere Labs")
        jj2 = db.query(Job).get(j2.id)
        bonus2, _ = referrals.referral_bonus(jj2, idx)
    assert 0 < bonus <= settings.referral_bonus_cap + 1e-9
    assert why
    assert bonus2 == 0.0


def test_discover_company_stores_deduped(monkeypatch):
    import jobbot.contacts as gc
    monkeypatch.setattr(gc, "search_configured", lambda: True)
    fixture = [{"name": "Jane Doe", "role": "Scientist", "email": "",
                "linkedin": "https://linkedin.com/in/jane", "source_url": "u"}]
    monkeypatch.setattr(gc, "_contacts_from_recruiter_search", lambda job: fixture)
    monkeypatch.setattr(gc, "_contacts_from_team_search", lambda job: fixture)  # dup
    init_db()
    n = referrals.discover_company("Acme Corporation")
    with session() as db:
        rows = db.query(Contact).filter_by(company="Acme Corporation").all()
    assert len(rows) == 1 and rows[0].source == "discovered"  # deduped across sources
    assert len(n) == 1
    # second run adds nothing new (dedupe on (name, company))
    referrals.discover_company("Acme Corporation")
    with session() as db:
        assert db.query(Contact).filter_by(name="Jane Doe").count() == 1


def test_discover_company_noop_without_provider(monkeypatch):
    import jobbot.contacts as gc
    monkeypatch.setattr(gc, "search_configured", lambda: False)
    init_db()
    assert referrals.discover_company("Anything") == []


def test_seed_sample_lead_idempotent():
    init_db()
    c1 = referrals.seed_sample_lead()
    c2 = referrals.seed_sample_lead()
    assert c1 is not None and c2 is None
    with session() as db:
        rows = db.query(Contact).filter(Contact.name.like("%Taylor%")).all()
    assert len(rows) == 1 and rows[0].pinned is True


def test_contacts_page_and_crud():
    from jobbot.web import create_app
    init_db()
    c = create_app().test_client()
    assert c.get("/contacts").status_code == 200
    r = c.post("/contacts", data={"name": "Jane Roe", "company": "Acme",
                                   "title": "Scientist", "relationship": "colleague"},
               follow_redirects=True)
    assert r.status_code == 200
    with session() as db:
        row = db.query(Contact).filter_by(name="Jane Roe").one()
        cid = row.id
    assert c.post(f"/contacts/{cid}/pin", follow_redirects=True).status_code == 200
    with session() as db:
        assert db.query(Contact).get(cid).pinned is True
    assert c.post(f"/contacts/{cid}/delete", follow_redirects=True).status_code == 200
    with session() as db:
        assert db.query(Contact).get(cid) is None


def test_index_shows_warm_intro_badge_for_matched_job():
    from jobbot.web import create_app
    init_db()
    with session() as db:
        _job(db, company="Acme Corporation", match_score=0.4)  # lower score
        db.add(Contact(name="Ada", company="Acme", pinned=True, source="manual"))
        db.commit()
    body = create_app().test_client().get("/?scope=all").get_data(as_text=True)
    assert body.count("Warm intro") >= 1 or "warm-intro" in body


def test_warm_intros_preset_filters(monkeypatch):
    from jobbot.web import create_app
    init_db()
    with session() as db:
        _job(db, hash="wi1", company="Acme Corporation")
        _job(db, hash="wi2", company="Nowhere Labs")
        db.add(Contact(name="Ada", company="Acme", source="manual"))
        db.commit()
    body = create_app().test_client().get("/?scope=all&preset=warm-intros").get_data(as_text=True)
    assert "Acme" in body and "Nowhere Labs" not in body


def test_kit_route_invokes_build_package(monkeypatch):
    from jobbot.web import create_app
    import jobbot.networking_package as np
    calls = {}
    monkeypatch.setattr(np, "build_package",
                        lambda **kw: calls.update(kw) or {"dir": "/tmp/kit"})
    init_db()
    with session() as db:
        j = _job(db, company="Acme Corporation")
        db.add(Contact(name="Jane", company="Acme", source="manual"))
        db.commit()
        cid = db.query(Contact).filter_by(name="Jane").one().id
        jid = j.id
    cl = create_app().test_client()
    r = cl.post(f"/contacts/{cid}/kit", data={"job_id": jid}, follow_redirects=True)
    assert r.status_code == 200
    assert calls.get("name") == "Jane"


def test_sort_param_orders_and_renders():
    """Each sort mode returns 200 and the sort selector reflects the choice."""
    from jobbot.web import create_app
    init_db()
    with session() as db:
        _job(db, hash="s1", company="Zeta Labs", title="Alpha role", match_score=0.9)
        _job(db, hash="s2", company="Acme Bio", title="Beta role", match_score=0.5)
        db.commit()
    cl = create_app().test_client()
    for mode in ("smart", "match", "new", "warm", "company"):
        r = cl.get(f"/?scope=all&sort={mode}")
        assert r.status_code == 200
        assert f'value="{mode}" selected' in r.get_data(as_text=True)


def test_warm_intro_top_job_not_cut_off():
    """A warm-intro job with a modest match_score must not be truncated below the
    display cap: scoring happens over the full set before the slice."""
    from jobbot.web import create_app
    init_db()
    with session() as db:
        # A warm-intro job at a lower raw score than a plain high-score job.
        _job(db, hash="warm", company="Acme Corporation",
             title="Referral Scientist", match_score=0.40)
        _job(db, hash="plain", company="Nowhere Labs",
             title="Plain Scientist", match_score=0.85)
        db.add(Contact(name="Ada", company="Acme", pinned=True, source="manual"))
        db.commit()
    body = create_app().test_client().get("/?scope=all&sort=warm").get_data(as_text=True)
    # Warm-intro job appears and sorts ahead of the plain higher-score job.
    assert body.index("Referral Scientist") < body.index("Plain Scientist")


class _CSnapSrc:
    def __init__(self, source="import", company="Acme Corporation",
                 title="", relationship="unknown", pinned=False, notes=""):
        self.name = "X"; self.company = company; self.title = title
        self.relationship = relationship; self.pinned = pinned; self.notes = notes
        self.source = source


def test_import_source_adds_warmth():
    terms = set()
    imp, why = referrals.warmth_score(_CSnapSrc(source="import"), terms)
    man, _ = referrals.warmth_score(_CSnapSrc(source="manual"), terms)
    assert imp > man
    assert "1st-degree connection" in why
    assert 0.0 <= imp <= 1.0


def test_warmth_still_failopen_without_source():
    # object() has no .source -> getattr default, no crash
    assert referrals.warmth_score(object(), set()) == (0.0, "")


def test_cli_contact_add_and_seed():
    from click.testing import CliRunner
    from jobbot.cli import cli
    init_db()
    r = CliRunner().invoke(cli, ["contact", "add", "Jane Roe", "--company", "Acme"])
    assert r.exit_code == 0
    with session() as db:
        assert db.query(Contact).filter_by(name="Jane Roe").count() == 1
    r2 = CliRunner().invoke(cli, ["referrals", "seed-lead"])
    assert r2.exit_code == 0
    with session() as db:
        assert db.query(Contact).filter(Contact.name.like("%Taylor%")).count() == 1
