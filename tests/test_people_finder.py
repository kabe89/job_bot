# tests/test_people_finder.py
from jobbot import people_finder as pf


PROFILE_MD = """\
# Jane Doe

## Summary
Computational structural biologist.

## Publications
- Doe J, Taylor NA, Smith J. Distributed consensus algorithms. J Mol Biol. 2023.
- Chen L and Doe J. High throughput data analysis. Biophys J. 2022.

## Skills
Python, OpenMM
"""


def test_publications_block_extracts_only_that_section():
    block = pf._publications_block(PROFILE_MD)
    assert "Distributed consensus algorithms" in block
    assert "Computational structural biologist" not in block
    assert "OpenMM" not in block


def test_seed_coauthors_from_text_drops_self(monkeypatch):
    # Isolate from real files + live PubMed.
    monkeypatch.setattr(pf, "_read_sources", lambda: PROFILE_MD)
    monkeypatch.setattr(pf, "_pubmed_coauthors", lambda *a, **k: set())
    monkeypatch.setattr(pf.settings, "applicant_name", "Jane Doe", raising=False)
    seeds = pf._seed_coauthors()
    assert "taylor na" in seeds
    assert "chen l" in seeds
    # the user themselves is excluded (matches surname "doe")
    assert not any("doe" in s for s in seeds)


def test_seed_coauthors_excludes_journal_abbreviations(monkeypatch):
    # Isolate from real files + live PubMed.
    monkeypatch.setattr(pf, "_read_sources", lambda: PROFILE_MD)
    monkeypatch.setattr(pf, "_pubmed_coauthors", lambda *a, **k: set())
    monkeypatch.setattr(pf.settings, "applicant_name", "Jane Doe", raising=False)
    seeds = pf._seed_coauthors()
    # real authors survive
    assert "taylor na" in seeds
    assert "chen l" in seeds
    # journal-abbreviation fragments are filtered out
    assert "mol biol" not in seeds
    assert "biophys j" not in seeds


def test_pubmed_coauthors_failopen(monkeypatch):
    # Any network error -> empty set, never raises.
    def boom(*a, **k):
        raise RuntimeError("no network")
    monkeypatch.setattr(pf.requests, "get", boom)
    assert pf._pubmed_coauthors("Jane Doe") == set()


def test_thread_score_ranks_coauthor_over_field():
    seeds = {"taylor na"}
    terms = {"rna", "docking"}
    affils = ["suny"]
    coauth, why1 = pf._thread_score(
        {"name": "Taylor NA", "role": "Scientist at Acme"}, seeds, terms, affils)
    field, why2 = pf._thread_score(
        {"name": "Someone Else", "role": "RNA docking scientist"}, seeds, terms, affils)
    inst, _ = pf._thread_score(
        {"name": "Third Person", "role": "SUNY-trained chemist"}, seeds, terms, affils)
    assert coauth > inst > field > 0
    assert "coauthor" in why1.lower()
    assert why2  # shared-field rationale present


def test_find_shared_thread_contacts_sorts_and_tags(monkeypatch):
    monkeypatch.setattr(pf, "_seed_coauthors", lambda: {"taylor na"})
    monkeypatch.setattr(pf.referrals, "profile_terms", lambda: {"rna"})
    monkeypatch.setattr(pf.settings, "affiliation_terms", "suny", raising=False)
    monkeypatch.setattr(pf._contacts, "find_contacts", lambda job_id, save=False: [
        {"name": "Zed Field", "role": "RNA scientist", "email": "", "linkedin": "l1"},
        {"name": "Taylor NA", "role": "PI at Acme", "email": "b@x.org", "linkedin": "l2"},
    ])
    out = pf.find_shared_thread_contacts(job_id=1)
    assert out[0]["name"] == "Taylor NA"        # coauthor ranks first
    assert "coauthor" in out[0]["thread"].lower()
    assert all("thread_score" in p for p in out)


def test_find_shared_thread_contacts_failopen(monkeypatch):
    def boom(*a, **k):
        raise RuntimeError("x")
    monkeypatch.setattr(pf._contacts, "find_contacts", boom)
    assert pf.find_shared_thread_contacts(job_id=1) == []


def test_thread_score_matches_display_order_coauthor():
    seeds = {"taylor a"}
    score, why = pf._thread_score(
        {"name": "Alex Taylor", "role": "Scientist at TechCorp"},
        seeds, set(), [])
    assert score == 0.6
    assert "coauthor" in why.lower()


def test_thread_score_no_coauthor_for_unrelated_name():
    seeds = {"taylor na"}
    score, why = pf._thread_score(
        {"name": "Wei Chen", "role": "Analyst at Acme"},
        seeds, set(), [])
    assert score == 0
    assert "coauthor" not in why.lower()
