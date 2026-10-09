"""Deterministic source-fidelity checks for a tailored resume.

WHY THIS IS CODE AND NOT A PROMPT RULE
--------------------------------------
The critic prompt already names all three defects below, explicitly, including
the literal example 'source says "Expected 2027", draft invents "Expected
January 2027"' and "any renamed institution or relocated employer". The local
model still missed all three on a real run (Innotech 3968) and self-scored 0.95.

Evidence that it was NOT truncation (measured, not assumed):
    critic prompt ~5,947 tokens vs num_ctx 8,192  -> fits
    Water 2022 DOI at char 4,494 of profile (cap 6,000) -> reaches the model
    same DOI at char 5,131 of base resume (cap 8,000)   -> reaches the model

What the critic DID report as omissions were three whole job roles -- large,
salient blocks that tailoring had correctly dropped as irrelevant. It found the
big things and missed a one-line citation. That is a needle-in-haystack limit of
a small model diffing 24k chars, and two rounds of prompt-strengthening did not
move it.

So the diff moves into code. A DOI is a literal string; a date is a token. These
need judgment from nobody. The LLM keeps the judgment work (is this bullet
relevant, is this phrasing strong); code keeps the verification work.

Each case below is from the real Innotech 3968 run.
"""
import pytest

from jobbot.source_fidelity import check_fidelity

_SOURCE = """
Jane M. Doe
Springfield, IL - jane.doe@example.com

Education
- Ph.D. Systems Engineering (Expected 2027) - Example State University. Doctoral Candidate, 1/2021-Present.
- B.S. Computer Science - Example State University, 8/2017-11/2020. Academic Honors.

Publications
- Smith, S.R.; Doe, J.M.; et al. Synthetic Models for Targeted Sensing. Journal of Systems 15, 32405 (2025).
  https://doi.org/10.1000/182-demo-systems
- Doe, J.M.; et al. Empirical Analysis of Network Latency in Distributed Systems. Network Science 2022, 14, 2137.
  https://doi.org/10.1000/182-demo-network
"""


def _draft(**over):
    base = {
        "doi_1": "https://doi.org/10.1000/182-demo-network",
        "doi_2": "https://doi.org/10.1000/182-demo-systems",
        "phd": "Ph.D. Systems Engineering (Expected 2027)",
        "uni": "Example State University",
        # The source says Academic Honors, so a FAITHFUL draft must keep it.
        # Omitting it here made the baseline draft genuinely defective once the
        # credential check landed -- the checker was right and this fixture was
        # incomplete.
        "honors": "Academic Honors",
    }
    base.update(over)
    return f"""# Jane M. Doe, Ph.D. Candidate
Springfield, IL

### Education
**{base['phd']}** | {base['uni']}
B.S. Computer Science - {base['honors']}

### Publications
- Journal of Systems 15, 32405 (2025). {base['doi_2']}
- Network Science 2022, 14, 2137. {base['doi_1']}
"""


def test_a_faithful_draft_reports_no_defects():
    """Guard against a checker that fires on everything."""
    r = check_fidelity(_SOURCE, _draft())
    assert r.ok, f"clean draft flagged: {r.omissions} {r.fabrications}"


# --- (1) the dropped peer-reviewed publication -------------------------------

def test_a_dropped_doi_is_caught():
    """Verify that omitting a publication DOI is flagged as an omission."""
    d = _draft(doi_1="")
    r = check_fidelity(_SOURCE, d)
    assert not r.ok
    assert any("10.1000/182-demo-network" in o for o in r.omissions)


def test_a_dropped_doi_is_severe_enough_to_block_early_stop():
    r = check_fidelity(_SOURCE, _draft(doi_2=""))
    assert r.severe is True


def test_every_source_doi_is_checked_not_just_the_first():
    r = check_fidelity(_SOURCE, _draft(doi_2=""))
    assert not r.ok
    assert any("182-demo-systems" in o for o in r.omissions)


def test_dois_are_matched_case_insensitively():
    """DOIs are case-insensitive by spec; a case flip is not an omission."""
    r = check_fidelity(_SOURCE, _draft(doi_2="https://doi.org/10.1000/182-DEMO-SYSTEMS"))
    assert r.ok


def test_a_bare_doi_without_the_url_prefix_still_counts():
    r = check_fidelity(_SOURCE, _draft(doi_1="doi:10.1000/182-demo-network"))
    assert r.ok


# --- (2) the re-invented date precision --------------------------------------

def test_inventing_a_month_on_an_expected_year_is_caught():
    """Ensure adding an unverified month to an expected graduation year is flagged."""
    r = check_fidelity(_SOURCE, _draft(phd="Ph.D. Systems Engineering (Expected Jan 2027)"))
    assert not r.ok
    assert any("2027" in f for f in r.fabrications)


@pytest.mark.parametrize("month", ["January", "Jan", "May", "December"])
def test_any_invented_month_is_caught(month):
    r = check_fidelity(_SOURCE, _draft(phd=f"Ph.D. Systems Engineering (Expected {month} 2027)"))
    assert not r.ok


def test_the_bare_expected_year_is_accepted():
    r = check_fidelity(_SOURCE, _draft(phd="Ph.D. Systems Engineering (Expected 2027)"))
    assert r.ok


# --- (3) the relocated institution -------------------------------------------

def test_relocating_the_university_to_the_home_town_is_caught():
    """Ensure the model does not improperly attach contact header cities to institutions."""
    r = check_fidelity(_SOURCE, _draft(uni="Example State University - Springfield, IL"))
    assert not r.ok
    assert any("Springfield" in f for f in r.fabrications)


def test_the_home_town_in_the_contact_header_is_still_fine():
    """Springfield belongs in the header -- only the employer pairing is wrong."""
    r = check_fidelity(_SOURCE, _draft())
    assert r.ok
    assert "Springfield, IL" in _draft()


# --- (4) banned AI buzzwords and corporate clichés ---------------------------

def test_banned_ai_buzzwords_are_detected_and_repaired():
    from jobbot.source_fidelity import check_banned_buzzwords, repair_fidelity
    bad_draft = (
        "# Jane M. Doe\n\n"
        "- Spearheaded a new engineering pipeline.\n"
        "- Utilized AutoDock Vina for modeling.\n"
        "- Executed end-to-end experimental workflows.\n"
        "- Leveraged Python for analytics.\n"
    )
    findings = check_banned_buzzwords(bad_draft)
    assert len(findings) >= 4
    assert any("spearheaded" in f.lower() for f in findings)
    assert any("utilized" in f.lower() for f in findings)
    assert any("executed" in f.lower() for f in findings)

    repaired, remaining = repair_fidelity(_SOURCE, bad_draft)
    assert "spearheaded" not in repaired.lower()
    assert "utilized" not in repaired.lower()
    assert "executed end-to-end experimental workflows" not in repaired.lower()
    assert "Led a new engineering pipeline" in repaired
    assert "Used AutoDock Vina" in repaired
    assert "Conducted experimental protocols" in repaired
