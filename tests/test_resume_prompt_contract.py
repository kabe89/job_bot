"""Regression guard for SYSTEM_RESUME_TAILOR, the contract every provider shares.

Each rule here exists because the local pipeline actually produced the defect on
a real application (Innotech R19096, job 3968):

  - "PyTorch/TensorFlow/Keras (learning phase)", "Exploring R/C++ for broader
    applicability", "expanding proficiency with advanced frameworks"
        -> forbidden from FABRICATING a skill, the model confessed weakness on
           the resume instead. Truthful, and catastrophic.
  - "Developed skills ... relevant to bioinformatics workflows" on a campus
    sustainability job -> editorialising relevance.
  - dropped the candidate's real first-author Water (2022) publication.
  - "Jane Doe, Ph.D." for a degree expected Jan 2027.
  - "Example State University - Springfield, IL" (the candidate's home town, not the
    university's location); "RT-qPCR" where the source says "RT-PCR".

These are content assertions on a prompt, so they are deliberately loose -- they
check the RULE is present, not its exact wording.
"""
import pytest

from jobbot.gemini_client import SYSTEM_RESUME_TAILOR as CONTRACT

_LOW = CONTRACT.lower()


def test_fabrication_is_still_forbidden():
    assert "only source of truth" in _LOW
    assert "must not add it" in _LOW


@pytest.mark.parametrize("phrase", [
    "learning phase",
    "exploring",
    "expanding proficiency",
    "currently learning",
])
def test_contract_names_the_hedges_it_forbids(phrase):
    """Naming the exact bad strings is what makes the rule stick for a small
    local model -- an abstract "don't hedge" did not."""
    assert phrase in _LOW


def test_hedging_rule_says_omit_rather_than_downgrade():
    assert "omit" in _LOW
    assert "drop it entirely" in _LOW


def test_contract_forbids_editorialising_relevance():
    assert "editorialize" in _LOW or "editorialise" in _LOW


def test_contract_forbids_dropping_the_record():
    assert "never drop a publication" in _LOW


def test_contract_forbids_upgrading_a_credential():
    assert "expected/in progress" in _LOW


def test_contract_requires_exact_institutions_and_techniques():
    assert "exactly as the source" in _LOW
    assert "rt-qpcr" in _LOW          # the real near-neighbour slip, named


def test_contract_is_shared_by_every_provider():
    """ollama_client and claude_client import the contract from gemini_client as
    _g rather than keeping their own copy -- so this guard covers all three."""
    from jobbot import claude_client, ollama_client
    assert ollama_client._g.SYSTEM_RESUME_TAILOR is CONTRACT
    assert claude_client._g.SYSTEM_RESUME_TAILOR is CONTRACT
