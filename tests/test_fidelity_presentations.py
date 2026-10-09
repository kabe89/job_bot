"""Dropping a conference presentation must be caught, like a dropped publication.

Real defect, 2026-09-10. `_check_dropped_publications` keys on DOIs, and
presentations have none, so the June 2026 conference talk silently vanished
from every tailored resume while the fidelity report came back clean.

The tailoring prompt already says "Never drop a publication, presentation,
degree, award, or role that is in the source". It was ignored. This is the
deterministic check that makes it stick.
"""
from jobbot.source_fidelity import _check_dropped_presentations, check_fidelity


SOURCE = """
## Presentations

* **Doe, J.M.**; Johnson, L. *Developing Neural Networks for Systems Using a Hybrid
  Optimization Machine Learning Approach.* The Computing Systems Conference;
  June 2026; Chicago, IL.
* **Doe, J.M.**; Ingenito, S.; Johnson, L. *Developing Neural Networks for Systems
  Using a Standard Optimization Approach.* The Computing Systems Conference; June 2024;
  Austin, TX.

## Awards
"""

KEPT_BOTH = """
### Publications & Presentations
- Doe, J.M.; Johnson, L. Developing Neural Networks for Systems Using a Hybrid
  Optimization Machine Learning Approach. The Computing Systems Conference; June
  2026; Chicago, IL.
- Doe, J.M.; Ingenito, S.; Johnson, L. Developing Neural Networks for Systems Using
  a Standard Optimization Approach. The Computing Systems Conference; June 2024;
  Austin, TX.
"""

DROPPED_THE_2026 = """
### Publications & Presentations
- Doe, J.M.; Ingenito, S.; Johnson, L. Developing Neural Networks for Systems Using
  a Standard Optimization Approach. The Computing Systems Conference; June 2024;
  Austin, TX.
"""


def test_keeping_both_presentations_is_clean():
    assert _check_dropped_presentations(SOURCE, KEPT_BOTH) == []


def test_dropping_one_presentation_is_reported():
    out = _check_dropped_presentations(SOURCE, DROPPED_THE_2026)
    assert len(out) == 1
    # The message names the talk by its discriminating words, lowercased.
    assert "hybrid" in out[0].lower(), out[0]


def test_the_similar_title_is_not_mistaken_for_the_dropped_one():
    """The two talks share most of their words; only the distinguishing part differs.

    A naive substring check passes here because 'Developing Neural Networks for
    Systems Using a Standard Optimization Approach' is present. The check must key on
    what makes the entry unique.
    """
    out = _check_dropped_presentations(SOURCE, DROPPED_THE_2026)
    assert out, "the 2026 talk was dropped but not reported"


def test_reformatting_is_not_treated_as_a_drop():
    """Tailoring reorders and restyles; that is allowed."""
    reflowed = ("Presentations: Johnson L. and Doe JM, Developing Neural Networks "
                "for Systems Using a Hybrid Optimization Machine Learning "
                "Approach, The Computing Systems Conference, Chicago IL, June 2026. "
                "Also: Developing Neural Networks for Systems Using a Standard "
                "Optimization Approach, The Computing Systems Conference, Austin TX, June 2024.")
    assert _check_dropped_presentations(SOURCE, reflowed) == []


def test_no_presentations_section_yields_nothing():
    assert _check_dropped_presentations("## Skills\n- Python\n", "anything") == []


def test_empty_inputs_are_safe():
    assert _check_dropped_presentations("", "") == []
    assert _check_dropped_presentations(SOURCE, "") != []


def test_wired_into_check_fidelity_as_an_omission():
    r = check_fidelity(SOURCE, DROPPED_THE_2026)
    assert r.omissions, "a dropped presentation must surface as an omission"
    assert not r.ok
    assert r.severe, "dropping a presentation must block the refinement early-stop"
