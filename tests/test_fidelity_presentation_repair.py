"""A dropped presentation is restored, not merely reported.

Detection alone was not enough. The refiner is handed the defect in its prompt
with instructions and still ships the draft without the talk, exactly as the
module docstring already records for invented dates: it is sampling noise, so
"the model will fix it next round" is not a plan.

Restoring a presentation IS mechanically unambiguous when the draft has a
Publications/Presentations section carrying the talk's siblings: the source line
is known verbatim and the destination is the section its siblings are already in.
Where that is not true, the entry stays a reported defect rather than being
guessed into place.
"""
from jobbot.source_fidelity import check_fidelity, repair_fidelity

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

DRAFT_MISSING_2026 = """# Jane M. Doe

### Publications & Presentations
- Doe, J.M.; Ingenito, S.; Johnson, L. Developing Neural Networks for Systems Using
  a Standard Optimization Approach. The Computing Systems Conference; June 2024;
  Austin, TX.

### Awards & Certifications
- Academic Honors
"""

DRAFT_NO_SECTION = """# Jane M. Doe

### Professional Summary
A scientist.

### Awards & Certifications
- Academic Honors
"""


def test_the_dropped_talk_is_restored():
    fixed, defects = repair_fidelity(SOURCE, DRAFT_MISSING_2026)
    assert "Chicago" in fixed, fixed
    assert "Hybrid" in fixed or "hybrid" in fixed.lower()


def test_restored_talk_clears_the_defect():
    _fixed, defects = repair_fidelity(SOURCE, DRAFT_MISSING_2026)
    assert not [d for d in defects if "presentation" in d.lower()], defects


def test_restored_into_the_presentations_section_not_appended_at_the_end():
    fixed, _ = repair_fidelity(SOURCE, DRAFT_MISSING_2026)
    pres_at = fixed.index("Publications & Presentations")
    awards_at = fixed.index("Awards & Certifications")
    chicago_at = fixed.index("Chicago")
    assert pres_at < chicago_at < awards_at, "restored entry landed outside its section"


def test_sibling_entry_is_left_intact():
    """The draft wraps entries across lines, so compare on collapsed whitespace."""
    import re
    fixed, _ = repair_fidelity(SOURCE, DRAFT_MISSING_2026)
    flat = re.sub(r"\s+", " ", fixed)
    assert flat.count("June 2024; Austin, TX.") == 1, flat
    assert "Standard Optimization Approach" in flat


def test_section_separation_is_preserved():
    fixed, _ = repair_fidelity(SOURCE, DRAFT_MISSING_2026)
    assert "\n\n### Awards & Certifications" in fixed, (
        "the blank line before the next heading was eaten")


def test_bullet_style_matches_the_surrounding_entries():
    fixed, _ = repair_fidelity(SOURCE, DRAFT_MISSING_2026)
    line = next(l for l in fixed.splitlines() if "Chicago" in l)
    assert line.lstrip().startswith("- "), repr(line)
    # source markdown emphasis must not leak into the tailored document
    assert "**" not in line and "*Developing" not in line, repr(line)


def test_no_presentations_section_leaves_a_reported_defect():
    fixed, defects = repair_fidelity(SOURCE, DRAFT_NO_SECTION)
    assert "Chicago" not in fixed, "must not guess a location for the entry"
    assert any("presentation" in d.lower() for d in defects)


def test_a_clean_draft_is_untouched():
    clean = DRAFT_MISSING_2026.replace(
        "### Awards & Certifications",
        "- Doe, J.M.; Johnson, L. Developing Neural Networks for Systems Using a "
        "Hybrid Optimization Machine Learning Approach. The Computing Systems "
        "Conference; June 2026; Chicago, IL.\n\n### Awards & Certifications")
    fixed, defects = repair_fidelity(SOURCE, clean)
    assert fixed.count("Chicago") == 1, "no duplicate insertion"
    assert not [d for d in defects if "presentation" in d.lower()]


def test_repair_never_raises_on_junk_input():
    for src, drf in (("", ""), (SOURCE, ""), ("", DRAFT_MISSING_2026)):
        fixed, defects = repair_fidelity(src, drf)
        assert isinstance(fixed, str)
        assert isinstance(defects, list)
