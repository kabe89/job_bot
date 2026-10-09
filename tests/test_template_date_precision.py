"""The house style must not order the model to invent a month.

The template's format rule reads:

    Dates as `Mon YYYY - Mon YYYY` (or `- Present`).

The source says "Expected 2027" -- no month. Obeying the format therefore
REQUIRES manufacturing one, and the model did, picking a different month on each
run of the same job (Innotech 3968):

    run 2: "Expected Jan 2027"
    run 3: "Expected Dec 2027"

The template already contradicted itself two rules later ("Copy URLs, DOIs,
dates, citations ... VERBATIM from the source - never reformat digits or
guess"), so the model was obeying whichever rule it read last. It was not being
careless; it was resolving a conflict we handed it.

The deterministic checker in source_fidelity catches the bad output, but
detection is not repair: the refinement loop stayed pinned at 0.50 for both
rounds because the model kept re-satisfying the format rule. The fix is to
remove the conflict.
"""
import re
from pathlib import Path

import pytest

_TEMPLATE = Path("data/resume_template.md")


@pytest.fixture(scope="module")
def style() -> str:
    """Whitespace-normalised, so these content assertions do not break the next
    time someone re-wraps a line in the template."""
    if not _TEMPLATE.exists():
        pytest.skip("resume_template.md not present")
    return re.sub(r"\s+", " ", _TEMPLATE.read_text(encoding="utf-8"))


def test_the_date_format_rule_still_exists(style):
    """Guard the fix's premise -- if the format rule is gone, this test is moot."""
    assert "Mon YYYY" in style


def test_the_format_rule_carries_an_explicit_escape_hatch(style):
    low = style.lower()
    assert "never invent precision" in low
    assert "only a year" in low


def test_the_exact_regression_string_is_named(style):
    """An abstract rule did not stick on the small local model; naming the exact
    case is what made the hedging and fabrication rules work."""
    assert "Expected 2027" in style


def test_the_verbatim_rule_is_declared_to_win(style):
    """The two rules genuinely conflict. Say which one loses."""
    low = style.lower()
    assert "verbatim" in low
    assert "loses to the verbatim rule" in low
