"""Derived dealbreakers must be ROLE types, never geography.

Dealbreakers hard-cap the fit judge at 0.1, and location is already handled
separately by user_locations.txt / Job.is_local. A location dealbreaker
double-counts it and silently caps every out-of-area job.

Real regression, 2026-09-10: a profile rebuild emitted

    "Positions outside the Pacific Northwest or remote-friendly tech
     settings (as per candidate preference)"

which would have capped Cambridge MA, San Diego and San Francisco - the entire
top tier of the shortlist - at 0.1.

The model is not reliable enough to be trusted on this by prompt alone, so the
filter is deterministic and lives in code.
"""
import pytest

from jobbot.profile import CandidateProfile, _drop_non_role_dealbreakers


LOCATION_SHAPED = [
    "Positions outside the Pacific Northwest or remote-friendly tech settings",
    "Roles that are not remote-friendly",
    "Jobs requiring relocation outside New York",
    "Positions located outside the Capital Region",
    "Roles requiring on-site presence in another state",
    "Anything not within commuting distance of Seattle",
]

ROLE_SHAPED = [
    "Roles whose primary function is synthetic medicinal chemistry bench work",
    "People-management roles requiring prior direct reports",
    "Field-based commercial roles such as Medical Science Liaison or sales",
    "Positions requiring clinical trial biostatistics and SAS programming",
    "Roles requiring extensive prior industry experience beyond doctoral training",
]


@pytest.mark.parametrize("entry", LOCATION_SHAPED)
def test_location_dealbreakers_are_dropped(entry):
    assert _drop_non_role_dealbreakers([entry]) == []


@pytest.mark.parametrize("entry", ROLE_SHAPED)
def test_role_dealbreakers_are_kept(entry):
    assert _drop_non_role_dealbreakers([entry]) == [entry]


def test_mixed_list_keeps_only_role_entries():
    mixed = [ROLE_SHAPED[0], LOCATION_SHAPED[0], ROLE_SHAPED[1]]
    assert _drop_non_role_dealbreakers(mixed) == [ROLE_SHAPED[0], ROLE_SHAPED[1]]


def test_empty_and_blank_entries_are_dropped():
    assert _drop_non_role_dealbreakers(["", "   ", None]) == []


def test_from_dict_applies_the_filter():
    """The filter must run on the real construction path, not just standalone."""
    p = CandidateProfile.from_dict({
        "skills": [], "role_titles": [], "role_archetypes": [], "domains": [],
        "seniority": "mid",
        "dealbreakers": [LOCATION_SHAPED[0], ROLE_SHAPED[0]],
        "summary": "",
    })
    assert LOCATION_SHAPED[0] not in p.dealbreakers
    assert ROLE_SHAPED[0] in p.dealbreakers


def test_salary_and_seniority_shaped_entries_are_dropped():
    """Same reasoning: both are scored elsewhere, so capping on them double-counts."""
    assert _drop_non_role_dealbreakers(["Roles paying below $100,000"]) == []
    assert _drop_non_role_dealbreakers(["Positions offering less than market salary"]) == []
