"""Leaf redaction, the truncation-fragment sweep, and the fail-closed gate.

The `.json` capture is the more dangerous of the two recon files: `.html` comes
from cloneNode(true), which copies ATTRIBUTES and not live control state, while
the JSON records `el.value` - the live property, i.e. what the applicant
actually typed. It was the file left unprotected.

Every test seeds `pairs=` explicitly. A bare identity_tokens() call would read
conftest's seeded fixtures and make these assertions depend on the environment.
No token here belongs to anybody.
"""
import json

from jobbot.recon_sanitize import (
    _MIN_FRAGMENT,
    _SENSITIVE_PLACEHOLDER,
    _TRUNC_MARK,
    identity_list_health,
    redact,
    redact_tree,
    redact_truncated_tail,
    sensitive_field_values,
    with_sensitive_values,
    with_value_strings,
)

PAIRS = [
    ("Acme Research Center", "Employer 1"),
    ("Zafira Quillon", "Alex Rivera"),
    ("Quillon’s Lab", "Employer 2"),
]


# --- Step 1: walk the leaves ---------------------------------------------

def test_redact_tree_reaches_every_nesting_shape():
    data = {
        "heading": "Zafira Quillon",
        "controls": [
            {"attrs": {"value": "Zafira Quillon", "aria-label": "Name"},
             "visible": True, "checked": None, "n": 3},
        ],
        "subtrees": ["<div>Acme Research Center</div>", ["Zafira Quillon"]],
    }
    out = redact_tree(data, pairs=PAIRS)
    flat = json.dumps(out, ensure_ascii=False)

    assert "Zafira Quillon" not in flat
    assert "Acme Research Center" not in flat
    assert flat.count("Alex Rivera") == 3
    assert "Employer 1" in flat

    # Keys are DOM attribute names, not user data, and the non-string leaves
    # must survive as their own types.
    assert "aria-label" in out["controls"][0]["attrs"]
    assert out["controls"][0]["visible"] is True
    assert out["controls"][0]["checked"] is None
    assert out["controls"][0]["n"] == 3


def test_redact_tree_beats_json_escaping_where_string_redaction_fails():
    """The regression proof for redacting BEFORE json.dumps, not after.

    ensure_ascii=True turns a curly apostrophe into \\u2019, so a literal
    pattern no longer matches the serialized text. Leaf redaction never sees
    an escape at all.
    """
    pairs = [("Acme’s Lab", "Employer 1")]
    data = {"page_text": "worked at Acme’s Lab for years"}

    # New ordering: redact the leaves, then serialize.
    assert "Acme" not in json.dumps(redact_tree(data, pairs=pairs))

    # Old ordering, same input: the token survives, because json.dumps has
    # already rewritten it to Acme’s Lab.
    assert "Acme" in redact(json.dumps(data), pairs=pairs)


# --- Step 2: the truncation-fragment sweep --------------------------------

def test_truncation_leaves_no_token_fragment():
    """`trunc()` can cut mid-token, leaving a prefix redact() never matches."""
    data = {"page_text": "xxxx" + "Acme Research Cent" + _TRUNC_MARK}
    out = redact_tree(data, pairs=PAIRS)
    text = out["page_text"]

    real = "Acme Research Center"
    for k in range(_MIN_FRAGMENT, len(real)):
        assert real[:k].lower() not in text.lower(), \
            f"a {k}-character prefix of a token survived truncation"
    assert text.endswith(_TRUNC_MARK)
    assert "Employer 1" in text
    assert text.startswith("xxxx")


def test_sweep_ignores_strings_that_were_not_truncated():
    """It must not mangle ordinary text that merely ends mid-word."""
    plain = "the applicant previously worked at Acme Research Cent"
    assert redact_truncated_tail(plain, PAIRS) == plain
    assert redact_truncated_tail("", PAIRS) == ""


def test_sweep_applies_at_most_one_token():
    s = "Zafira Quil" + _TRUNC_MARK
    out = redact_truncated_tail(s, PAIRS)
    assert out == "Alex Rivera" + _TRUNC_MARK


def test_sweep_leaves_a_truncated_string_with_no_fragment_alone():
    s = "nothing identifying here at all" + _TRUNC_MARK
    assert redact_truncated_tail(s, PAIRS) == s


# --- Step 3: fail closed on an unusable identity list ---------------------

_NAME_PAIR = ("Zafira Quillon", "Alex Rivera")
_EMAIL_PAIR = ("zafira@quillon.test", "alex.rivera@example.com")
_EMPLOYER_PAIR = ("Acme Research Center", "Employer 1")
_SCHOOL_PAIR = ("Some University", "School 1")


def test_identity_list_health_rejects_an_empty_list():
    ok, reason = identity_list_health(pairs=[])
    assert ok is False
    assert "0 token" in reason


def test_identity_list_health_rejects_name_only():
    ok, reason = identity_list_health(pairs=[_NAME_PAIR])
    assert ok is False
    assert "email" in reason
    assert "employer/school" in reason


def test_identity_list_health_rejects_name_and_email_without_employer():
    ok, reason = identity_list_health(pairs=[_NAME_PAIR, _EMAIL_PAIR])
    assert ok is False
    assert "employer/school" in reason


def test_identity_list_health_accepts_all_three_categories():
    for third in (_EMPLOYER_PAIR, _SCHOOL_PAIR):
        ok, reason = identity_list_health(
            pairs=[_NAME_PAIR, _EMAIL_PAIR, third])
        assert ok is True, reason
        assert "3 token" in reason


def test_identity_list_health_never_names_a_token():
    """A failure message gets pasted into reports and logs."""
    for pairs in ([], [_NAME_PAIR], [_NAME_PAIR, _EMAIL_PAIR],
                  [_NAME_PAIR, _EMAIL_PAIR, _EMPLOYER_PAIR]):
        _, reason = identity_list_health(pairs=pairs)
        for real, _placeholder in pairs:
            assert real not in reason
            assert real.lower() not in reason.lower()


# --- government ID masking (added in the Task 19b follow-up) -----------------
#
# An SSN is not an identity TOKEN: it appears in no profile file, so
# identity_tokens() can never learn it and the denylist can never match it.
# These assert the two halves of the structural rule.

def test_dashed_government_id_is_masked_even_with_an_explicit_pair_list():
    """A caller narrowing `pairs` is choosing which NAMES to replace, not
    opting out of ID masking."""
    out = redact("SSN: 123-45-6789 on file", pairs=[])
    assert "123-45-6789" not in out
    assert "000-00-0000" in out


def test_government_id_masking_survives_the_tree_walk():
    data = {"page_text": "Social Security Number 123-45-6789", "n": 1, "ok": None}
    out = redact_tree(data, pairs=[])
    assert "123-45-6789" not in json.dumps(out)
    assert out["n"] == 1 and out["ok"] is None


def test_bare_nine_digit_runs_are_left_alone():
    """Over-redaction is not free: the capture exists to collect DOM facts, and
    requisition ids, postal+4 and phone digits are all nine-digit runs."""
    keep = "req R1604206-1, zip 12222-1234, phone 8005551234, id 123456789"
    assert redact(keep, pairs=[]) == keep


# --- composite employer/school decomposition ---------------------------------
#
# Found live: resume_facts.json stores an employer as one long composite line,
# while a Workday form holds only a COMPONENT of it. redact() matches the full
# literal, so the composite token never fired and real employers went into
# captures in clear.

def test_composite_employer_is_split_into_redactable_components():
    from jobbot.recon_sanitize import _components
    parts = _components("Dr. Jane Doe Lab, University at Springfield (Springfield, ST)")
    assert "University at Springfield" in parts
    assert "Dr. Jane Doe Lab" in parts
    assert "ST" not in parts          # below the 6-char floor
    assert "Lab" not in parts


def test_component_split_drops_purely_generic_parts():
    from jobbot.recon_sanitize import _components
    assert _components("The Office of, University") == []


def test_a_form_value_that_is_only_a_component_still_redacts():
    """The exact live failure: the page holds a substring of the stored token."""
    pairs = [("University at Springfield", "Employer 1")]
    assert "Springfield" not in redact("University at Springfield", pairs)
    assert redact("University at Springfield", pairs) == "Employer 1"


def test_ordinary_workday_chrome_is_not_over_redacted():
    from jobbot.recon_sanitize import _components
    pairs = [(p, "Employer 1") for p in
             _components("Dr. Jane Doe Lab, University at Springfield (Springfield, ST)")]
    for keep in ("Save and Continue", "Application Questions 1 of 2",
                 "Principal Scientist, Protein Design"):
        assert redact(keep, pairs) == keep


# ----------------------------------------------------------------------
# Self-derived field values
#
# A street address is in no profile file, so identity_tokens() can never learn
# it. The capture is the only thing that knows what it is.
# ----------------------------------------------------------------------

NL = chr(10)

_ADDRESS_CAPTURE = {
    "automation_ids": [
        {"aid": "address--addressLine1", "value": "12 Nowhere Ct"},
        {"aid": "address--city", "value": "Placeville"},
        {"aid": "name--legalName--firstName", "value": "Zafira"},
    ],
    "controls": [
        {"attrs": {"id": "address--postalCode"}, "value": "00000"},
    ],
}


def test_an_address_field_value_is_harvested_out_of_the_capture():
    found = sensitive_field_values(_ADDRESS_CAPTURE)
    assert "12 Nowhere Ct" in found
    assert "Placeville" in found
    assert "00000" in found
    # A name is identity, but it is NOT a self-derived one: the denylist knows
    # names. Harvesting it here would mean redacting every occurrence of a
    # common first name out of the whole document.
    assert "Zafira" not in found


def test_a_harvested_value_is_redacted_out_of_prose_not_just_its_field():
    """The reason harvesting beats masking the field at capture time.

    Masking `address--addressLine1` would blank the input and leave the Review
    step's rendered copy of the same address sitting in `page_text`.
    """
    pairs = with_sensitive_values(_ADDRESS_CAPTURE, [])
    prose = ("Address" + NL + "12 Nowhere Ct" + NL + "Placeville, ST 00000")
    assert "12 Nowhere Ct" not in redact(prose, pairs)
    assert "Placeville" not in redact(prose, pairs)


def test_a_short_identity_token_cannot_eat_a_longer_harvested_address():
    """Longest-first, or the city redacts first and the street survives.

    "Placeville" is on the denylist (it is the profile location) AND a
    substring of the harvested address line. Applied in list order the city
    goes first, the full address literal then matches nothing, and the street
    number stays in the document while every check reports clean.
    """
    pairs = with_value_strings(["12 Nowhere Ct, Placeville ST"],
                               [("Placeville", "Springfield")])
    out = redact("I live at 12 Nowhere Ct, Placeville ST today", pairs)
    assert "Nowhere" not in out
    assert "12 Nowhere Ct" not in out
    assert _SENSITIVE_PLACEHOLDER in out


def test_an_already_redacted_capture_harvests_nothing():
    """Idempotence. A second sweep must not add the placeholder as a token."""
    swept = {"automation_ids": [
        {"aid": "address--addressLine1", "value": _SENSITIVE_PLACEHOLDER}]}
    assert sensitive_field_values(swept) == []
