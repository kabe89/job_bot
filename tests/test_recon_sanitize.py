"""The identity list and redaction. conftest isolates every data path, so the
tokens here come from seeded fixtures and never from the owner's real files."""
from jobbot import recon_sanitize
from jobbot.resume_facts import Education, Employment, ResumeFacts


def _seed_facts(monkeypatch):
    facts = ResumeFacts(
        employment=[Employment(employer="Northwind Labs", title="Scientist")],
        education=[Education(school="Bellweather University", degree="PhD")],
    )
    monkeypatch.setattr(recon_sanitize.resume_facts, "load", lambda: facts)


def test_tokens_include_name_email_phone_employer_and_school(monkeypatch):
    _seed_facts(monkeypatch)
    monkeypatch.setattr(recon_sanitize.settings, "applicant_name", "Dana Whitfield")
    monkeypatch.setattr(recon_sanitize.settings, "applicant_email", "dana@real.example")
    monkeypatch.setattr(recon_sanitize.settings, "applicant_phone", "(555) 555-0142")

    reals = [t for t, _ in recon_sanitize.identity_tokens()]

    assert "Dana Whitfield" in reals
    assert "dana@real.example" in reals
    assert "Northwind Labs" in reals
    assert "Bellweather University" in reals


def test_phone_appears_in_every_punctuation_form(monkeypatch):
    _seed_facts(monkeypatch)
    monkeypatch.setattr(recon_sanitize.settings, "applicant_phone", "(555) 555-0142")

    reals = [t for t, _ in recon_sanitize.identity_tokens()]

    for form in ("5555550142", "555-555-0142", "(555) 555-0142", "555.555.0142"):
        assert form in reals, form


def test_longest_token_first_so_full_name_beats_first_name(monkeypatch):
    _seed_facts(monkeypatch)
    monkeypatch.setattr(recon_sanitize.settings, "applicant_name", "Dana Whitfield")

    reals = [t for t, _ in recon_sanitize.identity_tokens()]

    assert reals.index("Dana Whitfield") < reals.index("Dana")


def test_missing_sources_do_not_raise(monkeypatch):
    monkeypatch.setattr(recon_sanitize.resume_facts, "load",
                        lambda: (_ for _ in ()).throw(FileNotFoundError()))
    monkeypatch.setattr(recon_sanitize.settings, "applicant_name", "")
    assert isinstance(recon_sanitize.identity_tokens(), list)


def test_redact_replaces_full_name_before_first_name():
    pairs = [("Dana Whitfield", "Alex Rivera"), ("Dana", "Alex")]
    out = recon_sanitize.redact("<p>Dana Whitfield</p>", pairs)
    assert out == "<p>Alex Rivera</p>"


def test_redact_is_case_insensitive_and_hits_attributes():
    pairs = [("dana@real.example", "alex.rivera@example.com")]
    html = '<input value="DANA@Real.Example" aria-label="dana@real.example">'
    out = recon_sanitize.redact(html, pairs)
    assert "real.example" not in out.lower()
    assert out.count("alex.rivera@example.com") == 2


def test_an_email_no_token_knows_about_is_still_scrubbed():
    """The identity list is NAMES, so it can only reach an address that
    contains one. A Workday capture records the signed-in account's address in
    accountSettingsButton's text, and a test account's address need not carry
    the owner's name at all. Before the shape-based sweep this string came out
    of redact() byte-for-byte unchanged, and the fixture gate could not flag it
    either, because it is not a token.
    """
    pairs = [("Dana Whitfield", "Alex Rivera")]
    out = recon_sanitize.redact("<span>testuser@example.com</span>", pairs)
    assert "testuser@example.com" not in out
    assert out == "<span>%s</span>" % recon_sanitize._EMAIL


def test_the_sweep_runs_after_the_token_pass_not_instead_of_it():
    """Order matters. A name inside an address must still be destroyed by the
    token pass first, so that no code path can ever see it -- the sweep
    replacing the whole address is the second line of defence, not the only
    one.
    """
    pairs = [("Whitfield", "Rivera")]
    out = recon_sanitize.redact("<span>dana.whitfield@corp.example</span>", pairs)
    assert "whitfield" not in out.lower()
    assert out == "<span>%s</span>" % recon_sanitize._EMAIL


def test_the_placeholder_survives_its_own_sweep():
    """redact() must be idempotent: sanitizing an already-sanitized capture is
    a normal thing to do, and a sweep that rewrote its own placeholder would
    churn the fixture on every pass.
    """
    once = recon_sanitize.redact("<span>a@b.example</span>", [])
    assert recon_sanitize.redact(once, []) == once


def test_the_sweep_does_not_eat_ordinary_dom_facts():
    """Over-redaction costs the DOM facts the capture exists to collect. Ids
    and automation ids carry no @, so nothing structural should move.
    """
    html = ('<input id="name--legalName--firstName" '
            'data-automation-id="formField-firstName" aria-required="true">')
    assert recon_sanitize.redact(html, []) == html
