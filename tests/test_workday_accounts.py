"""Per-tenant Workday credentials, and signing in without touching the
create-account form.

Every DOM shape asserted here is transcribed from the live capture
recon/20260813-085738-02-log-inv2.json via tests/fixtures/workday_signin.html.
That page holds TWO complete auth forms at once and ships invalid HTML: both
email inputs carry id="input-4", both password inputs carry id="input-5", and
three separate buttons read exactly "Sign In". So:

  - no assertion here may address a field by element id (both forms share
    them). Values are read as the ORDERED LIST of every matching automation
    id, index 0 being the create-account form and index 1 the sign-in form,
  - no expected value here is produced by calling the code under test,
  - the honeypot (data-automation-id="beecatcher", section 13 of
    workday-dom-facts.md) must still be empty afterwards.

Synthetic test fixture identity only. Mock data for testing.
"""
from __future__ import annotations

import json
import logging
from contextlib import contextmanager
from pathlib import Path

import pytest

from jobbot import workday_accounts
from jobbot.apply_questions import FormQuestion
from jobbot.browser_engine import BrowserEngine
from jobbot.config import settings

SIGNIN = Path(__file__).parent / "fixtures" / "workday_signin.html"
# A signed-in wizard page: carries NONE of the twelve gate automation ids.
WIZARD = Path(__file__).parent / "fixtures" / "workday_step1.html"

EMAIL = "alex@example.com"
PASSWORD = "pw-placeholder"

INNOTECH = "https://innotech.wd1.myworkdayjobs.com/en-US/InnoCareers/job/City/Sci_R1"
GLOBALCORP = "https://globalcorp.wd1.myworkdayjobs.com/en-US/GlobalCareers/job/City/Sci_R2"

# Captured before the autouse fixture can monkeypatch it: the only handle on
# the real, unpatched resolver, which the isolation test needs.
_REAL_PATH_FN = workday_accounts._path
_REPO_ROOT = Path(__file__).resolve().parent.parent
_REAL_DATA_DIR = (_REPO_ROOT / "data").resolve()


@pytest.fixture(autouse=True)
def _tmp_store(tmp_path, monkeypatch):
    monkeypatch.setattr(workday_accounts, "_path",
                        lambda: tmp_path / "workday_accounts.json")
    return tmp_path / "workday_accounts.json"


@contextmanager
def _engine_on(fixture=SIGNIN):
    if not fixture.exists():
        pytest.skip(f"Fixture {fixture.name} not present")
    with BrowserEngine(headed=False) as eng:
        eng.goto(fixture.resolve().as_uri())
        yield eng


def _values(eng, automation_id):
    """Every value carried by `automation_id`, in document order.

    Read straight from the DOM, never through a row the driver returned, and
    never by element id - both forms use id="input-4"/"input-5", so an id
    lookup silently reads the CREATE ACCOUNT field and a test built on it
    passes on the exact bug this file exists to catch."""
    return eng.page.eval_on_selector_all(
        "[data-automation-id='%s']" % automation_id, "els => els.map(e => e.value)")


def _all_input_values(eng):
    return eng.page.eval_on_selector_all("input, textarea", "els => els.map(e => e.value)")


def _clicks(eng):
    """Automation ids of every element clicked, recorded by the fixture's own
    inline listener rather than by spying on our own code."""
    return eng.page.evaluate("window.__clicks")


# Removes every gate marker from the page when signInSubmitButton is clicked,
# which is the ONLY way this fixture can ever produce a signed_in result: every
# button on it is type="button" and the forms carry onsubmit="return false", so
# without this the gate always survives the click. Installed from the test side
# so the fixture file itself stays a faithful transcript of the capture.
_CLEAR_GATE_ON_SUBMIT = """
() => {
  const GATE = ['signInFormo','signInContent','signInSubmitButton',
    'createAccountSubmitButton','signInLink','createAccountLink',
    'forgotPasswordLink','passwordRulesList','verifyPassword','popUpDialog',
    'beecatcher','utilityButtonSignIn'];
  document.querySelector("[data-automation-id='signInSubmitButton']")
    .addEventListener('click', () => {
      GATE.forEach(id => document.querySelectorAll(
        "[data-automation-id='" + id + "']").forEach(el => el.remove()));
    });
}
"""

# Hides the sign-in dialog so only the create-account form is on the page, and
# re-inserts the dialog when signInLink is clicked. That is the form-switcher
# branch of sign_in, which the as-captured fixture (dialog already open) cannot
# reach. workday-dom-facts.md section 11 records that whether the dialog is
# open on first landing is UNVERIFIED, so both shapes have to work.
_HIDE_DIALOG_UNTIL_SIGNIN_LINK = """
() => {
  const dlg = document.querySelector("[data-automation-id='popUpDialog']");
  const html = dlg.outerHTML;
  dlg.remove();
  document.querySelector("[data-automation-id='signInLink']")
    .addEventListener('click', () => {
      document.body.insertAdjacentHTML('beforeend', html);
    });
}
"""


# ---------------------------------------------------------------------------
# The store
# ---------------------------------------------------------------------------

def test_tenant_is_the_full_host_not_just_the_subdomain():
    """globalcorp.wd1 and innotech.wd1 are separate Workday deployments with
    separate accounts AND divergent DOM, so the key has to be the whole host."""
    assert workday_accounts.tenant_for(INNOTECH) == "innotech.wd1.myworkdayjobs.com"
    assert workday_accounts.tenant_for(GLOBALCORP) == "globalcorp.wd1.myworkdayjobs.com"


def test_a_bare_host_normalises_to_itself():
    """The CLI accepts either a full job URL or a bare tenant host."""
    assert (workday_accounts.tenant_for("globalcorp.wd1.myworkdayjobs.com")
            == "globalcorp.wd1.myworkdayjobs.com")
    assert (workday_accounts.tenant_for("GLOBALCORP.WD1.MyWorkdayJobs.com")
            == "globalcorp.wd1.myworkdayjobs.com")


def test_a_non_workday_url_has_no_tenant():
    assert workday_accounts.tenant_for("https://boards.greenhouse.io/acme/jobs/1") == ""
    assert workday_accounts.tenant_for("") == ""
    assert workday_accounts.tenant_for(None) == ""


def test_an_unknown_tenant_yields_no_credentials():
    assert workday_accounts.credentials_for(INNOTECH) is None


def test_credentials_are_per_tenant():
    """Storing innotech must not silently grant globalcorp. Sending one tenant's
    password to another tenant's login form is a credential leak."""
    workday_accounts.remember(INNOTECH, EMAIL, PASSWORD)
    assert workday_accounts.credentials_for(INNOTECH) == {
        "email": EMAIL, "password": PASSWORD}
    assert workday_accounts.credentials_for(GLOBALCORP) is None


def test_remember_accepts_a_bare_tenant_host():
    workday_accounts.remember("globalcorp.wd1.myworkdayjobs.com", EMAIL, PASSWORD)
    assert workday_accounts.credentials_for(GLOBALCORP)["email"] == EMAIL


def test_a_corrupt_store_is_empty_and_logged_not_raised(_tmp_store, caplog):
    """An apply run must not die because this file got truncated."""
    _tmp_store.write_text("{not json", encoding="utf-8")
    with caplog.at_level(logging.WARNING, logger="jobbot.workday_accounts"):
        assert workday_accounts._load() == {}
        assert workday_accounts.credentials_for(INNOTECH) is None
    assert any("workday accounts" in r.message.lower() or "unreadable" in r.message.lower()
               for r in caplog.records)


def test_known_tenants_never_exposes_the_password():
    """`jobbot workday-account list` is built on this. It must be safe to print."""
    workday_accounts.remember(INNOTECH, EMAIL, PASSWORD)
    listed = workday_accounts.known_tenants()
    assert listed == [{"tenant": "innotech.wd1.myworkdayjobs.com", "email": EMAIL}]
    assert PASSWORD not in json.dumps(listed)


def test_default_store_path_is_isolated_from_the_real_data_dir():
    """Every test above monkeypatches `_path`, so they would all pass while the
    conftest happily let the suite read and overwrite the user's real
    data/workday_accounts.json. This exercises the REAL resolver."""
    p = _REAL_PATH_FN()
    assert p.is_absolute(), (
        f"workday_accounts._path() returned the relative default {p!r}; conftest "
        f"is not isolating WORKDAY_ACCOUNTS_PATH")
    assert p.resolve().parent != _REAL_DATA_DIR, (
        f"workday_accounts._path() resolves into the real data/ dir ({p})")
    assert Path(settings.workday_accounts_path).resolve().parent != _REAL_DATA_DIR
