"""check_job_alive was structurally blind to dead Workday postings.

Workday serves a JavaScript app shell: a PULLED posting still answers the public
URL with HTTP 200 and no "no longer available" text -- the "The page you are
looking for doesn't exist" message is rendered client-side. So the generic
probe (404/410 or dead-phrase scan) reported every dead Workday job as alive.

Verified live against a posting confirmed dead by both its CXS endpoint
(403/S22) and its rendered page:

    KNOWN-DEAD: generic liveness -> alive=True   |   CXS -> gone

Workday is ~66% of this pipeline, so the sweep could never reap the majority of
its dead jobs. Raising liveness_check_per_cycle would NOT have helped -- it would
have probed thousands of dead jobs and pronounced them all alive.

Workday URLs now delegate to the CXS probe, which knows the difference.
"""
import pytest

from jobbot import liveness

_WD_DEAD = ("https://globalcorp.wd1.myworkdayjobs.com/en-US/globalcareers/job/"
            "United-States---California---Foster-City/Counsel--IP_R0052365-1")
_WD_LIVE = ("https://innotech.wd1.myworkdayjobs.com/en-US/InnoCareers/job/"
            "Cambridge-Massachusetts/Scientist--Platform-and-TA-Bioinformatics_R19096-1")
_GH = "https://boards.greenhouse.io/acme/jobs/123"


class _Shell:
    """What Workday really returns for a pulled posting: 200 + a JS app shell."""
    status_code = 200
    text = "<html><head><title>Careers</title></head><body><div id='root'></div></body></html>"


def _no_http(monkeypatch):
    def boom(*a, **k):
        raise AssertionError("should not have made a generic HTTP probe")
    monkeypatch.setattr(liveness, "_default_fetch", boom)


def test_pulled_workday_posting_is_detected_as_dead(monkeypatch):
    monkeypatch.setattr(liveness, "_workday_state", lambda url: "gone")
    _no_http(monkeypatch)
    alive, reason = liveness.check_job_alive(_WD_DEAD)
    assert alive is False
    assert "workday" in reason.lower()


def test_live_workday_posting_is_kept(monkeypatch):
    monkeypatch.setattr(liveness, "_workday_state", lambda url: "ok")
    _no_http(monkeypatch)
    alive, reason = liveness.check_job_alive(_WD_LIVE)
    assert alive is True
    assert reason == "alive"


def test_a_transient_workday_error_never_expires(monkeypatch):
    """A WAF hiccup or 5xx must not reap a live board."""
    monkeypatch.setattr(liveness, "_workday_state", lambda url: "error")
    _no_http(monkeypatch)
    alive, reason = liveness.check_job_alive(_WD_LIVE)
    assert alive is True
    assert reason.startswith("error:")


def test_the_js_shell_alone_would_have_said_alive():
    """Pins the bug: the generic path cannot see a dead Workday posting."""
    alive, _ = liveness.check_job_alive(_GH, fetch=lambda u, timeout=0: _Shell())
    assert alive is True          # 200 + no dead phrase -> "alive"


def test_a_workday_probe_that_raises_is_fail_open(monkeypatch):
    def boom(url):
        raise RuntimeError("cxs exploded")
    monkeypatch.setattr(liveness, "_workday_state", boom)
    alive, reason = liveness.check_job_alive(_WD_LIVE)
    assert alive is True
    assert reason.startswith("error:")


def test_non_workday_urls_still_use_the_generic_probe(monkeypatch):
    called = {}

    def fake_state(url):
        called["workday"] = True
        return "gone"

    monkeypatch.setattr(liveness, "_workday_state", fake_state)

    class _Gone:
        status_code = 404
        text = ""

    alive, reason = liveness.check_job_alive(_GH, fetch=lambda u, timeout=0: _Gone())
    assert alive is False and "404" in reason
    assert "workday" not in called, "greenhouse must not go through the CXS probe"


def test_generic_dead_phrase_still_works(monkeypatch):
    class _Closed:
        status_code = 200
        text = "This position is no longer available."

    alive, reason = liveness.check_job_alive(_GH, fetch=lambda u, timeout=0: _Closed())
    assert alive is False
    assert "no longer available" in reason


def test_sweep_expires_a_dead_workday_job(monkeypatch):
    from jobbot.models import Job, init_db, session
    init_db()
    with session() as db:
        j = Job(hash="lv_wd_dead", source="workday:globalcorp", title="Counsel, IP",
                company="GlobalCorp", url=_WD_DEAD, description="d", status="new")
        db.add(j); db.commit(); db.refresh(j)
        jid = j.id
    monkeypatch.setattr(liveness, "_workday_state", lambda url: "gone")
    _no_http(monkeypatch)
    stats = liveness.sweep_expired(limit=500)
    assert stats["expired"] >= 1
    with session() as db:
        assert db.get(Job, jid).status == "expired"
