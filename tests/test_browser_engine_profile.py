# tests/test_browser_engine_profile.py
from pathlib import Path

import pytest
from jobbot.browser_engine import BrowserEngine

_FIXTURE_PATH = Path(__file__).parent / "fixtures" / "greenhouse_form.html"
if not _FIXTURE_PATH.exists():
    pytest.skip("greenhouse_form.html fixture not present", allow_module_level=True)

FIXTURE = _FIXTURE_PATH.resolve().as_uri()


def test_persistent_profile_creates_dir_and_loads(tmp_path):
    profile = tmp_path / "profile"
    with BrowserEngine(headed=False, user_data_dir=str(profile)) as eng:
        eng.goto(FIXTURE)
        assert "First Name" in eng.labels_present()
    assert profile.exists()


def test_click_role_button(tmp_path):
    with BrowserEngine(headed=False) as eng:
        eng.goto(FIXTURE)
        assert eng.click_role("button", "Submit Application") is True
