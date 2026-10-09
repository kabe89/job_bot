from pathlib import Path

import pytest

from jobbot.browser_engine import BrowserEngine

FIXTURE = Path(__file__).parent / "fixtures" / "greenhouse_form.html"
if not FIXTURE.exists():
    pytest.skip("greenhouse_form.html fixture not present", allow_module_level=True)


@pytest.fixture
def page_url():
    return FIXTURE.resolve().as_uri()  # file:///.../greenhouse_form.html


def test_fill_and_read_back(page_url):
    with BrowserEngine(headed=False) as eng:
        eng.goto(page_url)
        assert eng.fill_by_label("First Name", "Jane") is True
        assert eng.read_by_label("First Name") == "Jane"


def test_labels_present(page_url):
    with BrowserEngine(headed=False) as eng:
        eng.goto(page_url)
        labels = eng.labels_present()
        assert "First Name" in labels
        assert "Why do you want to work here?" in labels


def test_screenshot_written(tmp_path, page_url):
    out = tmp_path / "shot.png"
    with BrowserEngine(headed=False) as eng:
        eng.goto(page_url)
        written = eng.screenshot(str(out))
    assert Path(written).exists()


def test_fill_missing_label_returns_false(page_url):
    with BrowserEngine(headed=False) as eng:
        eng.goto(page_url)
        assert eng.fill_by_label("Nonexistent Field", "x") is False
