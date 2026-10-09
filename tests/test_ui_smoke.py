import re
from pathlib import Path

STYLE = Path("jobbot/static/style.css")
FONTS = Path("jobbot/static/fonts")
BASE = Path("jobbot/templates/base.html")

GET_ROUTES = [
    "/", "/kanban", "/discover", "/search-builder", "/apply-queue", "/auto-apply",
    "/applications", "/followups", "/profile", "/skills", "/advice",
    "/locations", "/tags", "/limits", "/models", "/forecast",
]


def _client():
    from jobbot import web as web_mod
    app = web_mod.create_app()
    app.config.update(TESTING=True)
    return app.test_client()


def test_fonts_are_self_hosted():
    assert (FONTS / "SpaceGrotesk-var.woff2").exists()
    assert (FONTS / "JetBrainsMono-var.woff2").exists()
    css = STYLE.read_text(encoding="utf-8")
    assert "@font-face" in css
    # No remote font/CDN references anywhere in the stylesheet.
    assert "http://" not in css and "https://" not in css
    assert "@import" not in css


def test_theme_tokens_present_for_both_modes():
    css = STYLE.read_text(encoding="utf-8")
    assert ':root[data-theme="light"]' in css
    for var in ("--bg", "--panel", "--glass", "--accent", "--accent2",
                "--font-sans", "--font-mono", "--glow", "--ease"):
        assert var in css


# Every url_for endpoint the current shell links to — must all still be present.
SHELL_ENDPOINTS = [
    "index", "kanban_route", "discover_page", "search_builder", "apply_queue_page",
    "auto_apply_route", "applications", "followups_page", "profile",
    "skills_route", "advice", "locations_route", "tags_route",
    "limits_route", "models_route", "forecast_route", "scrape",
]


def test_shell_preserves_all_nav_endpoints():
    html = BASE.read_text(encoding="utf-8")
    for ep in SHELL_ENDPOINTS:
        assert re.search(r"url_for\(\s*'%s'" % ep, html), ep


def test_shell_has_theme_toggle_and_noflash_init():
    html = BASE.read_text(encoding="utf-8")
    assert "data-theme" in html and "jb-theme" in html   # persisted toggle
    assert 'class="sidebar"' in html and 'class="topbar"' in html
    assert "{% block content %}" in html


def test_all_get_routes_render_200():
    c = _client()
    for path in GET_ROUTES:
        assert c.get(path).status_code == 200, path


def test_dashboard_preserves_table_and_bulk_hooks():
    html = Path("jobbot/templates/index.html").read_text(encoding="utf-8")
    for hook in ('class="jobs"', 'id="bulk-all"', 'id="bulk-form"',
                 'id="sel-count"', 'class="row-check"', "url_for('star'",
                 "url_for('tailor'", "url_for('bulk_skip')"):
        assert hook in html, hook


def test_dashboard_renders_and_has_gauge_style():
    assert _client().get("/").status_code == 200
    css = STYLE.read_text(encoding="utf-8")
    assert ".gauge" in css          # animated score ring styling exists
    assert ".stat-tile" in css      # glass stat tiles exist


# Hooks read/used by apply_run.html's polling <script> and its form — must
# survive the restyle byte-for-byte (renaming any of these silently breaks
# the live browser-apply poll/submit flow).
APPLY_RUN_HOOKS = [
    'id="phase"',
    'id="panel"',
    'id="results"',
    'id="decide"',
    "url_for('download', filename=state.screenshot)",
    "url_for('browser_apply_decision', token=token)",
    'name="choice" value="submit"',
    'name="choice" value="skip"',
    "const token = {{ token }};",
    "/apply-run/status/${token}",
    "document.getElementById('phase')",
    "document.getElementById('decide')",
    "s.phase === 'done' || s.phase === 'error' || s.phase === 'checkpoint'",
]


def test_auto_apply_renders():
    assert _client().get("/auto-apply").status_code == 200


def test_apply_flow_preserves_hooks():
    ar = Path("jobbot/templates/apply_run.html").read_text(encoding="utf-8")
    au = Path("jobbot/templates/auto_apply.html").read_text(encoding="utf-8")
    for hook in APPLY_RUN_HOOKS:
        assert hook in ar, hook
    assert "<form" in au


def test_apply_flow_has_command_center_markup():
    ar = Path("jobbot/templates/apply_run.html").read_text(encoding="utf-8")
    assert 'class="cc-panel"' in ar
    assert 'class="stepper"' in ar
    assert 'class="field-list"' in ar
    assert 'class="shot"' in ar
    css = STYLE.read_text(encoding="utf-8")
    for cls in (".cc-panel", ".stepper", ".field-list", ".shot"):
        assert cls in css


def test_no_legacy_hardcoded_dark_hexes_in_css():
    # After the token migration, the old hardcoded panel/bg hexes should be
    # gone so light mode fully applies. Guards against stragglers.
    css = STYLE.read_text(encoding="utf-8")
    for legacy in ("#0e1117", "#161b22", "#1c2230", "#131822"):
        assert legacy not in css, legacy


def test_every_get_route_still_200_after_touchups():
    c = _client()
    for path in GET_ROUTES:
        assert c.get(path).status_code == 200, path
