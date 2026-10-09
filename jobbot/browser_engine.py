# jobbot/browser_engine.py
"""Thin Playwright wrapper — provider-agnostic browser primitives.

Knows nothing about any specific ATS. Label-oriented (Playwright get_by_label)
because that is the most robust way to address form fields. Context-managed
so the browser always closes.
"""
from __future__ import annotations

import logging
from typing import List

from playwright.sync_api import sync_playwright

log = logging.getLogger("jobbot.browser_engine")

DEFAULT_USER_AGENT: str = (
    "Mozilla/5.0 (Windows NT 10.0; Win64; x64) "
    "AppleWebKit/537.36 (KHTML, like Gecko) "
    "Chrome/126.0.0.0 Safari/537.36"
)

DEFAULT_VIEWPORT = {"width": 1280, "height": 800}


class BrowserEngine:
    def __init__(self, headed: bool = True, timeout_ms: int = 15000,
                 user_data_dir: "str | None" = None):
        self._headed = headed
        self._timeout = timeout_ms
        self._user_data_dir = user_data_dir
        self._pw = None
        self._browser = None
        self._context = None
        self.page = None

    def __enter__(self) -> "BrowserEngine":
        self._pw = sync_playwright().start()
        try:
            if self._user_data_dir:
                from pathlib import Path
                Path(self._user_data_dir).mkdir(parents=True, exist_ok=True)
                self._context = self._pw.chromium.launch_persistent_context(
                    self._user_data_dir,
                    headless=not self._headed,
                    user_agent=DEFAULT_USER_AGENT,
                    viewport=DEFAULT_VIEWPORT,
                )
                self.page = (self._context.pages[0] if self._context.pages
                             else self._context.new_page())
            else:
                self._browser = self._pw.chromium.launch(
                    headless=not self._headed,
                )
                self._context = self._browser.new_context(
                    user_agent=DEFAULT_USER_AGENT,
                    viewport=DEFAULT_VIEWPORT,
                )
                self.page = self._context.new_page()
            self.page.set_default_timeout(self._timeout)
        except Exception:
            try:
                if self._context:
                    self._context.close()
                if self._browser:
                    self._browser.close()
            finally:
                self._pw.stop()
            raise
        return self

    def __exit__(self, exc_type, exc, tb) -> None:
        try:
            if self._context:
                self._context.close()
            if self._browser:
                self._browser.close()
        finally:
            if self._pw:
                self._pw.stop()

    def goto(self, url: str) -> None:
        self.page.goto(url, wait_until="domcontentloaded")

    def labels_present(self) -> List[str]:
        labels = self.page.locator("label").all_inner_texts()
        return [l.strip() for l in labels if l.strip()]

    def fill_by_label(self, label: str, value: str) -> bool:
        try:
            self.page.get_by_label(label, exact=False).first.fill(value)
            return True
        except Exception as e:  # noqa: BLE001
            log.info("fill_by_label failed for %r: %s", label, e)
            return False

    def read_by_label(self, label: str) -> str:
        try:
            return self.page.get_by_label(label, exact=False).first.input_value()
        except Exception as e:  # noqa: BLE001
            log.info("read_by_label failed for %r: %s", label, e)
            return ""

    def fill_by_placeholder(self, placeholder: str, value: str, exact: bool = False) -> bool:
        try:
            loc = self.page.get_by_placeholder(placeholder, exact=exact)
            if exact and loc.count() != 1:
                log.info("fill_by_placeholder ambiguous for %r: %d matches",
                         placeholder, loc.count())
                return False
            loc.first.fill(value)
            return True
        except Exception as e:  # noqa: BLE001
            log.info("fill_by_placeholder failed for %r: %s", placeholder, e)
            return False

    def read_by_placeholder(self, placeholder: str, exact: bool = False) -> str:
        try:
            return self.page.get_by_placeholder(placeholder, exact=exact).first.input_value()
        except Exception as e:  # noqa: BLE001
            log.info("read_by_placeholder failed for %r: %s", placeholder, e)
            return ""

    def wait_for_form(self, timeout_ms: int = 5000) -> bool:
        try:
            self.page.wait_for_selector("form", timeout=timeout_ms)
            return True
        except Exception:  # noqa: BLE001
            return False

    def set_file_by_label(self, label: str, path: str) -> bool:
        try:
            self.page.get_by_label(label, exact=False).first.set_input_files(path)
            return True
        except Exception as e:  # noqa: BLE001
            log.info("set_file_by_label(%r) label path failed: %s", label, e)
        try:
            loc = self.page.locator("input[type=file]")
            if loc.count() >= 1:
                loc.first.set_input_files(path)
                log.info("set_file_by_label(%r): used file-input fallback.", label)
                return True
        except Exception as e:  # noqa: BLE001
            log.info("set_file_by_label(%r) file-input fallback failed: %s", label, e)
        return False

    def file_input_count(self) -> int:
        try:
            return self.page.locator("input[type=file]").count()
        except Exception as e:  # noqa: BLE001
            log.info("file_input_count failed: %s", e)
            return 0

    def set_file_by_index(self, index: int, path: str) -> bool:
        try:
            loc = self.page.locator("input[type=file]")
            if loc.count() > index:
                loc.nth(index).set_input_files(path)
                return True
        except Exception as e:  # noqa: BLE001
            log.info("set_file_by_index(%d) failed: %s", index, e)
        return False

    def click_text(self, text: str) -> bool:
        try:
            self.page.get_by_text(text, exact=False).first.click()
            return True
        except Exception as e:  # noqa: BLE001
            log.info("click_text failed for %r: %s", text, e)
            return False

    def click_role(self, role: str, name: str) -> bool:
        try:
            self.page.get_by_role(role, name=name, exact=False).first.click()
            return True
        except Exception as e:  # noqa: BLE001
            log.info("click_role failed for %r/%r: %s", role, name, e)
            return False

    def page_text(self) -> str:
        return self.page.inner_text("body")

    def screenshot(self, path: str) -> str:
        from pathlib import Path
        Path(path).parent.mkdir(parents=True, exist_ok=True)
        self.page.screenshot(path=path, full_page=True)
        return path
