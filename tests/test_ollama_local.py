"""Verification tests for the local Ollama tailoring path.

Goal: assert that tailor_resume() and tailor_resume_iterative() return
substantial non-empty Markdown (>= 800 chars, contains a heading) without
raising.

These tests skip automatically when Ollama is not reachable, so the main
test suite stays at 92 passed on machines without Ollama.

Run just these tests with:
    python -m pytest tests/test_ollama_local.py -v -s

Run the full suite (Ollama tests included when available):
    python -m pytest tests/ -q
"""
from __future__ import annotations

import os
import sys
import tempfile
import time
import unittest
from pathlib import Path
from unittest.mock import MagicMock, patch

# --- Isolation: temp paths before any jobbot import ---------------------------
TMP = Path(tempfile.mkdtemp(prefix="jobbot_ollama_test_"))
os.environ.setdefault("DB_PATH", str(TMP / "test.db"))
os.environ.setdefault("OUTPUT_DIR", str(TMP / "out"))
os.environ.setdefault("LOG_PATH", str(TMP / "logs" / "j.log"))
os.environ.setdefault("BASE_RESUME_PATH", str(TMP / "resume.md"))
# NOTE: deliberately do NOT override COMPANY_WATCHLIST or PROFILE_PATH here.
# conftest.py leaves those pointing at the real data/ files so WatchlistTests
# can assert real content; overriding them globally (settings is a singleton)
# corrupted that test. The Ollama tests don't depend on either file.
os.environ.setdefault("GEMINI_API_KEY", "")
os.environ.setdefault("DIGEST_EMAIL_TO", "")
os.environ.setdefault("OLLAMA_AUTOSTART", "false")

ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ROOT))

# Seed fixture files before importing settings
Path(os.environ["BASE_RESUME_PATH"]).write_text(
    "# Test Applicant\n\n## Summary\nComputational Scientist with 5 years of experience.\n\n"
    "## Skills\nPython, C++, PyTorch, modeling, distributed computing, SQL, data pipelines\n\n"
    "## Experience\n\n### Research Assistant, University Lab, 2021-Present\n"
    "- Developed computational models for distributed systems\n"
    "- Applied algorithmic optimization and high-performance computing\n"
    "- Published in IEEE Transactions 2025\n\n"
    "## Education\nPhD Computer Science (expected 2027) — State University\n"
    "BS Computer Science — State University, 2020\n",
    encoding="utf-8",
)
# Sample job description for tests
_SAMPLE_JD = """\
Senior Research Scientist — Drug Discovery (Biochemistry)
Company: BioTech Corp, Boston, MA

We seek a PhD biochemist to join our drug discovery team.
Requirements:
- PhD in biochemistry, chemistry, or related field
- Experience with protein purification (FPLC/HPLC/AKTA)
- PCR, cloning, molecular biology techniques
- Computational or structural biology experience a plus
- Mass spectrometry familiarity
- Strong publication record

Responsibilities:
- Design and execute biochemical assays
- Purify recombinant proteins for drug screening
- Collaborate with computational chemistry team
- Present findings at team meetings
"""

_SAMPLE_RESUME = Path(os.environ["BASE_RESUME_PATH"]).read_text(encoding="utf-8")


def _ollama_reachable() -> bool:
    """Check if Ollama is reachable without importing the full client (avoids
    side effects during collection)."""
    try:
        import requests
        r = requests.get("http://localhost:11434/api/tags", timeout=3)
        return r.status_code == 200
    except Exception:
        return False


def _ollama_model_present() -> bool:
    """Check whether the configured model is installed."""
    try:
        import requests
        from jobbot.config import settings
        r = requests.get("http://localhost:11434/api/tags", timeout=3)
        models = [m.get("name", "") for m in (r.json().get("models") or [])]
        wanted = getattr(settings, "ollama_model", "qwen3.5:latest")
        return any(wanted in m or m.startswith(wanted.split(":")[0]) for m in models)
    except Exception:
        return False


_OLLAMA_AVAILABLE = (os.environ.get("RUN_OLLAMA_LIVE") == "1"
                     and _ollama_reachable() and _ollama_model_present())
_SKIP_MSG = "Live Ollama tests require RUN_OLLAMA_LIVE=1 and reachable Ollama server"


# ---------------------------------------------------------------------------
# Unit tests (no real Ollama — mock _generate) — always run
# ---------------------------------------------------------------------------

class OllamaClientUnitTests(unittest.TestCase):
    """Unit-level tests that mock Ollama's HTTP call.  Always run."""

    def _make_mock_resp(self, content: str):
        """Build a mock requests.Response that looks like an Ollama reply."""
        mock = MagicMock()
        mock.raise_for_status.return_value = None
        mock.json.return_value = {"message": {"content": content, "thinking": ""}}
        return mock

    def test_generate_returns_content(self):
        from jobbot import ollama_client as oc
        long_text = "# Resume\n\n" + "- bullet point with real content\n" * 80
        with patch.object(oc, "_server_reachable", return_value=True), \
             patch("requests.post", return_value=self._make_mock_resp(long_text)):
            result = oc._generate("tailor this resume", include_profile=False)
        self.assertEqual(result, long_text.strip())

    def test_generate_retries_on_empty_content(self):
        """When Ollama returns empty content, _generate retries with smaller ctx."""
        from jobbot import ollama_client as oc
        call_count = {"n": 0, "ctx": []}

        def fake_post(url, json=None, timeout=None, headers=None):
            call_count["n"] += 1
            ctx = (json or {}).get("options", {}).get("num_ctx", 0)
            call_count["ctx"].append(ctx)
            # First call returns empty; second call returns real content
            if call_count["n"] == 1:
                return self._make_mock_resp("")
            return self._make_mock_resp("# Retried Resume\n\nReal content here.")

        with patch.object(oc, "_server_reachable", return_value=True), \
             patch("requests.post", side_effect=fake_post):
            result = oc._generate("tailor", include_profile=False)

        self.assertEqual(call_count["n"], 2, "Should have retried exactly once")
        self.assertGreater(call_count["ctx"][1], 0,
                           "Second call should specify a ctx")
        self.assertLess(call_count["ctx"][1], call_count["ctx"][0],
                        "Retry ctx should be smaller than first ctx")
        self.assertIn("Retried Resume", result)

    def test_generate_fallback_thinking_salvage(self):
        """If content is empty but thinking has text, salvage it."""
        from jobbot import ollama_client as oc
        mock_resp = MagicMock()
        mock_resp.raise_for_status.return_value = None
        # First call: content empty, thinking has output. No retry needed since
        # thinking content is salvaged before the retry logic triggers.
        # Second call (retry): also empty to confirm final fallback.
        responses = [
            {"message": {"content": "", "thinking": "THINKING TEXT"}},
            {"message": {"content": "", "thinking": "THINKING TEXT"}},
        ]
        call_idx = {"i": 0}
        def fake_json():
            r = responses[min(call_idx["i"], len(responses)-1)]
            call_idx["i"] += 1
            return r
        mock_resp.json = fake_json
        with patch.object(oc, "_server_reachable", return_value=True), \
             patch("requests.post", return_value=mock_resp):
            result = oc._generate("tailor", include_profile=False)
        # thinking text was salvaged so we should get "THINKING TEXT" back
        self.assertIn("THINKING TEXT", result)

    def test_truncated_response_is_reported(self):
        """A context-limit cut must be logged, not shipped as if it were done.

        Regression: job 5536 shipped a resume ending mid-sentence at
        "preparing wastewater samples for", with Education and Publications
        missing. Ollama had reported done_reason="length" and nothing looked,
        so a truncated document rendered straight to .md/.pdf/.docx.
        """
        from jobbot import ollama_client as oc
        mock = MagicMock()
        mock.raise_for_status.return_value = None
        mock.json.return_value = {
            "message": {"content": "# Resume\n\nCut off mid-sent", "thinking": ""},
            "done_reason": "length",
            "prompt_eval_count": 7387,
            "eval_count": 805,
        }
        # 7387 + 805 == 8192 exactly: the historical context wall, so pin
        # num_ctx to the value that actually produced the truncated resume.
        with patch.object(oc, "_server_reachable", return_value=True), \
             patch("requests.post", return_value=mock), \
             self.assertLogs("jobbot.ollama", level="WARNING") as logs:
            out = oc._generate("tailor", include_profile=False,
                               _ctx_override=8192)

        joined = "\n".join(logs.output)
        self.assertIn("truncated", joined.lower())
        # The operator needs to know WHICH limit bit, since the fix differs.
        self.assertIn("OLLAMA_NUM_CTX", joined)
        # The (partial) text is still returned; callers decide what to do.
        self.assertIn("Cut off mid-sent", out)
        # ...and it must NOT blame num_predict, which was nowhere near its cap.
        self.assertNotIn("OLLAMA_NUM_PREDICT", joined)

    def test_truncation_blames_num_predict_when_that_is_the_limit(self):
        """The two limits need different fixes, so don't name the wrong one."""
        from jobbot import ollama_client as oc
        from jobbot.config import settings
        predict = int(settings.ollama_num_predict)
        mock = MagicMock()
        mock.raise_for_status.return_value = None
        mock.json.return_value = {
            "message": {"content": "# Resume\n\nRan long", "thinking": ""},
            "done_reason": "length",
            # Plenty of context left; the generation cap is what bit.
            "prompt_eval_count": 1000,
            "eval_count": predict,
        }
        with patch.object(oc, "_server_reachable", return_value=True), \
             patch("requests.post", return_value=mock), \
             self.assertLogs("jobbot.ollama", level="WARNING") as logs:
            oc._generate("tailor", include_profile=False, _ctx_override=16384)

        joined = "\n".join(logs.output)
        self.assertIn("OLLAMA_NUM_PREDICT", joined)
        self.assertNotIn("OLLAMA_NUM_CTX", joined)

    def test_num_ctx_fits_a_whole_tailored_resume(self):
        """num_ctx is a SHARED prompt+response budget, so it must exceed both.

        Measured against the real tailoring prompt (job 5536, qwen3.5):
            prompt              ~7,387 tokens
            full resume needs   ~8,860 tokens end to end
        At num_ctx=8192 Ollama stopped with done_reason="length" after only
        805 generated tokens. Anything at or below the full-document budget
        guarantees a cut-off resume, so guard the floor with real headroom.
        """
        from jobbot.config import settings
        measured_full_document_budget = 8860
        self.assertGreaterEqual(
            int(settings.ollama_num_ctx), measured_full_document_budget,
            "ollama_num_ctx must leave room for prompt AND a complete resume; "
            f"{settings.ollama_num_ctx} truncates mid-document.",
        )

    def test_tailor_resume_iterative_raises_on_empty_draft(self):
        """tailor_resume_iterative must raise RuntimeError if draft is empty
        (after retries), rather than silently iterating on an empty string."""
        from jobbot import ollama_client as oc
        with patch.object(oc, "tailor_resume", return_value=""):
            with self.assertRaises(RuntimeError) as ctx:
                oc.tailor_resume_iterative(
                    _SAMPLE_RESUME, "Biochemist", "Acme", _SAMPLE_JD, rounds=1,
                )
        self.assertIn("empty content", str(ctx.exception).lower())

    def test_tailor_resume_iterative_skips_empty_revision(self):
        """If a refinement round returns empty, current draft is preserved."""
        from jobbot import ollama_client as oc
        good_draft = "# Good Draft\n\n## Summary\nSolid biochemist resume.\n" * 10
        call_count = {"n": 0}

        def fake_tailor(*args, **kwargs):
            return good_draft

        def fake_refine(*args, **kwargs):
            call_count["n"] += 1
            return ""  # Empty revision — should be skipped

        def fake_critique(*args, **kwargs):
            return {"score": 0.5, "issues": ["needs work"], "missing_keywords": [],
                    "fabrication_risks": []}

        with patch.object(oc, "tailor_resume", side_effect=fake_tailor), \
             patch.object(oc, "_critique_tailored", side_effect=fake_critique), \
             patch.object(oc, "_refine_tailored", side_effect=fake_refine):
            result = oc.tailor_resume_iterative(
                _SAMPLE_RESUME, "Biochemist", "Acme", _SAMPLE_JD, rounds=1,
            )
        # The empty revision should have been skipped; current stays as good_draft
        self.assertEqual(result["resume"], good_draft)

    def test_ctx_override_not_retried(self):
        """When _ctx_override is set (retry call), empty result is NOT retried again."""
        from jobbot import ollama_client as oc
        call_count = {"n": 0}
        def fake_post(url, json=None, timeout=None, headers=None):
            call_count["n"] += 1
            return self._make_mock_resp("")  # Always empty

        with patch.object(oc, "_server_reachable", return_value=True), \
             patch("requests.post", side_effect=fake_post):
            # Call with _ctx_override set — should NOT trigger a third retry
            result = oc._generate("test", include_profile=False, _ctx_override=4096)

        # Only 1 call (no further retry when _ctx_override is explicitly given)
        self.assertEqual(call_count["n"], 1)
        self.assertEqual(result, "")  # Returns empty, caller handles it


# ---------------------------------------------------------------------------
# Integration tests — skip if Ollama not reachable
# ---------------------------------------------------------------------------

@unittest.skipUnless(_OLLAMA_AVAILABLE, _SKIP_MSG)
class OllamaLiveTailorTests(unittest.TestCase):
    """Live integration tests against the running local Ollama server.

    These tests call the real model and assert:
    1. tailor_resume() returns >= 800 chars
    2. Output contains at least one Markdown heading (#)
    3. tailor_resume_iterative() returns a dict with 'resume' key >= 800 chars
    4. No exception is raised for the standard tailoring path

    Timing: each test may take 1-5 minutes on an 8 GB GPU (qwen3.5).
    """

    @classmethod
    def setUpClass(cls):
        from jobbot import ollama_client as oc
        oc.refresh_profile()  # Reset cached profile for test isolation

    def _assert_substantial_markdown(self, text: str, label: str = "output"):
        """Assert text is substantial non-empty Markdown."""
        self.assertIsInstance(text, str, f"{label} must be a string")
        self.assertGreater(len(text.strip()), 800,
                           f"{label} too short: {len(text.strip())} chars "
                           f"(first 200: {text[:200]!r})")
        self.assertTrue(
            any(line.startswith("#") for line in text.splitlines()),
            f"{label} has no Markdown heading (#). "
            f"First 300 chars: {text[:300]!r}"
        )

    def test_tailor_resume_basic(self):
        """tailor_resume() returns substantial Markdown without raising."""
        from jobbot import ollama_client as oc
        t0 = time.time()
        result = oc.tailor_resume(
            _SAMPLE_RESUME, "Senior Research Scientist", "BioTech Corp", _SAMPLE_JD
        )
        elapsed = time.time() - t0
        print(f"\n[tailor_resume] elapsed={elapsed:.1f}s, len={len(result)}")
        self._assert_substantial_markdown(result, "tailor_resume output")

    def test_tailor_resume_iterative_single_round(self):
        """tailor_resume_iterative with rounds=1 returns a valid result dict."""
        from jobbot import ollama_client as oc
        t0 = time.time()
        result = oc.tailor_resume_iterative(
            _SAMPLE_RESUME, "Senior Research Scientist", "BioTech Corp", _SAMPLE_JD,
            rounds=1,
        )
        elapsed = time.time() - t0
        print(f"\n[iterative rounds=1] elapsed={elapsed:.1f}s, "
              f"score={result.get('score')}, len={len(result.get('resume',''))}")
        self.assertIn("resume", result)
        self.assertIn("rounds", result)
        self.assertIn("score", result)
        self.assertIsInstance(result["score"], float)
        self._assert_substantial_markdown(result["resume"], "iterative resume (1 round)")

    def test_tailor_resume_iterative_zero_rounds(self):
        """rounds=0 (no refinement) should still produce a substantial draft."""
        from jobbot import ollama_client as oc
        t0 = time.time()
        result = oc.tailor_resume_iterative(
            _SAMPLE_RESUME, "Senior Research Scientist", "BioTech Corp", _SAMPLE_JD,
            rounds=0,
        )
        elapsed = time.time() - t0
        print(f"\n[iterative rounds=0] elapsed={elapsed:.1f}s, "
              f"len={len(result.get('resume',''))}")
        self._assert_substantial_markdown(result["resume"], "iterative resume (0 rounds)")
        self.assertEqual(result["rounds"], 0)


# ---------------------------------------------------------------------------
# Stability double-check — runs the unit tests twice (no extra live calls)
# ---------------------------------------------------------------------------

class OllamaUnitStabilityCheck(unittest.TestCase):
    """Re-run the core unit assertions a second time to verify stability."""

    def _make_mock_resp(self, content: str):
        mock = MagicMock()
        mock.raise_for_status.return_value = None
        mock.json.return_value = {"message": {"content": content, "thinking": ""}}
        return mock

    def test_retry_logic_stable_second_run(self):
        """Second run of retry test — confirms determinism."""
        from jobbot import ollama_client as oc
        call_count = {"n": 0}

        def fake_post(url, json=None, timeout=None, headers=None):
            call_count["n"] += 1
            if call_count["n"] == 1:
                return self._make_mock_resp("")
            return self._make_mock_resp("# Second Run\n\nContent ok.")

        with patch.object(oc, "_server_reachable", return_value=True), \
             patch("requests.post", side_effect=fake_post):
            result = oc._generate("tailor", include_profile=False)

        self.assertEqual(call_count["n"], 2)
        self.assertIn("Second Run", result)

    def test_empty_draft_raises_stable_second_run(self):
        """Second run of empty-draft raise test — confirms determinism."""
        from jobbot import ollama_client as oc
        with patch.object(oc, "tailor_resume", return_value=""):
            with self.assertRaises(RuntimeError):
                oc.tailor_resume_iterative(
                    _SAMPLE_RESUME, "Biochemist", "Acme", _SAMPLE_JD, rounds=1,
                )


if __name__ == "__main__":
    unittest.main(verbosity=2)
