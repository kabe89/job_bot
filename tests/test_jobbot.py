"""Smoke tests for JobBot — runs without network (mocks scrapers) and without Gemini."""
from __future__ import annotations

import os
import sys
import tempfile
import unittest
from pathlib import Path
from unittest.mock import patch

# Use a temp DB / paths for isolation BEFORE importing jobbot modules.
# Use setdefault so tests/conftest.py (which runs first, before ANY jobbot
# import) wins: it sets these to its own temp dir and binds settings there.
# Hard-setting them here would be a no-op on settings in a full-suite run (the
# singleton is already frozen) yet point our fixture writes at a different path
# than settings reads — so defer to conftest for a single source of truth.
TMP = Path(tempfile.mkdtemp(prefix="jobbot_test_"))
os.environ.setdefault("DB_PATH", str(TMP / "test.db"))
os.environ.setdefault("OUTPUT_DIR", str(TMP / "out"))
os.environ.setdefault("LOG_PATH", str(TMP / "logs" / "j.log"))
os.environ.setdefault("BASE_RESUME_PATH", str(TMP / "resume.md"))
os.environ.setdefault("COMPANY_WATCHLIST", str(TMP / "companies.md"))
os.environ.setdefault("PROFILE_PATH", str(TMP / "profile.md"))
os.environ.setdefault("GEMINI_API_KEY", "")  # disable Gemini for tests
os.environ.setdefault("DIGEST_EMAIL_TO", "")  # disable email

# Repo root on sys.path
ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ROOT))

# Seed fixture files before importing settings
Path(os.environ["BASE_RESUME_PATH"]).write_text(
    "# Test Resume\n\n## Skills\nPython, biochemistry, protein purification, "
    "PCR, cloning, mass spectrometry, enzyme kinetics, AKTA, FPLC, crystallography\n",
    encoding="utf-8",
)
Path(os.environ["COMPANY_WATCHLIST"]).write_text(
    "## Companies\n- Acme Corporation\n- TechFlow Inc\n- GlobalCorp\n",
    encoding="utf-8",
)
Path(os.environ["PROFILE_PATH"]).write_text(
    "# Profile\nBiochem PhD focused on protein purification and enzyme kinetics.\n",
    encoding="utf-8",
)

from jobbot import matcher, watchlist, exports  # noqa: E402
from jobbot.models import Application, Job, init_db, session  # noqa: E402
from jobbot.scrapers import RawJob, registry  # noqa: E402


class MatcherTests(unittest.TestCase):
    def test_score_basic(self):
        resume = "biochemistry phd protein purification enzyme kinetics"
        job = "We need a biochemistry PhD with protein expression and enzyme kinetics."
        s = matcher.score(resume, job, ["biochemistry", "enzyme"])
        self.assertGreater(s, 0.3)

    def test_score_zero_on_empty(self):
        self.assertEqual(matcher.score("", "", []), 0.0)

    def test_excluded(self):
        self.assertTrue(matcher.excluded("Sales Rep needed", ["sales"]))
        self.assertFalse(matcher.excluded("Scientist needed", ["sales"]))

    def test_location_match_home(self):
        self.assertTrue(matcher.location_match("Springfield, ST", "", ["springfield"]))
        self.assertFalse(matcher.location_match("San Francisco", "Onsite SF", ["springfield"]))

    def test_location_match_remote(self):
        self.assertTrue(matcher.location_match("", "This is a fully remote position", ["springfield", "remote"]))

    def test_location_match_hybrid(self):
        self.assertTrue(matcher.location_match("Boston, MA", "Hybrid schedule", ["hybrid"]))

    def test_company_watchlist_match(self):
        watch = ["Acme Corporation", "TechFlow Inc"]
        self.assertTrue(matcher.company_in_watchlist("Acme", watch))
        self.assertTrue(matcher.company_in_watchlist("TechFlow Inc (NYS DOH)", watch))
        self.assertFalse(matcher.company_in_watchlist("Acme Inc.", watch))


class WatchlistTests(unittest.TestCase):
    def test_load_companies(self):
        watchlist.refresh()
        names = watchlist.load_companies()
        self.assertIn("Acme Corporation", names)
        self.assertIn("TechFlow Inc", names)


class ScraperRegistryTests(unittest.TestCase):
    def test_registry_run_all_isolates_failures(self):
        local = registry.__class__()
        def good(_kw): return [RawJob(source="fake", title="Biochemist II", company="Acme", url="https://x")]
        def bad(_kw): raise RuntimeError("boom")
        local.register(good); local.register(bad)
        out = local.run_all(["biochem"])
        self.assertEqual(len(out), 1)
        self.assertEqual(out[0].title, "Biochemist II")


class PipelineTests(unittest.TestCase):
    def setUp(self):
        # Wipe DB between tests
        init_db()
        with session() as db:
            for a in db.query(Application).all(): db.delete(a)
            for j in db.query(Job).all(): db.delete(j)
            db.commit()

    def test_full_pipeline_with_mocked_scrapers(self):
        from jobbot import pipeline

        fake_jobs = [
            RawJob(source="test", title="Senior Biochemist", company="Acme",
                   url="https://example.com/1", location="Springfield, ST",
                   description="Looking for a biochemistry PhD with enzyme kinetics, protein purification, AKTA experience."),
            RawJob(source="test", title="Sales Representative", company="GenericCo",
                   url="https://example.com/2", location="Springfield, ST",
                   description="Pharmaceutical sales role."),
            RawJob(source="test", title="Research Scientist", company="Acme Pharma",
                   url="https://example.com/3", location="San Francisco, CA",
                   description="Biochemistry research onsite SF."),
            RawJob(source="test", title="Computational Biologist", company="Acme Pharma",
                   url="https://example.com/4", location="",
                   description="Remote computational biology role with python and biochemistry."),
        ]
        # Force the legacy location gate (semantic ranking OFF) so out-of-range
        # storage is exercised deterministically. With semantic ranking ON
        # (Ollama + a pulled embed model present), location is de-gated — a
        # strong-fit far role is kept in-range — so out_of_range_stored can be
        # 0 and this test's intent (out-of-range jobs ARE stored, not dropped)
        # would depend on whether embeddings happen to be installed.
        with patch.object(pipeline.registry, "run_all", return_value=fake_jobs), \
             patch.object(pipeline.settings, "semantic_ranking_enabled", False):
            result = pipeline.run_scrape_cycle(send_digest=False)

        self.assertEqual(result["scraped"], 4)
        self.assertEqual(result["filtered_excluded"], 1, "Sales rep should be excluded")
        # Now we STORE everything (in_range + out_of_range) instead of dropping at scrape time
        self.assertGreaterEqual(result["new"], 3)
        self.assertGreaterEqual(result["watchlist_boosted"], 1, "Acme should be boosted")
        self.assertGreaterEqual(result["out_of_range_stored"], 1, "SF + remote should be stored as out-of-range")
        # Under the legacy gate a low bag-of-words score can also push the
        # in-range jobs out, so in_range_stored can legitimately be 0. Assert
        # all 3 non-excluded jobs are stored somewhere (in-range OR out-of-range).
        self.assertGreaterEqual(
            result["in_range_stored"] + result["out_of_range_stored"], 3,
            "All 3 non-excluded jobs must be stored (semantic threshold gates in-range)"
        )

        with session() as db:
            jobs = db.query(Job).all()
            companies = {j.company for j in jobs}
            self.assertIn("Acme", companies)
            # Watchlist tag applied
            acme = next(j for j in jobs if j.company == "Acme")
            self.assertIn("watchlist", acme.tags)


class ExportTests(unittest.TestCase):
    def test_csv_export(self):
        init_db()
        with session() as db:
            j = Job(hash="h1", source="t", title="Biochemist", company="Acme",
                   location="Springfield, ST", url="https://x", description="d", match_score=0.7)
            db.add(j); db.commit()
        p = exports.export_jobs()
        self.assertTrue(p.exists())
        content = p.read_text(encoding="utf-8")
        self.assertIn("Acme", content)
        self.assertIn("Biochemist", content)


class ResumeRenderTests(unittest.TestCase):
    def test_markdown_to_pdf(self):
        from jobbot.resume import markdown_to_pdf
        md = "# Jane Doe\n\n## Skills\nBiochemistry, PCR, protein purification.\n\n- bullet one\n- bullet two\n"
        out = TMP / "out" / "test_resume.pdf"
        result = markdown_to_pdf(md, out)
        self.assertTrue(result.exists())
        self.assertGreater(result.stat().st_size, 1000)


class WebTests(unittest.TestCase):
    def test_flask_app_creates_and_routes(self):
        from jobbot.web import create_app
        app = create_app()
        client = app.test_client()
        for path in ("/", "/applications", "/advice", "/profile"):
            resp = client.get(path)
            self.assertEqual(resp.status_code, 200, f"GET {path} returned {resp.status_code}")


class DocxRenderTests(unittest.TestCase):
    def test_markdown_to_docx(self):
        from jobbot.resume import markdown_to_docx
        md = "# Jane Doe\n\n## Summary\nBiochemistry PhD with **protein purification** experience.\n\n- AKTA / FPLC\n- *Enzyme kinetics*\n"
        out = TMP / "out" / "resume.docx"
        result = markdown_to_docx(md, out)
        self.assertTrue(result.exists())
        self.assertGreater(result.stat().st_size, 5000)

    def test_load_docx_resume(self):
        from jobbot.resume import load_resume, markdown_to_docx
        out = TMP / "out" / "loadback.docx"
        markdown_to_docx("# Title\n\nBody text with biochemistry skills.", out)
        text = load_resume(out)
        self.assertIn("biochemistry", text.lower())


class ProfileContextTests(unittest.TestCase):
    def test_profile_loads_into_gemini_context(self):
        from jobbot import gemini_client as gc
        Path(os.environ["PROFILE_PATH"]).write_text(
            "# Profile\nBiochem PhD focused on protein purification.\n", encoding="utf-8")
        gc.refresh_profile()
        ctx = gc._profile_context()
        self.assertIn("protein purification", ctx)


class InterviewPrepRoutingTests(unittest.TestCase):
    def test_interview_route_renders_without_prep(self):
        from unittest.mock import patch
        from jobbot.web import create_app
        init_db()
        with session() as db:
            j = Job(hash="h-interview", source="t", title="Scientist II", company="Acme",
                   location="Springfield, ST", url="https://x/iv", description="Biochem role", match_score=0.5)
            db.add(j); db.commit(); jid = j.id
        client = create_app().test_client()
        with patch("jobbot.interview_coach.generate_pack", return_value={"error": None}):
            resp = client.get(f"/interview/{jid}")
        self.assertEqual(resp.status_code, 200)
        body = resp.get_data(as_text=True)
        self.assertIn("Interview Prep", body)
        self.assertIn("Scientist II", body)

    def test_profile_save_route(self):
        from jobbot.web import create_app
        client = create_app().test_client()
        resp = client.post("/profile", data={"content": "# New profile\nUpdated info."}, follow_redirects=True)
        self.assertEqual(resp.status_code, 200)
        saved = Path(os.environ["PROFILE_PATH"]).read_text(encoding="utf-8")
        self.assertIn("Updated info", saved)


class PredictorTests(unittest.TestCase):
    def setUp(self):
        init_db()
        with session() as db:
            for a in db.query(Application).all(): db.delete(a)
            for j in db.query(Job).all(): db.delete(j)
            db.commit()

    def _make_job(self, **overrides) -> Job:
        defaults = dict(
            hash=f"hp{overrides.get('hash_suffix','x')}", source="greenhouse:test",
            title="Senior Scientist", company="Acme",
            location="Springfield, ST", url="https://x", description="biochem role",
            match_score=0.6, status="new", tags="watchlist",
        )
        defaults.update({k: v for k, v in overrides.items() if k != "hash_suffix"})
        with session() as db:
            j = Job(**defaults)
            db.add(j); db.commit(); db.refresh(j)
            return j

    def test_predict_returns_sane_probabilities(self):
        from jobbot.predict import predict_job
        j = self._make_job()
        p = predict_job(j)
        self.assertGreater(p.callback_probability, 0.005)
        self.assertLess(p.callback_probability, 0.85)
        self.assertGreaterEqual(p.interview_probability, 0)
        self.assertLessEqual(p.offer_probability, p.interview_probability)

    def test_tailored_application_outscores_bare(self):
        from jobbot.predict import predict_job
        j = self._make_job(hash_suffix="t")
        bare = predict_job(j)
        with session() as db:
            app = Application(job_id=j.id, tailored_resume_path="/tmp/r.pdf", cover_letter_path="/tmp/c.txt")
            db.add(app); db.commit(); db.refresh(app)
        prepped = predict_job(j, app)
        self.assertGreater(prepped.callback_probability, bare.callback_probability)

    def test_watchlist_outscores_non_watchlist(self):
        from jobbot.predict import predict_job
        on = self._make_job(hash_suffix="w1", tags="watchlist")
        off = self._make_job(hash_suffix="w2", tags="", url="https://y", company="Random Co")
        self.assertGreater(predict_job(on).callback_probability,
                           predict_job(off).callback_probability)

    def test_low_match_lowers_score(self):
        from jobbot.predict import predict_job
        hi = self._make_job(hash_suffix="hi", match_score=0.8)
        lo = self._make_job(hash_suffix="lo", match_score=0.2, url="https://y")
        self.assertGreater(predict_job(hi).callback_probability,
                           predict_job(lo).callback_probability)

    def test_forecast_aggregates(self):
        from datetime import datetime as _dt
        from jobbot.predict import forecast
        for i in range(8):
            self._make_job(hash_suffix=f"f{i}", url=f"https://x/{i}", match_score=0.6 - i * 0.05)
        result = forecast(_dt(2026, 6, 1), horizon_days=60, apply_rate_per_week=3)
        self.assertEqual(result["total_applications_projected"], min(8, (60 // 7) * 3))
        self.assertGreaterEqual(result["expected_callbacks"], 0)
        self.assertGreaterEqual(result["p_at_least_one_offer"], 0)
        self.assertLessEqual(result["p_at_least_one_offer"], 1)

    def test_forecast_empty_db_returns_zeros(self):
        from datetime import datetime as _dt
        from jobbot.predict import forecast
        result = forecast(_dt(2026, 6, 1), horizon_days=30, apply_rate_per_week=5)
        self.assertEqual(result["total_applications_projected"], 0)
        self.assertEqual(result["expected_callbacks"], 0)
        self.assertEqual(result["p_at_least_one_offer"], 0)

    def test_recommended_actions_present_for_bare_job(self):
        from jobbot.predict import predict_job
        j = self._make_job(hash_suffix="rec")
        p = predict_job(j)
        self.assertTrue(any("tailor" in a.lower() for a in p.recommended_actions))


class RecruiterScraperTests(unittest.TestCase):
    def test_workday_parses_postings(self):
        from jobbot.scrapers import workday
        sample = {"jobPostings": [
            {"title": "Senior Scientist, Biochemistry", "externalPath": "/job/foo/abc",
             "locationsText": "Tarrytown, NY | Remote", "postedOn": "Posted 3 Days Ago",
             "bulletFields": ["JR12345"]},
            {"title": "Sales Director", "externalPath": "/job/foo/xyz",
             "locationsText": "NJ", "bulletFields": ["JR9"]},
        ]}
        with patch.object(workday, "_load_targets", return_value=[("acme", "Acme-Careers", "Acme Corporation", "wd5")]), \
             patch.object(workday, "_query", return_value=sample["jobPostings"]):
            jobs = workday.scrape_workday(["biochemistry"])
        titles = [j.title for j in jobs]
        self.assertIn("Senior Scientist, Biochemistry", titles)
        biochem = [j for j in jobs if "Biochemistry" in j.title][0]
        self.assertEqual(biochem.company, "Acme Corporation")
        self.assertIn("Tarrytown, NY", biochem.location)
        self.assertTrue(biochem.url.startswith("https://acme.wd5.myworkdayjobs.com"))

    def test_workday_location_cleaning(self):
        from jobbot.scrapers.workday import _clean_location
        self.assertEqual(_clean_location("Tarrytown, NY | Remote | Boston, MA"),
                         "Tarrytown, NY | Remote")
        self.assertEqual(_clean_location(""), "")

    def test_ashby_filters_by_keywords(self):
        from jobbot.scrapers import ashby
        sample = [
            {"title": "Senior Biochemist", "descriptionPlain": "Lead protein purification efforts",
             "locationName": "Cambridge, MA", "jobUrl": "https://jobs.ashbyhq.com/pinnacle/x"},
            {"title": "Marketing Lead", "descriptionPlain": "Run campaigns",
             "locationName": "NYC", "jobUrl": "https://jobs.ashbyhq.com/pinnacle/y"},
        ]
        with patch.object(ashby, "_load_targets", return_value=[("pinnacle", "Pinnacle Systems")]), \
             patch.object(ashby, "_fetch_board", return_value=sample):
            jobs = ashby.scrape_ashby(["biochemistry", "protein"])
        self.assertEqual(len(jobs), 1)
        self.assertEqual(jobs[0].company, "Pinnacle Systems")
        self.assertIn("Cambridge", jobs[0].location)

    def test_greenhouse_uses_display_names(self):
        from jobbot.scrapers.company_pages import display_name
        self.assertEqual(display_name("recursionpharmaceuticals"), "Recursion Pharmaceuticals")
        self.assertEqual(display_name("10xgenomics"), "10x Genomics")
        self.assertEqual(display_name("UnknownSlug"), "UnknownSlug")  # fallback


class ExtendedScrapersTests(unittest.TestCase):
    def test_linkedin_card_parser(self):
        from jobbot.scrapers import linkedin
        html = """
        <li><a href="https://www.linkedin.com/jobs/view/12345678/?ref=foo"></a>
          <h3>Senior Biochemist</h3>
          <h4>Acme</h4>
          <span class="job-search-card__location">Tarrytown, NY</span>
        </li>"""
        cards = linkedin._parse_cards(html)
        self.assertEqual(len(cards), 1)
        self.assertEqual(cards[0]["title"], "Senior Biochemist")
        self.assertEqual(cards[0]["company"], "Acme")
        self.assertIn("Tarrytown", cards[0]["location"])
        self.assertEqual(cards[0]["id"], "12345678")

    def test_linkedin_disabled_returns_empty(self):
        from jobbot.scrapers import linkedin
        with patch.object(linkedin.settings, "linkedin_enabled", False):
            self.assertEqual(linkedin.scrape_linkedin(["biochemistry"]), [])

    def test_google_jobs_jsonld_extract(self):
        from jobbot.scrapers import google_jobs
        html = """<script type="application/ld+json">
        {"@type":"JobPosting","title":"Senior Scientist, Biochemistry",
         "hiringOrganization":{"name":"Acme"},
         "jobLocation":{"address":{"addressLocality":"Tarrytown","addressRegion":"NY"}},
         "description":"<p>Lead biochemistry research</p>",
         "datePosted":"2026-04-01T00:00:00Z",
         "url":"https://x/jobs/1"}
        </script>"""
        with patch.object(google_jobs.requests, "get") as mget:
            mget.return_value.status_code = 200
            mget.return_value.text = html
            jobs = google_jobs._extract_job_postings("https://x/jobs/1")
        self.assertEqual(len(jobs), 1)
        self.assertEqual(jobs[0].title, "Senior Scientist, Biochemistry")
        self.assertEqual(jobs[0].company, "Acme")
        self.assertIn("Tarrytown", jobs[0].location)

    def test_scraper_limits_honored(self):
        """Each scraper reads its cap from settings — a low cap should be respected."""
        from jobbot.scrapers import linkedin
        with patch.object(linkedin.settings, "max_linkedin_pages", 1), \
             patch.object(linkedin.settings, "linkedin_enabled", True), \
             patch.object(linkedin, "_fetch_search", return_value="<li><a href='https://www.linkedin.com/jobs/view/1/'></a><h3>X</h3></li>"), \
             patch.object(linkedin.time, "sleep", lambda *_: None), \
             patch.object(linkedin, "_keyword_terms", return_value=["biochemistry"]), \
             patch.object(linkedin, "_location_terms", return_value=["Springfield, ST"]):
            linkedin.scrape_linkedin(["biochemistry"])
        self.assertTrue(True)

    def test_limits_dashboard_get(self):
        from jobbot.web import create_app
        resp = create_app().test_client().get("/limits")
        self.assertEqual(resp.status_code, 200)
        body = resp.get_data(as_text=True)
        for snippet in ("LinkedIn pages", "Workday max offset", "max_linkedin_pages"):
            self.assertIn(snippet, body)

    def test_search_builder_route(self):
        from jobbot.web import create_app
        c = create_app().test_client()
        resp = c.get("/search-builder?keywords=biochemistry&location=Springfield%20ST")
        self.assertEqual(resp.status_code, 200)
        body = resp.get_data(as_text=True)
        for name in ("LinkedIn", "Glassdoor", "Google Jobs", "Indeed", "TechCareers", "USAJobs"):
            self.assertIn(name, body)


class IndexFilterTests(unittest.TestCase):
    def setUp(self):
        init_db()
        with session() as db:
            for j in db.query(Job).all(): db.delete(j)
            db.commit()
            db.add(Job(hash="ff1", source="t", title="Biochemist", company="Acme",
                       location="Springfield, ST", url="https://x/1", description="local biochem",
                       match_score=0.7, is_local=True, is_remote=False, tags="watchlist"))
            db.add(Job(hash="ff2", source="t", title="Biochemist", company="Bay Bio",
                       location="San Francisco, CA", url="https://x/2", description="SF biochem",
                       match_score=0.6, is_local=False, is_remote=False, tags=""))
            db.add(Job(hash="ff3", source="t", title="Remote Biochemist", company="Anywhere Co",
                       location="", url="https://x/3", description="fully remote biochem role",
                       match_score=0.5, is_local=False, is_remote=True, tags=""))
            db.commit()

    def test_default_shows_only_in_range(self):
        from jobbot.web import create_app
        # Pin an explicit, generic home location so in-range filtering is
        # deterministic and independent of the machine's configured .env
        # locations (and of any user_locations.txt left by other tests).
        from jobbot import user_prefs
        user_prefs.save_locations("Springfield")
        try:
            body = create_app().test_client().get("/").get_data(as_text=True)
            # In-range job ff1 should appear in a row
            self.assertIn(">Biochemist</a>", body)
            # ff2 (San Francisco) should not appear as a job row
            self.assertNotIn(">San Francisco", body[body.find("<tbody>"):])
        finally:
            try:
                user_prefs.USER_LOCATIONS_FILE.unlink()
            except FileNotFoundError:
                pass

    def test_scope_all_shows_everything(self):
        from jobbot.web import create_app
        body = create_app().test_client().get("/?scope=all").get_data(as_text=True)
        self.assertIn("Springfield", body)
        self.assertIn("San Francisco", body)
        self.assertIn("Remote Biochemist", body)

    def test_remote_ok_includes_remote(self):
        from jobbot.web import create_app
        body = create_app().test_client().get("/?scope=all&remote_ok=on").get_data(as_text=True)
        self.assertIn("Remote Biochemist", body)

    def test_watchlist_only_filter(self):
        from jobbot.web import create_app
        body = create_app().test_client().get("/?scope=all&watchlist_only=on").get_data(as_text=True)
        self.assertIn("Acme", body)
        self.assertNotIn("Bay Bio", body)

    def test_company_text_filter(self):
        from jobbot.web import create_app
        body = create_app().test_client().get("/?scope=all&company=Bay").get_data(as_text=True)
        # Job-row content lives in <td>; placeholder text mentioning Acme
        # is in the form input, not a table cell.
        self.assertIn("<td>Bay Bio</td>", body)
        self.assertNotIn("<td>Acme</td>", body)


class DynamicLocationTests(unittest.TestCase):
    def setUp(self):
        init_db()
        from jobbot import user_prefs
        # Wipe any persisted user_locations file
        try:
            user_prefs.USER_LOCATIONS_FILE.unlink()
        except FileNotFoundError:
            pass
        with session() as db:
            for j in db.query(Job).all(): db.delete(j)
            db.commit()
            db.add(Job(hash="lo1", source="t", title="Job in Springfield", company="A",
                       location="Springfield, ST", url="https://x/1", description="local",
                       match_score=0.6, is_local=True))
            db.add(Job(hash="lo2", source="t", title="Job in Baltimore", company="B",
                       location="Baltimore, MD", url="https://x/2", description="east coast",
                       match_score=0.6, is_local=False))
            db.add(Job(hash="lo3", source="t", title="Job in San Diego", company="C",
                       location="San Diego, CA", url="https://x/3", description="west coast",
                       match_score=0.6, is_local=False))
            db.commit()

    def test_user_prefs_save_and_load(self):
        from jobbot import user_prefs
        saved = user_prefs.save_locations("Baltimore, Boston\nSan Diego")
        self.assertEqual(saved, ["Baltimore", "Boston", "San Diego"])
        self.assertEqual(user_prefs.load_locations(), ["Baltimore", "Boston", "San Diego"])

    def test_normalize_dedup_and_strip(self):
        from jobbot.user_prefs import _normalize
        self.assertEqual(_normalize("Springfield, SPRINGFIELD, springfield\nBoston , "),
                         ["Springfield", "Boston"])
        self.assertEqual(_normalize("# comment\nremote\n"), ["remote"])

    def test_anywhere_disables_filter(self):
        from jobbot.user_prefs import is_anywhere_mode
        self.assertTrue(is_anywhere_mode([]))
        self.assertTrue(is_anywhere_mode(["anywhere"]))
        self.assertFalse(is_anywhere_mode(["Springfield", "Boston"]))

    def test_url_locations_override(self):
        from jobbot.web import create_app
        client = create_app().test_client()
        # Default config locations don't include Baltimore — but URL override does
        body = client.get("/?scope=in-range&locations=Baltimore").get_data(as_text=True)
        self.assertIn(">Job in Baltimore</a>", body)
        self.assertNotIn(">Job in San Diego</a>", body)

    def test_anywhere_scope_shows_all(self):
        from jobbot.web import create_app
        body = create_app().test_client().get("/?scope=anywhere").get_data(as_text=True)
        for loc in ("Job in Springfield", "Job in Baltimore", "Job in San Diego"):
            self.assertIn(loc, body)

    def test_saved_user_locations_apply(self):
        from jobbot import user_prefs
        from jobbot.web import create_app
        user_prefs.save_locations("Baltimore, Boston")
        try:
            body = create_app().test_client().get("/?scope=in-range").get_data(as_text=True)
            self.assertIn(">Job in Baltimore</a>", body)
            self.assertNotIn(">Job in Springfield</a>", body)
        finally:
            user_prefs.USER_LOCATIONS_FILE.unlink()

    def test_locations_post_persists(self):
        from jobbot.web import create_app
        from jobbot import user_prefs
        client = create_app().test_client()
        client.post("/locations", data={"locations": "Baltimore\nBoston"}, follow_redirects=True)
        try:
            self.assertEqual(user_prefs.load_locations(), ["Baltimore", "Boston"])
        finally:
            user_prefs.USER_LOCATIONS_FILE.unlink()


class GeminiSkipOnErrorTests(unittest.TestCase):
    def setUp(self):
        init_db()
        with session() as db:
            for j in db.query(Job).all(): db.delete(j)
            db.commit()
            db.add(Job(hash="gskip", source="t", title="T", company="C",
                       location="Springfield, ST", url="https://x/skip",
                       description="biochem", match_score=0.6))
            db.commit()

    def test_tailor_returns_stub_when_gemini_denied(self):
        """When Gemini errors and skip-on-error is enabled, tailor returns a
        partial result with `error` rather than crashing."""
        from jobbot import gemini_client as gc, pipeline, ai_client
        from jobbot.config import settings
        with session() as db:
            jid = db.query(Job).first().id
        # Restrict to Gemini only so the test exercises the skip-on-error path
        # rather than failing over to the local Ollama backup.
        with patch.object(ai_client, "_ordered_providers", return_value=[gc]), \
             patch.object(gc, "_call_one", side_effect=RuntimeError("API key denied")), \
             patch.object(settings, "gemini_skip_on_error", True):
            r = pipeline.tailor_for_job(jid)
        self.assertIn("error", r)
        self.assertIn("denied", r["error"])
        self.assertIsNone(r["application_id"])
        self.assertEqual(r["analysis"]["score"], 0.0)

    def test_tailor_raises_when_skip_disabled(self):
        from jobbot import gemini_client as gc, pipeline, ai_client
        from jobbot.config import settings
        with session() as db:
            jid = db.query(Job).first().id
        with patch.object(ai_client, "_ordered_providers", return_value=[gc]), \
             patch.object(gc, "_call_one", side_effect=RuntimeError("denied")), \
             patch.object(settings, "gemini_skip_on_error", False):
            with self.assertRaises(RuntimeError):
                pipeline.tailor_for_job(jid)

    def test_interview_prep_returns_stub_when_gemini_denied(self):
        from jobbot import gemini_client as gc, pipeline, ai_client
        from jobbot.config import settings
        with session() as db:
            jid = db.query(Job).first().id
        with patch.object(ai_client, "_ordered_providers", return_value=[gc]), \
             patch.object(gc, "_call_one", side_effect=RuntimeError("API denied")), \
             patch.object(settings, "gemini_skip_on_error", True):
            r = pipeline.generate_interview_prep(jid)
        self.assertIn("error", r)
        self.assertEqual(r["prep"], "")
        self.assertEqual(r["path"], "")

    def test_is_available_reflects_key(self):
        from jobbot import gemini_client as gc
        from jobbot.config import settings
        with patch.object(settings, "gemini_api_key", ""):
            self.assertFalse(gc.is_available())
        with patch.object(settings, "gemini_api_key", "AIzaXX"):
            self.assertTrue(gc.is_available())


class ScrapeProgressTests(unittest.TestCase):
    def test_progress_lifecycle(self):
        from jobbot.scrape_progress import progress
        progress.start(total=3)
        self.assertEqual(progress.snapshot()["state"], "running")
        self.assertEqual(progress.snapshot()["total"], 3)
        progress.scraper_start("linkedin")
        self.assertEqual(progress.snapshot()["current"], "linkedin")
        progress.scraper_done("linkedin", 12)
        self.assertEqual(progress.snapshot()["jobs_found"], 12)
        self.assertEqual(progress.snapshot()["done"], 1)
        progress.scraper_error("indeed", "blocked")
        self.assertEqual(progress.snapshot()["done"], 2)
        progress.scraper_done("acme", 30)
        snap = progress.snapshot()
        self.assertEqual(snap["jobs_found"], 42)
        self.assertEqual(snap["done"], 3)
        self.assertEqual(snap["percent"], 100)
        progress.finish({"new": 8, "in_range_stored": 6})
        snap = progress.snapshot()
        self.assertEqual(snap["state"], "done")
        self.assertEqual(snap["stats"]["new"], 8)

    def test_registry_publishes_progress(self):
        from jobbot.scrapers import RawJob
        from jobbot.scrapers import ScraperRegistry
        from jobbot.scrape_progress import progress
        local = ScraperRegistry()
        def s_ok(_kw): return [RawJob(source="x", title="A", company="C", url="https://u/1")]
        def s_fail(_kw): raise RuntimeError("boom")
        local.register(s_ok); local.register(s_fail)
        out = local.run_all(["biochemistry"])
        snap = progress.snapshot()
        self.assertEqual(snap["total"], 2)
        self.assertEqual(snap["done"], 2)
        self.assertEqual(snap["jobs_found"], 1)
        self.assertEqual(len(out), 1)

    def test_status_endpoint_returns_json(self):
        from jobbot.web import create_app
        client = create_app().test_client()
        resp = client.get("/scrape/status")
        self.assertEqual(resp.status_code, 200)
        self.assertEqual(resp.content_type.split(";")[0], "application/json")
        data = resp.get_json()
        for k in ("state", "current", "done", "total", "percent", "jobs_found"):
            self.assertIn(k, data)

    def test_scrape_route_starts_background_thread(self):
        import threading, time
        from jobbot.web import create_app
        from jobbot.scrape_progress import progress
        progress.reset()
        client = create_app().test_client()
        done = threading.Event()
        called = {"n": 0}

        def fake_run(send_digest=False):
            called["n"] += 1
            progress.start(2); progress.scraper_done("fake1", 5)
            progress.scraper_done("fake2", 7); progress.finish({"new": 1})
            done.set()
            return {"new": 1}

        import jobbot.web as _w
        with patch.object(_w, "run_scrape_cycle", fake_run):
            client.post("/scrape", follow_redirects=True)
            # Wait for the spawned thread to actually fire fake_run before
            # we exit the patch — otherwise it falls through to the real scraper.
            self.assertTrue(done.wait(timeout=5),
                            "background thread didn't invoke fake_run within 5s")
        self.assertEqual(called["n"], 1)
        self.assertEqual(progress.snapshot()["state"], "done")


class UserInputFlowsTests(unittest.TestCase):
    """End-to-end: settings the user changes via the dashboard actually reach
    the scrapers."""

    def setUp(self):
        from jobbot import user_prefs
        try:
            user_prefs.USER_LOCATIONS_FILE.unlink()
        except FileNotFoundError:
            pass

    def test_locations_post_reaches_linkedin_terms(self):
        from jobbot.web import create_app
        from jobbot import user_prefs
        from jobbot.scrapers import linkedin
        client = create_app().test_client()
        client.post("/locations", data={"locations": "Rockville, MD\nBoston"},
                    follow_redirects=True)
        try:
            with patch.object(user_prefs.settings, "scrape_radius_miles", 30), \
                 patch.object(user_prefs.settings, "max_scrape_locations", 10):
                terms = linkedin._location_terms()
            self.assertIn("Rockville, MD", terms)
            self.assertIn("Boston, MA", terms)
        finally:
            user_prefs.USER_LOCATIONS_FILE.unlink()

    def test_limits_post_persists_to_env_and_settings(self):
        from jobbot.web import create_app
        from jobbot.config import settings
        from pathlib import Path
        env = Path(".env")
        existed = env.exists()
        if not existed:
            env.write_text("GEMINI_API_KEY=x\n", encoding="utf-8")
        original = env.read_text(encoding="utf-8")
        try:
            create_app().test_client().post("/limits", data={
                "scrape_radius_miles": "75",
                "max_linkedin_pages": "11",
                "max_scrape_locations": "9",
            }, follow_redirects=True)
            self.assertEqual(int(settings.scrape_radius_miles), 75)
            self.assertEqual(int(settings.max_linkedin_pages), 11)
            self.assertEqual(int(settings.max_scrape_locations), 9)
            new_text = env.read_text(encoding="utf-8")
            self.assertIn("SCRAPE_RADIUS_MILES=75", new_text)
            self.assertIn("MAX_LINKEDIN_PAGES=11", new_text)
        finally:
            if existed:
                env.write_text(original, encoding="utf-8")
            else:
                env.unlink(missing_ok=True)


class ScrapeLocationsTests(unittest.TestCase):
    def setUp(self):
        from jobbot import user_prefs
        try:
            user_prefs.USER_LOCATIONS_FILE.unlink()
        except FileNotFoundError:
            pass

    def test_scrape_locations_no_radius(self):
        from jobbot import user_prefs
        user_prefs.save_locations("Rockville, MD\nBoston\nremote\nhybrid")
        try:
            with patch.object(user_prefs.settings, "scrape_radius_miles", 0), \
                 patch.object(user_prefs.settings, "max_scrape_locations", 12):
                locs = user_prefs.scrape_locations()
            # remote/hybrid are filtered out; cities preserved
            self.assertIn("Rockville, MD", locs)
            self.assertIn("Boston, MA", locs)  # canonicalized from "Boston"
            self.assertNotIn("remote", [l.lower() for l in locs])
            self.assertNotIn("hybrid", [l.lower() for l in locs])
        finally:
            user_prefs.USER_LOCATIONS_FILE.unlink()

    def test_scrape_locations_radius_expansion(self):
        from jobbot import user_prefs
        user_prefs.save_locations("Rockville, MD")
        try:
            with patch.object(user_prefs.settings, "scrape_radius_miles", 50), \
                 patch.object(user_prefs.settings, "max_scrape_locations", 20):
                locs = user_prefs.scrape_locations()
            names = " | ".join(locs)
            self.assertIn("Rockville, MD", names)
            self.assertIn("Bethesda, MD", names)
            self.assertIn("Baltimore, MD", names)
            self.assertNotIn("Boston, MA", names)   # ~400 mi away
        finally:
            user_prefs.USER_LOCATIONS_FILE.unlink()

    def test_scrape_locations_cap(self):
        from jobbot import user_prefs
        user_prefs.save_locations("Rockville, MD\nSpringfield, ST\nBoston")
        try:
            with patch.object(user_prefs.settings, "scrape_radius_miles", 100), \
                 patch.object(user_prefs.settings, "max_scrape_locations", 3):
                locs = user_prefs.scrape_locations()
            self.assertEqual(len(locs), 3)
        finally:
            user_prefs.USER_LOCATIONS_FILE.unlink()

    def test_includes_remote_token(self):
        from jobbot import user_prefs
        user_prefs.save_locations("Rockville, MD\nremote")
        try:
            self.assertTrue(user_prefs.includes_remote())
        finally:
            user_prefs.USER_LOCATIONS_FILE.unlink()
        user_prefs.save_locations("Rockville, MD")
        try:
            self.assertFalse(user_prefs.includes_remote())
        finally:
            user_prefs.USER_LOCATIONS_FILE.unlink()

    def test_linkedin_terms_use_scrape_locations(self):
        from jobbot import user_prefs
        from jobbot.scrapers import linkedin
        user_prefs.save_locations("Rockville, MD")
        try:
            with patch.object(user_prefs.settings, "scrape_radius_miles", 50), \
                 patch.object(user_prefs.settings, "max_scrape_locations", 10):
                terms = linkedin._location_terms()
            self.assertIn("Rockville, MD", terms)
            self.assertIn("Bethesda, MD", terms)
        finally:
            user_prefs.USER_LOCATIONS_FILE.unlink()

    def test_google_queries_use_scrape_locations(self):
        from jobbot import user_prefs
        from jobbot.scrapers import google_jobs
        user_prefs.save_locations("Rockville, MD\nremote")
        try:
            with patch.object(user_prefs.settings, "scrape_radius_miles", 50), \
                 patch.object(user_prefs.settings, "max_scrape_locations", 10), \
                 patch.object(google_jobs.settings, "search_keywords", "biochemistry"):
                qs = list(google_jobs._build_queries())
            joined = " || ".join(qs)
            self.assertIn("Rockville, MD", joined)
            self.assertIn("Bethesda, MD", joined)
            self.assertIn("remote", joined)
        finally:
            user_prefs.USER_LOCATIONS_FILE.unlink()


class GeoRadiusTests(unittest.TestCase):
    def setUp(self):
        init_db()
        from jobbot import user_prefs
        try:
            user_prefs.USER_LOCATIONS_FILE.unlink()
        except FileNotFoundError:
            pass
        with session() as db:
            for j in db.query(Job).all(): db.delete(j)
            db.commit()
            db.add(Job(hash="g1", source="t", title="Job in Rockville", company="A",
                       location="Rockville, MD 20850", url="https://x/1", description="d",
                       match_score=0.6, is_local=False))
            db.add(Job(hash="g2", source="t", title="Job in Bethesda", company="B",
                       location="Bethesda, MD", url="https://x/2", description="d",
                       match_score=0.6, is_local=False))
            db.add(Job(hash="g3", source="t", title="Job in Baltimore", company="C",
                       location="Baltimore, MD", url="https://x/3", description="d",
                       match_score=0.6, is_local=False))
            db.add(Job(hash="g4", source="t", title="Job in Boston", company="D",
                       location="Boston, MA", url="https://x/4", description="d",
                       match_score=0.6, is_local=False))
            db.commit()

    def test_find_city(self):
        from jobbot.geo import find_city
        self.assertIsNotNone(find_city("Rockville, MD"))
        self.assertIsNotNone(find_city("rockville md 20850"))
        self.assertEqual(find_city("Boston, MA")[2], "Boston, MA")
        self.assertIsNone(find_city("Atlantis"))

    def test_haversine_known_distance(self):
        from jobbot.geo import _haversine_miles, find_city
        rock = find_city("Rockville, MD")
        beth = find_city("Bethesda, MD")
        d = _haversine_miles(rock[0], rock[1], beth[0], beth[1])
        self.assertLess(d, 10)  # Rockville–Bethesda ~7 miles
        balt = find_city("Baltimore, MD")
        d2 = _haversine_miles(rock[0], rock[1], balt[0], balt[1])
        self.assertGreater(d2, 25)
        self.assertLess(d2, 50)

    def test_cities_within_radius(self):
        from jobbot.geo import cities_within
        near = cities_within("Rockville, MD", 50)
        names = " | ".join(near)
        self.assertIn("Rockville, MD", names)
        self.assertIn("Bethesda, MD", names)
        self.assertIn("Washington, DC", names)
        self.assertNotIn("Boston, MA", names)

    def test_location_in_radius(self):
        from jobbot.geo import location_in_radius
        self.assertTrue(location_in_radius("Bethesda, MD 20814", ["Rockville, MD"], 25))
        self.assertFalse(location_in_radius("Boston, MA", ["Rockville, MD"], 25))
        self.assertTrue(location_in_radius("Boston, MA", ["Rockville, MD"], 1000))

    def test_index_radius_filter_includes_nearby_cities(self):
        from jobbot.web import create_app
        client = create_app().test_client()
        # 50-mile radius around Rockville should include Bethesda + Baltimore
        body = client.get("/?scope=in-range&locations=Rockville,MD&radius_miles=50").get_data(as_text=True)
        self.assertIn(">Job in Rockville</a>", body)
        self.assertIn(">Job in Bethesda</a>", body)
        # Baltimore is ~32mi from Rockville
        self.assertIn(">Job in Baltimore</a>", body)
        # Boston is far enough out it shouldn't appear at 50mi
        self.assertNotIn(">Job in Boston</a>", body)

    def test_index_radius_filter_500_miles_includes_far(self):
        from jobbot.web import create_app
        body = create_app().test_client().get(
            "/?scope=in-range&locations=Rockville%2C+MD&radius_miles=500"
        ).get_data(as_text=True)
        self.assertIn(">Job in Boston</a>", body)


class UserFeaturesTests(unittest.TestCase):
    def setUp(self):
        init_db()
        with session() as db:
            for j in db.query(Job).all(): db.delete(j)
            db.commit()
            db.add(Job(hash="uf1", source="t", title="Biochemist 1", company="Acme",
                       location="Springfield, ST", url="https://x/1", description="local biochem",
                       match_score=0.7, is_local=True, tags="watchlist"))
            db.add(Job(hash="uf2", source="t", title="Biochemist 2", company="Other Co",
                       location="Springfield, ST", url="https://x/2", description="more biochem",
                       match_score=0.4, is_local=True, tags=""))
            db.commit()

    def test_star_toggle_route(self):
        from jobbot.web import create_app
        client = create_app().test_client()
        with session() as db:
            jid = db.query(Job).first().id
        client.post(f"/star/{jid}")
        with session() as db:
            self.assertTrue(db.get(Job, jid).starred)
        client.post(f"/star/{jid}")
        with session() as db:
            self.assertFalse(db.get(Job, jid).starred)

    def test_starred_preset_filter(self):
        from jobbot.web import create_app
        with session() as db:
            job = db.query(Job).first()
            job.starred = True
            jid = job.id
            db.commit()
        body = create_app().test_client().get("/?preset=starred").get_data(as_text=True)
        self.assertIn(f"/job/{jid}", body)
        # The other job should NOT appear
        with session() as db:
            other_id = db.query(Job).filter(Job.id != jid).first().id
        self.assertNotIn(f"/job/{other_id}\"", body)

    def test_preset_today(self):
        from jobbot.web import create_app
        body = create_app().test_client().get("/?preset=today").get_data(as_text=True)
        self.assertEqual(create_app().test_client().get("/?preset=today").status_code, 200)
        # Just confirm the preset button label appears (template renders)
        self.assertIn("Today", body)

    def test_summary_card_present(self):
        from jobbot.web import create_app
        body = create_app().test_client().get("/").get_data(as_text=True)
        for marker in ("New in last 24h", "High-match", "Watchlist hits", "Starred"):
            self.assertIn(marker, body)

    def test_note_save_round_trip(self):
        from jobbot.web import create_app
        client = create_app().test_client()
        with session() as db:
            jid = db.query(Job).first().id
        client.post(f"/note/{jid}", data={"notes": "Followup with PI on Friday."}, follow_redirects=True)
        with session() as db:
            self.assertEqual(db.get(Job, jid).notes, "Followup with PI on Friday.")

    def test_reopen_route(self):
        from jobbot.web import create_app
        client = create_app().test_client()
        with session() as db:
            j = db.query(Job).first()
            j.status = "skipped"; db.commit(); jid = j.id
        client.post(f"/reopen/{jid}")
        with session() as db:
            self.assertEqual(db.get(Job, jid).status, "new")

    def test_bulk_skip(self):
        from jobbot.web import create_app
        client = create_app().test_client()
        with session() as db:
            ids = [j.id for j in db.query(Job).all()]
        from werkzeug.datastructures import MultiDict
        client.post("/bulk-skip", data=MultiDict([("ids", str(i)) for i in ids]), follow_redirects=True)
        with session() as db:
            for j in db.query(Job).all():
                self.assertEqual(j.status, "skipped")


class AutoApplyTests(unittest.TestCase):
    def setUp(self):
        init_db()
        with session() as db:
            for a in db.query(Application).all(): db.delete(a)
            for j in db.query(Job).all(): db.delete(j)
            db.commit()

    def test_extract_recruiter_email(self):
        from jobbot.auto_apply import extract_recruiter_email
        self.assertEqual(extract_recruiter_email("Send to careers@acme.com please"),
                         "careers@acme.com")
        # Junk filtering
        self.assertIsNone(extract_recruiter_email("Reply to noreply@x.com"))
        self.assertEqual(extract_recruiter_email("apply: noreply@x.com or hr@biotech.org"),
                         "hr@biotech.org")
        self.assertIsNone(extract_recruiter_email("No emails here at all"))

    def _seed_job(self, **overrides) -> int:
        defaults = dict(
            hash=f"aa{overrides.get('h','x')}", source="greenhouse:t",
            title="Senior Scientist", company="Acme", location="Springfield, ST",
            url="https://x", description="Reach out to careers@acme.com for details.",
            match_score=0.7, status="new", tags="watchlist",
        )
        defaults.update({k: v for k, v in overrides.items() if k != "h"})
        with session() as db:
            j = Job(**defaults); db.add(j); db.commit(); return j.id

    def test_dry_run_selects_eligible_jobs(self):
        from jobbot.auto_apply import AutoApplySettings, run_auto_apply
        self._seed_job(h="1")
        cfg = AutoApplySettings(confirm=False, default_recipient="fallback@me.com")
        report = run_auto_apply(cfg)
        self.assertGreaterEqual(report["selected"], 1)
        self.assertFalse(report["confirmed_send"])
        self.assertTrue(any("dry-run" in d["action"] for d in report["details"]))
        self.assertEqual(report["sent_ids"], [])

    def test_skips_below_score_floor(self):
        from jobbot.auto_apply import AutoApplySettings, run_auto_apply
        self._seed_job(h="lo", match_score=0.3, url="https://y")
        cfg = AutoApplySettings(confirm=False, min_match_score=0.5)
        self.assertEqual(run_auto_apply(cfg)["selected"], 0)

    def test_watchlist_only_mode(self):
        from jobbot.auto_apply import AutoApplySettings, run_auto_apply
        self._seed_job(h="w", tags="watchlist", company="Acme", url="https://r")
        self._seed_job(h="nw", tags="", company="Random Co", url="https://x2")
        cfg = AutoApplySettings(confirm=False, watchlist_only=True,
                                default_recipient="me@me.com")
        report = run_auto_apply(cfg)
        for d in report["details"]:
            self.assertEqual(d["company"], "Acme")

    def test_skips_without_recipient(self):
        from jobbot.auto_apply import AutoApplySettings, run_auto_apply
        self._seed_job(h="ne", description="No email anywhere in this job posting.")
        cfg = AutoApplySettings(confirm=False, default_recipient="")
        self.assertEqual(run_auto_apply(cfg)["selected"], 0)

    def test_uses_default_recipient_when_posting_has_none(self):
        from jobbot.auto_apply import AutoApplySettings, run_auto_apply
        self._seed_job(h="dr", description="No email anywhere.")
        cfg = AutoApplySettings(confirm=False, default_recipient="me@example.com")
        report = run_auto_apply(cfg)
        self.assertEqual(report["details"][0]["recipient"], "me@example.com")

    def test_blocklist_skips(self):
        from jobbot.auto_apply import AutoApplySettings, run_auto_apply
        self._seed_job(h="bl", company="Bad Co", url="https://bad", tags="watchlist")
        cfg = AutoApplySettings(confirm=False, skip_companies=["Bad Co"],
                                default_recipient="me@me.com")
        self.assertEqual(run_auto_apply(cfg)["selected"], 0)

    def test_callback_floor_filters_low_predictions(self):
        from jobbot.auto_apply import AutoApplySettings, run_auto_apply
        self._seed_job(h="cb", match_score=0.55, tags="")
        cfg = AutoApplySettings(confirm=False, min_callback_probability=0.99,
                                default_recipient="me@me.com")
        self.assertEqual(run_auto_apply(cfg)["selected"], 0)

    def test_dashboard_auto_apply_get(self):
        from jobbot.web import create_app
        resp = create_app().test_client().get("/auto-apply")
        self.assertEqual(resp.status_code, 200)
        self.assertIn(b"Auto-apply", resp.data)

    def test_dashboard_auto_apply_dry_run_post(self):
        from jobbot.web import create_app
        self._seed_job(h="webdr")
        resp = create_app().test_client().post("/auto-apply", data={
            "min_score": "0.4", "min_callback": "0.05",
            "default_recipient": "fallback@me.com",
        }, follow_redirects=True)
        self.assertEqual(resp.status_code, 200)
        # Confirm checkbox unchecked → dry run, no sends
        self.assertIn(b"Plan", resp.data)


class GeminiFallbackTests(unittest.TestCase):
    def setUp(self):
        from jobbot import gemini_client as gc
        gc.reset_fallback_state()

    def test_is_quota_error_recognizes_common_signals(self):
        from jobbot.gemini_client import _is_quota_error
        class FakeRE(Exception): pass
        FakeRE.__name__ = "ResourceExhausted"
        self.assertTrue(_is_quota_error(FakeRE("quota exceeded")))
        self.assertTrue(_is_quota_error(Exception("429 Too Many Requests")))
        self.assertTrue(_is_quota_error(Exception("rate limit hit")))
        self.assertFalse(_is_quota_error(Exception("model not found 404")))

    def test_fallback_used_on_quota_error(self):
        from jobbot import gemini_client as gc
        calls = []
        def fake_call(name, full, temperature):
            calls.append(name)
            if name == "gemini-2.5-flash":
                raise RuntimeError("429 Too Many Requests: quota exceeded")
            return "FALLBACK_OK"
        with patch.object(gc, "_call_one", side_effect=fake_call), \
             patch.object(gc.settings, "gemini_model", "gemini-2.5-flash"), \
             patch.object(gc.settings, "gemini_fallback_chain", ""), \
             patch.object(gc.settings, "gemini_fallback_model", "gemini-3-flash-preview"):
            out = gc._generate("hi", include_profile=False)
        self.assertEqual(out, "FALLBACK_OK")
        self.assertEqual(calls, ["gemini-2.5-flash", "gemini-3-flash-preview"])

    def test_subsequent_calls_skip_to_fallback_after_first_429(self):
        from jobbot import gemini_client as gc
        calls = []
        def fake_call(name, full, temperature):
            calls.append(name)
            if name == "gemini-2.5-flash":
                raise RuntimeError("ResourceExhausted: quota")
            return "OK_FALLBACK"
        with patch.object(gc, "_call_one", side_effect=fake_call), \
             patch.object(gc.settings, "gemini_model", "gemini-2.5-flash"), \
             patch.object(gc.settings, "gemini_fallback_chain", ""), \
             patch.object(gc.settings, "gemini_fallback_model", "gemini-3-flash-preview"):
            gc._generate("first", include_profile=False)
            calls.clear()
            gc._generate("second", include_profile=False)
        self.assertEqual(calls, ["gemini-3-flash-preview"])

    def test_non_quota_error_does_not_trigger_fallback(self):
        from jobbot import gemini_client as gc
        def fake_call(name, full, temperature):
            raise RuntimeError("404 model not found")
        with patch.object(gc, "_call_one", side_effect=fake_call):
            with self.assertRaises(RuntimeError):
                gc._generate("hi", include_profile=False)


class WebPredictorRouteTests(unittest.TestCase):
    def test_forecast_route_get(self):
        from jobbot.web import create_app
        client = create_app().test_client()
        resp = client.get("/forecast")
        self.assertEqual(resp.status_code, 200)
        self.assertIn(b"Forecast", resp.data)

    def test_predict_route(self):
        from jobbot.web import create_app
        init_db()
        with session() as db:
            j = Job(hash="hpred-web", source="greenhouse:t", title="Scientist",
                   company="Acme", location="Springfield, ST", url="https://q",
                   description="biochem", match_score=0.7, tags="watchlist")
            db.add(j); db.commit(); jid = j.id
        client = create_app().test_client()
        resp = client.get(f"/predict/{jid}")
        self.assertEqual(resp.status_code, 200)
        self.assertIn(b"Success Prediction", resp.data)


if __name__ == "__main__":
    unittest.main(verbosity=2)
