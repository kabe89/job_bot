"""Settings loaded from .env via pydantic-settings."""
from __future__ import annotations

from pathlib import Path
from typing import List

try:
    from pydantic_settings import (
        BaseSettings,
        PydanticBaseSettingsSource,
        SettingsConfigDict,
        YamlConfigSettingsSource,
    )
    _YAML_SUPPORT = True
except ImportError:
    from pydantic_settings import BaseSettings, SettingsConfigDict  # type: ignore
    _YAML_SUPPORT = False


def _csv(value: str) -> List[str]:
    return [v.strip() for v in value.split(",") if v.strip()]


class Settings(BaseSettings):
    model_config = SettingsConfigDict(
        env_file=".env",
        env_file_encoding="utf-8",
        yaml_file="config.yaml",
        yaml_file_encoding="utf-8",
        extra="ignore",
        case_sensitive=False,
    )

    if _YAML_SUPPORT:
        @classmethod
        def settings_customise_sources(
            cls,
            settings_cls: type[BaseSettings],
            init_settings: PydanticBaseSettingsSource,
            env_settings: PydanticBaseSettingsSource,
            dotenv_settings: PydanticBaseSettingsSource,
            file_secret_settings: PydanticBaseSettingsSource,
        ) -> tuple[PydanticBaseSettingsSource, ...]:
            return (
                init_settings,
                dotenv_settings,
                env_settings,
                YamlConfigSettingsSource(settings_cls),
                file_secret_settings,
            )

    # AI provider routing
    # "gemini"  -> Gemini first
    # "claude"  -> Claude first
    # "ollama"  -> local Ollama first (free, no key/quota)
    # "auto"    -> prefer Claude, then Gemini, then local Ollama
    # The dispatcher (ai_client) skips any provider that isn't usable (no key /
    # server down) and fails over to the next on quota/errors, so this safely
    # degrades to whichever provider actually works.
    ai_provider: str = "ollama"

    # --- Ollama (local open-source models — free, no API key, no quota) ---
    ollama_host: str = "http://localhost:11434"
    ollama_model: str = "qwen3.5:latest"
    ollama_timeout: int = 300   # local generation can be slow on CPU
    # Context window (tokens). This is a SHARED budget: prompt + generated
    # response must both fit inside it, so an oversized prompt silently steals
    # the room the answer needs and the resume stops mid-sentence.
    # Measured on the real tailoring prompt (job 5536, qwen3.5): the prompt is
    # ~7,400 tokens and a complete resume needs ~8,800 tokens end to end. At
    # 8192 Ollama returned done_reason="length" after only 805 generated
    # tokens, cutting Education and Publications off the document entirely.
    # 12288/16384/24576 all completed cleanly (done_reason="stop") on an 8 GB
    # RTX 4070 Laptop — the older "16384 OOMs on 8 GB" note was stale.
    # 16384 leaves ~9k tokens of headroom for longer JDs/resumes at ~73s/call.
    ollama_num_ctx: int = 16384
    # Max tokens to generate per call (bounds a runaway response).
    # Reduced from 8192 to stay within VRAM budget on the typical 8 GB laptop GPU.
    ollama_num_predict: int = 4096
    # Disable "thinking"/reasoning mode for reasoning models (e.g. Qwen3): on
    # long structured tasks they spend the whole budget thinking and return an
    # empty `content`. False -> the answer lands directly in `content`.
    ollama_think: bool = False
    # Iterative "smart refinement" for local tailoring: number of
    # draft -> self-critique -> revise rounds. 0 = single pass (no refinement).
    # Each round is 2 extra model calls, so keep modest on slow hardware.
    ollama_refine_rounds: int = 2
    # Stop refining early once a critique self-scores at or above this (0-1).
    ollama_refine_target_score: float = 0.9
    # When the bot needs the local model and the Ollama server isn't running,
    # auto-launch `ollama serve` (local hosts only — never a remote/tunneled
    # host). Requires the `ollama` binary on PATH.
    ollama_autostart: bool = True
    # If the configured model isn't pulled yet, auto `ollama pull` it on first
    # use (local only). First pull of a large model can take a while.
    ollama_autostart_pull: bool = True
    # Local embedding model for semantic job ranking (jobbot/embeddings.py).
    # Pull once with `ollama pull nomic-embed-text`. Small + fast; safe on 8 GB GPUs.
    ollama_embed_model: str = "nomic-embed-text"
    # Ollama hosted Web Search API key (https://ollama.com -> Settings -> API keys).
    # Free tier available. When set, jobbot gains live web search + page fetch
    # (jobbot/ollama_search.py) used as the top-priority research backend and an
    # optional model-driven (agentic) search tool. Empty = feature off.
    ollama_api_key: str = ""
    # Base URL for the hosted web-search/fetch API (override only if self-hosting
    # a compatible proxy).
    ollama_web_base: str = "https://ollama.com/api"
    # Base host for Ollama *cloud* models (the `-cloud` tagged large models).
    # Cloud chat requests go to <ollama_cloud_base>/api/chat with the
    # OLLAMA_API_KEY as a bearer token. Override only if self-hosting a proxy.
    ollama_cloud_base: str = "https://ollama.com"

    # Gemini
    gemini_api_key: str = ""
    gemini_model: str = "gemini-3-flash-preview"
    gemini_fallback_model: str = "gemini-2.5-flash"
    # Comma-separated fallback chain — walked in order on quota / 429 errors.
    # Default: primary -> 2.5-flash -> 3.1-flash-lite (cheapest, last-resort).
    gemini_fallback_chain: str = "gemini-2.5-flash,gemini-3.1-flash-lite-preview"
    # When True: if Gemini is denied / errors out, log + return a partial result
    # instead of raising. Lets the rest of the pipeline keep running.
    gemini_skip_on_error: bool = True

    # Claude (Anthropic) — alternative / fallback AI provider
    anthropic_api_key: str = ""
    claude_model: str = "claude-opus-4-8"
    # Used only if the primary Claude model is overloaded (529) or unavailable.
    claude_fallback_model: str = "claude-sonnet-4-6"
    # Reasoning effort: low | medium | high | xhigh | max (max is Opus-tier only).
    claude_effort: str = "high"
    # Hard ceiling on output tokens per call. With adaptive thinking + high
    # effort, thinking tokens count toward this budget, so 8k risked truncating
    # long outputs (full resume, interview-prep doc). 16k gives headroom and
    # still stays under the SDK's non-streaming HTTP-timeout guard; set higher
    # and the client streams automatically (see claude_client._call_one).
    claude_max_tokens: int = 16000

    # --- Assisted browser auto-apply (Claude-in-Chrome) ---
    # When True, the browser-apply flow is allowed to actually submit forms.
    # When False (default) it fills fields and STOPS before the final submit so
    # you can review. CAPTCHAs and login walls always pause for the user.
    browser_apply_autosubmit: bool = False
    # --- Playwright auto-apply (Greenhouse slice) ---
    # Headed = a visible Chromium window so you can eyeball the real filled
    # form before the CLI submit prompt. Set PLAYWRIGHT_HEADED=false to hide.
    playwright_headed: bool = True
    # Seconds a background web-apply worker waits at the review gate for the
    # user's Submit/Skip decision before auto-skipping (safety timeout).
    web_apply_decision_timeout: int = 900
    # Where apply-run screenshots are written.
    evidence_dir: str = "output/evidence"
    # Cross-job answer store (superset of data/answer_bank.json).
    answer_memory_path: str = "data/answer_memory.json"
    # Plain-text personal-info sheet (label / value lines) the user maintains to
    # auto-populate application answers the .env identity fields don't cover —
    # address lines, EEO self-ID, "how did you hear", etc. Parsed into answer-bank
    # entries at answer time; the curated data/answer_bank.json still wins on ties.
    personal_info_path: str = "My Information.txt"
    # Structured, human-verified resume facts (employment/education/identity).
    # Gitignored: contains the user's address and contact details.
    resume_facts_path: str = "data/resume_facts.json"
    # Learned ATS field-resolution recipes. Perishable cache, safe to delete.
    ats_recipes_path: str = "data/ats_recipes.json"
    # Escalate a form field the deterministic path could not resolve to the
    # local LLM DOM agent. Bounded per run, verified before anything is cached,
    # and never applied to a bot-detection honeypot. Set false to keep the fill
    # path purely deterministic.
    ats_escalation_enabled: bool = True
    # Persistent Chromium profile dir for Workday (keeps logins across runs).
    workday_profile_dir: str = "data/profiles/workday"
    # Per-tenant Workday credentials. Gitignored: holds a plaintext password.
    workday_accounts_path: str = "data/workday_accounts.json"
    # Treat these substrings (case-insensitive) on a page as a CAPTCHA / human
    # checkpoint: pause and hand control back to the user.
    browser_apply_captcha_markers: str = (
        "captcha,recaptcha,hcaptcha,i'm not a robot,im not a robot,cloudflare,"
        "verify you are human,are you a robot,challenge-form,turnstile"
    )

    # --- Apply-kit: default answers for screening questions ---
    # Used by apply_questions.py when a form asks the matching question.
    # Leave a value empty to have the AI answer (or flag it for you) instead.
    work_authorized: str = "Yes"
    requires_sponsorship: str = "No"
    salary_expectation: str = ""          # e.g. "Open to discussion" or "$95,000-$110,000"
    earliest_start_date: str = ""         # e.g. "Two weeks from offer"
    willing_onsite: str = "Yes"
    willing_relocate: str = ""            # e.g. "Open to discussion"
    how_heard: str = "Company careers site"
    # User-curated question->answer memory (regex match), grows via the
    # dashboard's "remember this answer" checkbox.
    answer_bank_path: str = "data/answer_bank.json"
    # HTTP timeout (seconds) when fetching application forms.
    apply_fetch_timeout: int = 60
    # After each scrape cycle, auto-build Apply Kits for this many of the best
    # new prospects (kit-mode auto-apply). 0 = off. Fires AI calls per kit.
    auto_prepare_kits: int = 0

    # --- Follow-ups ---
    # A sent application with no reply becomes "follow-up due" after this many days.
    followup_after_days: int = 8
    # Stop suggesting follow-ups after this many have been sent per application.
    followup_max: int = 2

    # --- Contact finder ---
    contacts_max_per_job: int = 3
    # When True, opening a job in the dashboard auto-finds + caches its contacts
    # on first view (uses the keyless ddgs backend, so no API key needed).
    contacts_autofind: bool = True
    # After each scrape cycle, bulk pre-fetch contacts for this many of the
    # top-scoring new-ish jobs that lack them. 0 = off. Uses the keyless ddgs
    # search with a polite delay between jobs.
    auto_prefetch_contacts: int = 0
    # Polite delay (seconds) between jobs during bulk contact prefetch.
    contacts_prefetch_delay: float = 2.0

    # --- Callback predictor (learned model) ---
    callback_model_path: str = "data/callback_model.json"
    # Train/use the learned model only once at least this many labeled
    # outcomes (callback or no-reply/rejected) exist; below it, heuristics.
    callback_model_min_outcomes: int = 20

    # SMTP
    smtp_host: str = "smtp.gmail.com"
    smtp_port: int = 587
    smtp_user: str = ""
    smtp_password: str = ""
    smtp_from_name: str = "JobBot"
    digest_email_to: str = ""

    # Profile
    applicant_name: str = "Applicant"
    applicant_title: str = "Your Title"
    applicant_location: str = "Your City, ST"
    applicant_phone: str = ""
    applicant_email: str = ""
    applicant_linkedin: str = ""
    applicant_github: str = ""
    applicant_portfolio: str = ""

    # Search
    search_keywords: str = "software engineer,python,data analyst"
    search_locations: str = "remote,hybrid"
    search_exclude: str = "intern,undergraduate,sales,recruiter"
    research_enricher_keywords: str = "software engineering,architecture,distributed systems,machine learning,cloud"
    scrapers_preferred_keywords: str = "software engineer,backend engineer,data scientist,machine learning engineer"
    # Matched against the TITLE ONLY, unlike search_exclude which reads
    # title + description. Freelance and gig work has to be filtered this way:
    # "contract" appears in the body of hundreds of legitimate staff postings
    # (contract research organization, contract negotiation, contracting
    # strategy), so a description-wide exclude deletes the good jobs along with
    # the gig work. In the title the signal is unambiguous.
    #
    # Deliberately NOT in this list, having been checked against the real pool:
    #   contract    - "Senior Manager, Contract Strategy" and 30 other staff roles
    #   consultant  - "Lead Consultant - Data Engineering" is a staff job
    #   temp        - "Temp to Perm" and "Temp Sr Associate Scientist" can convert
    # Add "temp,temporary,per diem" here if short-term W2 work is unwanted too.
    search_exclude_title: str = (
        "freelance,freelancer,contractor,independent contractor,"
        "contract to hire,contract-to-hire,1099,c2c,corp to corp,"
        "upwork,fiverr,gig work,self-employed"
    )
    min_match_score: float = 0.35
    # Max profile-derived search queries merged into the scrape. Each query is a
    # full keyword pass across all sites/locations, so this is a real runtime
    # multiplier — kept conservative; raise it if you want broader recall.
    query_expansion_max: int = 12
    # Master kill-switch for semantic ranking. False -> legacy bag-of-words score.
    semantic_ranking_enabled: bool = True
    # Stage-2 LLM re-rank: how many top jobs get the Ollama judge per cycle.
    rerank_top_k: int = 20
    # Kill-switch for stage-2 re-rank. False -> keep stage-1 score, no rationale.
    rerank_enabled: bool = True
    # In-range cutoff for the semantic recall filter (replaces the location gate).
    # PLACEHOLDER — calibrate with `jobbot profile calibrate` after a real cycle.
    semantic_recall_threshold: float = 0.55
    # --- Flexible answer matching (jobbot/answer_memory.py) ---
    # Above this token-set overlap, a reworded question is answered outright.
    # Deliberately high: thresholds are uniform across question kinds, so this
    # one value also has to be safe for work-authorisation and EEO questions.
    answer_match_high_threshold: float = 0.93
    # Between review and high, the match is not submitted -- it is offered as a
    # suggestion for the user to confirm instead.
    answer_match_review_threshold: float = 0.80
    # --- Outcome-tracking feedback loop (jobbot/outcomes.py) ---
    # Application 'applied' with no response after this many days -> auto 'ghosted'.
    ghost_after_days: int = 30
    # Beta-Binomial pseudo-count: how strongly per-feature rates are pulled toward
    # the global base rate (higher = more conservative on sparse data).
    outcome_prior_strength: float = 8
    # Max absolute ranking nudge from outcomes (on the 0..1 score). Tilt, not dominate.
    outcome_adjustment_cap: float = 0.08
    # Interview-rate signal weight relative to response-rate in the nudge.
    outcome_interview_weight: float = 1.5
    # A feature value needs at least this many trials to contribute a nudge.
    outcome_min_trials: int = 3
    # --- Referral-first routing (jobbot/referrals.py) ---
    # Max ranking boost from a warm intro (on the 0..1 score). Bigger than the
    # outcome nudge because a referral is a stronger lever.
    referral_bonus_cap: float = 0.15
    # Batch discovery: how many top companies (by high-fit job count) to search.
    referral_discover_max: int = 15
    # Max contacts stored per company from a single discovery pass.
    contacts_max_per_company: int = 5
    # Comma-separated institution terms that mark a contact as a warm affiliation
    # (e.g. "alumni_university,prior_lab,partner_institute"). Empty -> no institution boost.
    affiliation_terms: str = ""
    # Field gate for warm intros: only surface a warm-intro badge/draft when the
    # job's own match_score clears this bar, so an org match (e.g. a contact at a
    # big institution) doesn't drag in roles outside the candidate's field.
    warm_intro_min_score: float = 0.25
    # --- Interview coach (jobbot/interview_coach.py) ---
    # Master switch for pack auto-generation and practice. False -> the coach
    # never calls an LLM and routes/CLI report it is disabled.
    interview_coach_enabled: bool = True
    # Adaptive follow-ups per base question (bounded so a weak answer cannot
    # spiral into an endless drill-down).
    interview_followup_max: int = 1
    # Max ranking bonus (absolute, on the 0..1 callback probability) earned from
    # mock-interview practice. Never a penalty; zero sessions = neutral. Set via
    # `jobbot interview config --weight <x>` or INTERVIEW_PRACTICE_WEIGHT in .env.
    interview_practice_weight: float = 0.05
    # --- Outreach auto-draft queue (approve-then-send) ---
    outreach_enabled: bool = True
    outreach_cold_max: int = 10
    outreach_max_per_scan: int = 25
    # --- Company/source discovery (jobbot/discovery) ---
    discovery_pending_path: str = "data/discovered_pending.json"
    auto_discover_harvest: bool = True       # harvest ATS boards from scraped jobs each cycle
    discovery_websearch_queries: int = 8     # cap on web queries per `discover` run
    discovery_llm_max_companies: int = 15    # cap on LLM-suggested names to resolve
    discovery_min_fit_score: float = 0.0     # hide pending entries below this in review (0 = show all)
    location_required: bool = True
    location_radius_hint: str = "your metro area + remote/hybrid"

    # --- Auto-Apply Subsystem (EXPERIMENTAL / IN ACTIVE DEVELOPMENT) ---
    # WARNING: This feature is under active development. Direct automated form
    # submission should only be run in assisted review mode (headed browser).
    # Default is False. Never enable unattended submissions without prior testing.
    auto_apply: bool = False
    apply_rate_limit_per_day: int = 100
    auto_apply_min_score: float = 0.55
    auto_apply_min_callback_prob: float = 0.10
    auto_apply_require_watchlist_or_score: float = 0.65
    auto_apply_watchlist_only: bool = False
    auto_apply_default_recipient: str = ""
    auto_apply_skip_stale_days: int = 30

    # --- Liveness / expiration sweep (jobbot/liveness.py) ---
    # After each scrape cycle, probe up to N not-yet-applied active jobs and
    # soft-expire dead ones (status='expired'). 0 = off.
    liveness_check_per_cycle: int = 40
    # HTTP timeout (seconds) for one liveness probe.
    liveness_timeout: int = 20
    # A job whose posted_at/discovered_at is older than this shows a "stale"
    # badge in the dashboard (display only — not auto-expired).
    stale_after_days: int = 30
    schedule_interval_hours: int = 6

    # Paths
    base_resume_path: str = "data/base_resume.md"
    output_dir: str = "output"
    db_path: str = "data/jobbot.db"
    log_path: str = "logs/jobbot.log"
    company_watchlist: str = "data/companies.md"
    profile_path: str = "data/profile.md"
    # Hand-authored master skills intake (git-ignored). Parsed by jobbot/skills.py
    # into level-ranked skills that feed semantic matching + job-aware tailoring.
    skills_path: str = "MY_SKILLS_fill_in.md"
    # Structured, auto-derived candidate profile (jobbot/profile.py) — distinct
    # from profile_path above (which is the freeform markdown profile injected
    # into AI prompts). Cached JSON: extracted fields + embedding + resume hash.
    candidate_profile_path: str = "data/candidate_profile.json"
    # SOTA resume/cover-letter style template every AI tailoring path follows so
    # output is consistently professional. Leave the file in place; edit it to
    # change house style. Blank/missing = templates are simply not injected.
    resume_template_path: str = "data/resume_template.md"

    # Dashboard
    dashboard_host: str = "127.0.0.1"
    dashboard_port: int = 5000
    dashboard_secret: str = "change-me"
    # Local zone for rendering stored (naive-UTC) timestamps. Storage stays UTC
    # for consistency with every other timestamp in the schema; only display
    # converts.
    timezone: str = "America/New_York"

    # Output
    output_docx: bool = True

    # External job-search integrations
    linkedin_enabled: bool = True
    google_cse_key: str = ""        # optional — empty falls back to search engine HTML
    google_cse_cx: str = ""         # CSE engine ID
    # Serper.dev — Google results via one API key, no Cloud Console setup.
    # Preferred search backend for contact discovery when set (free tier ~2,500
    # queries). Tried before Google CSE and the bot-blocked HTML fallbacks.
    serper_api_key: str = ""

    # JobSpy unified scraper (replaces legacy LinkedIn + Indeed scrapers)
    # Sites: linkedin, indeed, glassdoor, zip_recruiter (comma-separated)
    jobspy_sites: str = "linkedin,indeed"
    # Only return postings published within this many hours (168 = 1 week)
    jobspy_hours_old: int = 168
    # Results to fetch per (keyword, location) call
    jobspy_results_wanted: int = 400
    # Concurrent scrape_jobs() threads — each call fetches all configured sites
    jobspy_workers: int = 3
    # Fetch full LinkedIn descriptions (extra HTTP request per job, richer content)
    jobspy_fetch_descriptions: bool = True

    # --- Per-scraper max-job limits (user tunable) ---
    max_linkedin_pages: int = 30           # 25 results × pages × (kw × loc) combos
    max_indeed_pages: int = 4             # 10 results × pages × (kw × loc) combos
    max_workday_offset: int = 400         # max offset depth per tenant × kw
    max_google_jobs_urls: int = 200       # JSON-LD detail fetches per cycle
    max_search_engine_results: int = 50   # per query, Bing/Brave/DDG fallbacks

    # Workday's search endpoint returns listing cards with NO description, so the
    # real JD body needs one extra HTTP call per new posting. False -> jobs keep
    # only the card metadata as their description, which makes their match_score
    # meaningless (it would be scored against scraper noise). Leave True unless
    # you are deliberately trading match quality for scrape speed.
    workday_fetch_descriptions: bool = True
    # Polite delay (seconds) between per-job Workday detail fetches.
    workday_detail_delay: float = 0.2

    # Mile-radius applied to scraping (expands user_locations.txt with nearby cities).
    # 0 disables — scrapers use raw user locations only.
    scrape_radius_miles: int = 50
    max_scrape_locations: int = 12        # cap on cities passed to location-aware scrapers

    @property
    def keywords(self) -> List[str]:
        return _csv(self.search_keywords)

    @property
    def locations(self) -> List[str]:
        return _csv(self.search_locations)

    @property
    def excludes(self) -> List[str]:
        return _csv(self.search_exclude)

    @property
    def excludes_title(self) -> List[str]:
        """Terms filtered on the job title alone. See `search_exclude_title`."""
        return _csv(self.search_exclude_title)

    @property
    def captcha_markers(self) -> List[str]:
        return [m.lower() for m in _csv(self.browser_apply_captcha_markers)]

    def ensure_dirs(self) -> None:
        for p in (self.output_dir, Path(self.db_path).parent, Path(self.log_path).parent, Path(self.base_resume_path).parent):
            Path(p).mkdir(parents=True, exist_ok=True)


settings = Settings()
settings.ensure_dirs()
