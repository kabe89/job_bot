# JobBot Development & Architectural Invariants

This document establishes the mandatory standards and invariants for developing, updating, and extending `jobbot`. All AI agents, contributors, and maintainers must strictly follow these rules.

---

## 1. Truthfulness & Anti-Hallucination Invariant (Strict Non-Negotiable)
- **Zero Hallucination Tolerance:** Under no circumstance may any AI model or agent invent, extrapolate, or hallucinate credentials, degrees, dates, employers, publications, presentations, or skills.
- **Source of Truth:** The candidate's real resume, `data/resume_facts.json`, and `data/profile_context.py` represent the authoritative boundary of truth.
- **Genuine Gaps:** When a job requirement is absent from the candidate's real record, leave it unfilled rather than fabricating or inflating experience.
- **Date Precision:** Never invent months or days not present in the source (e.g., "Expected 2027" stays "Expected 2027", never "Expected May 2027"). In any conflict between format and source, verbatim wins.

---

## 2. Professional Voice & Anti-AI Cliché Contract
- **Authentic Voice:** All generated resumes and cover letters must sound like an authentic technical expert speaking peer-to-peer to hiring managers and senior technical leads.
- **Banned AI Filler & Buzzwords:**
  - ❌ NEVER use: `spearheaded`, `leveraged`, `pioneered`, `utilized`, `streamlined`, `fostered`, `orchestrated`, `passionate`, `seasoned`, `beacon`, `testament`, `dynamic team player`, `results-driven`, `executed workflows`.
- **Mandated Active Technical Verbs:**
  - ✅ Use direct, natural technical verbs: *designed, built, developed, optimized, characterized, modeled, automated, implemented, formulated, validated, troubleshot, quantified, analyzed*.
- **Concrete Technical Context:** Name real frameworks, libraries, protocols, and tools (*PyTorch, Docker, PostgreSQL, React, Kubernetes, AWS, FastAPI*) rather than vague generalities.
- **Break the Robotic Formula:** Avoid mechanical repetition of `[Verb] [Object] by [Method] resulting in [Metric %]`. Mix problem-solving, protocol refinements, and technical discoveries naturally.

---

## 3. Self-Contained Repository & Relative Path Contract
- **Zero Explicit Directory Hardcoding:** No code, configuration, or test may reference explicit absolute directories (e.g. hardcoded filesystem paths or user home trees).
- **Relative Path Resolution:** All paths must resolve relative to the repository root (e.g. `Path("data/jobbot.db")` or `Path(__file__).resolve().parent.parent`).
- **Platform Agnostic:** Code must execute seamlessly on Windows, Linux, and macOS without path delimiter assumptions.

---

## 4. Windows SSL & Network Resilience
- **Native Truststore:** Python on Windows frequently fails to verify SSL certificates for hosts like LinkedIn or Workday when using static Mozilla `certifi` bundles (`SSLCertVerificationError`).
- **Standard Protocol:** Always initialize `truststore.inject_into_ssl()` at startup in `jobbot/__init__.py` so requests delegate to the Windows native Certificate Store.

---

## 5. Performance & Non-Blocking Design
- **Local Ollama Concurrency:** Local LLMs cannot process concurrent embeddings and generation requests without queuing stalls.
- **Embedding Cache:** Always use persistent caching (`data/skill_embeddings_cache.json`) for skill and keyword vector operations. Never trigger sequential embedding calls inside UI requests.
- **Sub-Millisecond Triage:** Index inspector views, status updates, and Kanban movements must execute asynchronously or sub-millisecond via zero-reload AJAX endpoints (`/job/<id>/inspect`, `/job/<id>/status-api`, `/api/kanban/move`).

---

## 6. Auto-Apply Subsystem Invariant (Experimental Safety)
- **Experimental Status:** The automated application engine (`jobbot/browser_apply.py`, `jobbot/ats/*`) is in active development.
- **Default Disabled:** `auto_apply` and `browser_apply_autosubmit` must default to `False`.
- **Interactive Review First:** Always default to headed browser review mode so the user can verify fields before submission.

---

## 7. Non-Destructive Database Management
- **Database Safety:** Always maintain backups (`data/jobbot.db.bak`). Never execute destructive migrations or drop user data without explicit approval.
- **Model Invariants:** When instantiating `Job`, always provide required attributes such as `url` (`url` is NOT NULL).
- **Test Compatibility:** Preserve all UI test hooks (`class="jobs"`, `id="bulk-form"`, `data-legacy-action`, `url_for('tailor')`, `url_for('star')`) so that existing test suites remain 100% green.
