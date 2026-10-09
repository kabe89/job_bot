<!--
SOTA HUMANIZED STYLE TEMPLATE — the standard every AI tailoring path follows.
This is a STYLE + STRUCTURE guide, NOT content. The AI must fill it using ONLY
verified facts from the candidate's resume/profile (truthfulness contract still
applies and overrides anything here).
-->

# Humanized Professional Resume & Cover-Letter Standard

## 1. Core Voice & Humanization Principles (Say NO to AI Clichés)
- **Authentic Professional Voice:** Write in the voice of an experienced practitioner speaking directly to a senior peer or hiring manager. Every line should sound like it was written by the candidate themselves, reflecting genuine hands-on expertise with systems, tools, and methodologies.
- **Strictly Banned AI Buzzwords & Fillers:**
  - ❌ NEVER use: "spearheaded", "leveraged", "pioneered", "utilized", "demonstrated proficiency in", "fostered a collaborative environment", "synergized", "executed end-to-end workflows", "results-driven", "dynamic team player", "passionate about", "seasoned".
  - ✅ Instead use direct, natural, active verbs: *Designed, Developed, Built, Automated, Engineered, Optimized, Implemented, Profiled, Modeled, Analyzed, Scaled, Deployed, Validated, Troubleshot, Quantified*.
- **Concrete Technical Context over Vague Generalities:**
  - Name real frameworks, architectures, protocols, and instruments (*e.g., Python, PostgreSQL, Docker, AWS, PyTorch, Linux/Bash, REST/gRPC, analytical pipelines*).
  - Describe the real problem, how it was solved at the terminal or bench, and the concrete technical outcome or performance improvement.
- **Natural Sentence Variety (Break the Robotic Formula):**
  - Do NOT mechanically repeat `[Action Verb] [Object] by [Method] resulting in [Metric %]` on every single bullet.
  - Mix technical implementations, architectural decisions, protocol optimizations, and problem-solving descriptions naturally.
  - When real metrics exist in the source, include them naturally; when they do not, state the clear technical outcome rather than inventing artificial percentages.

## 2. ATS & Recruiter Formatting Rules
- Clean GitHub-flavored Markdown only — no tables, columns, text boxes, images, emojis, or non-standard glyphs.
- 1–2 pages max. Concise, high-density, scannable.
- Reverse-chronological order within each section (most recent first).
- Present tense for current roles, past tense for completed roles. Zero first-person pronouns ("I", "my", "we").
- Dates as `Mon YYYY – Mon YYYY` (or `– Present`). Keep them aligned and consistent.
  **NEVER invent precision the source does not give, to satisfy this format.** If the source
  states only a year, write only that year. A source reading "Expected 2027" stays
  "Expected 2027" — it must NOT become "Expected May 2027". An unknown month is not a
  formatting problem to solve; it is a fact you do not have. This rule loses to the
  VERBATIM rule below every time.
- Copy URLs, DOIs, dates, citations, employer names, and titles VERBATIM from the source — never reformat digits or guess.
- Never drop a real publication, presentation, degree, award, or employer from the candidate's record.

## 3. Section Architecture & Layout
1. **Header** — Full Name (H1), Targeted Professional Title | City, ST | Email | Phone | LinkedIn / Portfolio | GitHub.
2. **Professional Summary** — 3–4 sentences. A compelling, authentic summary stating core identity, primary technical domains, notable milestones, and role alignment.
3. **Core Competencies** — Categorized, scannable groupings reflecting real workflows:
   - *Software Engineering & Backend Architecture*
   - *Cloud Infrastructure & DevOps*
   - *Data Systems, Modeling & High-Performance Computing*
   - *Testing, Observability & Reliability*
4. **Professional Experience** — 3–5 bullets per role, organized with bold role title, organization, location, and dates.
5. **Education** — Degrees, institutions, locations, honors, and focus areas.
6. **Projects & Publications** — Verbatim project descriptions or citations.
7. **Certifications & Awards** — Verbatim credentials and dates.

## 4. Professional Resume Skeleton
```markdown
# Alex Taylor
Senior Software Engineer | New York, NY
alex.taylor@example.com | (555) 012-3456 | linkedin.com/in/alextaylor | github.com/alextaylor

---

### Professional Summary
Software engineer with over 6 years of experience designing scalable distributed services and high-throughput data processing pipelines. Proven background optimizing system performance and automating cloud infrastructure across Linux and AWS environments. Focused on delivering reliable, maintainable code aligned with core business objectives.

---

### Core Competencies
- **Backend Architecture & APIs:** Python, Go, REST, gRPC, microservices, async concurrency, clean architecture
- **Data Engineering & Storage:** PostgreSQL, Redis, Apache Kafka, query optimization, schema migrations, ETL pipelines
- **Cloud Infrastructure & DevOps:** Docker, Kubernetes, AWS (ECS, S3, IAM), CI/CD workflows, Linux/Bash, Terraform
- **Quality & Observability:** Pytest, integration testing, Prometheus, Grafana, OpenTelemetry, structured logging

---

### Professional Experience

**Senior Backend Engineer** | Apex Innovations — New York, NY
*Jan 2023 – Present*
- Engineered event-driven microservices in Python and Go, handling over 25,000 real-time events per minute.
- Optimized PostgreSQL database queries and indexes, reducing p99 API latency from 450ms to under 120ms.
- Automated containerized CI/CD build and deployment pipelines using Docker and GitHub Actions.

**Software Engineer** | Pinnacle Systems — Boston, MA
*Jun 2020 – Dec 2022*
- Developed RESTful API endpoints and background worker tasks supporting customer analytics reporting.
- Built automated data validation suites and unit tests, expanding test coverage across core payment modules.
- Managed infrastructure provisioning with Terraform and maintained service monitoring with Prometheus and Grafana.

---

### Education
**B.S. in Computer Science** | State University — New York, NY
*Sep 2016 – May 2020*
- Honors: Magna Cum Laude | Focus: Distributed Systems and Database Architecture

---

### Projects & Open Source
- **Distributed Task Queue:** Built a lightweight distributed message queue in Go with persistent WAL and raft consensus.
- **Data Flow CLI:** Open-source Python developer tool for inspecting and transforming nested JSON datasets.

---

### Certifications & Honors
- AWS Certified Solutions Architect – Associate (2023)
```

## 5. Formal Executive Cover-Letter Standard
Follow the formal business letterhead structure:
1. **Sender Header Block:**
   ```
   [Full Name, Degree / Title]
   [Street Address]
   [City, State ZIP]
   [Month Day, Year]
   ```
2. **Recipient Header Block:**
   ```
   [Hiring Manager Name / Title / Search Committee]
   [Target Company / Organization]
   [Division / Department]
   [City, State ZIP]
   ```
3. **Salutation:**
   `Dear [Hiring Manager / Hiring Team],`
4. **Authoritative 4-Paragraph Peer-to-Peer Professional Body (~350–450 words):**
   - **Paragraph 1 (Strategic Alignment & Enthusiasm):** Express genuine enthusiasm for the specific role and team initiative. Directly bridge candidate's core background (e.g. backend architecture, cloud scalability, data engineering) with the company's specific mission, product initiative, or technical challenge.
   - **Paragraph 2 (Deep Technical Rigor & Concrete Execution):** Detail concrete engineering frameworks and tools (e.g., Python, PostgreSQL, distributed systems, Linux, cloud pipelines). Highlight specific achievements, architectural trade-offs solved, and quantitative performance improvements.
   - **Paragraph 3 (Scalable Solutions, Collaboration & Reliability):** Explain how your technical design translates into reliable systems, improved engineering velocity, and robust operational stability. Highlight commitment to peer collaboration, code quality, and technical documentation.
   - **Paragraph 4 (Value Proposition & Seamless Onboarding):** Emphasize immediate continuity with zero onboarding friction, intimate familiarity with standard toolchains, readiness from day one to deliver value, and a forward-looking call to discuss advancing team goals.
5. **Formal Complimentary Close & Signature Block:**
   ```
   Thank you very much for your time and consideration.

   Sincerely,

   [Full Name, Title]
   [Email | Phone]
   [LinkedIn / GitHub Profile]
   ```
