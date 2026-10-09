import json

from jobbot import profile as P


SAMPLE = {
    "skills": ["molecular dynamics", "Python", "free-energy calculation"],
    "role_titles": ["Computational Chemist", "Research Scientist"],
    "role_archetypes": ["MD simulation scientist", "molecular modeler"],
    "domains": ["biochemistry", "drug discovery"],
    "seniority": "mid",
    "dealbreakers": ["sales"],
    "summary": "Computational biochemist who builds and analyzes MD simulations.",
}


def test_seed_text_includes_skills_and_summary():
    prof = P.CandidateProfile(
        skills=SAMPLE["skills"], role_titles=SAMPLE["role_titles"],
        role_archetypes=SAMPLE["role_archetypes"], domains=SAMPLE["domains"],
        seniority="mid", dealbreakers=["sales"], summary=SAMPLE["summary"],
        source_resume_hash="abc", built_at="t")
    seed = prof.seed_text()
    assert "molecular dynamics" in seed
    assert "Computational Chemist" in seed
    assert SAMPLE["summary"] in seed


def test_build_profile_parses_ollama_json(monkeypatch):
    monkeypatch.setattr(P.oc, "_generate", lambda *a, **k: json.dumps(SAMPLE))
    prof = P.build_profile("RESUME TEXT")
    assert prof.skills == SAMPLE["skills"]
    assert prof.seniority == "mid"
    assert prof.source_resume_hash == P.resume_fingerprint("RESUME TEXT")


def test_load_profile_uses_cache_when_hash_matches(tmp_path, monkeypatch):
    path = tmp_path / "candidate_profile.json"
    monkeypatch.setattr(P.settings, "candidate_profile_path", str(path))
    monkeypatch.setattr(P, "_read_resume", lambda: "RESUME TEXT")
    # Pre-seed a cache file with the matching hash + an embedding.
    cached = dict(SAMPLE)
    cached["source_resume_hash"] = P.resume_fingerprint("RESUME TEXT")
    cached["built_at"] = "t"
    cached["embedding"] = [0.1, 0.2]
    path.write_text(json.dumps(cached), encoding="utf-8")
    # build_profile / embed must NOT be called on a cache hit.
    monkeypatch.setattr(P, "build_profile", lambda *a, **k: (_ for _ in ()).throw(AssertionError("rebuilt")))
    monkeypatch.setattr(P, "_safe_embed", lambda text: (_ for _ in ()).throw(AssertionError("re-embedded")))
    prof = P.load_profile()
    assert prof.skills == SAMPLE["skills"]
    assert prof.embedding == [0.1, 0.2]


def test_load_profile_falls_back_when_ollama_down(tmp_path, monkeypatch):
    path = tmp_path / "candidate_profile.json"
    monkeypatch.setattr(P.settings, "candidate_profile_path", str(path))
    monkeypatch.setattr(P, "_read_resume", lambda: "protein enzyme phd")
    def boom(*a, **k): raise RuntimeError("ollama down")
    monkeypatch.setattr(P, "build_profile", boom)
    monkeypatch.setattr(P, "_safe_embed", lambda text: None)
    prof = P.load_profile()
    assert isinstance(prof, P.CandidateProfile)
    assert prof.skills  # minimal profile is non-empty (from keywords/tags)
