from jobbot import skills


SAMPLE = """\
====================================================================
A. COMPUTATIONAL CHEMISTRY / CADD
====================================================================
- Molecular docking: Proficient — (which tools? Vina/Glide?) ... ZDock, MOE-DOCK, Vina
- Metadynamics / enhanced sampling: ... none
- MOE (Molecular Operating Environment): Some — structure prep ... Proficient over 6 years SVL
- PyMOL: ... Yep
- Schrodinger suite: ... Nope

====================================================================
B. MACHINE LEARNING
====================================================================
- Deep learning on molecules (GNNs): ...Yep GNNs are my specialty
- Machine learning (general): Proficient — built ML model predicting aptamer function
"""


def test_parse_extracts_name_level_notes_category():
    parsed = skills.parse_skills(SAMPLE)
    by_name = {s.name: s for s in parsed}
    assert by_name["Molecular docking"].level == "Proficient"
    assert "Vina" in by_name["Molecular docking"].notes
    assert by_name["Molecular docking"].category.startswith("A.")


def test_informal_answers_map_to_levels():
    parsed = {s.name: s for s in skills.parse_skills(SAMPLE)}
    # "Yep" -> Proficient, "Nope"/"none" -> None
    assert parsed["PyMOL"].level == "Proficient"
    assert parsed["Schrodinger suite"].level == "None"
    assert parsed["Metadynamics / enhanced sampling"].level == "None"


def test_explicit_level_after_notes_wins_highest():
    # "Some — ... Proficient over 6 years" should take the strongest declared level
    parsed = {s.name: s for s in skills.parse_skills(SAMPLE)}
    assert parsed["MOE (Molecular Operating Environment)"].level == "Proficient"


def test_known_skills_and_names_exclude_none():
    parsed = skills.parse_skills(SAMPLE)
    names = skills.skill_names(parsed, min_rank=2)
    assert "Molecular docking" in names
    assert "Schrodinger suite" not in names  # level None
    assert "Deep learning on molecules (GNNs)" in names


def test_inventory_markdown_groups_by_level():
    md = skills.inventory_markdown(skills.parse_skills(SAMPLE))
    assert "Proficient" in md
    assert "Molecular docking" in md
    assert "Schrodinger suite" not in md  # None excluded


def test_relevant_skills_ranks_by_keyword_when_no_embeddings(monkeypatch):
    # Force the embedding path to fail so we hit the keyword fallback.
    monkeypatch.setattr(skills, "_safe_embed", lambda text: None)
    parsed = skills.parse_skills(SAMPLE)
    monkeypatch.setattr(skills, "load_skills", lambda path=None: parsed)
    rel = skills.relevant_skills(
        "We need a scientist skilled in GNNs and deep learning for molecules.",
        job_embedding=None, top_n=3)
    assert rel and rel[0].name == "Deep learning on molecules (GNNs)"


def test_load_profile_merges_verified_skill_names(tmp_path, monkeypatch):
    from jobbot import profile as prof_mod
    from jobbot import skills as sk

    # Point profile cache + resume at temp; stub the resume + Ollama profile build.
    monkeypatch.setattr(prof_mod.settings, "candidate_profile_path",
                        str(tmp_path / "cand.json"))
    monkeypatch.setattr(prof_mod, "_read_resume", lambda: "RNA aptamer scientist resume")
    monkeypatch.setattr(prof_mod, "build_profile", lambda text: prof_mod.CandidateProfile(
        skills=["RNA"], role_titles=[], role_archetypes=[], domains=[],
        seniority="mid", dealbreakers=[], summary="RNA scientist",
        source_resume_hash="", built_at=""))
    monkeypatch.setattr(prof_mod, "_safe_embed", lambda text: [0.1, 0.2])
    sample = [sk.Skill("GNNs", "Expert", "graph nets", "B.")]
    monkeypatch.setattr(prof_mod, "load_skills", lambda path=None: sample)
    monkeypatch.setattr(prof_mod, "skills_fingerprint", lambda path=None: "abc")

    p = prof_mod.load_profile(force=True)
    assert "GNNs" in p.skills
