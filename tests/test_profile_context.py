from jobbot import profile_context


def test_context_merges_profile_md_and_skill_inventory(tmp_path, monkeypatch):
    prof = tmp_path / "profile.md"
    prof.write_text("Jane is a data scientist.", encoding="utf-8")
    monkeypatch.setattr(profile_context.settings, "profile_path", str(prof))

    from jobbot import skills as sk
    sample = [sk.Skill("GNNs", "Expert", "graph nets", "B."),
              sk.Skill("Docking", "Proficient", "Vina", "A.")]
    monkeypatch.setattr(profile_context, "load_skills", lambda path=None: sample)
    profile_context.refresh()

    ctx = profile_context.applicant_context()
    assert "data scientist" in ctx
    assert "VERIFIED SKILLS" in ctx
    assert "GNNs" in ctx


def test_context_empty_when_nothing_configured(tmp_path, monkeypatch):
    monkeypatch.setattr(profile_context.settings, "profile_path",
                        str(tmp_path / "missing.md"))
    monkeypatch.setattr(profile_context, "load_skills", lambda path=None: [])
    profile_context.refresh()
    assert profile_context.applicant_context() == ""
