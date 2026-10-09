from jobbot import pipeline


def test_augment_description_appends_relevant_skills(monkeypatch):
    monkeypatch.setattr(pipeline, "relevant_markdown",
                        lambda text, job_embedding=None, top_n=8: "REL: GNNs (Expert)")
    out = pipeline._augment_description("Base JD text", None)
    assert "Base JD text" in out
    assert "REL: GNNs (Expert)" in out


def test_augment_description_noop_when_no_skills(monkeypatch):
    monkeypatch.setattr(pipeline, "relevant_markdown",
                        lambda text, job_embedding=None, top_n=8: "")
    out = pipeline._augment_description("Base JD text", None)
    assert out == "Base JD text"
