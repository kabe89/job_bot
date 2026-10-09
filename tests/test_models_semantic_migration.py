from sqlalchemy import create_engine, inspect, text

from jobbot import models


def test_job_has_semantic_columns():
    cols = {c.name for c in models.Job.__table__.columns}
    assert {"embedding", "rerank_score", "rerank_rationale"} <= cols


def test_init_db_adds_missing_columns(tmp_path, monkeypatch):
    # A pre-existing DB whose jobs table lacks the new columns gets them added.
    db_file = tmp_path / "old.db"
    eng = create_engine(f"sqlite:///{db_file}", future=True)
    with eng.begin() as conn:
        conn.execute(text(
            "CREATE TABLE jobs (id INTEGER PRIMARY KEY, hash TEXT, source TEXT, "
            "title TEXT, company TEXT, url TEXT, match_score FLOAT)"))
    monkeypatch.setattr(models, "_engine", eng)
    models.init_db()
    cols = {c["name"] for c in inspect(eng).get_columns("jobs")}
    assert {"embedding", "rerank_score", "rerank_rationale"} <= cols
    # Idempotent: a second call must not raise.
    models.init_db()
