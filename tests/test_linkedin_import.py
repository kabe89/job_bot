# tests/test_linkedin_import.py
from pathlib import Path

import pytest

from jobbot.models import Contact, Job, init_db, session
from jobbot import linkedin_import


@pytest.fixture(autouse=True)
def _clean_db():
    def _wipe():
        init_db()
        with session() as db:
            db.query(Contact).delete()
            db.query(Job).delete()
            db.commit()
    _wipe()
    yield
    _wipe()


# A realistic LinkedIn export: a preamble "Notes:" block, then the header.
CSV_BODY = (
    "Notes:\n"
    '"When exporting your connection data, you may notice some emails are missing."\n'
    "\n"
    "First Name,Last Name,URL,Email Address,Company,Position,Connected On\n"
    "Ada,Lovelace,https://www.linkedin.com/in/ada/,ada@calc.org,Acme Technologies,Principal Scientist,01 Jan 2024\n"
    "Grace,Hopper,https://www.linkedin.com/in/grace/,,Navy Labs,Rear Admiral,02 Feb 2024\n"
    "Solo,,https://www.linkedin.com/in/solo/,,SomeCo,Engineer,03 Mar 2024\n"
    ",,,,,,\n"
)


def _write_csv(tmp_path: Path) -> str:
    p = tmp_path / "Connections.csv"
    p.write_text(CSV_BODY, encoding="utf-8")
    return str(p)


def test_import_parses_rows_and_skips_preamble_and_blanks(tmp_path):
    with session() as db:
        db.add(Job(hash="j1", source="s", title="Scientist",
                   company="Acme Technologies", url="u", description="d",
                   match_score=0.7))
        db.commit()
    stats = linkedin_import.import_connections(_write_csv(tmp_path))
    assert stats["imported"] == 3          # Ada, Grace, Solo (blank row skipped)
    assert stats["skipped"] == 1           # the ",,,,,," row
    assert stats["at_target"] == 1         # Ada @ Acme matches the job
    with session() as db:
        ada = db.query(Contact).filter_by(name="Ada Lovelace").one()
        assert ada.source == "import"
        assert ada.email == "ada@calc.org"
        assert ada.linkedin == "https://www.linkedin.com/in/ada"
        assert "connected 01 Jan 2024" in ada.notes
        # last-name-only / first-name-only still imported by whatever name exists
        assert db.query(Contact).filter_by(name="Solo").count() == 1


def test_reimport_is_idempotent(tmp_path):
    path = _write_csv(tmp_path)
    linkedin_import.import_connections(path)
    stats = linkedin_import.import_connections(path)
    assert stats["imported"] == 0
    assert stats["updated"] == 3
    with session() as db:
        assert db.query(Contact).count() == 3   # no duplicates


def test_missing_file_failopen(tmp_path):
    stats = linkedin_import.import_connections(tmp_path / "nope.csv")
    assert stats == {"imported": 0, "updated": 0, "skipped": 0, "at_target": 0}


def test_import_does_not_clobber_curated_manual_rows(tmp_path):
    # A hand-curated contact that matches the "Ada Lovelace" / Acme Technologies row in
    # CSV_BODY by name+company (blank linkedin forces the name+company fallback
    # path in import_connections).
    with session() as db:
        db.add(Contact(name="Ada Lovelace", company="Acme Technologies",
                        title="Curated Title", email="curated@x.org", linkedin="",
                        source="manual", relationship="pi", notes="hand-written note"))
        db.commit()

    linkedin_import.import_connections(_write_csv(tmp_path))

    with session() as db:
        rows = db.query(Contact).filter_by(name="Ada Lovelace").all()
        assert len(rows) == 1  # no duplicate created
        ada = rows[0]
        # Curated fields untouched.
        assert ada.title == "Curated Title"
        assert ada.email == "curated@x.org"
        assert ada.relationship == "pi"
        assert ada.notes == "hand-written note"
        assert ada.source == "manual"
        # Empty field enriched from the import.
        assert ada.linkedin == "https://www.linkedin.com/in/ada"
