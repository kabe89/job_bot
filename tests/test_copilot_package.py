import pytest
from jobbot.web import create_app
from jobbot.models import Job, Application, init_db, session
from jobbot.config import settings


@pytest.fixture
def client(tmp_path, monkeypatch):
    monkeypatch.setattr(settings, "db_path", str(tmp_path / "test.db"))
    init_db()
    app = create_app()
    app.config["TESTING"] = True
    with app.test_client() as client:
        yield client


def test_copilot_package_options_cors(client):
    res = client.options("/api/job/1/copilot-package")
    assert res.status_code == 200
    assert res.headers.get("Access-Control-Allow-Origin") == "*"
    assert "GET" in res.headers.get("Access-Control-Allow-Methods", "")


def test_copilot_package_404_when_missing(client):
    res = client.get("/api/job/99999/copilot-package")
    assert res.status_code == 404
    data = res.get_json()
    assert "error" in data


def test_copilot_package_by_id_and_url(client):
    init_db()
    with session() as db:
        job = Job(
            hash="copilot_test_1",
            source="test",
            title="Senior Platform Engineer",
            company="Acme Corp",
            url="https://jobs.example.com/acme/platform-eng",
            description="Requirements: Python, Distributed Systems",
            status="new"
        )
        db.add(job)
        db.commit()
        db.refresh(job)
        job_id = job.id

    # Test by ID
    res = client.get(f"/api/job/{job_id}/copilot-package")
    assert res.status_code == 200
    data = res.get_json()
    assert data["job_id"] == job_id
    assert data["company"] == "Acme Corp"
    assert data["title"] == "Senior Platform Engineer"
    assert "applicant" in data
    assert "screening_answers" in data

    # Test by URL
    res_url = client.get(f"/api/copilot-package?url=https://jobs.example.com/acme/platform-eng")
    assert res_url.status_code == 200
    data_url = res_url.get_json()
    assert data_url["job_id"] == job_id
    assert data_url["company"] == "Acme Corp"


def test_copilot_answer_question_options_cors(client):
    res = client.options("/api/copilot/answer-question")
    assert res.status_code == 200
    assert res.headers.get("Access-Control-Allow-Origin") == "*"
    assert "POST" in res.headers.get("Access-Control-Allow-Methods", "")


def test_copilot_answer_question_endpoint(client, monkeypatch):
    monkeypatch.setattr(settings, "work_authorized", "Yes")
    monkeypatch.setattr(settings, "requires_sponsorship", "No")

    # Standard work authorization question
    payload = {
        "question": "Are you legally authorized to work in the United States?",
        "options": ["Yes", "No"],
        "qtype": "select"
    }
    res = client.post("/api/copilot/answer-question", json=payload)
    assert res.status_code == 200
    data = res.get_json()
    assert data["ok"] is True
    assert data["answer"] == "Yes"

    # Visa sponsorship question
    payload_visa = {
        "question": "Will you now or in the future require visa sponsorship?",
        "options": ["Yes", "No"],
        "qtype": "select"
    }
    res_visa = client.post("/api/copilot/answer-question", json=payload_visa)
    assert res_visa.status_code == 200
    data_visa = res_visa.get_json()
    assert data_visa["ok"] is True
    assert data_visa["answer"] == "No"


def test_launch_copilot_function(monkeypatch, tmp_path):
    from jobbot import launch_copilot
    from unittest.mock import patch, MagicMock

    init_db()
    with session() as db:
        job = Job(
            hash="launch_copilot_test",
            source="test",
            title="Lead Backend Architect",
            company="InnoTech",
            url="https://jobs.example.com/innotech/lead-arch",
            description="Requirements: Python, Architecture",
            status="new"
        )
        db.add(job)
        db.commit()
        db.refresh(job)
        job_id = job.id

    with patch("webbrowser.open", return_value=True), \
         patch("jobbot.apply_runner.build_package", return_value=MagicMock()):
        res = launch_copilot(job_id=job_id, browser_mode="system")
        assert res["ok"] is True
        assert res["job_id"] == job_id
        assert res["company"] == "InnoTech"
        assert "javascript:" in res["bookmarklet"]


def test_launch_copilot_web_routes(client):
    init_db()
    with session() as db:
        job = Job(
            hash="launch_route_test",
            source="test",
            title="Senior Staff Scientist",
            company="Apex Labs",
            url="https://jobs.example.com/apex/scientist",
            description="Requirements: Python, Systems",
            status="new"
        )
        db.add(job)
        db.commit()
        db.refresh(job)
        job_id = job.id

    # Test OPTIONS CORS
    opt = client.options(f"/api/job/{job_id}/launch-copilot")
    assert opt.status_code == 200
    assert opt.headers.get("Access-Control-Allow-Origin") == "*"

    # Test JSON API
    api_res = client.post(f"/api/job/{job_id}/launch-copilot")
    assert api_res.status_code == 200
    data = api_res.get_json()
    assert data["ok"] is True
    assert data["job_id"] == job_id
    assert "bookmarklet" in data

    # Test Web redirect
    web_res = client.get(f"/job/{job_id}/launch-copilot", follow_redirects=False)
    assert web_res.status_code == 302
    assert f"/apply-kit/{job_id}" in web_res.headers["Location"]


def test_copilot_cli_command():
    from click.testing import CliRunner
    from jobbot.cli import cli
    from unittest.mock import patch, MagicMock

    init_db()
    with session() as db:
        job = Job(
            hash="cli_copilot_test",
            source="test",
            title="DevOps Specialist",
            company="CloudScale",
            url="https://jobs.example.com/cloudscale/devops",
            description="Requirements: Linux, Docker",
            status="new"
        )
        db.add(job)
        db.commit()
        db.refresh(job)
        job_id = job.id

    runner = CliRunner()
    with patch("webbrowser.open", return_value=True), \
         patch("jobbot.apply_runner.build_package", return_value=MagicMock()):
        result = runner.invoke(cli, ["copilot", str(job_id), "--browser", "system"])
        assert result.exit_code == 0
        assert "Copilot session finished" in result.output
