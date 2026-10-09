import json
import pytest
from jobbot import web as web_mod
from jobbot.models import Application, Job, session


@pytest.fixture
def client():
    app = web_mod.create_app()
    app.config.update(TESTING=True)
    return app.test_client()


def test_kanban_get_route(client):
    with session() as db:
        test_job = Job(
            hash=Job.make_hash("test", "http://example.com/kanban-get-test", "Lead Biologist"),
            source="test",
            title="Lead Biologist",
            company="KanbanBio",
            url="http://example.com/kanban-get-test",
            status="starred",
            starred=True,
            match_score=0.88,
        )
        db.add(test_job)
        db.commit()
        job_id = test_job.id

    try:
        res = client.get("/kanban")
        assert res.status_code == 200
        html = res.get_data(as_text=True)
        # Check board columns
        assert "Application Kanban Board" in html
        assert "Wishlist" in html
        assert "Tailored" in html
        assert "Applied" in html
        assert "Interview" in html
        assert "Offer" in html
        assert "Archive" in html
        # Check dropzones
        assert 'data-column="wishlist"' in html
        assert 'data-column="tailored"' in html
        assert 'data-column="applied"' in html
        assert 'data-column="interview"' in html
        assert 'data-column="offer"' in html
        assert 'data-column="rejected"' in html
        # Check card content and Tailor Studio links
        assert "Lead Biologist" in html
        assert f"/tailor-studio/{job_id}" in html
        assert f"/job/{job_id}" in html
    finally:
        with session() as db:
            j = db.get(Job, job_id)
            if j:
                db.delete(j)
                db.commit()


def test_kanban_move_api(client):
    # Create a test job
    with session() as db:
        test_job = Job(
            hash=Job.make_hash("test", "http://example.com/kanban-test", "Senior Scientist"),
            source="test",
            title="Senior Scientist",
            company="KanbanBio",
            url="http://example.com/kanban-test",
            status="starred",
            starred=True,
            match_score=0.92,
        )
        db.add(test_job)
        db.commit()
        job_id = test_job.id

    try:
        # Move to tailored
        res = client.post(
            "/api/kanban/move",
            data=json.dumps({"job_id": job_id, "target_column": "tailored"}),
            content_type="application/json",
        )
        assert res.status_code == 200
        data = res.get_json()
        assert data["ok"] is True
        assert data["job_id"] == job_id
        assert data["new_status"] == "tailored"

        # Verify DB
        with session() as db:
            j = db.get(Job, job_id)
            assert j.status == "tailored"

        # Move to applied
        res = client.post(
            "/api/kanban/move",
            data=json.dumps({"job_id": job_id, "target_column": "applied"}),
            content_type="application/json",
        )
        assert res.status_code == 200
        data = res.get_json()
        assert data["ok"] is True
        assert data["new_status"] == "applied"

        # Verify DB: Application should have status applied and sent_at set
        with session() as db:
            j = db.get(Job, job_id)
            assert j.status == "applied"
            assert len(j.applications) > 0
            latest = j.applications[-1]
            assert latest.status == "applied"
            assert latest.sent_at is not None
            app_id = latest.id

        # Move using application_id to interview
        res = client.post(
            "/api/kanban/move",
            data=json.dumps({"application_id": app_id, "target_column": "interview"}),
            content_type="application/json",
        )
        assert res.status_code == 200
        data = res.get_json()
        assert data["ok"] is True
        assert data["new_status"] == "interview"

        with session() as db:
            j = db.get(Job, job_id)
            assert j.status == "interview"
            assert j.applications[-1].status == "interview"

        # Move to offer
        res = client.post(
            "/api/kanban/move",
            data=json.dumps({"job_id": job_id, "target_column": "offer"}),
            content_type="application/json",
        )
        assert res.status_code == 200
        assert res.get_json()["new_status"] == "offer"

        # Move to rejected/archive
        res = client.post(
            "/api/kanban/move",
            data=json.dumps({"job_id": job_id, "target_column": "archive"}),
            content_type="application/json",
        )
        assert res.status_code == 200
        assert res.get_json()["new_status"] == "rejected"

        # Test invalid column
        res = client.post(
            "/api/kanban/move",
            data=json.dumps({"job_id": job_id, "target_column": "invalid_xyz"}),
            content_type="application/json",
        )
        assert res.status_code == 400
        assert res.get_json()["ok"] is False

        # Test invalid job_id
        res = client.post(
            "/api/kanban/move",
            data=json.dumps({"job_id": 999999999, "target_column": "offer"}),
            content_type="application/json",
        )
        assert res.status_code == 404
        assert res.get_json()["ok"] is False

    finally:
        # Clean up test job and applications
        with session() as db:
            j = db.get(Job, job_id)
            if j:
                db.delete(j)
                db.commit()
