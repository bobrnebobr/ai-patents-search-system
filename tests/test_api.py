from litestar.testing import TestClient

from src.app import app, create_app
from src.config import Settings


def test_healthz_returns_ok() -> None:
    with TestClient(app=app) as client:
        response = client.get("/healthz")

    assert response.status_code == 200
    assert response.json() == {"status": "ok"}


def test_version_returns_injected_version() -> None:
    settings = Settings(app_version="0.1.0", app_commit_sha="test-commit")

    with TestClient(app=create_app(settings=settings)) as client:
        response = client.get("/api/v1/version")

    assert response.status_code == 200
    assert response.json() == {
        "version": "0.1.0",
        "commit_sha": "test-commit",
    }


def test_healthz_includes_request_id() -> None:
    with TestClient(app=app) as client:
        response = client.get("/healthz")

    assert response.status_code == 200
    assert response.headers["x-request-id"]
