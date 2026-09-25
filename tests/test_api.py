from litestar.testing import TestClient

from src.api import health as health_api
from src.app import app
from src.schemas import DependencyHealth


def test_healthz_returns_ok() -> None:
    with TestClient(app=app) as client:
        response = client.get("/healthz")

    assert response.status_code == 200
    assert response.json() == {"status": "ok"}


def test_version_returns_local_version() -> None:
    with TestClient(app=app) as client:
        response = client.get("/api/v1/version")

    assert response.status_code == 200
    assert response.json() == {
        "version": "0.1.0",
        "commit_sha": "local",
    }


async def postgres_is_available(_: str) -> DependencyHealth:
    return DependencyHealth(
        status="ok",
        response_time_ms=4.2,
        version="16.4",
    )


def test_health_returns_ok_when_postgres_is_available(monkeypatch) -> None:
    monkeypatch.setattr(health_api, "check_postgres", postgres_is_available)

    with TestClient(app=app) as client:
        response = client.get("/api/v1/health")

    assert response.status_code == 200
    assert response.json()["status"] == "ok"
    assert response.json()["dependencies"]["postgres"]["version"] == "16.4"


async def postgres_is_unavailable(_: str) -> DependencyHealth:
    return DependencyHealth(
        status="error",
        response_time_ms=1.5,
        error="ConnectionRefusedError",
    )


def test_health_returns_503_when_postgres_is_unavailable(monkeypatch) -> None:
    monkeypatch.setattr(health_api, "check_postgres", postgres_is_unavailable)

    with TestClient(app=app) as client:
        response = client.get("/api/v1/health")

    assert response.status_code == 503
    assert response.json()["status"] == "degraded"


def test_healthz_includes_request_id() -> None:
    with TestClient(app=app) as client:
        response = client.get("/healthz")

    assert response.status_code == 200
    assert response.headers["x-request-id"]
