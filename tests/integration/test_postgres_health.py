from litestar.testing import TestClient
from testcontainers.community.postgres import PostgresContainer

from src.app import create_app
from src.config import Settings


def test_health_with_real_postgres() -> None:
    with PostgresContainer("postgres:16-alpine", driver=None) as postgres:
        settings = Settings(database_url=postgres.get_connection_url())
        with TestClient(app=create_app(settings=settings)) as client:
            healthy_response = client.get("/api/v1/health")

        assert healthy_response.status_code == 200

        postgres_health = healthy_response.json()["dependencies"]["postgres"]
        assert healthy_response.json()["status"] == "ok"
        assert postgres_health["status"] == "ok"
        assert postgres_health["version"]
        assert postgres_health["response_time_ms"] >= 0
        assert postgres_health["error"] is None

    with TestClient(app=create_app(settings=settings)) as client:
        unavailable_response = client.get("/api/v1/health")

    postgres_health = unavailable_response.json()["dependencies"]["postgres"]
    assert unavailable_response.status_code == 503
    assert unavailable_response.json()["status"] == "degraded"
    assert postgres_health["status"] == "error"
    assert postgres_health["version"] is None
    assert postgres_health["response_time_ms"] >= 0
    assert postgres_health["error"]
