from unittest.mock import AsyncMock

import pytest
from litestar import Litestar
from litestar.di import Provide
from litestar.testing import TestClient

from src.api.ml import MLController
from src.app import create_app
from src.clients.errors import UpstreamError
from src.config import Settings
from src.schemas import PatentSearchResponse
from src.services.ml import PatentSearchService


def test_patent_search_uses_injected_service() -> None:
    service = AsyncMock(spec=PatentSearchService)
    service.search.return_value = PatentSearchResponse(patents=[])
    app = Litestar(
        route_handlers=[MLController],
        dependencies={"patent_search_service": Provide(lambda: service, sync_to_thread=False)},
    )
    with TestClient(app=app) as client:
        response = client.post(
            "/api/v1/ml/patent-search/process",
            json={"text": "Battery recycling", "limit": 10},
        )

    assert response.status_code == 200
    assert response.json() == {"patents": []}
    service.search.assert_awaited_once_with("Battery recycling", 10)


@pytest.mark.parametrize("status", [502, 503, 504])
def test_patent_search_handles_upstream_errors(status: int) -> None:
    service = AsyncMock(spec=PatentSearchService)
    service.search.side_effect = UpstreamError("External service failed", status)
    app = Litestar(
        route_handlers=[MLController],
        dependencies={"patent_search_service": Provide(lambda: service, sync_to_thread=False)},
    )
    with TestClient(app=app) as client:
        response = client.post(
            "/api/v1/ml/patent-search/process", json={"text": "Battery recycling"}
        )
    assert response.status_code == status
    assert response.json()["detail"] == "External service failed"


@pytest.mark.parametrize(
    "payload",
    [
        {"text": "   "},
        {"text": "Battery recycling", "limit": 0},
        {"text": "Battery recycling", "limit": 101},
        {"text": "x" * 20001},
        {},
    ],
)
def test_patent_search_validates_input(payload: dict) -> None:
    with TestClient(app=create_app(settings=Settings(_env_file=None))) as client:
        response = client.post("/api/v1/ml/patent-search/process", json=payload)

    assert response.status_code == 400
