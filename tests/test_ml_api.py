from unittest.mock import AsyncMock

import pytest
from litestar import Litestar
from litestar.di import Provide
from litestar.testing import TestClient

from src.api.ml import MLController
from src.app import create_app
from src.clients.errors import UpstreamError
from src.clients.mlserver import MLServerClient, PatentClassifierClient
from src.clients.qdrant import PatentSearchClient
from src.config import Settings
from src.schemas import PatentClassification, PatentMatch, PatentSearchResponse
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


def test_patent_search_returns_classification_without_filtering_or_reordering() -> None:
    encoder = AsyncMock(spec=MLServerClient)
    search = AsyncMock(spec=PatentSearchClient)
    classifier = AsyncMock(spec=PatentClassifierClient)
    encoder.encode.return_value = [0.1]
    encoder.encode_batch.return_value = [[0.2], [0.3]]
    search.search.return_value = [
        PatentMatch(publication_id="EP1", title="First", abstract="Text", score=0.8),
        PatentMatch(publication_id="EP2", title="Second", abstract="Text", score=0.7),
    ]
    classifier.classify.return_value = [
        PatentClassification(label="not_relevant", score=0.1),
        PatentClassification(label="relevant", score=0.9),
    ]
    service = PatentSearchService(encoder, search, classifier)
    app = Litestar(
        route_handlers=[MLController],
        dependencies={"patent_search_service": Provide(lambda: service, sync_to_thread=False)},
    )
    with TestClient(app=app) as client:
        response = client.post(
            "/api/v1/ml/patent-search/process", json={"text": "Battery recycling", "limit": 2}
        )
    assert response.status_code == 200
    patents = response.json()["patents"]
    assert [patent["publication_id"] for patent in patents] == ["EP1", "EP2"]
    assert [patent["score"] for patent in patents] == [0.8, 0.7]
    assert [patent["relevance_label"] for patent in patents] == ["not_relevant", "relevant"]
    assert [patent["relevance_score"] for patent in patents] == [0.1, 0.9]
    encoder.encode_batch.assert_awaited_once_with(["First. Text", "Second. Text"])


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
