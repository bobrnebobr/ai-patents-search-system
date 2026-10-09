import asyncio
from unittest.mock import AsyncMock

import httpx
import pytest
from litestar.testing import TestClient
from qdrant_client import AsyncQdrantClient
from qdrant_client.http import models
from qdrant_client.http.exceptions import ResponseHandlingException, UnexpectedResponse

from src.app import create_app
from src.clients.errors import UpstreamError
from src.clients.mlserver import MLServerClient
from src.clients.qdrant import PatentSearchClient
from src.config import Settings
from src.schemas import PatentMatch
from src.services.ml import PatentSearchService


def test_encoder_request_and_vector() -> None:
    def respond(request: httpx.Request) -> httpx.Response:
        assert request.url.path == "/v2/models/patent-bi-encoder/infer"
        assert b"Battery recycling" in request.content
        return httpx.Response(200, json={"outputs": [{"shape": [1, 1024], "data": [0.1] * 1024}]})

    async def run() -> None:
        async with httpx.AsyncClient(
            base_url="http://ml:8080", transport=httpx.MockTransport(respond)
        ) as http:
            vector = await MLServerClient(http, "patent-bi-encoder").encode("Battery recycling")
        assert vector == [0.1] * 1024

    asyncio.run(run())


@pytest.mark.parametrize(
    ("error", "expected_status"),
    [(httpx.ReadTimeout("private"), 504), (httpx.ConnectError("private"), 503)],
)
def test_encoder_transport_errors(error: httpx.RequestError, expected_status: int) -> None:
    def respond(request: httpx.Request) -> httpx.Response:
        raise error

    async def run() -> None:
        async with httpx.AsyncClient(
            base_url="http://ml:8080", transport=httpx.MockTransport(respond)
        ) as http:
            with pytest.raises(UpstreamError) as caught:
                await MLServerClient(http, "encoder").encode("query")
        assert caught.value.status_code == expected_status
        assert "private" not in str(caught.value)

    asyncio.run(run())


@pytest.mark.parametrize("status", [404, 500, 503])
def test_encoder_http_errors(status: int) -> None:
    async def run() -> None:
        async with httpx.AsyncClient(
            base_url="http://ml:8080",
            transport=httpx.MockTransport(lambda request: httpx.Response(status)),
        ) as http:
            with pytest.raises(UpstreamError) as caught:
                await MLServerClient(http, "encoder").encode("query")
        assert caught.value.status_code == (502 if status == 500 else 503)

    asyncio.run(run())


@pytest.mark.parametrize(
    "payload",
    [
        {},
        {"outputs": []},
        {"outputs": [{"shape": [1, 2], "data": [1, 2]}]},
        {"outputs": [{"shape": [1, 1024], "data": [1]}]},
        {"outputs": [{"shape": [1, 1024], "data": ["nan"] * 1024}]},
    ],
)
def test_encoder_invalid_response(payload: dict) -> None:
    async def run() -> None:
        async with httpx.AsyncClient(
            base_url="http://ml:8080",
            transport=httpx.MockTransport(lambda request: httpx.Response(200, json=payload)),
        ) as http:
            with pytest.raises(UpstreamError) as caught:
                await MLServerClient(http, "encoder").encode("query")
        assert caught.value.status_code == 502

    asyncio.run(run())


def test_qdrant_search_maps_payload() -> None:
    sdk = AsyncMock(spec=AsyncQdrantClient)
    sdk.query_points.return_value = models.QueryResponse(
        points=[
            models.ScoredPoint(
                id=1,
                version=1,
                score=0.8,
                payload={"publication_id": "EP123", "title": "Battery", "abstract": "Recycling"},
            )
        ]
    )
    result = asyncio.run(PatentSearchClient(sdk, "patents").search([0.1] * 1024, 5))
    assert result[0].publication_id == "EP123"
    assert result[0].score == 0.8
    sdk.query_points.assert_awaited_once_with(
        collection_name="patents",
        query=[0.1] * 1024,
        limit=5,
        with_payload=True,
        with_vectors=False,
        timeout=300,
    )


@pytest.mark.parametrize(
    ("error", "status"),
    [
        (ResponseHandlingException(httpx.ReadTimeout("private")), 504),
        (ResponseHandlingException(httpx.ConnectError("private")), 503),
        (UnexpectedResponse(403, "Forbidden", b"private", httpx.Headers()), 503),
        (UnexpectedResponse(504, "Timeout", b"private", httpx.Headers()), 504),
    ],
)
def test_qdrant_errors(error: Exception, status: int) -> None:
    sdk = AsyncMock(spec=AsyncQdrantClient)
    sdk.query_points.side_effect = error
    with pytest.raises(UpstreamError) as caught:
        asyncio.run(PatentSearchClient(sdk, "patents").search([0.1], 1))
    assert caught.value.status_code == status
    assert "private" not in str(caught.value)


def test_qdrant_invalid_payload() -> None:
    sdk = AsyncMock(spec=AsyncQdrantClient)
    sdk.query_points.return_value = models.QueryResponse(
        points=[models.ScoredPoint(id=1, version=1, score=0.8, payload={})]
    )
    with pytest.raises(UpstreamError) as caught:
        asyncio.run(PatentSearchClient(sdk, "patents").search([0.1], 1))
    assert caught.value.status_code == 502


def test_qdrant_requires_collection() -> None:
    sdk = AsyncMock(spec=AsyncQdrantClient)
    with pytest.raises(UpstreamError) as caught:
        asyncio.run(PatentSearchClient(sdk, "").search([0.1], 1))
    assert caught.value.status_code == 503
    sdk.query_points.assert_not_called()


def test_search_service() -> None:
    encoder = AsyncMock(spec=MLServerClient)
    search = AsyncMock(spec=PatentSearchClient)
    encoder.encode.return_value = [0.1]
    search.search.return_value = [
        PatentMatch(publication_id="EP123", title="Battery", abstract="Recycling", score=0.8)
    ]
    result = asyncio.run(PatentSearchService(encoder, search).search("query", 5))
    encoder.encode.assert_awaited_once_with("query")
    search.search.assert_awaited_once_with([0.1], 5)
    assert result.patents[0].publication_id == "EP123"


def test_clients_are_shared_and_closed() -> None:
    app = create_app(settings=Settings(_env_file=None))
    with TestClient(app=app) as client:
        http = app.state.encoder_client.http
        encoder = app.state.encoder_client
        assert not http.is_closed
        assert client.get("/healthz").status_code == 200
        assert app.state.encoder_client is encoder
    assert http.is_closed
