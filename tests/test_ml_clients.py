import asyncio
import json
from unittest.mock import AsyncMock

import httpx
import pytest
from litestar.testing import TestClient
from qdrant_client import AsyncQdrantClient
from qdrant_client.http import models
from qdrant_client.http.exceptions import ResponseHandlingException, UnexpectedResponse

from src.app import create_app
from src.clients.errors import UpstreamError
from src.clients.mlserver import MLServerClient, PatentClassifierClient
from src.clients.qdrant import PatentSearchClient
from src.config import Settings
from src.schemas import PatentClassification, PatentMatch
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
    classifier = AsyncMock(spec=PatentClassifierClient)
    encoder.encode.return_value = [0.1]
    encoder.encode_batch.return_value = [[0.2]]
    classifier.classify.return_value = [PatentClassification(label="relevant", score=0.9)]
    search.search.return_value = [
        PatentMatch(publication_id="EP123", title="Battery", abstract="Recycling", score=0.8)
    ]
    result = asyncio.run(PatentSearchService(encoder, search, classifier).search("query", 5))
    encoder.encode.assert_awaited_once_with("query")
    search.search.assert_awaited_once_with([0.1], 5)
    encoder.encode_batch.assert_awaited_once_with(["Battery. Recycling"])
    classifier.classify.assert_awaited_once_with([0.1], [[0.2]])
    assert result.patents[0].publication_id == "EP123"
    assert result.patents[0].score == 0.8
    assert result.patents[0].relevance_label == "relevant"
    assert result.patents[0].relevance_score == 0.9


def test_empty_search_skips_classification() -> None:
    encoder = AsyncMock(spec=MLServerClient)
    search = AsyncMock(spec=PatentSearchClient)
    classifier = AsyncMock(spec=PatentClassifierClient)
    search.search.return_value = []
    result = asyncio.run(PatentSearchService(encoder, search, classifier).search("query", 5))
    assert result.patents == []
    encoder.encode_batch.assert_not_awaited()
    classifier.classify.assert_not_awaited()


def test_encoder_batches_candidate_texts() -> None:
    sizes = []

    def respond(request: httpx.Request) -> httpx.Response:
        payload = json.loads(request.content)
        texts = payload["inputs"][0]["data"]
        sizes.append(len(texts))
        return httpx.Response(
            200,
            json={"outputs": [{"shape": [len(texts), 1024], "data": [0.1] * (1024 * len(texts))}]},
        )

    async def run() -> None:
        async with httpx.AsyncClient(
            base_url="http://ml:8080", transport=httpx.MockTransport(respond)
        ) as http:
            encoder = MLServerClient(http, "encoder", batch_size=2)
            assert await encoder.encode_batch([]) == []
            vectors = await encoder.encode_batch(["one", "two", "three"])
        assert len(vectors) == 3
        assert all(len(vector) == 1024 for vector in vectors)

    asyncio.run(run())
    assert sizes == [2, 1]


def test_classifier_request_and_predictions() -> None:
    def respond(request: httpx.Request) -> httpx.Response:
        assert request.url.path == "/v2/models/patent-classifier/infer"
        payload = json.loads(request.content)["inputs"][0]
        assert payload["name"] == "embeddings"
        assert payload["datatype"] == "FP32"
        assert payload["shape"] == [2, 2048]
        assert payload["data"] == [0.1] * 1024 + [0.2] * 1024 + [0.1] * 1024 + [0.3] * 1024
        return httpx.Response(
            200,
            json={
                "outputs": [
                    {"name": "label", "shape": [2, 1], "data": ["relevant", "not_relevant"]},
                    {"name": "p_relevant", "shape": [2, 1], "data": [0.7, 0.1]},
                ]
            },
        )

    async def run() -> None:
        async with httpx.AsyncClient(
            base_url="http://ml:8080", transport=httpx.MockTransport(respond)
        ) as http:
            classifier = PatentClassifierClient(http, "patent-classifier")
            assert await classifier.classify([0.1] * 1024, []) == []
            result = await classifier.classify([0.1] * 1024, [[0.2] * 1024, [0.3] * 1024])
        assert result == [
            PatentClassification(label="relevant", score=0.7),
            PatentClassification(label="not_relevant", score=0.1),
        ]

    asyncio.run(run())


@pytest.mark.parametrize(
    ("label", "score", "shape"),
    [
        ("unknown", 0.5, [1, 1]),
        ("relevant", 1.5, [1, 1]),
        ("relevant", 0.5, [2, 1]),
    ],
)
def test_classifier_rejects_invalid_response(label: str, score: float, shape: list[int]) -> None:
    async def run() -> None:
        async with httpx.AsyncClient(
            base_url="http://ml:8080",
            transport=httpx.MockTransport(
                lambda request: httpx.Response(
                    200,
                    json={
                        "outputs": [
                            {"name": "label", "shape": shape, "data": [label]},
                            {"name": "p_relevant", "shape": shape, "data": [score]},
                        ]
                    },
                )
            ),
        ) as http:
            with pytest.raises(UpstreamError) as caught:
                await PatentClassifierClient(http, "classifier").classify(
                    [0.1] * 1024, [[0.2] * 1024]
                )
        assert caught.value.status_code == 502

    asyncio.run(run())


def test_clients_are_shared_and_closed() -> None:
    app = create_app(settings=Settings(_env_file=None))
    with TestClient(app=app) as client:
        http = app.state.encoder_client.http
        encoder = app.state.encoder_client
        classifier = app.state.classifier_client
        assert classifier.http is http
        assert not http.is_closed
        assert client.get("/healthz").status_code == 200
        assert app.state.encoder_client is encoder
        assert app.state.classifier_client is classifier
    assert http.is_closed
