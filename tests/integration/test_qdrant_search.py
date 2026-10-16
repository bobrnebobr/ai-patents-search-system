import asyncio
from collections.abc import Iterator
from unittest.mock import AsyncMock
from uuid import uuid4

import pytest
from qdrant_client import models
from testcontainers.community.qdrant import QdrantContainer

from src.clients.errors import UpstreamError
from src.clients.mlserver import MLServerClient, PatentClassifierClient
from src.clients.qdrant import PatentSearchClient
from src.schemas import PatentClassification
from src.services.ml import PatentSearchService

pytestmark = pytest.mark.integration


def vector(first: float, second: float = 0.0) -> list[float]:
    return [first, second, *([0.0] * 1022)]


@pytest.fixture(scope="module")
def qdrant() -> Iterator[QdrantContainer]:
    with QdrantContainer("qdrant/qdrant:v1.19.2", api_key="integration-test-key") as container:
        yield container


@pytest.fixture
def collection(qdrant: QdrantContainer) -> Iterator[str]:
    name = f"test_patents_{uuid4().hex}"
    client = qdrant.get_client(check_compatibility=False, timeout=10)
    try:
        client.create_collection(
            collection_name=name,
            vectors_config=models.VectorParams(size=1024, distance=models.Distance.COSINE),
        )
        yield name
    finally:
        client.delete_collection(name)
        client.close()


def seed_patents(qdrant: QdrantContainer, collection: str) -> None:
    client = qdrant.get_client(check_compatibility=False, timeout=10)
    try:
        client.upsert(
            collection_name=collection,
            wait=True,
            points=[
                models.PointStruct(
                    id=1,
                    vector=vector(1.0),
                    payload={
                        "publication_id": "EP-BATTERY",
                        "title": "Battery recycling",
                        "abstract": "Recycling lithium-ion batteries",
                        "cpc_codes": ["H01M10/54"],
                        "ipc_codes": ["H01M10/54"],
                    },
                ),
                models.PointStruct(
                    id=2,
                    vector=vector(0.8, 0.6),
                    payload={
                        "publication_id": "EP-RELATED",
                        "title": "Battery materials",
                        "abstract": "Recovery of electrode materials",
                    },
                ),
                models.PointStruct(
                    id=3,
                    vector=vector(0.0, 1.0),
                    payload={
                        "publication_id": "EP-OPTICS",
                        "title": "Optical sensor",
                        "abstract": "An optical navigation sensor",
                    },
                ),
            ],
        )
    finally:
        client.close()


def test_search_ranks_patents_and_returns_payload(
    qdrant: QdrantContainer,
    collection: str,
) -> None:
    seed_patents(qdrant, collection)

    async def run() -> None:
        sdk = qdrant.get_async_client(check_compatibility=False, timeout=10)
        try:
            patents = await PatentSearchClient(sdk, collection, timeout=5).search(vector(1.0), 3)
        finally:
            await sdk.close()
        assert [patent.publication_id for patent in patents] == [
            "EP-BATTERY",
            "EP-RELATED",
            "EP-OPTICS",
        ]
        assert [patent.score for patent in patents] == pytest.approx([1.0, 0.8, 0.0])
        assert patents[0].title == "Battery recycling"
        assert patents[0].abstract == "Recycling lithium-ion batteries"
        assert patents[0].cpc_codes == ["H01M10/54"]
        assert patents[0].ipc_codes == ["H01M10/54"]
        assert patents[1].cpc_codes == []
        assert patents[1].ipc_codes == []

    asyncio.run(run())


def test_search_respects_limit(qdrant: QdrantContainer, collection: str) -> None:
    seed_patents(qdrant, collection)

    async def run() -> None:
        sdk = qdrant.get_async_client(check_compatibility=False, timeout=10)
        try:
            patents = await PatentSearchClient(sdk, collection, timeout=5).search(vector(1.0), 1)
        finally:
            await sdk.close()
        assert len(patents) == 1
        assert patents[0].publication_id == "EP-BATTERY"

    asyncio.run(run())


def test_empty_collection_returns_empty_results(
    qdrant: QdrantContainer,
    collection: str,
) -> None:
    async def run() -> None:
        sdk = qdrant.get_async_client(check_compatibility=False, timeout=10)
        try:
            assert await PatentSearchClient(sdk, collection, timeout=5).search(vector(1.0), 5) == []
        finally:
            await sdk.close()

    asyncio.run(run())


def test_missing_collection_returns_safe_error(qdrant: QdrantContainer) -> None:
    async def run() -> None:
        sdk = qdrant.get_async_client(check_compatibility=False, timeout=10)
        try:
            with pytest.raises(UpstreamError) as caught:
                await PatentSearchClient(sdk, "missing_collection", timeout=5).search(
                    vector(1.0), 1
                )
        finally:
            await sdk.close()
        assert caught.value.status_code == 503
        assert str(caught.value) == "Qdrant search failed"

    asyncio.run(run())


def test_invalid_payload_returns_safe_error(
    qdrant: QdrantContainer,
    collection: str,
) -> None:
    client = qdrant.get_client(check_compatibility=False, timeout=10)
    try:
        client.upsert(
            collection, [models.PointStruct(id=1, vector=vector(1.0), payload={})], wait=True
        )
    finally:
        client.close()

    async def run() -> None:
        sdk = qdrant.get_async_client(check_compatibility=False, timeout=10)
        try:
            with pytest.raises(UpstreamError) as caught:
                await PatentSearchClient(sdk, collection, timeout=5).search(vector(1.0), 1)
        finally:
            await sdk.close()
        assert caught.value.status_code == 502

    asyncio.run(run())


def test_search_service_with_real_qdrant(
    qdrant: QdrantContainer,
    collection: str,
) -> None:
    seed_patents(qdrant, collection)
    encoder = AsyncMock(spec=MLServerClient)
    encoder.encode.return_value = vector(1.0)
    encoder.encode_batch.return_value = [vector(1.0), vector(0.8, 0.6)]
    classifier = AsyncMock(spec=PatentClassifierClient)
    classifier.classify.return_value = [
        PatentClassification(label="relevant", score=0.9),
        PatentClassification(label="not_relevant", score=0.1),
    ]

    async def run() -> None:
        sdk = qdrant.get_async_client(check_compatibility=False, timeout=10)
        try:
            service = PatentSearchService(
                encoder, PatentSearchClient(sdk, collection, timeout=5), classifier
            )
            response = await service.search("Battery recycling", 2)
        finally:
            await sdk.close()
        assert len(response.patents) == 2
        assert response.patents[0].publication_id == "EP-BATTERY"
        assert response.patents[0].relevance_label == "relevant"
        assert response.patents[1].relevance_label == "not_relevant"
        encoder.encode.assert_awaited_once_with("Battery recycling")

    asyncio.run(run())
