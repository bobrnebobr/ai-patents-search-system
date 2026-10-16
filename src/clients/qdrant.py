import httpx
from pydantic import ValidationError
from qdrant_client import AsyncQdrantClient
from qdrant_client.http.exceptions import ResponseHandlingException, UnexpectedResponse

from src.clients.errors import UpstreamError
from src.schemas import PatentMatch


class PatentSearchClient:
    def __init__(self, client: AsyncQdrantClient, collection: str, timeout: int = 300) -> None:
        self.client = client
        self.collection = collection
        self.timeout = timeout

    async def search(self, vector: list[float], limit: int) -> list[PatentMatch]:
        if not self.collection:
            raise UpstreamError("Patent search collection is not configured", 503)
        try:
            result = await self.client.query_points(
                collection_name=self.collection,
                query=vector,
                limit=limit,
                with_payload=True,
                with_vectors=False,
                timeout=self.timeout,
            )
        except ResponseHandlingException as exc:
            if isinstance(exc.source, httpx.TimeoutException):
                raise UpstreamError("Qdrant request timed out", 504) from exc
            raise UpstreamError("Qdrant is unavailable", 503) from exc
        except UnexpectedResponse as exc:
            status = 504 if exc.status_code == 504 else 503
            raise UpstreamError("Qdrant search failed", status) from exc

        try:
            return [
                PatentMatch.model_validate({**(point.payload or {}), "score": point.score})
                for point in result.points
            ]
        except ValidationError as exc:
            raise UpstreamError("Qdrant returned an invalid patent payload", 502) from exc
