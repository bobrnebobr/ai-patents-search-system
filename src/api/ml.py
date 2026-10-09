from typing import ClassVar

from litestar import Controller, post
from litestar.di import NamedDependency
from litestar.exceptions import HTTPException
from litestar.status_codes import HTTP_200_OK

from src.clients.errors import UpstreamError
from src.schemas import PatentSearchRequest, PatentSearchResponse
from src.services.ml import PatentSearchService


class MLController(Controller):
    path = "/api/v1/ml"
    tags: ClassVar[list[str]] = ["ML"]

    @post(
        "/patent-search/process",
        status_code=HTTP_200_OK,
        summary="Search patents by text",
        description="Encode a text query with MLServer and retrieve patents from Qdrant.",
    )
    async def search_patents(
        self,
        data: PatentSearchRequest,
        patent_search_service: NamedDependency[PatentSearchService],
    ) -> PatentSearchResponse:
        try:
            return await patent_search_service.search(data.text, data.limit)
        except UpstreamError as exc:
            raise HTTPException(status_code=exc.status_code, detail=str(exc)) from exc
