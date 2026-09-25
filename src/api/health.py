from litestar import Response, get
from litestar.status_codes import HTTP_200_OK, HTTP_503_SERVICE_UNAVAILABLE

from src.config import get_settings
from src.schemas import HealthResponse
from src.services.health import check_postgres


@get("/healthz")
async def healthz() -> dict[str, str]:
    return {"status": "ok"}


@get("/api/v1/health")
async def health() -> Response[HealthResponse]:
    settings = get_settings()
    postgres = await check_postgres(settings.database_url)

    is_healthy = postgres.status == "ok"
    response = HealthResponse(
        status="ok" if is_healthy else "degraded",
        dependencies={"postgres": postgres},
    )

    return Response(
        content=response,
        status_code=HTTP_200_OK if is_healthy else HTTP_503_SERVICE_UNAVAILABLE,
    )
