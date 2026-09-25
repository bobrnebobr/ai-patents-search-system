from litestar import Response, get
from litestar.di import NamedDependency
from litestar.status_codes import HTTP_200_OK, HTTP_503_SERVICE_UNAVAILABLE

from src.config import Settings
from src.schemas import HealthResponse
from src.services.health import check_postgres


@get("/healthz")
async def healthz() -> dict[str, str]:
    return {"status": "ok"}


@get("/api/v1/health")
async def health(
    settings: NamedDependency[Settings],
) -> Response[HealthResponse]:
    postgres_health = await check_postgres(settings.database_url)
    is_healthy = postgres_health.status == "ok"
    response = HealthResponse(
        status="ok" if is_healthy else "degraded",
        dependencies={"postgres": postgres_health},
    )

    return Response(
        content=response,
        status_code=HTTP_200_OK if is_healthy else HTTP_503_SERVICE_UNAVAILABLE,
    )
