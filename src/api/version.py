from litestar import get
from litestar.di import NamedDependency

from src.config import Settings


@get("/api/v1/version")
async def version(settings: NamedDependency[Settings]) -> dict[str, str]:
    return {
        "version": settings.app_version,
        "commit_sha": settings.app_commit_sha,
    }
