from litestar import get

from src.config import get_settings


@get("/api/v1/version")
async def version() -> dict[str, str]:
    settings = get_settings()

    return {
        "version": settings.app_version,
        "commit_sha": settings.app_commit_sha,
    }
