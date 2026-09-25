from time import perf_counter
from typing import Any
from uuid import uuid4

from litestar import Litestar
from litestar.middleware import DefineMiddleware
from litestar.types import ASGIApp, Receive, Scope, Send
from loguru import logger

from src.api.health import health, healthz
from src.api.version import version
from src.config import Settings
from src.dependencies import create_dependencies
from src.logging_config import setup_logging


def request_logging_middleware(*, app: ASGIApp) -> ASGIApp:
    async def middleware(scope: Scope, receive: Receive, send: Send) -> None:
        if scope["type"] != "http":
            await app(scope, receive, send)
            return

        request_id = uuid4().hex[:8]
        started_at = perf_counter()
        status_code = 500

        async def send_with_request_id(message: dict[str, Any]) -> None:
            nonlocal status_code

            if message["type"] == "http.response.start":
                status_code = message["status"]
                headers = list(message.get("headers", []))
                headers.append((b"x-request-id", request_id.encode()))
                message["headers"] = headers

            await send(message)

        with logger.contextualize(request_id=request_id):
            try:
                await app(scope, receive, send_with_request_id)
            finally:
                duration_ms = (perf_counter() - started_at) * 1000
                logger.info(
                    "{} {} -> {} ({:.1f} ms)",
                    scope["method"],
                    scope["path"],
                    status_code,
                    duration_ms,
                )

    return middleware


def create_app(
    *,
    settings: Settings | None = None,
) -> Litestar:
    app_settings = settings if settings is not None else Settings()
    dependencies = create_dependencies(app_settings)

    def configure_app_logging() -> None:
        setup_logging(app_settings.log_level)

    return Litestar(
        route_handlers=[healthz, health, version],
        dependencies=dependencies,
        middleware=[DefineMiddleware(request_logging_middleware)],
        on_startup=[configure_app_logging],
    )


app = create_app()
