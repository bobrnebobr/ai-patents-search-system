import logging
from time import perf_counter
from typing import Any
from uuid import uuid4

from litestar import Litestar
from litestar.middleware import DefineMiddleware
from litestar.types import ASGIApp, Receive, Scope, Send

from src.api.health import health, healthz
from src.api.version import version
from src.config import get_settings
from src.logging_config import REQUEST_ID, setup_logging

logger = logging.getLogger(__name__)


def request_logging_middleware(*, app: ASGIApp) -> ASGIApp:
    async def middleware(scope: Scope, receive: Receive, send: Send) -> None:
        if scope["type"] != "http":
            await app(scope, receive, send)
            return

        request_id = uuid4().hex[:8]
        request_id_token = REQUEST_ID.set(request_id)
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

        try:
            await app(scope, receive, send_with_request_id)
        finally:
            duration_ms = (perf_counter() - started_at) * 1000
            logger.info(
                "%s %s -> %d (%.1f ms)",
                scope["method"],
                scope["path"],
                status_code,
                duration_ms,
            )
            REQUEST_ID.reset(request_id_token)

    return middleware


def configure_app_logging() -> None:
    settings = get_settings()
    setup_logging(settings.log_level)


app = Litestar(
    route_handlers=[healthz, health, version],
    middleware=[DefineMiddleware(request_logging_middleware)],
    on_startup=[configure_app_logging],
)
