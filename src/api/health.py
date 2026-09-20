from litestar import get


@get("/healthz")
async def healthz() -> dict[str, str]:
    return {"status": "ok"}