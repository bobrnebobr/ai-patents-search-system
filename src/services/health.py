from time import perf_counter

import asyncpg

from src.schemas import DependencyHealth


async def check_postgres(database_url: str) -> DependencyHealth:
    started_at = perf_counter()
    connection: asyncpg.Connection | None = None

    try:
        connection = await asyncpg.connect(database_url, timeout=2)
        version = await connection.fetchval("SHOW server_version")
    except (asyncpg.PostgresError, OSError, TimeoutError) as error:
        response_time_ms = round((perf_counter() - started_at) * 1000, 2)

        return DependencyHealth(
            status="error",
            response_time_ms=response_time_ms,
            error=type(error).__name__,
        )
    finally:
        if connection is not None:
            await connection.close()

    response_time_ms = round((perf_counter() - started_at) * 1000, 2)

    return DependencyHealth(
        status="ok",
        response_time_ms=response_time_ms,
        version=str(version),
    )