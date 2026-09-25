from typing import Literal

from pydantic import BaseModel

ComponentStatus = Literal["ok", "error"]
HealthStatus = Literal["ok", "degraded"]


class DependencyHealth(BaseModel):
    status: ComponentStatus
    response_time_ms: float
    version: str | None = None
    error: str | None = None


class HealthResponse(BaseModel):
    status: HealthStatus
    dependencies: dict[str, DependencyHealth]
