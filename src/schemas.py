from typing import Literal

from pydantic import BaseModel, Field

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


class PatentSearchRequest(BaseModel):
    model_config = {"str_strip_whitespace": True}

    text: str = Field(min_length=1, max_length=20000)
    limit: int = Field(default=10, ge=1, le=100)


class PatentMatch(BaseModel):
    publication_id: str
    title: str
    abstract: str
    score: float = Field(allow_inf_nan=False)
    cpc_codes: list[str] = Field(default_factory=list)
    ipc_codes: list[str] = Field(default_factory=list)


class PatentSearchResponse(BaseModel):
    patents: list[PatentMatch]
