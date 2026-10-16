from pydantic import Field, SecretStr
from pydantic_settings import BaseSettings, SettingsConfigDict


class Settings(BaseSettings):
    model_config = SettingsConfigDict(env_file=".env", extra="ignore")

    app_version: str = "0.1.1"
    app_commit_sha: str = "local"
    database_url: str = "postgresql://postgres:postgres@localhost:5432/ai_patent_search"
    log_level: str = "INFO"

    mlserver_url: str = "http://localhost:8080"
    encoder_model_name: str = "patent-bi-encoder"
    mlserver_timeout: float = Field(default=60.0, gt=0)

    qdrant_url: str = "http://localhost"
    qdrant_port: int = Field(default=6333, ge=1, le=65535)
    qdrant_api_key: SecretStr = SecretStr("")
    qdrant_collection: SecretStr = SecretStr("")
    qdrant_timeout: int = Field(default=300, gt=0)
