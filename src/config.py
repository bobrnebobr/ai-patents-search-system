from pydantic_settings import BaseSettings, SettingsConfigDict


class Settings(BaseSettings):
    model_config = SettingsConfigDict(env_file=".env", extra="ignore")

    app_version: str = "0.1.1"
    app_commit_sha: str = "local"
    database_url: str = "postgresql://postgres:postgres@localhost:5432/ai_patent_search"
    log_level: str = "INFO"
