from functools import lru_cache

from pydantic_settings import BaseSettings, SettingsConfigDict


class Settings(BaseSettings):
    model_config = SettingsConfigDict(env_file=".env", extra="ignore")

    app_version: str = "0.1.0"
    app_commit_sha: str = "local"


@lru_cache
def get_settings() -> Settings:
    return Settings()