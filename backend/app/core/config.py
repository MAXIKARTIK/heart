"""Application settings, loaded from environment / .env via pydantic-settings."""
from __future__ import annotations

from functools import lru_cache

from pydantic_settings import BaseSettings, SettingsConfigDict


class Settings(BaseSettings):
    model_config = SettingsConfigDict(
        env_file=".env", env_file_encoding="utf-8", extra="ignore"
    )

    app_name: str = "Heart Disease Risk API"
    environment: str = "development"
    api_v1_prefix: str = "/api/v1"

    # Persistence. SQLite by default so the API runs with zero infra locally.
    # In Docker/compose this is overridden with a PostgreSQL URL.
    database_url: str = "sqlite:///./heart.db"

    # Path to the serialized model bundle. If empty, heart_ml.config.MODEL_PATH
    # (the training artifact) is used.
    model_path: str = ""

    # Async task queue. If no broker is configured, batch requests are served
    # synchronously (see routers/batch.py).
    redis_url: str = ""
    celery_broker_url: str = ""
    celery_result_backend: str = ""

    # Optional API-key auth. When set, /predict and /batch require the
    # `X-API-Key` header. Left empty in local dev for convenience.
    api_key: str = ""

    # CORS: comma-separated list of allowed origins, or "*".
    cors_origins: str = "*"

    @property
    def celery_enabled(self) -> bool:
        return bool(self.celery_broker_url or self.redis_url)

    @property
    def effective_broker(self) -> str:
        return self.celery_broker_url or self.redis_url

    @property
    def effective_backend(self) -> str:
        return self.celery_result_backend or self.redis_url

    @property
    def cors_origin_list(self) -> list[str]:
        if self.cors_origins.strip() == "*":
            return ["*"]
        return [o.strip() for o in self.cors_origins.split(",") if o.strip()]


@lru_cache
def get_settings() -> Settings:
    return Settings()


settings = get_settings()
