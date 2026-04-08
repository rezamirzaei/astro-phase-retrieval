"""Application settings — loaded from environment or ``.env`` file."""

from __future__ import annotations

from pathlib import Path

from pydantic_settings import BaseSettings, SettingsConfigDict


class Settings(BaseSettings):
    """All knobs for the web backend, validated at startup."""

    model_config = SettingsConfigDict(env_prefix="PR_", env_file=".env", extra="ignore")

    # Database (SQLite default; PostgreSQL in Docker)
    database_url: str = "sqlite:///./web/phase_retrieval.db"

    # JWT
    secret_key: str = "dev-only-change-me-in-production"
    jwt_algorithm: str = "HS256"
    access_token_expire_minutes: int = 60 * 24  # 24 h

    # Paths
    data_dir: Path = Path("data")
    output_dir: Path = Path("web_outputs")

    # CORS
    cors_origins: list[str] = [
        "http://localhost:4532",
        "http://localhost:4533",
        "http://localhost",
    ]


settings = Settings()
