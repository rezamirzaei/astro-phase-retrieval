"""Application settings — loaded from environment or ``.env`` file.

All sensitive values **must** be supplied via environment variables in
production.  Set ``PR_SECRET_KEY`` and ``PR_ADMIN_PASSWORD`` before starting
the server.  The defaults here are intentionally weak and will trigger a
``ValueError`` when the application starts in non-development mode.

Example ``.env``::

    PR_SECRET_KEY=<64-hex-char random string>
    PR_ADMIN_PASSWORD=<strong password>
    PR_DATABASE_URL=postgresql+psycopg2://user:pass@db:5432/phase_retrieval
"""

from __future__ import annotations

import os
from pathlib import Path

from pydantic import Field, field_validator
from pydantic_settings import BaseSettings, SettingsConfigDict


class Settings(BaseSettings):
    """All knobs for the web backend, validated at startup."""

    model_config = SettingsConfigDict(env_prefix="PR_", env_file=".env", extra="ignore")

    # Database (SQLite default; PostgreSQL in production via PR_DATABASE_URL)
    database_url: str = "sqlite:///./web/phase_retrieval.db"

    # JWT — generate a strong key with: python -c "import secrets; print(secrets.token_hex(32))"
    secret_key: str = Field(
        default="dev-only-change-me-in-production",
        min_length=16,
        description="JWT signing secret.  Set PR_SECRET_KEY in production.",
    )
    jwt_algorithm: str = "HS256"
    access_token_expire_minutes: int = 15  # short-lived access tokens
    refresh_token_expire_days: int = 7  # long-lived refresh tokens

    # Admin seed account (created on first startup if the users table is empty)
    admin_password: str = Field(
        default="admin123",
        min_length=8,
        description="Seed admin password.  Set PR_ADMIN_PASSWORD in production.",
    )

    # Paths
    data_dir: Path = Path("data")
    output_dir: Path = Path("web_outputs")

    # CORS
    cors_origins: list[str] = [
        "http://localhost:4532",
        "http://localhost:4533",
        "http://localhost",
    ]

    # Rate-limiting: max concurrent heavy algorithm runs (0 = unlimited)
    max_concurrent_jobs: int = Field(default=4, ge=0)

    # Graceful shutdown: max seconds to wait for running jobs to complete
    shutdown_timeout_seconds: float = Field(default=30.0, ge=0)

    # Upload limits
    upload_max_bytes: int = Field(default=100 * 1024 * 1024, ge=0)

    @field_validator("secret_key")
    @classmethod
    def _warn_insecure_key(cls, v: str) -> str:
        if v == "dev-only-change-me-in-production" and os.getenv("PR_ENV", "dev") == "prod":
            raise ValueError(
                "PR_SECRET_KEY must be set to a strong random value in production. "
                'Generate one with: python -c "import secrets; print(secrets.token_hex(32))"'
            )
        return v

    @field_validator("admin_password")
    @classmethod
    def _warn_insecure_admin(cls, v: str) -> str:
        if v == "admin123" and os.getenv("PR_ENV", "dev") == "prod":
            raise ValueError("PR_ADMIN_PASSWORD must be set to a strong value in production.")
        return v


settings = Settings()
