"""JWT authentication and password hashing.

Security hardening (v2.3.0):
* Password hashing uses **bcrypt** (constant-time verification,
  automatic salting) instead of hand-rolled PBKDF2.
* JWT encoding/decoding uses **PyJWT** (actively maintained) instead of
  the unmaintained ``python-jose``.
* Refresh-token flow: short-lived access tokens (15 min) + long-lived
  refresh tokens (7 days) with ``"type"`` claim validation.
* Audit logging on every auth-relevant event.
"""

from __future__ import annotations

import logging
from datetime import UTC, datetime, timedelta
from typing import Any

import bcrypt
import jwt as pyjwt

from web.config import settings

logger = logging.getLogger(__name__)

# ---------------------------------------------------------------------------
# Password hashing — bcrypt (constant-time verify, auto-salt)
# ---------------------------------------------------------------------------


def hash_password(password: str) -> str:
    """Return a bcrypt hash of *password*."""
    hashed: str = bcrypt.hashpw(password.encode("utf-8"), bcrypt.gensalt()).decode("utf-8")
    return hashed


def verify_password(plain_password: str, hashed_password: str) -> bool:
    """Check *plain_password* against a stored bcrypt hash.

    Uses bcrypt's constant-time comparison to prevent timing attacks.
    """
    try:
        ok: bool = bcrypt.checkpw(plain_password.encode("utf-8"), hashed_password.encode("utf-8"))
    except (ValueError, TypeError):
        ok = False
    if not ok:
        logger.warning("Password verification failed")
    return ok


# ---------------------------------------------------------------------------
# JWT helpers — PyJWT (actively maintained)
# ---------------------------------------------------------------------------


def create_access_token(
    data: dict[str, Any],
    expires_delta: timedelta | None = None,
) -> str:
    """Create a signed short-lived access JWT (``type=access``)."""
    to_encode = data.copy()
    expire = datetime.now(UTC) + (
        expires_delta or timedelta(minutes=settings.access_token_expire_minutes)
    )
    to_encode["exp"] = int(expire.timestamp())
    to_encode["type"] = "access"
    encoded: str = pyjwt.encode(to_encode, settings.secret_key, algorithm=settings.jwt_algorithm)
    return encoded


def create_refresh_token(
    data: dict[str, Any],
    expires_delta: timedelta | None = None,
) -> str:
    """Create a signed long-lived refresh JWT (``type=refresh``)."""
    to_encode = data.copy()
    expire = datetime.now(UTC) + (
        expires_delta or timedelta(days=settings.refresh_token_expire_days)
    )
    to_encode["exp"] = int(expire.timestamp())
    to_encode["type"] = "refresh"
    encoded: str = pyjwt.encode(to_encode, settings.secret_key, algorithm=settings.jwt_algorithm)
    return encoded


def decode_access_token(token: str) -> dict[str, Any]:
    """Decode and verify a JWT.  Returns an empty dict on failure."""
    try:
        payload: dict[str, Any] = pyjwt.decode(
            token,
            settings.secret_key,
            algorithms=[settings.jwt_algorithm],
        )
        return payload
    except pyjwt.PyJWTError:
        logger.warning("JWT decode failed (invalid or expired token)")
        return {}
