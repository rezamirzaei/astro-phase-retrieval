"""JWT authentication and password hashing."""

from __future__ import annotations

import hashlib
import os
from datetime import UTC, datetime, timedelta
from typing import Any

from jose import JWTError, jwt

from web.config import settings


def hash_password(password: str) -> str:
    """Return a PBKDF2-SHA256 hash of *password*."""
    salt = os.urandom(32)
    key = hashlib.pbkdf2_hmac("sha256", password.encode(), salt, 260_000)
    return salt.hex() + ":" + key.hex()


def verify_password(plain_password: str, hashed_password: str) -> bool:
    """Check *plain_password* against a PBKDF2-SHA256 hash."""
    try:
        salt_hex, key_hex = hashed_password.split(":", 1)
    except ValueError:
        return False
    salt = bytes.fromhex(salt_hex)
    key = hashlib.pbkdf2_hmac("sha256", plain_password.encode(), salt, 260_000)
    return key.hex() == key_hex


def create_access_token(
    data: dict[str, Any],
    expires_delta: timedelta | None = None,
) -> str:
    """Create a signed JWT."""
    to_encode = data.copy()
    expire = datetime.now(UTC) + (
        expires_delta or timedelta(minutes=settings.access_token_expire_minutes)
    )
    to_encode["exp"] = int(expire.timestamp())
    encoded: str = jwt.encode(to_encode, settings.secret_key, algorithm=settings.jwt_algorithm)
    return encoded


def decode_access_token(token: str) -> dict[str, Any]:
    """Decode and verify a JWT.  Returns an empty dict on failure."""
    try:
        payload: dict[str, Any] = jwt.decode(
            token,
            settings.secret_key,
            algorithms=[settings.jwt_algorithm],
        )
        return payload
    except JWTError:
        return {}
