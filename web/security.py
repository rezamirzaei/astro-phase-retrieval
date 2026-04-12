"""JWT authentication and password hashing.

Uses ``bcrypt`` for industry-standard password hashing with
automatic salt generation and configurable work factor.
"""

from __future__ import annotations

from datetime import UTC, datetime, timedelta
from typing import Any

import bcrypt
from jose import JWTError, jwt

from web.config import settings


def hash_password(password: str) -> str:
    """Return a bcrypt hash of *password*."""
    pwd_bytes = password.encode("utf-8")
    hashed = bcrypt.hashpw(pwd_bytes, bcrypt.gensalt(rounds=12))
    return hashed.decode("utf-8")


def verify_password(plain_password: str, hashed_password: str) -> bool:
    """Check *plain_password* against a stored hash.

    Supports both bcrypt (new) and legacy PBKDF2 hashes for migration.
    """
    pwd_bytes = plain_password.encode("utf-8")

    # Try bcrypt first (new format — starts with $2b$)
    if hashed_password.startswith("$2"):
        try:
            return bcrypt.checkpw(pwd_bytes, hashed_password.encode("utf-8"))
        except (ValueError, TypeError):
            return False

    # Fall back to legacy PBKDF2-SHA256 for pre-migration hashes
    try:
        import hashlib

        salt_hex, key_hex = hashed_password.split(":", 1)
        salt = bytes.fromhex(salt_hex)
        key = hashlib.pbkdf2_hmac("sha256", pwd_bytes, salt, 260_000)
        if key.hex() == key_hex:
            return True
    except (ValueError, AttributeError):
        pass

    return False


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
