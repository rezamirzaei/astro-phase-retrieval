"""Authentication endpoints — register, login, refresh, current-user.

Security features (v2.3.0):
* Audit logging on every auth-relevant event.
* In-memory per-IP login rate limiting (5 attempts / 60 s).
* Refresh-token endpoint for silent access-token renewal.
"""

from __future__ import annotations

import logging
import time
from collections import OrderedDict

from fastapi import APIRouter, HTTPException, Request, status
from sqlalchemy import select

from web.dependencies import CurrentUser, DbSession
from web.models import User
from web.schemas import ChangePasswordRequest, LoginRequest, RefreshRequest, Token, UserCreate, UserResponse
from web.security import (
    create_access_token,
    create_refresh_token,
    decode_access_token,
    hash_password,
    verify_password,
)

router = APIRouter(prefix="/api/auth", tags=["auth"])
logger = logging.getLogger(__name__)

# ---------------------------------------------------------------------------
# Lightweight in-memory login rate limiter (no extra dependency)
# Uses OrderedDict for LRU eviction to prevent unbounded memory growth.
# ---------------------------------------------------------------------------
_LOGIN_WINDOW = 60  # seconds
_LOGIN_MAX_ATTEMPTS = 5
_MAX_TRACKED_IPS = 10_000  # hard cap — evict oldest on overflow

# ip -> list of timestamps (OrderedDict for LRU eviction)
_login_attempts: OrderedDict[str, list[float]] = OrderedDict()


def _check_rate_limit(ip: str) -> None:
    """Raise 429 if *ip* has exceeded the login attempt threshold."""
    now = time.monotonic()
    window = [t for t in _login_attempts.get(ip, []) if now - t < _LOGIN_WINDOW]
    _login_attempts[ip] = window
    # Move to end (most recently used)
    _login_attempts.move_to_end(ip)
    if len(window) >= _LOGIN_MAX_ATTEMPTS:
        logger.warning("Login rate limit exceeded for IP %s", ip)
        raise HTTPException(
            status_code=status.HTTP_429_TOO_MANY_REQUESTS,
            detail="Too many login attempts. Please try again later.",
        )


def _record_attempt(ip: str) -> None:
    if ip not in _login_attempts:
        _login_attempts[ip] = []
    _login_attempts[ip].append(time.monotonic())
    _login_attempts.move_to_end(ip)
    # Evict oldest entries if we exceed the cap
    while len(_login_attempts) > _MAX_TRACKED_IPS:
        _login_attempts.popitem(last=False)


# ---------------------------------------------------------------------------
# Endpoints
# ---------------------------------------------------------------------------


@router.post("/register", response_model=UserResponse, status_code=status.HTTP_201_CREATED)
def register(body: UserCreate, db: DbSession) -> User:
    """Create a new user account."""
    existing = db.execute(
        select(User).where((User.email == body.email) | (User.username == body.username))
    ).scalar_one_or_none()
    if existing is not None:
        raise HTTPException(
            status_code=status.HTTP_409_CONFLICT,
            detail="Email or username already registered",
        )
    user = User(
        email=body.email,
        username=body.username,
        hashed_password=hash_password(body.password),
    )
    db.add(user)
    db.commit()
    db.refresh(user)
    logger.info("New user registered: %s (id=%s)", user.username, user.id)
    return user


@router.post("/login", response_model=Token)
def login(body: LoginRequest, db: DbSession, request: Request) -> dict[str, str]:
    """Authenticate and return an access + refresh token pair."""
    client_ip = request.client.host if request.client else "unknown"
    _check_rate_limit(client_ip)
    _record_attempt(client_ip)

    user = db.execute(select(User).where(User.username == body.username)).scalar_one_or_none()
    if user is None or not verify_password(body.password, user.hashed_password):
        logger.warning("Failed login attempt for username=%s from IP=%s", body.username, client_ip)
        raise HTTPException(
            status_code=status.HTTP_401_UNAUTHORIZED,
            detail="Invalid credentials",
        )
    claims = {"sub": str(user.id), "username": user.username, "tv": user.token_version}
    access = create_access_token(claims)
    refresh = create_refresh_token(claims)
    logger.info("User logged in: %s (id=%s)", user.username, user.id)
    return {"access_token": access, "refresh_token": refresh, "token_type": "bearer"}


@router.post("/refresh", response_model=Token)
def refresh(body: RefreshRequest, db: DbSession) -> dict[str, str]:
    """Exchange a valid refresh token for a new access + refresh pair."""
    payload = decode_access_token(body.refresh_token)
    if payload.get("type") != "refresh":
        raise HTTPException(
            status_code=status.HTTP_401_UNAUTHORIZED,
            detail="Invalid refresh token",
        )
    sub = payload.get("sub")
    if sub is None:
        raise HTTPException(
            status_code=status.HTTP_401_UNAUTHORIZED,
            detail="Invalid refresh token",
        )
    user = db.get(User, int(str(sub)))
    if user is None or not user.is_active:
        raise HTTPException(
            status_code=status.HTTP_401_UNAUTHORIZED,
            detail="User not found or inactive",
        )
    # Verify token_version — rejects tokens issued before a password change
    if payload.get("tv", 0) != user.token_version:
        raise HTTPException(
            status_code=status.HTTP_401_UNAUTHORIZED,
            detail="Token has been revoked (password changed)",
        )
    claims = {"sub": str(user.id), "username": user.username, "tv": user.token_version}
    access = create_access_token(claims)
    new_refresh = create_refresh_token(claims)
    logger.info("Token refreshed for user %s (id=%s)", user.username, user.id)
    return {"access_token": access, "refresh_token": new_refresh, "token_type": "bearer"}


@router.get("/me", response_model=UserResponse)
def me(current_user: CurrentUser) -> User:
    """Return the currently authenticated user."""
    return current_user


@router.post("/change-password", response_model=Token)
def change_password(
    body: ChangePasswordRequest,
    current_user: CurrentUser,
    db: DbSession,
) -> dict[str, str]:
    """Change password and revoke all existing tokens.

    Verifies the current password, updates the hash, and increments
    ``token_version`` so that all previously issued JWTs become invalid.
    Returns a fresh access + refresh token pair.
    """
    if not verify_password(body.current_password, current_user.hashed_password):
        raise HTTPException(
            status_code=status.HTTP_401_UNAUTHORIZED,
            detail="Current password is incorrect",
        )
    current_user.hashed_password = hash_password(body.new_password)
    current_user.token_version += 1
    db.commit()
    db.refresh(current_user)

    claims = {"sub": str(current_user.id), "username": current_user.username, "tv": current_user.token_version}
    access = create_access_token(claims)
    refresh_tok = create_refresh_token(claims)
    logger.info("Password changed for user %s (id=%s), all tokens revoked", current_user.username, current_user.id)
    return {"access_token": access, "refresh_token": refresh_tok, "token_type": "bearer"}

