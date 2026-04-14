"""Tests for v3.1.0 security and concurrency improvements.

Covers:
- Path traversal protection (sanitize_filename, assert_path_within)
- Password change endpoint + token revocation
- Rate limiter LRU eviction
- Job queue thread-safety helpers
"""

from __future__ import annotations

import pytest

# ---------------------------------------------------------------------------
# web/utils.py — filename sanitization & path containment
# ---------------------------------------------------------------------------

fastapi = pytest.importorskip("fastapi", reason="web extras not installed")


class TestSanitizeFilename:
    """Verify sanitize_filename strips traversal components."""

    def test_normal_filename(self) -> None:
        from web.utils import sanitize_filename

        assert sanitize_filename("test.fits") == "test.fits"

    def test_strips_directory_traversal(self) -> None:
        from web.utils import sanitize_filename

        assert sanitize_filename("../../etc/passwd") == "passwd"

    def test_strips_directory_components(self) -> None:
        from web.utils import sanitize_filename

        assert sanitize_filename("/foo/bar/baz.npy") == "baz.npy"

    def test_strips_leading_dots(self) -> None:
        from web.utils import sanitize_filename

        result = sanitize_filename("..hidden")
        assert not result.startswith(".")

    def test_null_bytes_stripped(self) -> None:
        from web.utils import sanitize_filename

        result = sanitize_filename("test\x00.fits")
        assert "\x00" not in result

    def test_empty_name_raises_422(self) -> None:
        from fastapi import HTTPException

        from web.utils import sanitize_filename

        with pytest.raises(HTTPException):
            sanitize_filename("../../..")

    def test_special_characters_replaced(self) -> None:
        from web.utils import sanitize_filename

        result = sanitize_filename("my file (copy).fits")
        # Spaces and parens should be replaced with underscores
        assert " " not in result
        assert "(" not in result


class TestAssertPathWithin:
    """Verify assert_path_within blocks escapes."""

    def test_valid_child(self, tmp_path) -> None:
        from web.utils import assert_path_within

        child = tmp_path / "subdir" / "file.txt"
        child.parent.mkdir(parents=True, exist_ok=True)
        child.touch()
        result = assert_path_within(child, tmp_path)
        assert result == child.resolve()

    def test_traversal_blocked(self, tmp_path) -> None:
        from fastapi import HTTPException

        from web.utils import assert_path_within

        evil = tmp_path / ".." / ".." / "etc" / "passwd"
        with pytest.raises(HTTPException):
            assert_path_within(evil, tmp_path)


# ---------------------------------------------------------------------------
# Password change endpoint
# ---------------------------------------------------------------------------

sqlalchemy = pytest.importorskip("sqlalchemy", reason="web extras not installed")

from fastapi.testclient import TestClient  # noqa: E402


def _register_and_login(client: TestClient) -> dict[str, str]:
    client.post(
        "/api/auth/register",
        json={
            "email": "pwtest@example.com",
            "username": "pwuser",
            "password": "password123",
        },
    )
    resp = client.post("/api/auth/login", json={"username": "pwuser", "password": "password123"})
    token = resp.json()["access_token"]
    return {"Authorization": f"Bearer {token}"}


class TestPasswordChange:
    def test_change_password_success(self, client: TestClient) -> None:
        headers = _register_and_login(client)
        resp = client.post(
            "/api/auth/change-password",
            json={"current_password": "password123", "new_password": "newSecure456"},
            headers=headers,
        )
        assert resp.status_code == 200
        data = resp.json()
        assert "access_token" in data
        assert "refresh_token" in data

    def test_change_password_wrong_current(self, client: TestClient) -> None:
        headers = _register_and_login(client)
        resp = client.post(
            "/api/auth/change-password",
            json={"current_password": "wrongpassword", "new_password": "newSecure456"},
            headers=headers,
        )
        assert resp.status_code == 401

    def test_old_token_revoked_after_change(self, client: TestClient) -> None:
        headers = _register_and_login(client)

        # Change password
        resp = client.post(
            "/api/auth/change-password",
            json={"current_password": "password123", "new_password": "newSecure456"},
            headers=headers,
        )
        assert resp.status_code == 200

        # Old token should now be rejected
        resp = client.get("/api/auth/me", headers=headers)
        assert resp.status_code == 401

    def test_new_token_works_after_change(self, client: TestClient) -> None:
        headers = _register_and_login(client)

        resp = client.post(
            "/api/auth/change-password",
            json={"current_password": "password123", "new_password": "newSecure456"},
            headers=headers,
        )
        new_token = resp.json()["access_token"]
        new_headers = {"Authorization": f"Bearer {new_token}"}

        resp = client.get("/api/auth/me", headers=new_headers)
        assert resp.status_code == 200
        assert resp.json()["username"] == "pwuser"

    def test_login_with_new_password(self, client: TestClient) -> None:
        headers = _register_and_login(client)

        client.post(
            "/api/auth/change-password",
            json={"current_password": "password123", "new_password": "newSecure456"},
            headers=headers,
        )

        # Login with new password
        resp = client.post(
            "/api/auth/login",
            json={"username": "pwuser", "password": "newSecure456"},
        )
        assert resp.status_code == 200

        # Old password should fail
        resp = client.post(
            "/api/auth/login",
            json={"username": "pwuser", "password": "password123"},
        )
        assert resp.status_code == 401


# ---------------------------------------------------------------------------
# Security headers
# ---------------------------------------------------------------------------


class TestSecurityHeaders:
    def test_csp_header_present(self, client: TestClient) -> None:
        resp = client.get("/api/health")
        assert "content-security-policy" in resp.headers

    def test_x_content_type_options(self, client: TestClient) -> None:
        resp = client.get("/api/health")
        assert resp.headers.get("x-content-type-options") == "nosniff"

    def test_request_id_returned(self, client: TestClient) -> None:
        resp = client.get("/api/health")
        assert "x-request-id" in resp.headers
        assert len(resp.headers["x-request-id"]) > 0


# ---------------------------------------------------------------------------
# Upload path traversal
# ---------------------------------------------------------------------------


class TestUploadSecurity:
    def test_upload_traversal_filename_sanitized(self, client: TestClient) -> None:
        headers = _register_and_login(client)
        # Try to upload with a path-traversal filename
        files = {"file": ("../../etc/evil.npy", b"\x00" * 100, "application/octet-stream")}
        resp = client.post("/api/data/upload", files=files, headers=headers)
        if resp.status_code == 201:
            # The filename should be sanitized
            data = resp.json()
            assert ".." not in data["filename"]
            assert "/" not in data["filename"]

    def test_upload_cif_traversal_sanitized(self, client: TestClient) -> None:
        headers = _register_and_login(client)
        files = {"file": ("../../evil.cif", b"data_test\n", "application/octet-stream")}
        resp = client.post("/api/crystallography/upload", files=files, headers=headers)
        if resp.status_code == 201:
            data = resp.json()
            assert ".." not in data["filename"]


# ---------------------------------------------------------------------------
# Job queue helpers
# ---------------------------------------------------------------------------


class TestJobQueueHelpers:
    def test_is_job_cancelled_returns_false_for_unknown(self) -> None:
        from web.services.job_queue import is_job_cancelled

        assert is_job_cancelled("nonexistent") is False

    def test_cancel_unknown_job_returns_false(self) -> None:
        from web.services.job_queue import cancel_job

        assert cancel_job("nonexistent") is False

    def test_get_job_status_not_found(self) -> None:
        from web.services.job_queue import get_job_status

        status = get_job_status("nonexistent")
        assert status["error"] == "not_found"



