"""End-to-end tests for the web API — auth, data, algorithms, results, explain."""

from __future__ import annotations

import io
import zipfile

import pytest

fastapi = pytest.importorskip("fastapi", reason="web extras not installed")
sqlalchemy = pytest.importorskip("sqlalchemy", reason="web extras not installed")

from fastapi.testclient import TestClient  # noqa: E402


def _register_and_login(client: TestClient) -> dict[str, str]:
    client.post(
        "/api/auth/register",
        json={
            "email": "test@example.com",
            "username": "tester",
            "password": "password123",
        },
    )
    resp = client.post("/api/auth/login", json={"username": "tester", "password": "password123"})
    token = resp.json()["access_token"]
    return {"Authorization": f"Bearer {token}"}


# ---------------------------------------------------------------------------
# Auth tests
# ---------------------------------------------------------------------------


class TestAuth:
    def test_register(self, client: TestClient) -> None:
        resp = client.post(
            "/api/auth/register",
            json={
                "email": "a@b.com",
                "username": "alice",
                "password": "securepass",
            },
        )
        assert resp.status_code == 201
        assert resp.json()["username"] == "alice"

    def test_register_duplicate(self, client: TestClient) -> None:
        client.post(
            "/api/auth/register",
            json={
                "email": "a@b.com",
                "username": "alice",
                "password": "securepass",
            },
        )
        resp = client.post(
            "/api/auth/register",
            json={
                "email": "a@b.com",
                "username": "alice2",
                "password": "securepass",
            },
        )
        assert resp.status_code == 409

    def test_login_success(self, client: TestClient) -> None:
        client.post(
            "/api/auth/register",
            json={
                "email": "a@b.com",
                "username": "alice",
                "password": "securepass",
            },
        )
        resp = client.post("/api/auth/login", json={"username": "alice", "password": "securepass"})
        assert resp.status_code == 200
        body = resp.json()
        assert "access_token" in body
        assert "refresh_token" in body
        assert body["token_type"] == "bearer"

    def test_login_wrong_password(self, client: TestClient) -> None:
        client.post(
            "/api/auth/register",
            json={
                "email": "a@b.com",
                "username": "alice",
                "password": "securepass",
            },
        )
        resp = client.post("/api/auth/login", json={"username": "alice", "password": "wrong"})
        assert resp.status_code == 401

    def test_me_unauthenticated(self, client: TestClient) -> None:
        resp = client.get("/api/auth/me")
        assert resp.status_code == 401

    def test_me_authenticated(self, client: TestClient) -> None:
        headers = _register_and_login(client)
        resp = client.get("/api/auth/me", headers=headers)
        assert resp.status_code == 200
        assert resp.json()["username"] == "tester"

    def test_refresh_token(self, client: TestClient) -> None:
        """Refresh endpoint returns a new access+refresh pair."""
        client.post(
            "/api/auth/register",
            json={"email": "r@b.com", "username": "refresher", "password": "securepass"},
        )
        login_resp = client.post(
            "/api/auth/login",
            json={"username": "refresher", "password": "securepass"},
        )
        refresh_tok = login_resp.json()["refresh_token"]
        resp = client.post("/api/auth/refresh", json={"refresh_token": refresh_tok})
        assert resp.status_code == 200
        body = resp.json()
        assert "access_token" in body
        assert "refresh_token" in body

    def test_refresh_with_access_token_rejected(self, client: TestClient) -> None:
        """Using an access token as a refresh token must fail."""
        client.post(
            "/api/auth/register",
            json={"email": "x@b.com", "username": "xuser", "password": "securepass"},
        )
        login_resp = client.post(
            "/api/auth/login",
            json={"username": "xuser", "password": "securepass"},
        )
        access_tok = login_resp.json()["access_token"]
        resp = client.post("/api/auth/refresh", json={"refresh_token": access_tok})
        assert resp.status_code == 401

    def test_refresh_token_rejected_as_bearer(self, client: TestClient) -> None:
        """Using a refresh token as a bearer token must fail."""
        client.post(
            "/api/auth/register",
            json={"email": "y@b.com", "username": "yuser", "password": "securepass"},
        )
        login_resp = client.post(
            "/api/auth/login",
            json={"username": "yuser", "password": "securepass"},
        )
        refresh_tok = login_resp.json()["refresh_token"]
        resp = client.get("/api/auth/me", headers={"Authorization": f"Bearer {refresh_tok}"})
        assert resp.status_code == 401

    def test_security_headers(self, client: TestClient) -> None:
        """All responses must include hardening headers."""
        resp = client.get("/api/health")
        assert resp.headers["X-Content-Type-Options"] == "nosniff"
        assert resp.headers["X-Frame-Options"] == "DENY"
        assert "strict-origin" in resp.headers.get("Referrer-Policy", "")

    def test_auth_cache_control(self, client: TestClient) -> None:
        """Auth responses must have Cache-Control: no-store."""
        client.post(
            "/api/auth/register",
            json={"email": "cc@b.com", "username": "ccuser", "password": "securepass"},
        )
        resp = client.post(
            "/api/auth/login",
            json={"username": "ccuser", "password": "securepass"},
        )
        assert resp.headers.get("Cache-Control") == "no-store"

    def test_login_rate_limit(self, client: TestClient) -> None:
        """After 5 failed login attempts the 6th should get 429."""
        client.post(
            "/api/auth/register",
            json={"email": "rl@b.com", "username": "rluser", "password": "securepass"},
        )
        for _ in range(5):
            client.post("/api/auth/login", json={"username": "rluser", "password": "wrong"})
        resp = client.post("/api/auth/login", json={"username": "rluser", "password": "securepass"})
        assert resp.status_code == 429


# ---------------------------------------------------------------------------
# Health
# ---------------------------------------------------------------------------


class TestHealth:
    def test_health(self, client: TestClient) -> None:
        assert client.get("/api/health").json() == {"status": "ok"}


# ---------------------------------------------------------------------------
# Explain (no auth required for explanations)
# ---------------------------------------------------------------------------


class TestExplain:
    def test_algorithms(self, client: TestClient) -> None:
        headers = _register_and_login(client)
        resp = client.get("/api/explain/algorithms", headers=headers)
        assert resp.status_code == 200
        algos = resp.json()
        keys = {a["key"] for a in algos}
        assert "er" in keys
        assert "raar" in keys
        assert "wf" in keys

    def test_metrics(self, client: TestClient) -> None:
        headers = _register_and_login(client)
        resp = client.get("/api/explain/metrics", headers=headers)
        assert resp.status_code == 200
        names = {m["name"] for m in resp.json()}
        assert "Strehl Ratio" in names

    def test_science(self, client: TestClient) -> None:
        headers = _register_and_login(client)
        resp = client.get("/api/explain/science", headers=headers)
        assert resp.status_code == 200
        assert "title" in resp.json()


# ---------------------------------------------------------------------------
# Data
# ---------------------------------------------------------------------------


class TestData:
    def test_synthetic_and_list(self, client: TestClient) -> None:
        headers = _register_and_login(client)
        resp = client.post(
            "/api/data/synthetic",
            json={
                "name": "test_synth",
                "grid_size": 64,
                "aberration_rms": 0.5,
                "telescope": "hst",
            },
            headers=headers,
        )
        assert resp.status_code == 200
        assert resp.json()["filename"].endswith(".npy")

        resp = client.get("/api/data/fits", headers=headers)
        assert resp.status_code == 200
        names = [f["filename"] for f in resp.json()]
        assert any("test_synth" in n for n in names)

    def test_presets(self, client: TestClient) -> None:
        headers = _register_and_login(client)
        resp = client.get("/api/data/presets", headers=headers)
        assert resp.status_code == 200
        assert len(resp.json()) > 0


# ---------------------------------------------------------------------------
# Algorithms — run
# ---------------------------------------------------------------------------


class TestAlgorithms:
    def test_list_algorithms(self, client: TestClient) -> None:
        headers = _register_and_login(client)
        resp = client.get("/api/algorithms/", headers=headers)
        assert resp.status_code == 200
        keys = {a["key"] for a in resp.json()}
        assert "hio" in keys

    def test_run_algorithm(self, client: TestClient) -> None:
        headers = _register_and_login(client)
        # Generate data first
        client.post(
            "/api/data/synthetic",
            json={
                "name": "run_test",
                "grid_size": 64,
                "aberration_rms": 0.3,
                "telescope": "hst",
            },
            headers=headers,
        )
        files = client.get("/api/data/fits", headers=headers).json()
        fname = [f["filename"] for f in files if "run_test" in f["filename"]][0]

        resp = client.post(
            "/api/algorithms/run",
            json={
                "fits_filename": fname,
                "algorithm": "er",
                "max_iterations": 10,
                "grid_size": 64,
            },
            headers=headers,
        )
        assert resp.status_code == 200
        j = resp.json()
        assert j["status"] == "completed"
        assert j["strehl_ratio"] is not None
        assert len(j["plots"]) > 0
        assert "metrics.json" in j["artifacts"]
        assert "evaluation_report.json" in j["artifacts"]

    def test_run_missing_file(self, client: TestClient) -> None:
        headers = _register_and_login(client)
        resp = client.post(
            "/api/algorithms/run",
            json={
                "fits_filename": "nonexistent.fits",
                "algorithm": "er",
                "max_iterations": 10,
            },
            headers=headers,
        )
        assert resp.status_code == 404


# ---------------------------------------------------------------------------
# Results
# ---------------------------------------------------------------------------


class TestResults:
    def test_list_empty(self, client: TestClient) -> None:
        headers = _register_and_login(client)
        resp = client.get("/api/results/", headers=headers)
        assert resp.status_code == 200
        assert resp.json() == []

    def test_dashboard_empty(self, client: TestClient) -> None:
        headers = _register_and_login(client)
        resp = client.get("/api/results/dashboard", headers=headers)
        assert resp.status_code == 200
        assert resp.json()["total_runs"] == 0

    def test_get_result_and_delete(self, client: TestClient) -> None:
        headers = _register_and_login(client)
        # Generate + run
        client.post(
            "/api/data/synthetic",
            json={
                "name": "del_test",
                "grid_size": 64,
                "aberration_rms": 0.3,
                "telescope": "hst",
            },
            headers=headers,
        )
        files = client.get("/api/data/fits", headers=headers).json()
        fname = [f["filename"] for f in files if "del_test" in f["filename"]][0]
        run_resp = client.post(
            "/api/algorithms/run",
            json={
                "fits_filename": fname,
                "algorithm": "er",
                "max_iterations": 5,
                "grid_size": 64,
            },
            headers=headers,
        )
        job_id = run_resp.json()["id"]

        # Get single
        resp = client.get(f"/api/results/{job_id}", headers=headers)
        assert resp.status_code == 200

        # Get plot
        plots = resp.json()["plots"]
        if plots:
            plot_resp = client.get(
                f"/api/results/{job_id}/plots/{plots[0]}",
                headers=headers,
            )
            assert plot_resp.status_code == 200
            assert plot_resp.headers["content-type"] == "image/png"

        export_resp = client.get(f"/api/results/{job_id}/export", headers=headers)
        assert export_resp.status_code == 200
        with zipfile.ZipFile(io.BytesIO(export_resp.content)) as archive:
            exported = set(archive.namelist())
        assert "metrics.json" in exported
        assert "provenance.json" in exported
        assert "evaluation_report.json" in exported

        # Delete
        del_resp = client.delete(f"/api/results/{job_id}", headers=headers)
        assert del_resp.status_code == 204

        # Verify gone
        assert client.get(f"/api/results/{job_id}", headers=headers).status_code == 404
