"""End-to-end tests for the web API — auth, data, algorithms, results, explain,
middleware, upload, jobs, batch export.
"""

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
        data = client.get("/api/health").json()
        assert data["status"] == "ok"
        assert "version" in data
        assert "uptime_seconds" in data

    def test_readiness(self, client: TestClient) -> None:
        data = client.get("/api/readiness").json()
        assert data["db"] == "ok"
        assert data["disk"] == "ok"
        assert "version" in data


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
        data = resp.json()
        names = [f["filename"] for f in data["items"]]
        assert any("test_synth" in n for n in names)
        assert data["total"] >= 1

    def test_synthetic_accepts_richer_generation_controls(self, client: TestClient) -> None:
        headers = _register_and_login(client)
        resp = client.post(
            "/api/data/synthetic",
            json={
                "name": "rich_synth",
                "grid_size": 64,
                "aberration_rms": 0.5,
                "n_zernike": 12,
                "telescope": "jwst",
                "photon_count": 25000,
                "read_noise_std": 1e-4,
                "center_offset_row_pixels": 0.4,
                "center_offset_col_pixels": -0.25,
                "background_level": 1e-6,
                "bandwidth_fraction": 0.1,
                "spectral_samples": 3,
                "spectral_weighting": "gaussian",
                "field_defocus_waves": 0.2,
                "detector_sigma_pixels": 0.3,
                "jitter_sigma_pixels": 0.15,
                "pixel_integration_width": 1.2,
                "random_seed": 7,
            },
            headers=headers,
        )
        assert resp.status_code == 200
        assert resp.json()["filename"].startswith("rich_synth")

    def test_presets(self, client: TestClient) -> None:
        headers = _register_and_login(client)
        resp = client.get("/api/data/presets", headers=headers)
        assert resp.status_code == 200
        presets = resp.json()
        assert len(presets) > 0
        assert all("verification_supported" in preset for preset in presets)
        assert any(preset["verification_supported"] for preset in presets)
        supported = {preset["key"] for preset in presets if preset["verification_supported"]}
        assert {
            "hst-wfc3-uvis-f606w",
            "hst-wfc3-uvis-f814w",
            "hst-wfc3-uvis-f438w",
            "hst-wfc3-uvis-f275w",
            "hst-acs-wfc-f606w",
            "hst-acs-wfc-f814w",
            "jwst-nircam-f200w",
            "jwst-nircam-f356w",
        }.issubset(supported)


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
        files = client.get("/api/data/fits", headers=headers).json()["items"]
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

    def test_run_algorithm_accepts_advanced_method_controls(self, client: TestClient) -> None:
        headers = _register_and_login(client)
        client.post(
            "/api/data/synthetic",
            json={
                "name": "advanced_run",
                "grid_size": 64,
                "aberration_rms": 0.35,
                "telescope": "hst",
                "photon_count": 20000,
            },
            headers=headers,
        )
        files = client.get("/api/data/fits", headers=headers).json()["items"]
        fname = [f["filename"] for f in files if "advanced_run" in f["filename"]][0]

        resp = client.post(
            "/api/algorithms/run",
            json={
                "fits_filename": fname,
                "algorithm": "admm",
                "max_iterations": 12,
                "grid_size": 64,
                "tolerance": 1e-6,
                "noise_model": "poisson",
                "n_starts": 1,
                "uncertainty_samples": 2,
                "admm_rho": 1.25,
                "wf_step_size": 0.4,
                "wf_spectral_init": True,
                "spectral_init": True,
                "regulariser": "tv",
                "proximal_weight": 5e-4,
                "sparsity_threshold": 0.08,
                "sparsity_keep_fraction": 0.6,
            },
            headers=headers,
        )
        assert resp.status_code == 200
        assert resp.json()["status"] == "completed"

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

    def test_benchmark_endpoint(self, client: TestClient) -> None:
        headers = _register_and_login(client)
        cases_resp = client.get("/api/algorithms/benchmark/cases", headers=headers)
        assert cases_resp.status_code == 200
        assert any(case["key"] == "clean-low" for case in cases_resp.json())

        resp = client.post(
            "/api/algorithms/benchmark",
            json={
                "algorithms": ["er", "hio"],
                "cases": ["clean-low", "poisson-hst"],
                "max_iterations": 3,
                "beta": 0.9,
            },
            headers=headers,
        )
        assert resp.status_code == 200
        payload = resp.json()
        assert payload["records_count"] == 4
        assert {row["algorithm"] for row in payload["aggregate"]} == {"er", "hio"}
        assert any(case["key"] == "clean-low" for case in payload["selected_cases"])


# ---------------------------------------------------------------------------
# Results
# ---------------------------------------------------------------------------


class TestResults:
    def test_list_empty(self, client: TestClient) -> None:
        headers = _register_and_login(client)
        resp = client.get("/api/results/", headers=headers)
        assert resp.status_code == 200
        data = resp.json()
        assert data["items"] == []
        assert data["total"] == 0

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
        files = client.get("/api/data/fits", headers=headers).json()["items"]
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

        artifact_resp = client.get(
            f"/api/results/{job_id}/artifacts/evaluation_report.json",
            headers=headers,
        )
        assert artifact_resp.status_code == 200
        assert artifact_resp.json()["format"] == "json"
        assert artifact_resp.json()["content"]["report_type"] == "single_run_evaluation"

        provenance_resp = client.get(
            f"/api/results/{job_id}/artifacts/provenance.json",
            headers=headers,
        )
        assert provenance_resp.status_code == 200
        assert provenance_resp.json()["content"]["reference_validation_available"] in {True, False}

        # Delete
        del_resp = client.delete(f"/api/results/{job_id}", headers=headers)
        assert del_resp.status_code == 204

        # Verify gone
        assert client.get(f"/api/results/{job_id}", headers=headers).status_code == 404


class TestStudies:
    def test_validation_campaign_endpoint(self, client: TestClient) -> None:
        headers = _register_and_login(client)
        client.post(
            "/api/data/synthetic",
            json={
                "name": "campaign_input",
                "grid_size": 64,
                "aberration_rms": 0.3,
                "telescope": "hst",
            },
            headers=headers,
        )
        files = client.get("/api/data/fits", headers=headers).json()["items"]
        filename = [f["filename"] for f in files if "campaign_input" in f["filename"]][0]

        resp = client.post(
            "/api/studies/validation-campaign",
            json={
                "fits_filenames": [filename],
                "algorithm": "er",
                "max_iterations": 4,
                "grid_size": 64,
            },
            headers=headers,
        )
        assert resp.status_code == 200
        payload = resp.json()
        assert payload["summary"]["n_observations"] == 1
        assert payload["selected_files"] == [filename]
        assert "validation_campaign.json" in payload["artifacts"]
        assert "reference_summary" in payload
        assert "validation_campaign.md" in payload["artifacts"]

        artifact_resp = client.get(
            f"/api/studies/validation-campaigns/{payload['campaign_id']}/artifacts/validation_campaign.json",
            headers=headers,
        )
        assert artifact_resp.status_code == 200
        assert artifact_resp.json()["format"] == "json"
        assert artifact_resp.json()["content"]["summary"]["n_observations"] == 1

        md_resp = client.get(
            f"/api/studies/validation-campaigns/{payload['campaign_id']}/artifacts/validation_campaign.md",
            headers=headers,
        )
        assert md_resp.status_code == 200
        assert md_resp.json()["format"] == "markdown"
        assert "Validation Campaign Report" in md_resp.json()["content"]


# ---------------------------------------------------------------------------
# Middleware tests
# ---------------------------------------------------------------------------


class TestMiddleware:
    def test_request_id_header_returned(self, client: TestClient) -> None:
        """Every response must include an X-Request-ID header."""
        resp = client.get("/api/health")
        assert "X-Request-ID" in resp.headers
        assert len(resp.headers["X-Request-ID"]) > 0

    def test_request_id_propagated(self, client: TestClient) -> None:
        """If the client sends X-Request-ID, the server should echo it."""
        custom_id = "my-trace-id-12345"
        resp = client.get("/api/health", headers={"X-Request-ID": custom_id})
        assert resp.headers["X-Request-ID"] == custom_id

    def test_security_headers_present(self, client: TestClient) -> None:
        """Full set of security headers on every response."""
        resp = client.get("/api/version")
        assert resp.headers["X-Content-Type-Options"] == "nosniff"
        assert resp.headers["X-Frame-Options"] == "DENY"
        assert resp.headers["X-XSS-Protection"] == "1; mode=block"
        assert "strict-origin" in resp.headers.get("Referrer-Policy", "")
        assert "camera=()" in resp.headers.get("Permissions-Policy", "")
        assert "max-age=" in resp.headers.get("Strict-Transport-Security", "")

    def test_cors_expose_request_id(self, client: TestClient) -> None:
        """CORS headers should expose X-Request-ID for browser clients."""
        resp = client.options(
            "/api/health",
            headers={
                "Origin": "http://localhost:4532",
                "Access-Control-Request-Method": "GET",
            },
        )
        # The CORS middleware should expose our custom header
        expose = resp.headers.get("Access-Control-Expose-Headers", "")
        # Note: options may not include expose headers, but GET should
        get_resp = client.get(
            "/api/health",
            headers={"Origin": "http://localhost:4532"},
        )
        expose_get = get_resp.headers.get("Access-Control-Expose-Headers", "")
        assert "X-Request-ID" in expose_get or "X-Request-ID" in expose


# ---------------------------------------------------------------------------
# File upload tests
# ---------------------------------------------------------------------------


class TestUpload:
    def test_upload_npy_file(self, client: TestClient) -> None:
        """Upload a .npy file and verify it appears in the file list."""
        import numpy as np

        headers = _register_and_login(client)
        # Create a small .npy file in memory
        buf = io.BytesIO()
        np.save(buf, np.zeros((64, 64)))
        buf.seek(0)

        resp = client.post(
            "/api/data/upload",
            files={"file": ("upload_test.npy", buf, "application/octet-stream")},
            headers=headers,
        )
        assert resp.status_code == 201
        body = resp.json()
        assert body["filename"] == "upload_test.npy"
        assert body["size_bytes"] > 0
        assert "Upload successful" in body["message"] or "uploads" in body["message"]

    def test_upload_rejected_bad_extension(self, client: TestClient) -> None:
        """Uploads with unsupported extensions must be rejected."""
        headers = _register_and_login(client)
        buf = io.BytesIO(b"not a real file")
        resp = client.post(
            "/api/data/upload",
            files={"file": ("bad.txt", buf, "text/plain")},
            headers=headers,
        )
        assert resp.status_code == 422
        assert "Unsupported" in resp.json()["detail"]

    def test_upload_cif_file(self, client: TestClient) -> None:
        """Upload a .cif file to crystallography endpoint."""
        headers = _register_and_login(client)
        buf = io.BytesIO(b"data_test\n_cell_length_a 5.64\n")
        resp = client.post(
            "/api/crystallography/upload",
            files={"file": ("test_upload.cif", buf, "application/octet-stream")},
            headers=headers,
        )
        assert resp.status_code == 201
        body = resp.json()
        assert body["filename"] == "test_upload.cif"

    def test_upload_cif_rejected_bad_extension(self, client: TestClient) -> None:
        """Non-CIF files must be rejected by the crystallography upload."""
        headers = _register_and_login(client)
        buf = io.BytesIO(b"not a cif")
        resp = client.post(
            "/api/crystallography/upload",
            files={"file": ("bad.xyz", buf, "application/octet-stream")},
            headers=headers,
        )
        assert resp.status_code == 422


# ---------------------------------------------------------------------------
# Background job tests
# ---------------------------------------------------------------------------


class TestBackgroundJobs:
    def test_list_jobs_empty(self, client: TestClient) -> None:
        """Job list is empty initially."""
        headers = _register_and_login(client)
        resp = client.get("/api/jobs/", headers=headers)
        assert resp.status_code == 200
        assert resp.json() == []

    def test_poll_nonexistent_job(self, client: TestClient) -> None:
        """Polling a non-existent job returns 404."""
        headers = _register_and_login(client)
        resp = client.get("/api/jobs/nonexistent123", headers=headers)
        assert resp.status_code == 404

    def test_cancel_nonexistent_job(self, client: TestClient) -> None:
        """Cancelling a non-existent job returns 404."""
        headers = _register_and_login(client)
        resp = client.post("/api/jobs/nonexistent123/cancel", headers=headers)
        assert resp.status_code == 404


# ---------------------------------------------------------------------------
# Batch export tests
# ---------------------------------------------------------------------------


class TestBatchExport:
    def test_batch_export(self, client: TestClient) -> None:
        """Batch export combines multiple job outputs into a single ZIP."""
        headers = _register_and_login(client)
        # Generate data + run two jobs
        client.post(
            "/api/data/synthetic",
            json={
                "name": "batch_test",
                "grid_size": 64,
                "aberration_rms": 0.3,
                "telescope": "hst",
            },
            headers=headers,
        )
        files = client.get("/api/data/fits", headers=headers).json()["items"]
        fname = [f["filename"] for f in files if "batch_test" in f["filename"]][0]

        # Run two algorithms
        ids = []
        for algo in ["er", "gs"]:
            run_resp = client.post(
                "/api/algorithms/run",
                json={
                    "fits_filename": fname,
                    "algorithm": algo,
                    "max_iterations": 5,
                    "grid_size": 64,
                },
                headers=headers,
            )
            assert run_resp.status_code == 200
            ids.append(run_resp.json()["id"])

        # Batch export
        resp = client.post(
            "/api/results/export-batch",
            json={"job_ids": ids},
            headers=headers,
        )
        assert resp.status_code == 200
        assert resp.headers["content-type"] == "application/zip"
        with zipfile.ZipFile(io.BytesIO(resp.content)) as archive:
            names = archive.namelist()
        # Each job should have a metadata.json
        assert any("metadata.json" in n for n in names)

    def test_batch_export_empty(self, client: TestClient) -> None:
        """Batch export with non-existent job IDs returns 404."""
        headers = _register_and_login(client)
        resp = client.post(
            "/api/results/export-batch",
            json={"job_ids": [99999]},
            headers=headers,
        )
        assert resp.status_code == 404


# ---------------------------------------------------------------------------
# Version endpoint
# ---------------------------------------------------------------------------


class TestVersion:
    def test_version_endpoint(self, client: TestClient) -> None:
        resp = client.get("/api/version")
        assert resp.status_code == 200
        body = resp.json()
        assert "api_version" in body
        assert "python" in body

    def test_health_includes_uptime(self, client: TestClient) -> None:
        data = client.get("/api/health").json()
        assert data["status"] == "ok"
        assert data["uptime_seconds"] >= 0


# ---------------------------------------------------------------------------
# Pagination tests
# ---------------------------------------------------------------------------


class TestPagination:
    def test_fits_pagination(self, client: TestClient) -> None:
        """GET /api/data/fits supports skip/limit pagination."""
        headers = _register_and_login(client)
        # Create a few files
        for i in range(3):
            client.post(
                "/api/data/synthetic",
                json={
                    "name": f"page_test_{i}",
                    "grid_size": 64,
                    "aberration_rms": 0.3,
                    "telescope": "hst",
                },
                headers=headers,
            )

        # Request with limit=1
        resp = client.get("/api/data/fits?skip=0&limit=1", headers=headers)
        assert resp.status_code == 200
        data = resp.json()
        assert len(data["items"]) <= 1
        assert data["total"] >= 3
        assert data["limit"] == 1

    def test_results_pagination(self, client: TestClient) -> None:
        """GET /api/results/ supports skip/limit pagination."""
        headers = _register_and_login(client)
        resp = client.get("/api/results/?skip=0&limit=10", headers=headers)
        assert resp.status_code == 200
        data = resp.json()
        assert "items" in data
        assert "total" in data
        assert data["limit"] == 10

