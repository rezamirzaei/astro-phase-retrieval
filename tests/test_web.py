"""End-to-end tests for the web API — auth, data, algorithms, results, explain."""

from __future__ import annotations

import pytest

fastapi = pytest.importorskip("fastapi", reason="web extras not installed")
sqlalchemy = pytest.importorskip("sqlalchemy", reason="web extras not installed")

from fastapi.testclient import TestClient  # noqa: E402
from sqlalchemy import create_engine  # noqa: E402
from sqlalchemy.orm import Session, sessionmaker  # noqa: E402
from sqlalchemy.pool import StaticPool  # noqa: E402

from web.database import Base, get_db  # noqa: E402
from web.main import app  # noqa: E402

# ---------------------------------------------------------------------------
# Fixtures — in-memory SQLite, isolated per-test
# ---------------------------------------------------------------------------


@pytest.fixture()
def db_session() -> Session:  # type: ignore[misc]
    engine = create_engine(
        "sqlite://",
        connect_args={"check_same_thread": False},
        poolclass=StaticPool,
    )
    Base.metadata.create_all(bind=engine)
    _SessionLocal = sessionmaker(bind=engine)
    session = _SessionLocal()
    try:
        yield session  # type: ignore[misc]
    finally:
        session.close()
        Base.metadata.drop_all(bind=engine)


@pytest.fixture()
def client(db_session: Session) -> TestClient:  # type: ignore[misc]
    def _override_get_db():  # type: ignore[no-untyped-def]
        yield db_session

    app.dependency_overrides[get_db] = _override_get_db
    with TestClient(app) as c:
        yield c  # type: ignore[misc]
    app.dependency_overrides.clear()


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
        assert "access_token" in resp.json()

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

    def test_login_nonexistent_user(self, client: TestClient) -> None:
        resp = client.post(
            "/api/auth/login",
            json={"username": "ghost", "password": "nope1234"},
        )
        assert resp.status_code == 401

    def test_me_unauthenticated(self, client: TestClient) -> None:
        resp = client.get("/api/auth/me")
        assert resp.status_code == 401

    def test_me_authenticated(self, client: TestClient) -> None:
        headers = _register_and_login(client)
        resp = client.get("/api/auth/me", headers=headers)
        assert resp.status_code == 200
        assert resp.json()["username"] == "tester"


# ---------------------------------------------------------------------------
# Health
# ---------------------------------------------------------------------------


class TestHealth:
    def test_health(self, client: TestClient) -> None:
        assert client.get("/api/health").json() == {"status": "ok"}

    def test_version(self, client: TestClient) -> None:
        resp = client.get("/api/version")
        assert resp.status_code == 200
        assert "api_version" in resp.json()
        assert "python" in resp.json()


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

    def test_download_unknown_preset(self, client: TestClient) -> None:
        headers = _register_and_login(client)
        resp = client.post("/api/data/download/nonexistent_preset_xyz", headers=headers)
        assert resp.status_code == 404


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

    def test_compare_algorithms(self, client: TestClient) -> None:
        headers = _register_and_login(client)
        # Generate data first
        client.post(
            "/api/data/synthetic",
            json={
                "name": "cmp_test",
                "grid_size": 64,
                "aberration_rms": 0.3,
                "telescope": "hst",
            },
            headers=headers,
        )
        files = client.get("/api/data/fits", headers=headers).json()
        fname = [f["filename"] for f in files if "cmp_test" in f["filename"]][0]

        resp = client.post(
            "/api/algorithms/compare",
            json={
                "fits_filename": fname,
                "max_iterations": 5,
                "grid_size": 64,
                "algorithms": ["er", "hio"],
            },
            headers=headers,
        )
        assert resp.status_code == 200
        data = resp.json()
        assert len(data["results"]) == 2
        assert all(r["status"] == "completed" for r in data["results"])

    def test_compare_missing_file(self, client: TestClient) -> None:
        headers = _register_and_login(client)
        resp = client.post(
            "/api/algorithms/compare",
            json={
                "fits_filename": "nonexistent.fits",
                "max_iterations": 5,
                "grid_size": 64,
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

    def test_list_with_pagination(self, client: TestClient) -> None:
        headers = _register_and_login(client)
        resp = client.get("/api/results/?skip=0&limit=10", headers=headers)
        assert resp.status_code == 200

    def test_dashboard_empty(self, client: TestClient) -> None:
        headers = _register_and_login(client)
        resp = client.get("/api/results/dashboard", headers=headers)
        assert resp.status_code == 200
        assert resp.json()["total_runs"] == 0

    def test_get_nonexistent_result(self, client: TestClient) -> None:
        headers = _register_and_login(client)
        resp = client.get("/api/results/99999", headers=headers)
        assert resp.status_code == 404

    def test_delete_nonexistent_result(self, client: TestClient) -> None:
        headers = _register_and_login(client)
        resp = client.delete("/api/results/99999", headers=headers)
        assert resp.status_code == 404

    def test_export_nonexistent(self, client: TestClient) -> None:
        headers = _register_and_login(client)
        resp = client.get("/api/results/99999/export", headers=headers)
        assert resp.status_code == 404

    def test_plot_nonexistent_job(self, client: TestClient) -> None:
        headers = _register_and_login(client)
        resp = client.get("/api/results/99999/plots/test.png", headers=headers)
        assert resp.status_code == 404

    def test_full_lifecycle(self, client: TestClient) -> None:
        """Run, list, dashboard, get, export, plot, delete."""
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

        # List results
        list_resp = client.get("/api/results/", headers=headers)
        assert list_resp.status_code == 200
        assert len(list_resp.json()) >= 1

        # Dashboard with data
        dash_resp = client.get("/api/results/dashboard", headers=headers)
        assert dash_resp.status_code == 200
        assert dash_resp.json()["total_runs"] >= 1
        assert dash_resp.json()["completed_runs"] >= 1

        # Get single
        resp = client.get(f"/api/results/{job_id}", headers=headers)
        assert resp.status_code == 200

        # Export ZIP
        export_resp = client.get(f"/api/results/{job_id}/export", headers=headers)
        assert export_resp.status_code == 200
        assert export_resp.headers["content-type"] == "application/zip"

        # Get plot
        plots = resp.json()["plots"]
        if plots:
            plot_resp = client.get(
                f"/api/results/{job_id}/plots/{plots[0]}",
                headers=headers,
            )
            assert plot_resp.status_code == 200
            assert plot_resp.headers["content-type"] == "image/png"

            # Non-png plot name
            bad_plot = client.get(
                f"/api/results/{job_id}/plots/evil.txt",
                headers=headers,
            )
            assert bad_plot.status_code == 404

        # Delete
        del_resp = client.delete(f"/api/results/{job_id}", headers=headers)
        assert del_resp.status_code == 204

        # Verify gone
        assert client.get(f"/api/results/{job_id}", headers=headers).status_code == 404
