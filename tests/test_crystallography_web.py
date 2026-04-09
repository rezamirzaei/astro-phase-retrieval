"""End-to-end tests for the crystallography web API endpoints."""

from __future__ import annotations

from pathlib import Path

import pytest

fastapi = pytest.importorskip("fastapi", reason="web extras not installed")
sqlalchemy = pytest.importorskip("sqlalchemy", reason="web extras not installed")

from fastapi.testclient import TestClient  # noqa: E402
from sqlalchemy import create_engine  # noqa: E402
from sqlalchemy.orm import Session, sessionmaker  # noqa: E402
from sqlalchemy.pool import StaticPool  # noqa: E402

from web.database import Base, get_db  # noqa: E402
from web.main import app  # noqa: E402


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
            "email": "cryst@example.com",
            "username": "crystuser",
            "password": "password123",
        },
    )
    resp = client.post(
        "/api/auth/login",
        json={"username": "crystuser", "password": "password123"},
    )
    token = resp.json()["access_token"]
    return {"Authorization": f"Bearer {token}"}


def _create_cif_file(data_dir: Path) -> str:
    """Create a minimal CIF file in the expected location."""
    cif_dir = data_dir / "crystallography"
    cif_dir.mkdir(parents=True, exist_ok=True)
    cif_path = cif_dir / "test_nacl.cif"
    cif_path.write_text(
        "data_test\n"
        "_cell_length_a 5.64\n"
        "_cell_length_b 5.64\n"
        "_cell_length_c 5.64\n"
        "_cell_angle_alpha 90.0\n"
        "_cell_angle_beta 90.0\n"
        "_cell_angle_gamma 90.0\n"
        "_symmetry_space_group_name_H-M 'F m -3 m'\n"
        "_chemical_formula_sum 'Cl Na'\n"
        "loop_\n"
        "_atom_site_label\n"
        "_atom_site_type_symbol\n"
        "_atom_site_fract_x\n"
        "_atom_site_fract_y\n"
        "_atom_site_fract_z\n"
        "_atom_site_occupancy\n"
        "Na1 Na 0.0000 0.0000 0.0000 1.000\n"
        "Cl1 Cl 0.5000 0.5000 0.5000 1.000\n"
    )
    return cif_path.name


class TestCrystallographyPresets:
    def test_list_presets(self, client: TestClient) -> None:
        headers = _register_and_login(client)
        resp = client.get("/api/crystallography/presets", headers=headers)
        assert resp.status_code == 200
        data = resp.json()
        assert len(data) > 0
        keys = {p["key"] for p in data}
        assert "nacl" in keys


class TestCrystallographyCIFFiles:
    def test_list_cif_files(self, client: TestClient) -> None:
        headers = _register_and_login(client)
        # Create a CIF file
        from web.config import settings
        _create_cif_file(settings.data_dir)

        resp = client.get("/api/crystallography/cif-files", headers=headers)
        assert resp.status_code == 200
        files = resp.json()
        assert any("test_nacl" in f["filename"] for f in files)


class TestCrystallographySimulate:
    def test_simulate(self, client: TestClient) -> None:
        headers = _register_and_login(client)
        from web.config import settings
        fname = _create_cif_file(settings.data_dir)

        resp = client.post(
            "/api/crystallography/simulate",
            json={"cif_filename": fname, "grid_size": 64},
            headers=headers,
        )
        assert resp.status_code == 200
        data = resp.json()
        assert "formula" in data
        assert data["grid_size"] == 64

    def test_simulate_missing_file(self, client: TestClient) -> None:
        headers = _register_and_login(client)
        resp = client.post(
            "/api/crystallography/simulate",
            json={"cif_filename": "nonexistent.cif", "grid_size": 64},
            headers=headers,
        )
        assert resp.status_code == 404


class TestCrystallographyRun:
    def test_run_algorithm(self, client: TestClient) -> None:
        headers = _register_and_login(client)
        from web.config import settings
        fname = _create_cif_file(settings.data_dir)

        resp = client.post(
            "/api/crystallography/run",
            json={
                "cif_filename": fname,
                "algorithm": "er",
                "max_iterations": 5,
                "grid_size": 64,
            },
            headers=headers,
        )
        assert resp.status_code == 200
        data = resp.json()
        assert data["status"] == "completed"
        assert data["r_factor"] is not None
        assert len(data["plots"]) > 0

    def test_run_missing_file(self, client: TestClient) -> None:
        headers = _register_and_login(client)
        resp = client.post(
            "/api/crystallography/run",
            json={
                "cif_filename": "nonexistent.cif",
                "algorithm": "er",
                "max_iterations": 5,
            },
            headers=headers,
        )
        assert resp.status_code == 404


class TestCrystallographyResults:
    def test_get_result_and_delete(self, client: TestClient) -> None:
        headers = _register_and_login(client)
        from web.config import settings
        fname = _create_cif_file(settings.data_dir)

        # Run
        run_resp = client.post(
            "/api/crystallography/run",
            json={
                "cif_filename": fname,
                "algorithm": "er",
                "max_iterations": 5,
                "grid_size": 64,
            },
            headers=headers,
        )
        job_id = run_resp.json()["id"]

        # Get
        resp = client.get(f"/api/crystallography/{job_id}", headers=headers)
        assert resp.status_code == 200

        # Get plot
        plots = resp.json()["plots"]
        if plots:
            plot_resp = client.get(
                f"/api/crystallography/{job_id}/plots/{plots[0]}",
                headers=headers,
            )
            assert plot_resp.status_code == 200
            assert plot_resp.headers["content-type"] == "image/png"

        # Delete
        del_resp = client.delete(f"/api/crystallography/{job_id}", headers=headers)
        assert del_resp.status_code == 204

        # Verify gone
        assert client.get(
            f"/api/crystallography/{job_id}", headers=headers
        ).status_code == 404

    def test_get_nonexistent(self, client: TestClient) -> None:
        headers = _register_and_login(client)
        resp = client.get("/api/crystallography/99999", headers=headers)
        assert resp.status_code == 404

    def test_delete_nonexistent(self, client: TestClient) -> None:
        headers = _register_and_login(client)
        resp = client.delete("/api/crystallography/99999", headers=headers)
        assert resp.status_code == 404

    def test_get_plot_nonexistent_job(self, client: TestClient) -> None:
        headers = _register_and_login(client)
        resp = client.get("/api/crystallography/99999/plots/test.png", headers=headers)
        assert resp.status_code == 404


class TestCrystallographyCompare:
    def test_compare(self, client: TestClient) -> None:
        headers = _register_and_login(client)
        from web.config import settings
        fname = _create_cif_file(settings.data_dir)

        resp = client.post(
            "/api/crystallography/compare",
            json={
                "cif_filename": fname,
                "max_iterations": 5,
                "grid_size": 64,
                "algorithms": ["er", "hio"],
            },
            headers=headers,
        )
        assert resp.status_code == 200
        data = resp.json()
        assert len(data["results"]) == 2

    def test_compare_missing_file(self, client: TestClient) -> None:
        headers = _register_and_login(client)
        resp = client.post(
            "/api/crystallography/compare",
            json={
                "cif_filename": "nonexistent.cif",
                "max_iterations": 5,
                "grid_size": 64,
            },
            headers=headers,
        )
        assert resp.status_code == 404


class TestCrystallographyDownload:
    def test_download_unknown_preset(self, client: TestClient) -> None:
        headers = _register_and_login(client)
        resp = client.post(
            "/api/crystallography/download/nonexistent",
            headers=headers,
        )
        assert resp.status_code == 404


