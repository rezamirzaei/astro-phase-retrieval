"""Crystallography endpoints — COD data, diffraction, phase retrieval, upload."""

from __future__ import annotations

import asyncio
import json
from pathlib import Path

from fastapi import APIRouter, HTTPException, UploadFile, status
from fastapi.responses import FileResponse

from web.concurrency import get_job_semaphore
from web.config import settings
from web.dependencies import CurrentUser, DbSession
from web.utils import assert_path_within, sanitize_filename
from web.models import CrystallographyJob
from web.schemas import (
    CifFileInfo,
    CodPresetInfo,
    CrystallographyCompareRequest,
    CrystallographyCompareResponse,
    CrystallographyJobResponse,
    CrystallographyRunRequest,
    SimulateDiffractionRequest,
    UploadedFileResponse,
)
from web.services.crystallography_service import (
    compare_crystallography_algorithms,
    list_cif_files,
    list_cod_presets,
    list_crystallography_job_plots,
    resolve_cif_path,
    run_crystallography_job,
    simulate_from_cif,
)

router = APIRouter(prefix="/api/crystallography", tags=["crystallography"])


@router.get("/presets", response_model=list[CodPresetInfo])
def get_cod_presets(_user: CurrentUser) -> list[CodPresetInfo]:
    """List curated COD crystal structure presets."""
    return list_cod_presets()


@router.post("/download/{key}", status_code=status.HTTP_202_ACCEPTED)
async def download_cod_preset(key: str, _user: CurrentUser) -> dict[str, str]:
    """Download a CIF file from COD by preset key."""
    from src.data.crystallography import available_cod_presets
    from src.data.crystallography import download_cod_preset as _dl

    presets = available_cod_presets()
    if key not in presets:
        raise HTTPException(status_code=404, detail=f"Unknown COD preset '{key}'")
    try:
        path = await asyncio.to_thread(_dl, key, settings.data_dir)
        return {"status": "ok", "file": str(path)}
    except Exception as exc:
        raise HTTPException(status_code=502, detail=str(exc)) from exc


@router.get("/cif-files", response_model=list[CifFileInfo])
def get_cif_files(_user: CurrentUser) -> list[dict[str, object]]:
    """List all available CIF files."""
    return list_cif_files()


@router.post("/simulate")
async def simulate_diffraction_endpoint(
    body: SimulateDiffractionRequest, _user: CurrentUser
) -> dict[str, object]:
    """Simulate a 2-D diffraction pattern from a CIF file."""
    try:
        cif_path = resolve_cif_path(body.cif_filename)
    except FileNotFoundError as exc:
        raise HTTPException(status_code=404, detail=str(exc)) from exc
    pattern, formula = await asyncio.to_thread(
        simulate_from_cif, cif_path, grid_size=body.grid_size
    )
    return {
        "formula": formula,
        "space_group": pattern.space_group,
        "grid_size": body.grid_size,
        "source_id": pattern.source_id,
    }


@router.post("/run", response_model=CrystallographyJobResponse)
async def run_crystallography(
    body: CrystallographyRunRequest,
    user: CurrentUser,
    db: DbSession,
) -> CrystallographyJobResponse:
    """Run phase retrieval on a crystallographic diffraction pattern."""
    try:
        cif_path = resolve_cif_path(body.cif_filename)
    except FileNotFoundError as exc:
        raise HTTPException(status_code=404, detail=str(exc)) from exc

    job = CrystallographyJob(
        user_id=user.id,
        algorithm=body.algorithm,
        cif_filename=body.cif_filename,
        config_json=json.dumps(
            {
                "max_iterations": body.max_iterations,
                "beta": body.beta,
            }
        ),
    )
    db.add(job)
    db.flush()

    async with get_job_semaphore():
        job = await asyncio.to_thread(
            run_crystallography_job, db, job, cif_path, grid_size=body.grid_size
        )
    if job.status == "failed":
        raise HTTPException(status_code=500, detail=job.error_message or "Unknown error")

    plots = list_crystallography_job_plots(job)
    resp = CrystallographyJobResponse.model_validate(job)
    resp.plots = plots
    return resp


@router.post("/compare", response_model=CrystallographyCompareResponse)
async def compare_crystallography(
    body: CrystallographyCompareRequest,
    user: CurrentUser,
    db: DbSession,
) -> CrystallographyCompareResponse:
    """Run multiple algorithms on the same crystallographic data."""
    try:
        cif_path = resolve_cif_path(body.cif_filename)
    except FileNotFoundError as exc:
        raise HTTPException(status_code=404, detail=str(exc)) from exc

    async with get_job_semaphore():
        jobs = await asyncio.to_thread(
            compare_crystallography_algorithms,
            db,
            user_id=user.id,
            cif_path=cif_path,
            grid_size=body.grid_size,
            max_iterations=body.max_iterations,
            algorithm_keys=[a.value for a in body.algorithms] if body.algorithms else None,
        )

    results: list[CrystallographyJobResponse] = []
    for j in jobs:
        resp = CrystallographyJobResponse.model_validate(j)
        resp.plots = list_crystallography_job_plots(j)
        results.append(resp)

    return CrystallographyCompareResponse(results=results)


@router.get("/{job_id}", response_model=CrystallographyJobResponse)
def get_crystallography_result(
    job_id: int, user: CurrentUser, db: DbSession
) -> CrystallographyJobResponse:
    """Get a single crystallography result by ID."""
    job = db.get(CrystallographyJob, job_id)
    if job is None or job.user_id != user.id:
        raise HTTPException(status_code=404, detail="Crystallography result not found")
    resp = CrystallographyJobResponse.model_validate(job)
    resp.plots = list_crystallography_job_plots(job)
    return resp


@router.get("/{job_id}/plots/{plot_name}")
def get_crystallography_plot(
    job_id: int, plot_name: str, user: CurrentUser, db: DbSession
) -> FileResponse:
    """Serve a plot PNG for a crystallography result."""
    job = db.get(CrystallographyJob, job_id)
    if job is None or job.user_id != user.id:
        raise HTTPException(status_code=404, detail="Crystallography result not found")
    if not job.output_dir:
        raise HTTPException(status_code=404, detail="No plots available")
    safe_plot = sanitize_filename(plot_name)
    out_dir = Path(job.output_dir)
    plot_path = out_dir / safe_plot
    assert_path_within(plot_path, out_dir)
    if not plot_path.exists() or not plot_path.name.endswith(".png"):
        raise HTTPException(status_code=404, detail=f"Plot '{safe_plot}' not found")
    return FileResponse(plot_path, media_type="image/png")


@router.delete("/{job_id}", status_code=status.HTTP_204_NO_CONTENT)
def delete_crystallography_result(job_id: int, user: CurrentUser, db: DbSession) -> None:
    """Delete a crystallography result and its plots."""
    import shutil

    job = db.get(CrystallographyJob, job_id)
    if job is None or job.user_id != user.id:
        raise HTTPException(status_code=404, detail="Crystallography result not found")
    if job.output_dir and Path(job.output_dir).exists():
        shutil.rmtree(job.output_dir, ignore_errors=True)
    db.delete(job)
    db.commit()


# ---------------------------------------------------------------------------
# CIF upload
# ---------------------------------------------------------------------------

_ALLOWED_CIF_EXTENSIONS = {".cif"}


@router.post(
    "/upload",
    response_model=UploadedFileResponse,
    status_code=status.HTTP_201_CREATED,
)
async def upload_cif(file: UploadFile, _user: CurrentUser) -> UploadedFileResponse:
    """Upload a custom CIF file for crystallography analysis.

    * Maximum file size: controlled by ``PR_UPLOAD_MAX_BYTES``
    * Allowed extension: ``.cif``
    * Files are stored under ``data/crystallography/``
    """
    max_bytes = settings.upload_max_bytes
    if not file.filename:
        raise HTTPException(status_code=422, detail="No filename provided")

    safe_name = sanitize_filename(file.filename)
    ext = Path(safe_name).suffix.lower()
    if ext not in _ALLOWED_CIF_EXTENSIONS:
        raise HTTPException(
            status_code=422,
            detail=f"Unsupported file type '{ext}'. Only .cif files are allowed.",
        )

    cryst_dir = settings.data_dir / "crystallography"
    cryst_dir.mkdir(parents=True, exist_ok=True)
    dest = cryst_dir / safe_name

    total = 0
    with open(dest, "wb") as f:
        while chunk := await file.read(1024 * 64):
            total += len(chunk)
            if total > max_bytes:
                dest.unlink(missing_ok=True)
                raise HTTPException(
                    status_code=413,
                    detail=f"File exceeds {max_bytes // (1024 * 1024)} MB limit",
                )
            f.write(chunk)

    return UploadedFileResponse(
        filename=safe_name,
        size_bytes=total,
        message=f"Uploaded to crystallography/{safe_name}",
    )

