"""Data management endpoints — presets, FITS listing, upload, synthetic generation."""

from __future__ import annotations

import asyncio
from pathlib import Path

from fastapi import APIRouter, HTTPException, Query, UploadFile, status

from web.config import settings
from web.dependencies import CurrentUser
from web.schemas import (
    FitsFileInfo,
    PaginatedResponse,
    PresetInfo,
    SyntheticRequest,
    UploadedFileResponse,
)
from web.services.data_service import generate_synthetic_psf, list_fits_files

router = APIRouter(prefix="/api/data", tags=["data"])

_ALLOWED_EXTENSIONS = {".fits", ".fit", ".npy", ".npz"}


@router.get("/presets", response_model=list[PresetInfo])
def get_presets(_user: CurrentUser) -> list[PresetInfo]:
    """List curated observation presets available for download."""
    from src.data.downloader import available_presets
    from src.validation import available_reference_baselines

    baselines = available_reference_baselines()
    return [
        PresetInfo(
            key=k,
            description=v,
            verification_supported=k in baselines,
            baseline_key=k if k in baselines else None,
        )
        for k, v in available_presets().items()
    ]


@router.post("/download/{key}", status_code=status.HTTP_202_ACCEPTED)
async def download_preset(key: str, _user: CurrentUser) -> dict[str, str]:
    """Trigger download of a preset from MAST (requires network)."""
    from src.data.downloader import available_presets
    from src.data.downloader import download_preset as _dl

    presets = available_presets()
    if key not in presets:
        raise HTTPException(status_code=404, detail=f"Unknown preset '{key}'")
    try:
        from web.config import settings

        paths = await asyncio.to_thread(_dl, key, settings.data_dir)
        return {"status": "ok", "files": str([str(p) for p in paths])}
    except Exception as exc:
        raise HTTPException(status_code=502, detail=str(exc)) from exc


@router.get("/fits", response_model=PaginatedResponse[FitsFileInfo])
def get_fits_files(
    _user: CurrentUser,
    skip: int = Query(0, ge=0),
    limit: int = Query(50, ge=1, le=200),
) -> PaginatedResponse[FitsFileInfo]:
    """List all available FITS / ``.npy`` data files (paginated)."""
    all_files = list_fits_files()
    total = len(all_files)
    items = all_files[skip : skip + limit]
    return PaginatedResponse(items=items, total=total, skip=skip, limit=limit)


@router.post(
    "/upload",
    response_model=UploadedFileResponse,
    status_code=status.HTTP_201_CREATED,
)
async def upload_file(file: UploadFile, _user: CurrentUser) -> UploadedFileResponse:
    """Upload a custom FITS or NPY file for analysis.

    * Maximum file size: controlled by ``PR_UPLOAD_MAX_BYTES`` (default 100 MB)
    * Allowed extensions: ``.fits``, ``.fit``, ``.npy``, ``.npz``
    * Files are stored under ``data/uploads/``
    """
    max_bytes = settings.upload_max_bytes
    if not file.filename:
        raise HTTPException(status_code=422, detail="No filename provided")

    ext = Path(file.filename).suffix.lower()
    if ext not in _ALLOWED_EXTENSIONS:
        raise HTTPException(
            status_code=422,
            detail=f"Unsupported file type '{ext}'. Allowed: {sorted(_ALLOWED_EXTENSIONS)}",
        )

    upload_dir = settings.data_dir / "uploads"
    upload_dir.mkdir(parents=True, exist_ok=True)
    dest = upload_dir / file.filename

    # Stream to disk respecting the size limit
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
        filename=file.filename,
        size_bytes=total,
        message=f"Uploaded to {dest.relative_to(settings.data_dir)}",
    )


@router.post("/synthetic", response_model=FitsFileInfo)
async def create_synthetic(body: SyntheticRequest, _user: CurrentUser) -> dict[str, object]:
    """Generate a synthetic PSF for demo / testing."""
    path = await asyncio.to_thread(
        generate_synthetic_psf,
        name=body.name,
        grid_size=body.grid_size,
        aberration_rms=body.aberration_rms,
        n_zernike=body.n_zernike,
        telescope=body.telescope.value,
        filter_name=body.filter_name,
        photon_count=body.photon_count,
        read_noise_std=body.read_noise_std,
        center_offset_pixels=(body.center_offset_row_pixels, body.center_offset_col_pixels),
        background_level=body.background_level,
        bandwidth_fraction=body.bandwidth_fraction,
        spectral_samples=body.spectral_samples,
        spectral_weighting=body.spectral_weighting,
        field_defocus_waves=body.field_defocus_waves,
        detector_sigma_pixels=body.detector_sigma_pixels,
        jitter_sigma_pixels=body.jitter_sigma_pixels,
        pixel_integration_width=body.pixel_integration_width,
        random_seed=body.random_seed,
    )
    return {
        "filename": path.name,
        "filepath": str(path),
        "size_bytes": path.stat().st_size,
    }




