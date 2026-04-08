"""Data management endpoints — presets, FITS listing, synthetic generation."""

from __future__ import annotations

from fastapi import APIRouter, HTTPException, status

from web.dependencies import CurrentUser
from web.schemas import FitsFileInfo, PresetInfo, SyntheticRequest
from web.services.data_service import generate_synthetic_psf, list_fits_files

router = APIRouter(prefix="/api/data", tags=["data"])


@router.get("/presets", response_model=list[PresetInfo])
def get_presets(_user: CurrentUser) -> list[dict[str, str]]:
    """List curated observation presets available for download."""
    from src.data.downloader import available_presets

    return [{"key": k, "description": v} for k, v in available_presets().items()]


@router.post("/download/{key}", status_code=status.HTTP_202_ACCEPTED)
def download_preset(key: str, _user: CurrentUser) -> dict[str, str]:
    """Trigger download of a preset from MAST (requires network)."""
    from src.data.downloader import available_presets
    from src.data.downloader import download_preset as _dl

    presets = available_presets()
    if key not in presets:
        raise HTTPException(status_code=404, detail=f"Unknown preset '{key}'")
    try:
        from web.config import settings

        paths = _dl(key, settings.data_dir)
        return {"status": "ok", "files": str([str(p) for p in paths])}
    except Exception as exc:
        raise HTTPException(status_code=502, detail=str(exc)) from exc


@router.get("/fits", response_model=list[FitsFileInfo])
def get_fits_files(_user: CurrentUser) -> list[dict[str, object]]:
    """List all available FITS / ``.npy`` data files."""
    return list_fits_files()


@router.post("/synthetic", response_model=FitsFileInfo)
def create_synthetic(body: SyntheticRequest, _user: CurrentUser) -> dict[str, object]:
    """Generate a synthetic PSF for demo / testing."""
    path = generate_synthetic_psf(
        name=body.name,
        grid_size=body.grid_size,
        aberration_rms=body.aberration_rms,
        telescope=body.telescope,
        filter_name=body.filter_name,
    )
    return {
        "filename": path.name,
        "filepath": str(path),
        "size_bytes": path.stat().st_size,
    }

