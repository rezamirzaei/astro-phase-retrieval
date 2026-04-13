"""Load FITS images and extract clean PSF cutouts from real HST/JWST data."""

from __future__ import annotations

import hashlib
import logging
from pathlib import Path

import numpy as np
from astropy.io import fits

from src.models.config import DataConfig, PupilConfig
from src.models.optics import PSFData

logger = logging.getLogger(__name__)

# Guard limit: refuse to open FITS files larger than 2 GiB to prevent OOM
_MAX_FITS_BYTES: int = 2 * 1024**3


def _file_sha256(filepath: Path) -> str:
    """Return a SHA-256 checksum for *filepath*."""
    digest = hashlib.sha256()
    with filepath.open("rb") as handle:
        for chunk in iter(lambda: handle.read(65536), b""):
            digest.update(chunk)
    return digest.hexdigest()


def load_fits_image(filepath: Path, *, ext: int | str = 1) -> tuple[np.ndarray, dict]:
    """Read a 2-D image array and header from a FITS file.

    Parameters
    ----------
    filepath : Path
        Path to the FITS file.
    ext : int | str
        FITS extension to read (default 1 = first science extension).

    Returns
    -------
    data : ndarray
        2-D float64 image.
    header : dict
        Merged FITS header (primary + science extension) as a plain dict.
    """
    filepath = Path(filepath)

    # Guard: refuse to open suspiciously large files to prevent OOM
    try:
        file_size = filepath.stat().st_size
    except OSError:
        file_size = 0
    if file_size > _MAX_FITS_BYTES:
        raise ValueError(
            f"FITS file is too large ({file_size / 1024**3:.1f} GiB > 2 GiB limit): {filepath}"
        )

    with fits.open(filepath) as hdul:
        # Always capture the primary header for metadata (FILTER, INSTRUME, etc.)
        primary_header = dict(hdul[0].header)

        # Try requested extension, fall back to first image extension
        try:
            data = hdul[ext].data
            header = dict(hdul[ext].header)
        except (KeyError, IndexError):
            for h in hdul:
                if h.data is not None and h.data.ndim == 2:
                    data = h.data
                    header = dict(h.header)
                    break
            else:
                raise ValueError(f"No 2-D image extension found in {filepath}")

    # Merge: primary header values fill in missing keys in the science header
    merged = {**primary_header, **header}
    # But keep primary's FILTER/INSTRUME/DETECTOR since science ext may lack them
    for key in (
        "FILTER",
        "FILTER1",
        "FILTER2",
        "INSTRUME",
        "DETECTOR",
        "ROOTNAME",
        "TARGNAME",
        "EXPTIME",
        "DATE-OBS",
    ):
        if key in primary_header and primary_header[key]:
            merged[key] = primary_header[key]

    return data.astype(np.float64), merged


def find_brightest_source(image: np.ndarray, *, border: int = 50) -> tuple[int, int]:
    """Locate the brightest pixel away from the detector edges.

    Parameters
    ----------
    image : ndarray
        2-D image.
    border : int
        Pixel margin to exclude (avoids edge artifacts).

    Returns
    -------
    (row, col) of the brightest pixel.
    """
    sub = image.copy()
    sub[:border, :] = 0
    sub[-border:, :] = 0
    sub[:, :border] = 0
    sub[:, -border:] = 0
    # Mask NaN/inf
    sub = np.nan_to_num(sub, nan=0.0, posinf=0.0, neginf=0.0)
    idx = np.unravel_index(np.argmax(sub), sub.shape)
    return int(idx[0]), int(idx[1])


def extract_psf_cutout(
    image: np.ndarray,
    center: tuple[int, int],
    half_width: int,
) -> np.ndarray:
    """Cut a square PSF patch centred on *center*.

    Parameters
    ----------
    image : ndarray
        Full detector image.
    center : tuple[int, int]
        (row, col) of the star centre.
    half_width : int
        Half-side of the cutout square.

    Returns
    -------
    ndarray
        Square cutout of size (2*half_width, 2*half_width).
    """
    r, c = center
    size = half_width
    nr, nc = image.shape

    r0 = max(r - size, 0)
    r1 = min(r + size, nr)
    c0 = max(c - size, 0)
    c1 = min(c + size, nc)

    cutout = image[r0:r1, c0:c1].copy()

    # Pad if the cutout is smaller than expected (star near edge)
    expected = 2 * size
    if cutout.shape[0] != expected or cutout.shape[1] != expected:
        padded = np.zeros((expected, expected), dtype=np.float64)
        padded[: cutout.shape[0], : cutout.shape[1]] = cutout
        cutout = padded

    return cutout


def subtract_background(image: np.ndarray, *, percentile: float = 10.0) -> np.ndarray:
    """Simple background subtraction using a low percentile of the image."""
    bg = np.percentile(image[np.isfinite(image)], percentile)
    result = image - bg
    result[result < 0] = 0.0
    return result  # type: ignore[no-any-return]


def normalise_psf(psf: np.ndarray) -> np.ndarray:
    """Normalise a PSF so that it sums to 1."""
    total = psf.sum()
    if total > 0:
        return psf / total  # type: ignore[no-any-return]
    return psf


def load_psf_from_fits(
    filepath: Path,
    data_cfg: DataConfig,
    pupil_cfg: PupilConfig,
) -> PSFData:
    """End-to-end: load a FITS file, find the star, extract & clean the PSF.

    Parameters
    ----------
    filepath : Path
        Path to a calibrated FITS file.
    data_cfg : DataConfig
        Data configuration.
    pupil_cfg : PupilConfig
        Pupil/optics configuration (for wavelength, pixel scale).

    Returns
    -------
    PSFData
        Validated PSF data model.
    """
    image, header = load_fits_image(filepath)
    logger.info("Loaded %s — shape %s", filepath.name, image.shape)

    # Background subtraction
    image = subtract_background(image)
    image = np.nan_to_num(image, nan=0.0, posinf=0.0, neginf=0.0)

    # Find brightest star
    center = find_brightest_source(image)
    logger.info("Brightest source at row=%d, col=%d", *center)

    # Extract cutout
    half = data_cfg.cutout_size // 2
    cutout = extract_psf_cutout(image, center, half)

    # Normalise
    cutout = normalise_psf(cutout)

    try:
        file_size = filepath.stat().st_size
    except OSError:
        file_size = 0

    header_subset = {
        key: str(header[key])
        for key in (
            "ROOTNAME",
            "FILTER",
            "FILTER1",
            "FILTER2",
            "INSTRUME",
            "DETECTOR",
            "TARGNAME",
            "EXPTIME",
            "DATE-OBS",
        )
        if key in header and header[key] not in (None, "")
    }

    return PSFData(
        image=cutout,
        pixel_scale_arcsec=pupil_cfg.pixel_scale_arcsec,
        wavelength_m=_header_wavelength(header, pupil_cfg.wavelength_m),
        filter_name=_header_filter(header, data_cfg.filter_name),
        telescope=str(pupil_cfg.telescope.value),
        obs_id=header.get("ROOTNAME", filepath.stem),
        metadata={
            "source_kind": "fits",
            "source_path": str(filepath),
            "source_filename": filepath.name,
            "source_sha256": _file_sha256(filepath),
            "file_size_bytes": file_size,
            "brightest_source_center_rowcol": [int(center[0]), int(center[1])],
            "cutout_size": int(data_cfg.cutout_size),
            "prepared_shape": [int(cutout.shape[0]), int(cutout.shape[1])],
            "preprocessing": [
                "background_subtraction_percentile_10",
                "nan_to_num",
                "brightest_source_cutout",
                "unit_sum_normalisation",
            ],
            "header": header_subset,
        },
    )


def _header_filter(header: dict, fallback: str) -> str:
    """Extract the actual filter name from a FITS header."""
    for key in ("FILTER", "FILTER1", "FILTER2"):
        val = header.get(key, "")
        if val and "CLEAR" not in str(val).upper():
            return str(val).strip()
    return fallback


def _header_wavelength(header: dict, fallback: float) -> float:
    """Look up wavelength from the FITS header filter name."""
    from src.data.downloader import FILTER_WAVELENGTH_M

    filt = _header_filter(header, "")
    return FILTER_WAVELENGTH_M.get(filt, fallback)


def prepare_psf_for_retrieval(
    psf: PSFData,
    target_size: int,
) -> np.ndarray:
    """Resize / pad the PSF image to the algorithm's working grid size.

    Parameters
    ----------
    psf : PSFData
        Input PSF data.
    target_size : int
        Desired grid side length (must be power of 2).

    Returns
    -------
    ndarray
        Square PSF image of shape (target_size, target_size), normalised.
    """
    img = psf.image
    current = img.shape[0]

    if current == target_size:
        return img.copy()

    if current < target_size:
        # Zero-pad
        pad = (target_size - current) // 2
        result = np.zeros((target_size, target_size), dtype=np.float64)
        result[pad : pad + current, pad : pad + current] = img
    else:
        # Centre-crop
        crop = (current - target_size) // 2
        result = img[crop : crop + target_size, crop : crop + target_size].copy()

    total = result.sum()
    if total > 0:
        result /= total
    return result
