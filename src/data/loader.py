"""Load FITS images and extract clean PSF cutouts from real HST/JWST data."""

from __future__ import annotations

import hashlib
import logging
from dataclasses import dataclass
from pathlib import Path

import numpy as np
from astropy.io import fits
from scipy.ndimage import median_filter  # type: ignore[import-untyped]
from scipy.ndimage import shift as ndimage_shift

from src.models.config import DataConfig, PupilConfig, TelescopeType
from src.models.optics import PSFData

logger = logging.getLogger(__name__)

# Guard limit: refuse to open FITS files larger than 2 GiB to prevent OOM
_MAX_FITS_BYTES: int = 2 * 1024**3


@dataclass(frozen=True, slots=True)
class _CalibrationPreset:
    key: str
    telescope: TelescopeType
    detector: str
    pixel_scale_arcsec: float
    default_centroid_method: str
    default_saturation_percentile: float
    default_hot_pixel_sigma: float
    dq_bad_values: tuple[int, ...]
    header_keys: tuple[str, ...]


_CALIBRATION_PRESETS: dict[str, _CalibrationPreset] = {
    "hst-wfc3-uvis": _CalibrationPreset(
        key="hst-wfc3-uvis",
        telescope=TelescopeType.HST,
        detector="WFC3/UVIS",
        pixel_scale_arcsec=0.0395,
        default_centroid_method="moments",
        default_saturation_percentile=99.97,
        default_hot_pixel_sigma=7.0,
        dq_bad_values=(1, 4, 16, 32, 64, 256, 512, 1024, 2048, 4096),
        header_keys=(
            "TELESCOP",
            "INSTRUME",
            "DETECTOR",
            "FILTER",
            "FILTER1",
            "FILTER2",
            "ROOTNAME",
            "TARGNAME",
            "EXPTIME",
            "DATE-OBS",
            "BUNIT",
            "APERTURE",
            "SUBARRAY",
            "PHOTFLAM",
            "PHOTPLAM",
            "PCTECORR",
            "DQICORR",
            "DARKCORR",
            "FLATCORR",
        ),
    ),
    "hst-acs-wfc": _CalibrationPreset(
        key="hst-acs-wfc",
        telescope=TelescopeType.HST,
        detector="ACS/WFC",
        pixel_scale_arcsec=0.05,
        default_centroid_method="moments",
        default_saturation_percentile=99.98,
        default_hot_pixel_sigma=7.0,
        dq_bad_values=(1, 4, 16, 32, 64, 256, 512, 1024, 2048, 4096),
        header_keys=(
            "TELESCOP",
            "INSTRUME",
            "DETECTOR",
            "FILTER",
            "ROOTNAME",
            "TARGNAME",
            "EXPTIME",
            "DATE-OBS",
            "BUNIT",
            "APERTURE",
            "SUBARRAY",
            "PFLTFILE",
            "DFLTFILE",
            "FLSHCORR",
            "DARKCORR",
            "FLATCORR",
        ),
    ),
    "jwst-nircam-sw": _CalibrationPreset(
        key="jwst-nircam-sw",
        telescope=TelescopeType.JWST,
        detector="NIRCam",
        pixel_scale_arcsec=0.031,
        default_centroid_method="quadratic_peak",
        default_saturation_percentile=99.9,
        default_hot_pixel_sigma=6.0,
        dq_bad_values=(1, 2, 4, 8, 16, 32, 64, 128, 256, 512, 1024, 2048),
        header_keys=(
            "TELESCOP",
            "INSTRUME",
            "DETECTOR",
            "FILTER",
            "OBS_ID",
            "TARGNAME",
            "EXPTIME",
            "DATE-OBS",
            "BUNIT",
            "SUBARRAY",
            "PUPIL",
            "CHANNEL",
            "MODULE",
            "CRDS_CTX",
            "CAL_VER",
        ),
    ),
    "jwst-nircam-lw": _CalibrationPreset(
        key="jwst-nircam-lw",
        telescope=TelescopeType.JWST,
        detector="NIRCam",
        pixel_scale_arcsec=0.063,
        default_centroid_method="quadratic_peak",
        default_saturation_percentile=99.9,
        default_hot_pixel_sigma=6.0,
        dq_bad_values=(1, 2, 4, 8, 16, 32, 64, 128, 256, 512, 1024, 2048),
        header_keys=(
            "TELESCOP",
            "INSTRUME",
            "DETECTOR",
            "FILTER",
            "OBS_ID",
            "TARGNAME",
            "EXPTIME",
            "DATE-OBS",
            "BUNIT",
            "SUBARRAY",
            "PUPIL",
            "CHANNEL",
            "MODULE",
            "CRDS_CTX",
            "CAL_VER",
        ),
    ),
}


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


def load_fits_dq_mask(
    filepath: Path,
    image_shape: tuple[int, int],
    *,
    bad_values: tuple[int, ...] | None = None,
) -> np.ndarray | None:
    """Load a FITS data-quality mask when a matching DQ extension exists."""
    with fits.open(filepath) as hdul:
        for hdu in hdul:
            extname = str(hdu.header.get("EXTNAME", "")).upper()
            if "DQ" not in extname or hdu.data is None:
                continue
            dq = np.asarray(hdu.data)
            if dq.ndim == 2 and dq.shape == image_shape:
                if bad_values is None:
                    return np.asarray(dq != 0, dtype=bool)
                mask = np.zeros_like(dq, dtype=bool)
                dq_i64 = dq.astype(np.int64, copy=False)
                for value in bad_values:
                    mask |= (dq_i64 & int(value)) != 0
                return mask
    return None


def load_fits_error_image(filepath: Path, image_shape: tuple[int, int]) -> np.ndarray | None:
    """Load a FITS uncertainty image when an ERR extension exists."""
    with fits.open(filepath) as hdul:
        for hdu in hdul:
            extname = str(hdu.header.get("EXTNAME", "")).upper()
            if "ERR" not in extname or hdu.data is None:
                continue
            err = np.asarray(hdu.data, dtype=np.float64)
            if err.ndim == 2 and err.shape == image_shape:
                return err
    return None


def estimate_background_level(image: np.ndarray, *, percentile: float = 10.0) -> float:
    """Estimate a robust background level from image-edge pixels."""
    finite = np.isfinite(image)
    if not np.any(finite):
        return 0.0

    nrows, ncols = image.shape
    border = max(2, min(min(nrows, ncols) // 8, 32))
    edge_mask = np.zeros_like(image, dtype=bool)
    edge_mask[:border, :] = True
    edge_mask[-border:, :] = True
    edge_mask[:, :border] = True
    edge_mask[:, -border:] = True

    edge_values = image[edge_mask & finite]
    if edge_values.size == 0:
        edge_values = image[finite]
    return float(np.percentile(edge_values, percentile))


def refine_source_centroid(
    image: np.ndarray,
    center: tuple[int, int],
    *,
    window_radius: int = 12,
    method: str = "moments",
) -> tuple[float, float]:
    """Refine an integer source location to a subpixel flux centroid."""
    r, c = center
    nr, nc = image.shape
    r0 = max(r - window_radius, 0)
    r1 = min(r + window_radius + 1, nr)
    c0 = max(c - window_radius, 0)
    c1 = min(c + window_radius + 1, nc)

    local = np.nan_to_num(image[r0:r1, c0:c1], nan=0.0, posinf=0.0, neginf=0.0)
    weights = np.maximum(local, 0.0)
    total = float(weights.sum())
    if total <= 0:
        return float(r), float(c)

    if method == "quadratic_peak":
        peak_r, peak_c = np.unravel_index(np.argmax(weights), weights.shape)

        def _quadratic_offset(left: float, mid: float, right: float) -> float:
            denom = left - 2.0 * mid + right
            if abs(denom) < 1e-12:
                return 0.0
            return 0.5 * (left - right) / denom

        row_offset = 0.0
        col_offset = 0.0
        if 0 < peak_r < weights.shape[0] - 1:
            row_offset = _quadratic_offset(
                float(weights[peak_r - 1, peak_c]),
                float(weights[peak_r, peak_c]),
                float(weights[peak_r + 1, peak_c]),
            )
        if 0 < peak_c < weights.shape[1] - 1:
            col_offset = _quadratic_offset(
                float(weights[peak_r, peak_c - 1]),
                float(weights[peak_r, peak_c]),
                float(weights[peak_r, peak_c + 1]),
            )
        return float(r0 + peak_r + row_offset), float(c0 + peak_c + col_offset)

    y_idx, x_idx = np.indices(local.shape, dtype=np.float64)
    row = float((weights * (y_idx + r0)).sum() / total)
    col = float((weights * (x_idx + c0)).sum() / total)
    return row, col


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
    """Background subtraction using a robust edge-based percentile estimate."""
    bg = estimate_background_level(image, percentile=percentile)
    result = image - bg
    result[result < 0] = 0.0
    return result  # type: ignore[no-any-return]


def recenter_psf(psf: np.ndarray) -> tuple[np.ndarray, tuple[float, float]]:
    """Shift a PSF cutout so its flux centroid lies at the array centre."""
    weights = np.maximum(np.nan_to_num(psf, nan=0.0, posinf=0.0, neginf=0.0), 0.0)
    total = float(weights.sum())
    if total <= 0:
        return psf.copy(), (0.0, 0.0)

    y_idx, x_idx = np.indices(weights.shape, dtype=np.float64)
    row = float((weights * y_idx).sum() / total)
    col = float((weights * x_idx).sum() / total)
    target = (psf.shape[0] - 1) / 2.0
    shift = (target - row, target - col)

    if abs(shift[0]) < 1e-12 and abs(shift[1]) < 1e-12:
        return psf.copy(), shift

    shifted = ndimage_shift(
        psf,
        shift=shift,
        order=1,
        mode="constant",
        cval=0.0,
        prefilter=False,
    )
    return shifted.astype(np.float64), shift


def build_quality_mask(
    image: np.ndarray,
    *,
    saturation_percentile: float = 99.95,
    hot_pixel_sigma: float = 8.0,
) -> tuple[np.ndarray, dict[str, float]]:
    """Build a mask for non-finite, saturated, and isolated hot pixels."""
    finite = np.isfinite(image)
    mask = ~finite
    info: dict[str, float] = {
        "nonfinite_pixels": float(mask.sum()),
        "saturated_pixels": 0.0,
        "hot_pixels": 0.0,
        "masked_fraction": 0.0,
    }
    if not np.any(finite):
        info["masked_fraction"] = 1.0
        return mask, info

    finite_values = image[finite]
    saturation_level = float(np.percentile(finite_values, saturation_percentile))
    if np.isfinite(saturation_level):
        saturated = image >= saturation_level
        info["saturated_pixels"] = float(np.count_nonzero(saturated))
        if np.count_nonzero(saturated) >= 4:
            mask |= saturated

    local_median = median_filter(np.nan_to_num(image, nan=0.0), size=3, mode="nearest")
    residual = np.abs(np.nan_to_num(image, nan=0.0) - local_median)
    mad = float(np.median(residual[finite]))
    robust_scale = max(1.4826 * mad, 1e-12)
    hot_pixels = finite & (residual > hot_pixel_sigma * robust_scale)
    hot_pixels &= residual > max(5.0 * robust_scale, 0.0)
    mask |= hot_pixels
    info["hot_pixels"] = float(np.count_nonzero(hot_pixels))
    info["masked_fraction"] = float(np.count_nonzero(mask) / mask.size)
    info["saturation_level"] = saturation_level
    info["hot_pixel_sigma"] = float(hot_pixel_sigma)
    return mask, info


def apply_quality_mask(image: np.ndarray, mask: np.ndarray) -> np.ndarray:
    """Replace masked pixels with local-median values."""
    if not np.any(mask):
        return np.asarray(image.copy(), dtype=np.float64)
    filled = np.nan_to_num(image, nan=0.0, posinf=0.0, neginf=0.0).astype(np.float64, copy=True)
    local_median = median_filter(filled, size=3, mode="nearest")
    filled[mask] = local_median[mask]
    return np.asarray(filled, dtype=np.float64)


def normalise_psf(psf: np.ndarray) -> np.ndarray:
    """Normalise a PSF so that it sums to 1."""
    total = psf.sum()
    if total > 0:
        return np.asarray(psf / total, dtype=np.float64)
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
    filter_name = _header_filter(header, data_cfg.filter_name)
    image_shape = (int(image.shape[0]), int(image.shape[1]))
    preset = _resolve_calibration_preset(header, data_cfg, filter_name)
    centroid_method = _effective_data_setting(
        current_value=data_cfg.centroid_method,
        default_value=DataConfig().centroid_method,
        preset_value=preset.default_centroid_method,
    )
    saturation_percentile = _effective_data_setting(
        current_value=data_cfg.saturation_percentile,
        default_value=DataConfig().saturation_percentile,
        preset_value=preset.default_saturation_percentile,
    )
    hot_pixel_sigma = _effective_data_setting(
        current_value=data_cfg.hot_pixel_sigma,
        default_value=DataConfig().hot_pixel_sigma,
        preset_value=preset.default_hot_pixel_sigma,
    )

    # Background subtraction
    background_level = estimate_background_level(image, percentile=data_cfg.background_percentile)
    image = subtract_background(image, percentile=data_cfg.background_percentile)
    image = np.nan_to_num(image, nan=0.0, posinf=0.0, neginf=0.0)

    # Find brightest star
    center = find_brightest_source(image, border=data_cfg.source_detection_border)
    refined_center = refine_source_centroid(
        image,
        center,
        window_radius=max(2, data_cfg.centroid_window_size // 2),
        method=str(centroid_method),
    )
    logger.info(
        "Brightest source at row=%d, col=%d; refined centroid row=%.3f, col=%.3f",
        center[0],
        center[1],
        refined_center[0],
        refined_center[1],
    )

    # Extract cutout
    half = data_cfg.cutout_size // 2
    cutout = extract_psf_cutout(
        image,
        (int(round(refined_center[0])), int(round(refined_center[1]))),
        half,
    )

    quality_mask, quality_info = build_quality_mask(
        cutout,
        saturation_percentile=float(saturation_percentile),
        hot_pixel_sigma=float(hot_pixel_sigma),
    )
    dq_cutout = None
    err_cutout = None
    if data_cfg.use_dq_mask:
        dq_mask = load_fits_dq_mask(filepath, image_shape, bad_values=preset.dq_bad_values)
        if dq_mask is not None:
            dq_cutout = (
                extract_psf_cutout(
                    dq_mask.astype(np.float64),
                    (int(round(refined_center[0])), int(round(refined_center[1]))),
                    half,
                )
                > 0
            )
            quality_mask |= dq_cutout
    err_image = load_fits_error_image(filepath, image_shape)
    if err_image is not None:
        err_cutout = extract_psf_cutout(
            err_image,
            (int(round(refined_center[0])), int(round(refined_center[1]))),
            half,
        )
    cutout = apply_quality_mask(cutout, quality_mask)

    recenter_shift = (0.0, 0.0)
    if data_cfg.recenter_psf:
        cutout, recenter_shift = recenter_psf(cutout)

    # Normalise
    cutout = normalise_psf(cutout)

    try:
        file_size = filepath.stat().st_size
    except OSError:
        file_size = 0

    header_subset = {
        key: str(header[key])
        for key in preset.header_keys
        if key in header and header[key] not in (None, "")
    }

    return PSFData(
        image=cutout,
        pixel_scale_arcsec=preset.pixel_scale_arcsec,
        wavelength_m=_header_wavelength(header, pupil_cfg.wavelength_m),
        filter_name=filter_name,
        telescope=str(preset.telescope.value),
        obs_id=header.get("ROOTNAME", header.get("OBS_ID", filepath.stem)),
        metadata={
            "source_kind": "fits",
            "source_path": str(filepath),
            "source_filename": filepath.name,
            "source_sha256": _file_sha256(filepath),
            "file_size_bytes": file_size,
            "detector": preset.detector,
            "background_level": background_level,
            "background_percentile": float(data_cfg.background_percentile),
            "brightest_source_center_rowcol": [int(center[0]), int(center[1])],
            "refined_source_centroid_rowcol": [float(refined_center[0]), float(refined_center[1])],
            "cutout_size": int(data_cfg.cutout_size),
            "prepared_shape": [int(cutout.shape[0]), int(cutout.shape[1])],
            "centroid_window_size": int(data_cfg.centroid_window_size),
            "centroid_method": str(centroid_method),
            "recenter_psf": bool(data_cfg.recenter_psf),
            "recenter_shift_rowcol": [float(recenter_shift[0]), float(recenter_shift[1])],
            "quality_mask": {
                "masked_pixels": int(np.count_nonzero(quality_mask)),
                "masked_fraction": float(quality_info["masked_fraction"]),
                "nonfinite_pixels": int(quality_info["nonfinite_pixels"]),
                "saturated_pixels": int(quality_info["saturated_pixels"]),
                "hot_pixels": int(quality_info["hot_pixels"]),
                "dq_pixels": int(np.count_nonzero(dq_cutout)) if dq_cutout is not None else 0,
                "saturation_level": float(quality_info.get("saturation_level", 0.0)),
                "hot_pixel_sigma": float(quality_info.get("hot_pixel_sigma", hot_pixel_sigma)),
            },
            "calibration": {
                "preset": preset.key,
                "dq_bad_values": list(preset.dq_bad_values),
                "error_extension_present": err_cutout is not None,
                "error_mean": float(np.mean(err_cutout)) if err_cutout is not None else None,
                "error_std": float(np.std(err_cutout)) if err_cutout is not None else None,
                "header_keys_recorded": list(header_subset.keys()),
                "suggested_pupil": {
                    "telescope": preset.telescope.value,
                    "pixel_scale_arcsec": preset.pixel_scale_arcsec,
                    "wavelength_m": _header_wavelength(header, pupil_cfg.wavelength_m),
                },
            },
            "preprocessing": [
                f"calibration_preset_{preset.key}",
                f"background_subtraction_edge_percentile_{data_cfg.background_percentile:g}",
                "nan_to_num",
                "brightest_source_detection",
                "local_flux_centroid_refinement",
                f"local_flux_centroid_refinement_{centroid_method}",
                "brightest_source_cutout",
                "dq_mask_repair" if dq_cutout is not None else "dq_mask_unavailable",
                "quality_mask_repair",
                (
                    "subpixel_recentering"
                    if data_cfg.recenter_psf
                    else "subpixel_recentering_disabled"
                ),
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


def _resolve_calibration_preset(
    header: dict,
    data_cfg: DataConfig,
    filter_name: str,
) -> _CalibrationPreset:
    if data_cfg.calibration_preset != "auto":
        preset = _CALIBRATION_PRESETS.get(data_cfg.calibration_preset)
        if preset is not None:
            return preset

    detector = str(header.get("DETECTOR", data_cfg.detector)).upper()
    instrument = str(header.get("INSTRUME", "")).upper()
    wavelength = _header_wavelength(header, 0.0)
    filter_upper = filter_name.upper()

    if "ACS" in instrument or detector == "ACS/WFC":
        return _CALIBRATION_PRESETS["hst-acs-wfc"]
    if detector == "NIRCAM" or instrument == "NIRCAM" or filter_upper in {"F200W", "F356W"}:
        key = "jwst-nircam-lw" if wavelength >= 2.4e-6 else "jwst-nircam-sw"
        return _CALIBRATION_PRESETS[key]
    return _CALIBRATION_PRESETS["hst-wfc3-uvis"]


def _effective_data_setting(
    *,
    current_value: float | str,
    default_value: float | str,
    preset_value: float | str,
) -> float | str:
    if current_value == default_value:
        return preset_value
    return current_value


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
