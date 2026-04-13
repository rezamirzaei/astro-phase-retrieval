"""Reference-baseline validation against trusted instrument documentation."""

from __future__ import annotations

from dataclasses import asdict, dataclass
from typing import Any

import numpy as np


@dataclass(frozen=True, slots=True)
class InstrumentReferenceBaseline:
    """Curated baseline from instrument documentation or literature."""

    key: str
    telescope: str
    detector: str
    filter_name: str
    citation_title: str
    citation_url: str
    notes: str
    fwhm_arcsec: float | None = None
    encircled_energy_radius_arcsec: float | None = None
    encircled_energy_fraction: float | None = None
    fwhm_tolerance_arcsec: float = 0.02
    encircled_energy_tolerance: float = 0.05

    def to_dict(self) -> dict[str, Any]:
        """Return a JSON-serialisable representation."""
        return asdict(self)


_REFERENCE_BASELINES: dict[str, InstrumentReferenceBaseline] = {
    "hst-wfc3-uvis-f606w": InstrumentReferenceBaseline(
        key="hst-wfc3-uvis-f606w",
        telescope="HST",
        detector="WFC3/UVIS",
        filter_name="F606W",
        citation_title="WFC3 Instrument Handbook: UVIS Optical Performance",
        citation_url=(
            "https://hst-docs.stsci.edu/wfc3ihb/chapter-6-uvis-imaging-with-wfc3/"
            "6-6-uvis-optical-performance"
        ),
        notes=(
            "STScI handbook values for WFC3/UVIS F606W PSF width and encircled-energy "
            "behavior near the field center."
        ),
        fwhm_arcsec=0.08,
        encircled_energy_radius_arcsec=0.20,
        encircled_energy_fraction=0.87,
        fwhm_tolerance_arcsec=0.02,
        encircled_energy_tolerance=0.05,
    ),
    "hst-acs-wfc-f814w": InstrumentReferenceBaseline(
        key="hst-acs-wfc-f814w",
        telescope="HST",
        detector="ACS/WFC",
        filter_name="F814W",
        citation_title="ACS Instrument Handbook: ACS Point Spread Functions",
        citation_url="https://hst-docs.stsci.edu/acsihb/chapter-5-imaging/5-6-acs-point-spread-functions",
        notes=(
            "STScI ACS handbook aperture-correction and PSF guidance for F814W near "
            "the field center."
        ),
        fwhm_arcsec=0.105,
        encircled_energy_radius_arcsec=0.25,
        encircled_energy_fraction=0.85,
        fwhm_tolerance_arcsec=0.025,
        encircled_energy_tolerance=0.06,
    ),
    "jwst-nircam-f200w": InstrumentReferenceBaseline(
        key="jwst-nircam-f200w",
        telescope="JWST",
        detector="NIRCam",
        filter_name="F200W",
        citation_title="JWST User Documentation: NIRCam Point Spread Functions",
        citation_url=(
            "https://jwst-docs.stsci.edu/jwst-near-infrared-camera/"
            "nircam-performance/nircam-point-spread-functions"
        ),
        notes="STScI NIRCam PSF FWHM reference for the short-wavelength F200W channel.",
        fwhm_arcsec=0.068,
        fwhm_tolerance_arcsec=0.015,
    ),
    "jwst-nircam-f356w": InstrumentReferenceBaseline(
        key="jwst-nircam-f356w",
        telescope="JWST",
        detector="NIRCam",
        filter_name="F356W",
        citation_title="JWST User Documentation: NIRCam Point Spread Functions",
        citation_url=(
            "https://jwst-docs.stsci.edu/jwst-near-infrared-camera/"
            "nircam-performance/nircam-point-spread-functions"
        ),
        notes="STScI NIRCam PSF FWHM reference for the long-wavelength F356W channel.",
        fwhm_arcsec=0.103,
        fwhm_tolerance_arcsec=0.02,
    ),
}


def available_reference_baselines() -> dict[str, dict[str, Any]]:
    """Return all curated instrument baselines."""
    return {key: baseline.to_dict() for key, baseline in _REFERENCE_BASELINES.items()}


def _centroid(image: np.ndarray) -> tuple[float, float]:
    weights = np.maximum(np.nan_to_num(image, nan=0.0, posinf=0.0, neginf=0.0), 0.0)
    total = float(weights.sum())
    if total <= 0:
        centre = (image.shape[0] - 1) / 2.0
        return centre, centre
    y_idx, x_idx = np.indices(weights.shape, dtype=np.float64)
    row = float((weights * y_idx).sum() / total)
    col = float((weights * x_idx).sum() / total)
    return row, col


def compute_fwhm_arcsec(image: np.ndarray, pixel_scale_arcsec: float) -> float:
    """Estimate PSF FWHM from an azimuthally averaged radial profile."""
    row, col = _centroid(image)
    y_idx, x_idx = np.indices(image.shape, dtype=np.float64)
    radii = np.sqrt((y_idx - row) ** 2 + (x_idx - col) ** 2)
    values = np.maximum(np.nan_to_num(image, nan=0.0, posinf=0.0, neginf=0.0), 0.0)

    oversample = 4
    radial_index = np.floor(radii * oversample).astype(np.int64)
    radial_sum = np.bincount(radial_index.ravel(), weights=values.ravel())
    radial_count = np.bincount(radial_index.ravel())
    valid = radial_count > 0
    profile = np.divide(
        radial_sum[valid],
        radial_count[valid],
        out=np.zeros_like(radial_sum[valid], dtype=np.float64),
        where=radial_count[valid] > 0,
    )
    if profile.size == 0 or float(profile.max()) <= 0:
        return 0.0

    radii_px = np.flatnonzero(valid).astype(np.float64) / oversample
    profile /= float(profile.max())
    below = np.flatnonzero(profile <= 0.5)
    if below.size == 0:
        half_radius_px = radii_px[-1]
    else:
        idx = int(below[0])
        if idx == 0:
            half_radius_px = radii_px[0]
        else:
            left_r = radii_px[idx - 1]
            right_r = radii_px[idx]
            left_v = profile[idx - 1]
            right_v = profile[idx]
            if abs(right_v - left_v) < 1e-12:
                half_radius_px = right_r
            else:
                frac = (0.5 - left_v) / (right_v - left_v)
                half_radius_px = left_r + frac * (right_r - left_r)

    return float(max(0.0, 2.0 * half_radius_px * pixel_scale_arcsec))


def compute_encircled_energy_fraction(
    image: np.ndarray,
    *,
    pixel_scale_arcsec: float,
    radius_arcsec: float,
) -> float:
    """Return the encircled-energy fraction inside the requested radius."""
    row, col = _centroid(image)
    y_idx, x_idx = np.indices(image.shape, dtype=np.float64)
    radii_arcsec = np.sqrt((y_idx - row) ** 2 + (x_idx - col) ** 2) * pixel_scale_arcsec
    values = np.maximum(np.nan_to_num(image, nan=0.0, posinf=0.0, neginf=0.0), 0.0)
    total = float(values.sum())
    if total <= 0:
        return 0.0
    enclosed = float(values[radii_arcsec <= radius_arcsec].sum())
    return enclosed / total


def infer_reference_baseline(
    *,
    telescope: str,
    detector: str,
    filter_name: str,
) -> InstrumentReferenceBaseline | None:
    """Match metadata to a curated external baseline when available."""
    telescope_upper = telescope.upper()
    detector_upper = detector.upper()
    filter_upper = filter_name.upper()

    for baseline in _REFERENCE_BASELINES.values():
        if (
            baseline.telescope.upper() == telescope_upper
            and baseline.detector.upper() == detector_upper
            and baseline.filter_name.upper() == filter_upper
        ):
            return baseline
    return None


def compare_against_reference(
    *,
    observed_psf: np.ndarray,
    reconstructed_psf: np.ndarray | None,
    pixel_scale_arcsec: float,
    telescope: str,
    detector: str,
    filter_name: str,
) -> dict[str, Any]:
    """Compare observed and reconstructed PSFs against a trusted baseline."""
    baseline = infer_reference_baseline(
        telescope=telescope,
        detector=detector,
        filter_name=filter_name,
    )
    if baseline is None:
        return {}

    observed: dict[str, float] = {}
    reconstructed: dict[str, float] = {}
    deviations: dict[str, float] = {}
    summary_flags: dict[str, str] = {}

    if baseline.fwhm_arcsec is not None:
        observed_fwhm = compute_fwhm_arcsec(observed_psf, pixel_scale_arcsec)
        observed["fwhm_arcsec"] = observed_fwhm
        deviations["observed_fwhm_error_arcsec"] = observed_fwhm - baseline.fwhm_arcsec
        if reconstructed_psf is not None:
            reconstructed_fwhm = compute_fwhm_arcsec(reconstructed_psf, pixel_scale_arcsec)
            reconstructed["fwhm_arcsec"] = reconstructed_fwhm
            deviations["reconstructed_fwhm_error_arcsec"] = (
                reconstructed_fwhm - baseline.fwhm_arcsec
            )
            summary_flags["fwhm_agreement"] = (
                "strong"
                if abs(reconstructed_fwhm - baseline.fwhm_arcsec) <= baseline.fwhm_tolerance_arcsec
                else "weak"
            )

    if (
        baseline.encircled_energy_radius_arcsec is not None
        and baseline.encircled_energy_fraction is not None
    ):
        observed_ee = compute_encircled_energy_fraction(
            observed_psf,
            pixel_scale_arcsec=pixel_scale_arcsec,
            radius_arcsec=baseline.encircled_energy_radius_arcsec,
        )
        observed["encircled_energy_fraction"] = observed_ee
        deviations["observed_encircled_energy_error"] = (
            observed_ee - baseline.encircled_energy_fraction
        )
        if reconstructed_psf is not None:
            reconstructed_ee = compute_encircled_energy_fraction(
                reconstructed_psf,
                pixel_scale_arcsec=pixel_scale_arcsec,
                radius_arcsec=baseline.encircled_energy_radius_arcsec,
            )
            reconstructed["encircled_energy_fraction"] = reconstructed_ee
            deviations["reconstructed_encircled_energy_error"] = (
                reconstructed_ee - baseline.encircled_energy_fraction
            )
            summary_flags["encircled_energy_agreement"] = (
                "strong"
                if abs(reconstructed_ee - baseline.encircled_energy_fraction)
                <= baseline.encircled_energy_tolerance
                else "weak"
            )

    return {
        "baseline": baseline.to_dict(),
        "observed": observed,
        "reconstructed": reconstructed,
        "deviations": deviations,
        "summary": summary_flags,
    }
