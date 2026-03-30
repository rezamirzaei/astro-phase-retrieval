"""Download real HST / JWST stellar PSF observations from the MAST archive.

Uses astroquery.mast to search for and retrieve calibrated FITS files of
bright, isolated standard stars — the gold-standard data for phase retrieval.
"""

from __future__ import annotations

import logging
from pathlib import Path

from astroquery.mast import Observations

from src.models.config import DataConfig

logger = logging.getLogger(__name__)


# ---------------------------------------------------------------------------
# Curated observation list – known-good stellar PSF data
# ---------------------------------------------------------------------------

_CURATED_OBS: dict[str, dict] = {
    # HST / WFC3-UVIS: white dwarf standard star GRW+70D5824
    "hst-wfc3-uvis-f606w": dict(
        target_name="GRW+70D5824",
        instrument_name="WFC3/UVIS",
        filters="F606W",
        obs_collection="HST",
        dataproduct_type="image",
        calib_level=2,
        description="HST WFC3/UVIS F606W — WD standard GRW+70D5824",
    ),
    # HST / WFC3-UVIS: near-infrared
    "hst-wfc3-uvis-f814w": dict(
        target_name="GRW+70D5824",
        instrument_name="WFC3/UVIS",
        filters="F814W",
        obs_collection="HST",
        calib_level=2,
        dataproduct_type="image",
        description="HST WFC3/UVIS F814W — WD standard GRW+70D5824",
    ),
    # HST / WFC3-UVIS: blue filter
    "hst-wfc3-uvis-f438w": dict(
        target_name="GRW+70D5824",
        instrument_name="WFC3/UVIS",
        filters="F438W",
        obs_collection="HST",
        calib_level=2,
        dataproduct_type="image",
        description="HST WFC3/UVIS F438W — WD standard GRW+70D5824",
    ),
    # HST / WFC3-UVIS: ultraviolet
    "hst-wfc3-uvis-f275w": dict(
        target_name="GRW+70D5824",
        instrument_name="WFC3/UVIS",
        filters="F275W",
        obs_collection="HST",
        calib_level=2,
        dataproduct_type="image",
        description="HST WFC3/UVIS F275W — WD standard GRW+70D5824",
    ),
    # HST / ACS-WFC
    "hst-acs-wfc-f606w": dict(
        target_name="GRW+70D5824",
        instrument_name="ACS/WFC",
        filters="F606W",
        obs_collection="HST",
        calib_level=2,
        dataproduct_type="image",
        description="HST ACS/WFC F606W — WD standard GRW+70D5824",
    ),
    # HST / ACS-WFC: red
    "hst-acs-wfc-f814w": dict(
        target_name="GRW+70D5824",
        instrument_name="ACS/WFC",
        filters="F814W",
        obs_collection="HST",
        calib_level=2,
        dataproduct_type="image",
        description="HST ACS/WFC F814W — WD standard GRW+70D5824",
    ),
}


# Wavelength lookup for common HST filters (metres)
FILTER_WAVELENGTH_M: dict[str, float] = {
    "F200LP": 500e-9,
    "F218W": 218e-9,
    "F225W": 225e-9,
    "F275W": 275e-9,
    "F336W": 336e-9,
    "F390W": 390e-9,
    "F438W": 438e-9,
    "F475W": 475e-9,
    "F555W": 555e-9,
    "F606W": 606e-9,
    "F625W": 625e-9,
    "F775W": 775e-9,
    "F814W": 814e-9,
    "F850LP": 905e-9,
}


def search_and_download(
    cfg: DataConfig,
    *,
    max_products: int = 1,
) -> list[Path]:
    """Search MAST for a curated observation and download the calibrated FITS file.

    Parameters
    ----------
    cfg : DataConfig
        Data configuration specifying detector, filter, etc.
    max_products : int
        Maximum number of data products to download.

    Returns
    -------
    list[Path]
        Paths to the downloaded FITS files.
    """
    output_dir = Path(cfg.data_dir)
    output_dir.mkdir(parents=True, exist_ok=True)

    # Build a curated-key guess
    key_guess = f"hst-{cfg.detector.lower().replace('/', '-')}-{cfg.filter_name.lower()}"
    curated = _CURATED_OBS.get(key_guess)

    if curated is not None:
        logger.info("Using curated observation preset: %s — %s", key_guess, curated["description"])
        obs_table = Observations.query_criteria(
            target_name=curated["target_name"],
            instrument_name=curated["instrument_name"],
            filters=curated["filters"],
            obs_collection=curated["obs_collection"],
            dataproduct_type=curated["dataproduct_type"],
        )
    else:
        logger.info("No curated preset for '%s'; running general MAST query.", key_guess)
        obs_table = Observations.query_criteria(
            instrument_name=cfg.detector,
            filters=cfg.filter_name,
            obs_collection="HST",
            dataproduct_type="image",
        )

    if len(obs_table) == 0:
        raise RuntimeError(
            f"No observations found for {cfg.detector} / {cfg.filter_name}. "
            "Check your DataConfig or internet connection."
        )

    # Filter out HAP skycell reprocessed products — keep individual exposures
    keep_mask = [("skycell" not in str(row["obs_id"])) for row in obs_table]
    obs_table = obs_table[keep_mask]

    if len(obs_table) == 0:
        raise RuntimeError("All results were HAP skycell products; no individual exposures found.")

    logger.info("Found %d matching observations after filtering.", len(obs_table))

    # Sort by exposure time, prefer moderate exposures (2–30 s) for good SNR
    # without saturation
    obs_table.sort("t_exptime")
    moderate = [r for r in obs_table if 1.0 <= float(r["t_exptime"]) <= 30.0]
    if moderate:
        import astropy.table
        selected = astropy.table.Table(rows=[moderate[0]], names=obs_table.colnames)
    else:
        selected = obs_table[:1]

    logger.info(
        "Selected observation: %s  (target=%s, exptime=%.1fs)",
        selected["obs_id"][0],
        selected["target_name"][0],
        selected["t_exptime"][0],
    )

    # Get associated data products (we want _flt.fits or _flc.fits)
    products = Observations.get_product_list(selected)
    # Filter for calibrated individual exposures
    filtered = Observations.filter_products(
        products,
        productSubGroupDescription=["FLT", "FLC", "CAL"],
        extension="fits",
    )

    if len(filtered) == 0:
        # Fallback: take any FITS product
        filtered = Observations.filter_products(products, extension="fits")

    if len(filtered) == 0:
        raise RuntimeError("No suitable FITS products found for the selected observation.")

    filtered = filtered[:max_products]

    # Download
    manifest = Observations.download_products(
        filtered,
        download_dir=str(output_dir),
    )
    paths = [Path(row["Local Path"]) for row in manifest if row["Status"] == "COMPLETE"]
    logger.info("Downloaded %d file(s) to %s", len(paths), output_dir)
    return paths


def list_cached_fits(data_dir: Path) -> list[Path]:
    """Return all FITS files already present in *data_dir*."""
    if not data_dir.exists():
        return []
    return sorted(data_dir.rglob("*.fits"))


def available_presets() -> dict[str, str]:
    """Return a dict of ``{preset_key: description}`` for all curated observations."""
    return {k: v["description"] for k, v in _CURATED_OBS.items()}


def download_preset(
    key: str,
    data_dir: Path,
    *,
    max_products: int = 1,
) -> list[Path]:
    """Download a single curated observation preset by its key.

    Parameters
    ----------
    key : str
        One of the keys from :func:`available_presets`.
    data_dir : Path
        Root download directory.
    max_products : int
        Maximum number of FITS products to fetch.

    Returns
    -------
    list[Path]
        Paths to downloaded FITS files.
    """
    curated = _CURATED_OBS.get(key)
    if curated is None:
        raise KeyError(f"Unknown preset '{key}'. Available: {list(_CURATED_OBS)}")

    data_dir = Path(data_dir)
    data_dir.mkdir(parents=True, exist_ok=True)

    logger.info("Downloading preset '%s': %s", key, curated["description"])

    obs_table = Observations.query_criteria(
        target_name=curated["target_name"],
        instrument_name=curated["instrument_name"],
        filters=curated["filters"],
        obs_collection=curated["obs_collection"],
        dataproduct_type=curated["dataproduct_type"],
    )

    if len(obs_table) == 0:
        logger.warning("No observations found for preset '%s'.", key)
        return []

    # Filter out HAP skycell reprocessed products
    keep_mask = [("skycell" not in str(row["obs_id"])) for row in obs_table]
    obs_table = obs_table[keep_mask]
    if len(obs_table) == 0:
        logger.warning("All results were HAP skycell products for preset '%s'.", key)
        return []

    # Prefer moderate exposure times
    obs_table.sort("t_exptime")
    moderate = [r for r in obs_table if 1.0 <= float(r["t_exptime"]) <= 30.0]
    if moderate:
        import astropy.table
        selected = astropy.table.Table(rows=[moderate[0]], names=obs_table.colnames)
    else:
        selected = obs_table[:1]

    products = Observations.get_product_list(selected)
    filtered = Observations.filter_products(
        products,
        productSubGroupDescription=["FLT", "FLC", "CAL"],
        extension="fits",
    )
    if len(filtered) == 0:
        filtered = Observations.filter_products(products, extension="fits")
    if len(filtered) == 0:
        logger.warning("No FITS products for preset '%s'.", key)
        return []

    filtered = filtered[:max_products]
    manifest = Observations.download_products(filtered, download_dir=str(data_dir))
    paths = [Path(row["Local Path"]) for row in manifest if row["Status"] == "COMPLETE"]
    logger.info("Downloaded %d file(s) for preset '%s'.", len(paths), key)
    return paths


def download_all_presets(
    data_dir: Path,
    *,
    keys: list[str] | None = None,
    max_products_each: int = 1,
) -> dict[str, list[Path]]:
    """Download multiple curated observation presets.

    Parameters
    ----------
    data_dir : Path
        Root download directory.
    keys : list[str] | None
        Preset keys to download. If ``None``, downloads all available.
    max_products_each : int
        Max FITS files per preset.

    Returns
    -------
    dict[str, list[Path]]
        ``{preset_key: [path, ...]}`` for each successfully downloaded preset.
    """
    if keys is None:
        keys = list(_CURATED_OBS.keys())

    results: dict[str, list[Path]] = {}
    for key in keys:
        try:
            paths = download_preset(key, data_dir, max_products=max_products_each)
            if paths:
                results[key] = paths
        except Exception as exc:
            logger.error("Failed to download preset '%s': %s", key, exc)
    return results





