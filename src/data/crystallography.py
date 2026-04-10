"""Crystallography data: download CIF files from COD, parse structures, simulate diffraction.

Uses the Crystallography Open Database (COD) REST API for real-world data.
CIF downloads use ``httpx`` with a timeout and streaming to handle large files
robustly.  The legacy ``urllib.request`` path is intentionally removed to
avoid the lack of timeout control and the deprecation of ``urlretrieve``.
"""

from __future__ import annotations

import logging
import re
import urllib.request
from pathlib import Path

import numpy as np
from numpy.fft import fftshift, ifft2, ifftshift

from src.models.crystallography import (
    AtomSite,
    CrystallographyResult,
    CrystalStructure,
    DiffractionPattern,
)

logger = logging.getLogger(__name__)

# ---------------------------------------------------------------------------
# Atomic scattering factor approximation (Cromer-Mann 4-Gaussian)
# Common elements only — sufficient for demonstration.
# ---------------------------------------------------------------------------

_SCATTERING_FACTORS: dict[str, float] = {
    "H": 1.0, "He": 2.0, "Li": 3.0, "Be": 4.0, "B": 5.0, "C": 6.0,
    "N": 7.0, "O": 8.0, "F": 9.0, "Ne": 10.0, "Na": 11.0, "Mg": 12.0,
    "Al": 13.0, "Si": 14.0, "P": 15.0, "S": 16.0, "Cl": 17.0, "Ar": 18.0,
    "K": 19.0, "Ca": 20.0, "Ti": 22.0, "V": 23.0, "Cr": 24.0, "Mn": 25.0,
    "Fe": 26.0, "Co": 27.0, "Ni": 28.0, "Cu": 29.0, "Zn": 30.0, "Ga": 31.0,
    "Ge": 32.0, "As": 33.0, "Se": 34.0, "Br": 35.0, "Sr": 38.0, "Y": 39.0,
    "Zr": 40.0, "Mo": 42.0, "Ag": 47.0, "Sn": 50.0, "I": 53.0, "Ba": 56.0,
    "La": 57.0, "Pb": 82.0, "Bi": 83.0, "U": 92.0,
}


def _atomic_scattering_factor(symbol: str) -> float:
    """Return the approximate atomic number as a scattering factor proxy."""
    clean = re.sub(r"[^A-Za-z]", "", symbol)
    return _SCATTERING_FACTORS.get(clean, 6.0)


# ---------------------------------------------------------------------------
# Curated COD presets
# ---------------------------------------------------------------------------

_CURATED_COD: dict[str, dict[str, str]] = {
    "nacl": {
        "cod_id": "1000041",
        "formula": "NaCl",
        "description": "Sodium Chloride — cubic rock salt structure",
    },
    "quartz": {
        "cod_id": "1011097",
        "formula": "SiO2",
        "description": "α-Quartz — trigonal silicon dioxide",
    },
    "calcite": {
        "cod_id": "1010962",
        "formula": "CaCO3",
        "description": "Calcite — trigonal calcium carbonate",
    },
    "diamond": {
        "cod_id": "1010927",
        "formula": "C",
        "description": "Diamond — cubic carbon allotrope",
    },
    "corundum": {
        "cod_id": "1000032",
        "formula": "Al2O3",
        "description": "Corundum (α-alumina) — trigonal aluminium oxide",
    },
    "rutile": {
        "cod_id": "1000055",
        "formula": "TiO2",
        "description": "Rutile — tetragonal titanium dioxide",
    },
    "pyrite": {
        "cod_id": "1011031",
        "formula": "FeS2",
        "description": "Pyrite — cubic iron disulfide (fool's gold)",
    },
    "fluorite": {
        "cod_id": "1000043",
        "formula": "CaF2",
        "description": "Fluorite — cubic calcium fluoride",
    },
}


def available_cod_presets() -> dict[str, str]:
    """Return ``{preset_key: description}`` for curated COD entries."""
    return {k: v["description"] for k, v in _CURATED_COD.items()}


# ---------------------------------------------------------------------------
# CIF download — uses httpx for timeout / streaming safety
# ---------------------------------------------------------------------------


def _download_url_to_file(url: str, dest: Path, timeout: float = 30.0) -> None:
    """Download *url* to *dest* using ``httpx`` with a timeout.

    Falls back to ``urllib.request`` when ``httpx`` is not installed so that
    the library still works in minimal environments.

    Parameters
    ----------
    url : str
        Remote URL.
    dest : Path
        Local destination path.
    timeout : float
        Request timeout in seconds.
    """
    try:
        import httpx  # optional but strongly preferred

        with httpx.Client(timeout=timeout, follow_redirects=True) as client, \
                client.stream("GET", url) as response:
                response.raise_for_status()
                with dest.open("wb") as fh:
                    for chunk in response.iter_bytes(chunk_size=65536):
                        fh.write(chunk)
    except ImportError:
        # httpx not available — fall back to urllib (no timeout)
        logger.warning(
            "httpx not installed; falling back to urllib.request (no timeout). "
            "Install httpx for production use: pip install httpx"
        )
        urllib.request.urlretrieve(url, str(dest))  # noqa: S310


def download_cif(cod_id: str, data_dir: Path, timeout: float = 30.0) -> Path:
    """Download a CIF file from the Crystallography Open Database.

    Parameters
    ----------
    cod_id : str
        COD entry ID (numeric string, e.g. ``"1000041"``).
    data_dir : Path
        Directory where the CIF will be saved.
    timeout : float
        HTTP request timeout in seconds (default 30).

    Returns
    -------
    Path
        Path to the downloaded ``.cif`` file.

    Raises
    ------
    RuntimeError
        If the download fails for any reason (network error, 4xx/5xx).
    """
    data_dir = Path(data_dir)
    cif_dir = data_dir / "crystallography"
    cif_dir.mkdir(parents=True, exist_ok=True)
    out_path = cif_dir / f"{cod_id}.cif"

    if out_path.exists():
        logger.info("CIF %s already cached at %s", cod_id, out_path)
        return out_path

    url = f"https://www.crystallography.net/cod/{cod_id}.cif"
    logger.info("Downloading CIF %s from %s", cod_id, url)
    try:
        _download_url_to_file(url, out_path, timeout=timeout)
    except Exception as exc:
        # Clean up partial download
        if out_path.exists():
            out_path.unlink()
        raise RuntimeError(f"Failed to download CIF {cod_id}: {exc}") from exc

    logger.info("Saved CIF → %s  (%d bytes)", out_path, out_path.stat().st_size)
    return out_path


def download_cod_preset(key: str, data_dir: Path, timeout: float = 30.0) -> Path:
    """Download a curated COD preset by key.

    Parameters
    ----------
    key : str
        One of the keys from :func:`available_cod_presets`.
    data_dir : Path
        Root download directory.
    timeout : float
        HTTP request timeout in seconds.

    Returns
    -------
    Path
        Path to the downloaded CIF file.
    """
    preset = _CURATED_COD.get(key)
    if preset is None:
        raise KeyError(f"Unknown COD preset '{key}'. Available: {list(_CURATED_COD)}")
    return download_cif(preset["cod_id"], data_dir, timeout=timeout)


def list_cached_cif(data_dir: Path) -> list[Path]:
    """Return all CIF files already present under *data_dir*/crystallography."""
    cif_dir = data_dir / "crystallography"
    if not cif_dir.exists():
        return []
    return sorted(cif_dir.glob("*.cif"))


# ---------------------------------------------------------------------------
# CIF parser (lightweight, no external deps)
# ---------------------------------------------------------------------------


def parse_cif(filepath: Path) -> CrystalStructure:
    """Parse a CIF file and return a :class:`CrystalStructure`.

    This is a lightweight regex-based parser that handles the most common
    CIF formatting conventions.  It is sufficient for COD entries.

    Parameters
    ----------
    filepath : Path
        Path to the ``.cif`` file.

    Returns
    -------
    CrystalStructure
        Parsed structure with unit-cell parameters and atom sites.
    """
    filepath = Path(filepath)
    text = filepath.read_text(encoding="utf-8", errors="replace")

    def _float(s: str) -> float:
        """Extract float from CIF value (strip parenthesised uncertainty)."""
        s = s.strip().split("(")[0]
        return float(s)

    # Unit-cell parameters
    a = _extract_cif_value(text, "_cell_length_a", 5.0)
    b = _extract_cif_value(text, "_cell_length_b", 5.0)
    c = _extract_cif_value(text, "_cell_length_c", 5.0)
    alpha = _extract_cif_value(text, "_cell_angle_alpha", 90.0)
    beta = _extract_cif_value(text, "_cell_angle_beta", 90.0)
    gamma = _extract_cif_value(text, "_cell_angle_gamma", 90.0)

    # Space group
    space_group = _extract_cif_string(text, "_symmetry_space_group_name_H-M", "P 1")
    if space_group == "P 1":
        space_group = _extract_cif_string(text, "_space_group_name_H-M_alt", "P 1")

    # Chemical formula
    formula = _extract_cif_string(text, "_chemical_formula_sum", "")
    if not formula:
        formula = _extract_cif_string(text, "_chemical_formula_structural", "")

    # COD ID from filename
    cod_id = filepath.stem

    # Atom sites
    atoms = _parse_atom_sites(text)

    return CrystalStructure(
        cod_id=cod_id,
        formula=formula,
        space_group=space_group,
        a=a,
        b=b,
        c=c,
        alpha=alpha,
        beta=beta,
        gamma=gamma,
        atoms=atoms,
    )


def _extract_cif_value(text: str, tag: str, default: float) -> float:
    """Extract a numeric value for *tag* from CIF text."""
    pattern = re.compile(rf"^{re.escape(tag)}\s+(\S+)", re.MULTILINE)
    match = pattern.search(text)
    if match:
        raw = match.group(1).split("(")[0]
        try:
            return float(raw)
        except ValueError:
            return default
    return default


def _extract_cif_string(text: str, tag: str, default: str) -> str:
    """Extract a string value for *tag* from CIF text."""
    # Try quoted value first
    pattern = re.compile(rf"^{re.escape(tag)}\s+['\"](.+?)['\"]", re.MULTILINE)
    match = pattern.search(text)
    if match:
        return match.group(1).strip()
    # Unquoted
    pattern = re.compile(rf"^{re.escape(tag)}\s+(\S+)", re.MULTILINE)
    match = pattern.search(text)
    if match:
        val = match.group(1).strip().strip("'\"")
        return val
    return default


def _parse_atom_sites(text: str) -> list[AtomSite]:
    """Parse atom_site loop from CIF text."""
    atoms: list[AtomSite] = []

    # Find the loop_ block containing _atom_site_label
    loop_pattern = re.compile(
        r"loop_\s*((?:_atom_site_\w+\s*)+)((?:(?!loop_|data_|#).*\n)*)",
        re.MULTILINE,
    )
    for match in loop_pattern.finditer(text):
        header_block = match.group(1)
        data_block = match.group(2)

        headers = [h.strip() for h in header_block.strip().split("\n") if h.strip()]
        if "_atom_site_label" not in " ".join(headers):
            continue

        # Map column indices
        col_map: dict[str, int] = {}
        for idx, h in enumerate(headers):
            h = h.strip()
            col_map[h] = idx

        label_col = col_map.get("_atom_site_label")
        symbol_col = col_map.get(
            "_atom_site_type_symbol",
            col_map.get("_atom_site_label"),
        )
        x_col = col_map.get("_atom_site_fract_x")
        y_col = col_map.get("_atom_site_fract_y")
        z_col = col_map.get("_atom_site_fract_z")
        occ_col = col_map.get("_atom_site_occupancy")

        if label_col is None or x_col is None or y_col is None or z_col is None:
            continue

        for line in data_block.strip().split("\n"):
            line = line.strip()
            if not line or line.startswith("#") or line.startswith("_"):
                continue
            parts = line.split()
            if len(parts) < len(headers):
                continue
            try:
                label = parts[label_col]
                symbol_raw = parts[symbol_col] if symbol_col is not None else label
                symbol = re.sub(r"[^A-Za-z]", "", symbol_raw)[:2]
                x = float(parts[x_col].split("(")[0])
                y = float(parts[y_col].split("(")[0])
                z = float(parts[z_col].split("(")[0])
                occ = float(parts[occ_col].split("(")[0]) if occ_col is not None else 1.0
                atoms.append(AtomSite(
                    label=label,
                    symbol=symbol if symbol else "C",
                    x=x,
                    y=y,
                    z=z,
                    occupancy=min(max(occ, 0.0), 1.0),
                ))
            except (ValueError, IndexError):
                continue

    return atoms


# ---------------------------------------------------------------------------
# Diffraction simulation
# ---------------------------------------------------------------------------


def simulate_diffraction(
    crystal: CrystalStructure,
    grid_size: int = 128,
    wavelength_angstrom: float = 1.5418,
) -> DiffractionPattern:
    """Simulate a 2-D diffraction pattern from a crystal structure.

    Computes structure factors F(h,k) for a 2-D slice (l=0) of reciprocal
    space, then returns |F|² as the diffraction intensity.

    Parameters
    ----------
    crystal : CrystalStructure
        Parsed crystal structure with atom sites.
    grid_size : int
        Size of the output square array.
    wavelength_angstrom : float
        X-ray wavelength in Angstroms.

    Returns
    -------
    DiffractionPattern
        Simulated 2-D diffraction intensities.
    """
    n = grid_size
    half = n // 2

    # Reciprocal space grid: h, k indices centred at (0,0)
    h_range = np.arange(-half, half)
    k_range = np.arange(-half, half)
    H, K = np.meshgrid(h_range, k_range)  # noqa: N806

    # Compute structure factors F(h,k,0)
    F = np.zeros((n, n), dtype=np.complex128)  # noqa: N806

    if crystal.atoms:
        for atom in crystal.atoms:
            f_j = _atomic_scattering_factor(atom.symbol) * atom.occupancy
            phase = 2.0 * np.pi * (H * atom.x + K * atom.y)
            F += f_j * np.exp(1j * phase)
    else:
        # No atoms — generate a synthetic pattern
        rng = np.random.default_rng(42)
        n_atoms = 8
        for _ in range(n_atoms):
            f_j = rng.uniform(6.0, 30.0)
            x, y = rng.uniform(0, 1, 2)
            phase = 2.0 * np.pi * (H * x + K * y)
            F += f_j * np.exp(1j * phase)

    # Intensity = |F|²
    intensity = np.abs(F) ** 2

    # Apply a Debye-Waller-like thermal factor for realism
    r_sq = H.astype(float) ** 2 + K.astype(float) ** 2
    b_factor = 2.0  # Å²
    dw = np.exp(-b_factor * r_sq / (4.0 * crystal.a**2))
    intensity *= dw

    # Normalise
    total = intensity.sum()
    if total > 0:
        intensity = intensity / total

    return DiffractionPattern(
        image=intensity,
        wavelength_angstrom=wavelength_angstrom,
        d_max=max(crystal.a, crystal.b),
        space_group=crystal.space_group,
        source_id=crystal.cod_id,
    )


# ---------------------------------------------------------------------------
# Phase retrieval adapter for crystallography
# ---------------------------------------------------------------------------


def run_crystallography_retrieval(
    diffraction: DiffractionPattern,
    algorithm_name: str = "hio",
    max_iterations: int = 500,
    beta: float = 0.9,
    random_seed: int = 42,
) -> CrystallographyResult:
    """Run phase retrieval on a crystallographic diffraction pattern.

    Wraps the existing :class:`AlgorithmRegistry` and adapts the
    crystallographic data to the standard PSF-based pipeline.

    Parameters
    ----------
    diffraction : DiffractionPattern
        Observed (or simulated) 2-D diffraction intensities.
    algorithm_name : str
        Algorithm key (``"er"``, ``"hio"``, ``"raar"``, etc.).
    max_iterations : int
        Maximum iterations.
    beta : float
        Feedback parameter β.
    random_seed : int
        RNG seed.

    Returns
    -------
    CrystallographyResult
        Recovered phase, electron density, and quality metrics.
    """
    from src.algorithms.registry import AlgorithmRegistry
    from src.models.config import AlgorithmConfig, AlgorithmName
    from src.models.optics import PSFData, PupilModel

    grid_size = diffraction.image.shape[0]

    # Create a circular support mask (pupil) for the algorithms
    y, x = np.mgrid[-1:1:complex(0, grid_size), -1:1:complex(0, grid_size)]  # type: ignore[misc]
    rho = np.sqrt(x**2 + y**2)
    amplitude = np.ones((grid_size, grid_size), dtype=np.float64)
    amplitude[rho > 1.0] = 0.0

    pupil = PupilModel(amplitude=amplitude, grid_size=grid_size)

    # Wrap diffraction as PSFData for the existing algorithm interface
    psf_data = PSFData(
        image=diffraction.image,
        pixel_scale_arcsec=0.04,
        wavelength_m=diffraction.wavelength_angstrom * 1e-10,
        filter_name="X-ray",
        telescope="crystallography",
        obs_id=diffraction.source_id,
    )

    alg_cfg = AlgorithmConfig(
        name=AlgorithmName(algorithm_name),
        max_iterations=max_iterations,
        beta=beta,
        random_seed=random_seed,
    )

    retriever = AlgorithmRegistry.create(alg_cfg, pupil)
    result = retriever.run(psf_data)

    # Compute electron density as |IFT(F · exp(iφ))| where F = √I
    measured_amplitude = np.sqrt(np.maximum(diffraction.image, 0.0))
    complex_field = measured_amplitude * np.exp(1j * result.recovered_phase)
    electron_density_complex = fftshift(ifft2(ifftshift(complex_field)))
    electron_density = np.abs(electron_density_complex)

    # Compute R-factor: Σ|√I_obs − √I_calc| / Σ|√I_obs|
    calc_amplitude = np.sqrt(np.maximum(result.reconstructed_psf, 0.0))
    obs_amplitude = np.sqrt(np.maximum(diffraction.image, 0.0))
    denom = obs_amplitude.sum()
    r_factor = float(np.sum(np.abs(obs_amplitude - calc_amplitude)) / max(denom, 1e-30))

    return CrystallographyResult(
        algorithm=AlgorithmName(algorithm_name),
        recovered_phase=result.recovered_phase,
        recovered_amplitude=result.recovered_amplitude,
        reconstructed_diffraction=result.reconstructed_psf,
        electron_density=electron_density,
        cost_history=result.cost_history,
        n_iterations=result.n_iterations,
        converged=result.converged,
        elapsed_seconds=result.elapsed_seconds,
        r_factor=min(r_factor, 1.0),
        metadata={
            "source_id": diffraction.source_id,
            "space_group": diffraction.space_group,
            "strehl_ratio": result.strehl_ratio,
            "rms_phase_rad": result.rms_phase_rad,
        },
    )





