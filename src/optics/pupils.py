"""Telescope pupil amplitude models (HST, JWST, generic circular)."""

from __future__ import annotations

import numpy as np

from src.models.config import PupilConfig, TelescopeType
from src.models.optics import PupilModel


def _make_grid(n: int) -> tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray]:
    """Create normalised coordinate grids [-1, 1] × [-1, 1]."""
    y, x = np.mgrid[-1:1:complex(0, n), -1:1:complex(0, n)]
    rho = np.sqrt(x ** 2 + y ** 2)
    theta = np.arctan2(y, x)
    return x, y, rho, theta


def _spider_mask(
    x: np.ndarray,
    y: np.ndarray,
    n_spiders: int,
    width_frac: float,
) -> np.ndarray:
    """Binary mask blocking the spider vanes (1 = open, 0 = blocked)."""
    mask = np.ones_like(x, dtype=np.float64)
    if n_spiders == 0 or width_frac <= 0:
        return mask
    half_w = width_frac / 2.0
    angles = np.linspace(0, np.pi, n_spiders, endpoint=False)
    for angle in angles:
        ca, sa = np.cos(angle), np.sin(angle)
        # Project (x, y) onto the perpendicular of the vane direction
        perp = np.abs(-sa * x + ca * y)
        along = ca * x + sa * y
        blocked = (perp < half_w)
        mask[blocked] = 0.0
    return mask


def build_hst_pupil(cfg: PupilConfig) -> PupilModel:
    """Build the HST pupil: circular primary, circular secondary obstruction, 4 spider vanes."""
    n = cfg.grid_size
    x, y, rho, theta = _make_grid(n)

    # Radii normalised to primary
    outer = 1.0
    inner = cfg.secondary_radius / cfg.primary_radius

    # Annular aperture
    amp = np.zeros((n, n), dtype=np.float64)
    amp[(rho <= outer) & (rho >= inner)] = 1.0

    # Spider vanes – width normalised to primary radius
    spider_frac = cfg.spider_width / cfg.primary_radius
    amp *= _spider_mask(x, y, cfg.n_spiders, spider_frac)

    return PupilModel(amplitude=amp, grid_size=n)


def build_jwst_pupil(cfg: PupilConfig) -> PupilModel:
    """Build an approximate JWST pupil: hexagonal segments + 3 struts.

    This is a simplified but physically representative model.
    """
    n = cfg.grid_size
    x, y, rho, theta = _make_grid(n)

    # Overall circular envelope
    amp = np.zeros((n, n), dtype=np.float64)
    amp[rho <= 1.0] = 1.0

    # Central obstruction (secondary mirror)
    inner = cfg.secondary_radius / cfg.primary_radius
    amp[rho < inner] = 0.0

    # 3 spider struts at 60° spacing (V-shaped support structure)
    spider_frac = cfg.spider_width / cfg.primary_radius * 3  # JWST struts are wider
    for angle_deg in [0, 120, 240]:
        angle = np.radians(angle_deg)
        ca, sa = np.cos(angle), np.sin(angle)
        perp = np.abs(-sa * x + ca * y)
        amp[perp < spider_frac / 2] = 0.0

    # Segment gaps – simplified as radial lines + a hexagonal ring gap
    gap_width = 0.007 / cfg.primary_radius
    for angle_deg in [60, 180, 300]:
        angle = np.radians(angle_deg)
        ca, sa = np.cos(angle), np.sin(angle)
        perp = np.abs(-sa * x + ca * y)
        amp[(perp < gap_width) & (rho > inner) & (rho < 1.0)] = 0.0

    # Hexagonal ring gap
    ring_rho = 0.52
    ring_width = 0.005 / cfg.primary_radius
    amp[(np.abs(rho - ring_rho) < ring_width)] = 0.0

    return PupilModel(amplitude=amp, grid_size=n)


def build_generic_circular_pupil(cfg: PupilConfig) -> PupilModel:
    """Simple circular aperture with optional central obstruction."""
    n = cfg.grid_size
    _, _, rho, _ = _make_grid(n)
    inner = cfg.secondary_radius / cfg.primary_radius
    amp = np.zeros((n, n), dtype=np.float64)
    amp[(rho <= 1.0) & (rho >= inner)] = 1.0
    return PupilModel(amplitude=amp, grid_size=n)


def build_pupil(cfg: PupilConfig) -> PupilModel:
    """Factory function: build the appropriate pupil model from config."""
    builders = {
        TelescopeType.HST: build_hst_pupil,
        TelescopeType.JWST: build_jwst_pupil,
        TelescopeType.GENERIC_CIRCULAR: build_generic_circular_pupil,
    }
    builder = builders.get(cfg.telescope)
    if builder is None:
        raise ValueError(f"Unknown telescope type: {cfg.telescope}")
    return builder(cfg)

