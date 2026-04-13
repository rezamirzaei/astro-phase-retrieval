"""Pydantic configuration models for every stage of the pipeline.

Design
------
* ``AlgorithmConfig`` carries all hyper-parameters for every algorithm.
  PINN-specific fields are grouped at the bottom with a ``pinn_`` prefix so
  they are easily identified but remain in one flat model (avoids the need
  for callers to construct nested configs).
* ``admm_rho`` is the ADMM augmented-Lagrangian penalty parameter ρ.
  It is separate from ``beta`` (which is a relaxation / feedback coefficient)
  because the two parameters play fundamentally different roles.
"""

from __future__ import annotations

from enum import StrEnum
from pathlib import Path
from typing import Literal

from pydantic import BaseModel, Field, field_validator, model_validator


class Regulariser(StrEnum):
    """Regulariser for proximal-gradient algorithms."""

    NONE = "none"
    TV = "tv"
    L1_WAVELET = "l1_wavelet"


# ---------------------------------------------------------------------------
# Enums
# ---------------------------------------------------------------------------


class TelescopeType(StrEnum):
    """Supported telescope pupil geometries."""

    HST = "hst"
    JWST = "jwst"
    GENERIC_CIRCULAR = "generic_circular"


class AlgorithmName(StrEnum):
    """Registered phase-retrieval algorithm identifiers."""

    ERROR_REDUCTION = "er"
    GERCHBERG_SAXTON = "gs"
    HYBRID_INPUT_OUTPUT = "hio"
    RAAR = "raar"
    PHASE_DIVERSITY = "phase_diversity"
    WIRTINGER_FLOW = "wf"
    DOUGLAS_RACHFORD = "dr"
    ADMM = "admm"
    PINN = "pinn"
    FISTA = "fista"
    SPARSE_PR = "sparse_pr"


class BetaSchedule(StrEnum):
    """Adaptive β scheduling strategies."""

    CONSTANT = "constant"
    LINEAR = "linear"
    COSINE = "cosine"


class NoiseModel(StrEnum):
    """Noise model for focal-plane projection.

    ``GAUSSIAN``
        Standard amplitude replacement: G' = \|y\| · exp(iφ).  Fast and
        works well for high-SNR data.
    ``POISSON``
        Maximum-likelihood (ML) projection appropriate for photon-counting
        detectors.  Thibault & Guizar-Sicairos, NJP 14, 063004 (2012).
    """

    GAUSSIAN = "gaussian"
    POISSON = "poisson"


# ---------------------------------------------------------------------------
# Data configuration
# ---------------------------------------------------------------------------


class DataConfig(BaseModel):
    """Where to find / download the observation data."""

    data_dir: Path = Field(
        default=Path("data"), description="Root directory for downloaded FITS files"
    )
    obs_id: str = Field(
        default="JDOX-HST-WFC3",
        description="MAST observation identifier or preset name",
    )
    detector: Literal["WFC3/UVIS", "WFC3/IR", "ACS/WFC", "NIRCam"] = Field(
        default="WFC3/UVIS",
        description="Detector used for the observation",
    )
    cutout_size: int = Field(
        default=128, ge=32, le=1024, description="PSF cutout half-width in pixels"
    )
    filter_name: str = Field(default="F606W", description="Optical filter bandpass name")
    background_percentile: float = Field(
        default=10.0,
        ge=0.0,
        le=50.0,
        description="Percentile used for robust edge-based background estimation",
    )
    source_detection_border: int = Field(
        default=50,
        ge=0,
        le=2048,
        description="Border margin excluded during source detection to avoid edge artifacts",
    )
    centroid_window_size: int = Field(
        default=24,
        ge=4,
        le=256,
        description="Local window width used for subpixel centroid refinement",
    )
    recenter_psf: bool = Field(
        default=True,
        description="Apply subpixel recentering to extracted PSF cutouts before normalisation",
    )
    centroid_method: Literal["moments", "quadratic_peak"] = Field(
        default="moments",
        description="Subpixel centroid refinement strategy",
    )
    use_dq_mask: bool = Field(
        default=True,
        description="Use FITS data-quality (DQ) extensions when available",
    )
    saturation_percentile: float = Field(
        default=99.95,
        ge=90.0,
        le=100.0,
        description="Percentile threshold used to flag saturated or clipped bright pixels",
    )
    hot_pixel_sigma: float = Field(
        default=8.0,
        gt=0.0,
        le=50.0,
        description="MAD-based sigma threshold used to flag isolated hot pixels in PSF cutouts",
    )
    calibration_preset: str = Field(
        default="auto",
        description="Instrument-specific preprocessing preset (`auto` infers from FITS headers)",
    )

    @field_validator("cutout_size")
    @classmethod
    def _power_of_two(cls, v: int) -> int:
        if v & (v - 1) != 0:
            raise ValueError(f"cutout_size must be a power of 2, got {v}")
        return v


# ---------------------------------------------------------------------------
# Pupil / optics configuration
# ---------------------------------------------------------------------------


class PupilConfig(BaseModel):
    """Telescope pupil model parameters."""

    telescope: TelescopeType = Field(default=TelescopeType.HST)
    grid_size: int = Field(
        default=256, ge=64, le=2048, description="Pupil-plane grid side length (px)"
    )
    primary_radius: float = Field(default=1.2, gt=0, description="Primary mirror radius (m)")
    secondary_radius: float = Field(
        default=0.396, ge=0, description="Secondary mirror obstruction radius (m)"
    )
    spider_width: float = Field(default=0.0254, ge=0, description="Spider vane width (m)")
    n_spiders: int = Field(default=4, ge=0, le=8, description="Number of spider vanes")
    wavelength_m: float = Field(default=606e-9, gt=0, description="Observation wavelength (m)")
    pixel_scale_arcsec: float = Field(
        default=0.04, gt=0, description="Detector pixel scale (arcsec/px)"
    )
    bandwidth_fraction: float = Field(
        default=0.0,
        ge=0.0,
        le=1.0,
        description="Fractional wavelength span for simple polychromatic averaging",
    )
    spectral_samples: int = Field(
        default=1,
        ge=1,
        le=15,
        description="Number of wavelength samples for polychromatic averaging",
    )
    spectral_weighting: Literal["delta", "gaussian", "uniform"] = Field(
        default="delta",
        description="Spectral weighting model used for broadband forward propagation",
    )
    field_defocus_waves: float = Field(
        default=0.0,
        ge=-5.0,
        le=5.0,
        description="Field-dependent defocus term added in the forward model (waves)",
    )
    detector_sigma_pixels: float = Field(
        default=0.0,
        ge=0.0,
        le=10.0,
        description="Gaussian detector blur sigma applied in the focal plane (pixels)",
    )
    jitter_sigma_pixels: float = Field(
        default=0.0,
        ge=0.0,
        le=10.0,
        description="Gaussian pointing-jitter blur sigma applied in the focal plane (pixels)",
    )
    pixel_integration_width: float = Field(
        default=1.0,
        ge=0.5,
        le=4.0,
        description="Approximate detector pixel-response width in focal-plane pixels",
    )

    @field_validator("grid_size")
    @classmethod
    def _power_of_two(cls, v: int) -> int:
        if v & (v - 1) != 0:
            raise ValueError(f"grid_size must be a power of 2, got {v}")
        return v


# ---------------------------------------------------------------------------
# Algorithm configuration
# ---------------------------------------------------------------------------


class AlgorithmConfig(BaseModel):
    """Phase-retrieval algorithm hyper-parameters.

    Core parameters
    ~~~~~~~~~~~~~~~
    ``name``             : Algorithm key (see :class:`AlgorithmName`).
    ``max_iterations``   : Hard iteration limit.
    ``tolerance``        : Relative cost-change threshold for early stopping.
    ``beta``             : Feedback / relaxation parameter β (HIO, RAAR, DR).
    ``beta_schedule``    : Cosine / linear annealing of β.
    ``beta_min``         : Minimum β for adaptive schedules.

    Regularisation
    ~~~~~~~~~~~~~~
    ``tv_weight``        : TV proximal weight λ (0 = off).
    ``noise_model``      : Gaussian (default) or Poisson ML projection.
    ``use_sw_constraint``: Enable Shrink-Wrap dynamic support (Marchesini 2003).
    ``support_threshold``: Fraction of max amplitude for SW thresholding.

    ADMM
    ~~~~
    ``admm_rho``         : Augmented-Lagrangian penalty ρ (distinct from β).

    Enhancements
    ~~~~~~~~~~~~
    ``momentum``         : Nesterov / heavy-ball momentum coefficient.
    ``n_starts``         : Multi-start restarts (best result returned).
    ``wf_step_size``     : Wirtinger Flow learning rate μ.
    ``wf_spectral_init`` : Use spectral init for WF (recommended).
    ``spectral_init``    : Use spectral init for all algorithms.

    PINN (neural field)
    ~~~~~~~~~~~~~~~~~~~
    All ``pinn_*`` fields configure the PINN solver (only used when
    ``name == AlgorithmName.PINN``).
    """

    name: AlgorithmName = Field(default=AlgorithmName.HYBRID_INPUT_OUTPUT)
    max_iterations: int = Field(default=300, ge=1, le=100_000)
    tolerance: float = Field(
        default=1e-8, gt=0, description="Convergence tolerance on cost-function change"
    )
    beta: float = Field(default=0.9, gt=0, le=1.0, description="HIO / RAAR feedback parameter β")
    beta_schedule: BetaSchedule = Field(
        default=BetaSchedule.CONSTANT,
        description="Adaptive β scheduling: constant, linear ramp-down, or cosine annealing",
    )
    beta_min: float = Field(
        default=0.5, ge=0, le=1.0, description="Minimum β for adaptive schedules"
    )
    defocus_waves: float = Field(
        default=1.0,
        description="Defocus amount (waves) for phase-diversity second image",
    )
    use_sw_constraint: bool = Field(
        default=True,
        description=(
            "Apply Shrink-Wrap dynamic support refinement (Marchesini et al., PRB 68, 140101, 2003)"
        ),
    )
    support_threshold: float = Field(
        default=0.04,
        ge=0,
        le=1,
        description="Fraction of max smoothed amplitude for Shrink-Wrap thresholding",
    )
    random_seed: int | None = Field(default=42, description="RNG seed for reproducibility")

    # ── ADMM-specific ─────────────────────────────────────────────────
    admm_rho: float = Field(
        default=1.0,
        gt=0,
        description=(
            "ADMM augmented-Lagrangian penalty ρ.  "
            "Larger ρ enforces G = F{g} more aggressively at the cost of "
            "slower sub-problem convergence.  Distinct from β."
        ),
    )

    # ── State-of-the-art enhancements ─────────────────────────────────
    momentum: float = Field(
        default=0.0,
        ge=0,
        le=0.99,
        description="Nesterov/heavy-ball momentum coefficient (0 = off)",
    )
    tv_weight: float = Field(
        default=0.0,
        ge=0,
        description="Total-variation regularization weight on recovered phase (0 = off)",
    )
    noise_model: NoiseModel = Field(
        default=NoiseModel.GAUSSIAN,
        description="Noise model for focal-plane amplitude projection",
    )
    n_starts: int = Field(
        default=1,
        ge=1,
        le=100,
        description="Number of random restarts (multi-start); best result is returned",
    )
    wf_step_size: float = Field(
        default=0.5,
        gt=0,
        le=10.0,
        description="Wirtinger Flow step size (learning rate)",
    )
    wf_spectral_init: bool = Field(
        default=True,
        description="Use spectral initialization for Wirtinger Flow",
    )
    spectral_init: bool = Field(
        default=True,
        description="Use spectral initialization (power iteration) for all algorithms",
    )

    # ── PINN (neural field solver) — only used when name == "pinn" ────
    pinn_hidden_features: int = Field(
        default=128,
        ge=8,
        le=512,
        description="Width of PINN hidden layers",
    )
    pinn_hidden_layers: int = Field(
        default=4,
        ge=1,
        le=8,
        description="Number of PINN hidden layers",
    )
    pinn_learning_rate: float = Field(
        default=2e-3,
        gt=0,
        le=1.0,
        description="Adam learning rate for PINN solver",
    )
    pinn_smoothness_weight: float = Field(
        default=5e-5,
        ge=0,
        description="Smoothness regularization weight for PINN",
    )
    pinn_sqrt_weight: float = Field(
        default=0.2,
        ge=0,
        description="Weight of square-root intensity loss term",
    )
    pinn_log_weight: float = Field(
        default=0.05,
        ge=0,
        description="Weight of log1p-intensity loss term",
    )
    pinn_grad_clip: float = Field(
        default=1.0,
        ge=0,
        description="Gradient clipping threshold (0 = disabled)",
    )
    pinn_fourier_features: int = Field(
        default=64,
        ge=8,
        le=512,
        description="Random Fourier features for coordinate encoding",
    )
    pinn_fourier_sigma: float = Field(
        default=4.0,
        gt=0,
        le=20.0,
        description="Bandwidth of random Fourier feature encoding",
    )
    pinn_lbfgs_lr: float = Field(
        default=0.5,
        gt=0,
        le=5.0,
        description="L-BFGS refinement phase learning rate",
    )
    pinn_warm_start: bool = Field(
        default=True,
        description="Warm-start PINN from a short RAAR reconstruction",
    )
    pinn_warm_start_iterations: int = Field(
        default=200,
        ge=1,
        le=10_000,
        description="RAAR iterations for PINN warm start",
    )
    pinn_residual_scale: float = Field(
        default=0.5,
        ge=0,
        le=2.0,
        description="Scale of neural residual phase (units of π)",
    )
    pinn_device: Literal["auto", "cpu", "mps", "cuda"] = Field(
        default="auto",
        description="Preferred device for PINN solver",
    )
    pinn_lr_step: int = Field(
        default=50,
        ge=1,
        description="LR scheduler step size (epochs) for PINN",
    )
    pinn_lr_gamma: float = Field(
        default=0.7,
        gt=0,
        le=1.0,
        description="LR scheduler multiplicative decay γ for PINN",
    )

    # ── ER-finish (shared across HIO, RAAR, DR) ──────────────────────
    er_finish_fraction: float = Field(
        default=0.1,
        ge=0.0,
        le=0.5,
        description="Fraction of total iterations used for ER polish at the end",
    )

    # ── Proximal-gradient / FISTA ─────────────────────────────────────
    regulariser: Regulariser = Field(
        default=Regulariser.NONE,
        description="Regulariser for FISTA/proximal gradient: none, tv, l1_wavelet",
    )
    proximal_weight: float = Field(
        default=1e-3,
        ge=0,
        description="Regularisation weight λ for FISTA proximal step",
    )
    fista_lipschitz: float = Field(
        default=1.0,
        gt=0,
        description="Lipschitz constant estimate for FISTA step-size (auto-estimated if 0)",
    )

    # ── Sparse phase retrieval ────────────────────────────────────────
    sparsity_threshold: float = Field(
        default=0.1,
        ge=0,
        le=1.0,
        description="Thresholding level for sparse PR (fraction of max)",
    )
    sparsity_keep_fraction: float = Field(
        default=1.0,
        gt=0,
        le=1.0,
        description="Fraction of support pixels retained by hard-threshold sparse PR",
    )

    # ── Shrink-Wrap annealing ─────────────────────────────────────────
    sw_sigma_start: float = Field(
        default=3.0,
        ge=1.0,
        le=20.0,
        description="Initial Gaussian σ for Shrink-Wrap smoothing",
    )
    sw_sigma_end: float = Field(
        default=1.0,
        ge=0.5,
        le=10.0,
        description="Final Gaussian σ for Shrink-Wrap smoothing",
    )

    # ── Cross-field validators ────────────────────────────────────────
    @model_validator(mode="after")
    def _validate_cross_fields(self) -> AlgorithmConfig:
        """Validate field combinations for consistency."""
        import warnings

        # admm_rho only meaningful for ADMM
        if self.admm_rho != 1.0 and self.name != AlgorithmName.ADMM:
            warnings.warn(
                f"admm_rho={self.admm_rho} is set but algorithm is "
                f"'{self.name.value}' (not ADMM) — parameter will be ignored.",
                UserWarning,
                stacklevel=2,
            )

        # Regulariser only meaningful for FISTA
        if self.regulariser != Regulariser.NONE and self.name not in (
            AlgorithmName.FISTA,
            AlgorithmName.SPARSE_PR,
        ):
            warnings.warn(
                f"regulariser='{self.regulariser.value}' is set but algorithm is "
                f"'{self.name.value}' — regulariser will be ignored; "
                f"use tv_weight for TV regularisation in the base loop.",
                UserWarning,
                stacklevel=2,
            )

        # sw_sigma_start >= sw_sigma_end
        if self.sw_sigma_start < self.sw_sigma_end:
            warnings.warn(
                f"sw_sigma_start ({self.sw_sigma_start}) < sw_sigma_end "
                f"({self.sw_sigma_end}) — sigma will be clamped.",
                UserWarning,
                stacklevel=2,
            )

        return self


# ---------------------------------------------------------------------------
# Pipeline (master) configuration
# ---------------------------------------------------------------------------


class PipelineConfig(BaseModel):
    """Top-level configuration aggregating all sub-configs."""

    data: DataConfig = Field(default_factory=DataConfig)
    pupil: PupilConfig = Field(default_factory=PupilConfig)
    algorithm: AlgorithmConfig = Field(default_factory=AlgorithmConfig)
    output_dir: Path = Field(default=Path("outputs"))
    uncertainty_samples: int = Field(
        default=0,
        ge=0,
        le=100,
        description="Number of perturbation runs for uncertainty estimation (0 = disabled)",
    )
    uncertainty_shift_sigma_pixels: float = Field(
        default=0.15,
        ge=0.0,
        le=5.0,
        description="Stddev of random image shifts used in uncertainty runs",
    )
    uncertainty_background_sigma_fraction: float = Field(
        default=0.002,
        ge=0.0,
        le=1.0,
        description="Stddev of additive background perturbations relative to image peak",
    )
    uncertainty_noise_sigma_fraction: float = Field(
        default=0.001,
        ge=0.0,
        le=1.0,
        description="Stddev of additive Gaussian noise perturbations relative to image peak",
    )
    uncertainty_seed: int = Field(default=123, description="RNG seed for uncertainty estimation")

    @model_validator(mode="after")
    def _sync_wavelength_with_filter(self) -> PipelineConfig:
        """Auto-set wavelength from filter name when using defaults."""
        # Import lazily to avoid circular dependency: config → downloader → config
        from src.data.downloader import FILTER_WAVELENGTH_M  # noqa: PLC0415

        wl = FILTER_WAVELENGTH_M.get(self.data.filter_name)
        if wl is not None:
            self.pupil.wavelength_m = wl
        return self


# ---------------------------------------------------------------------------
# Convenience constructor
# ---------------------------------------------------------------------------


def default_hst_config() -> PipelineConfig:
    """Return a sensible default config for HST/WFC3/UVIS phase retrieval."""
    return PipelineConfig(
        data=DataConfig(
            detector="WFC3/UVIS",
            filter_name="F606W",
            cutout_size=128,
        ),
        pupil=PupilConfig(
            telescope=TelescopeType.HST,
            grid_size=256,
            primary_radius=1.2,
            secondary_radius=0.396,
            spider_width=0.0254,
            n_spiders=4,
            pixel_scale_arcsec=0.04,
            spectral_weighting="gaussian",
        ),
        algorithm=AlgorithmConfig(
            name=AlgorithmName.HYBRID_INPUT_OUTPUT,
            max_iterations=500,
            beta=0.9,
        ),
    )


def default_jwst_config() -> PipelineConfig:
    """Return a sensible default config for JWST/NIRCam phase retrieval."""
    return PipelineConfig(
        data=DataConfig(
            detector="NIRCam",
            filter_name="F200W",
            cutout_size=128,
        ),
        pupil=PupilConfig(
            telescope=TelescopeType.JWST,
            grid_size=256,
            primary_radius=3.25,
            secondary_radius=0.74,
            spider_width=0.025,
            n_spiders=3,
            pixel_scale_arcsec=0.031,
            spectral_weighting="gaussian",
        ),
        algorithm=AlgorithmConfig(
            name=AlgorithmName.HYBRID_INPUT_OUTPUT,
            max_iterations=500,
            beta=0.9,
        ),
    )
