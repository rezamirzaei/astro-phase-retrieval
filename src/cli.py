"""Command-line interface for the phase-retrieval pipeline.

Usage
-----
    phase-retrieval run   [--algorithm hio] [--iterations 500] [--fits PATH]
    phase-retrieval compare [--fits PATH]
    phase-retrieval download [--preset hst-wfc3-uvis-f606w]
"""

from __future__ import annotations

import argparse
import json
import logging
import sys
from pathlib import Path

logger = logging.getLogger(__name__)


def _configure_logging(verbose: bool = False) -> None:
    logging.basicConfig(
        level=logging.DEBUG if verbose else logging.INFO,
        format="%(asctime)s  %(name)-28s  %(levelname)-8s  %(message)s",
        stream=sys.stdout,
    )


# ── run ───────────────────────────────────────────────────────────────────

def _cmd_run(args: argparse.Namespace) -> None:
    """Run a single phase-retrieval algorithm on a FITS file."""
    from src.algorithms.multi_start import multi_start_run
    from src.algorithms.registry import AlgorithmRegistry
    from src.data.loader import load_psf_from_fits, prepare_psf_for_retrieval
    from src.metrics.quality import zernike_decomposition
    from src.models.config import (
        AlgorithmConfig, AlgorithmName, BetaSchedule, NoiseModel, default_hst_config,
    )
    from src.models.optics import PSFData
    from src.optics.pupils import build_pupil

    config = default_hst_config()
    config.output_dir = Path(args.output_dir)

    # Resolve FITS file
    if args.fits:
        fits_path = Path(args.fits)
    else:
        from src.data.downloader import list_cached_fits
        cached = list_cached_fits(config.data.data_dir)
        if not cached:
            logger.error("No FITS files found. Run 'phase-retrieval download' first.")
            sys.exit(1)
        fits_path = cached[0]

    pupil = build_pupil(config.pupil)
    support = pupil.amplitude > 0

    psf_data = load_psf_from_fits(fits_path, config.data, config.pupil)
    psf_image = prepare_psf_for_retrieval(psf_data, config.pupil.grid_size)
    psf_resized = PSFData(
        image=psf_image,
        pixel_scale_arcsec=psf_data.pixel_scale_arcsec,
        wavelength_m=psf_data.wavelength_m,
        filter_name=psf_data.filter_name,
        telescope=psf_data.telescope,
        obs_id=psf_data.obs_id,
    )

    alg_name = AlgorithmName(args.algorithm)
    alg_cfg = AlgorithmConfig(
        name=alg_name,
        max_iterations=args.iterations,
        beta=args.beta,
        beta_schedule=BetaSchedule(args.beta_schedule),
        random_seed=args.seed,
        momentum=args.momentum,
        tv_weight=args.tv_weight,
        noise_model=NoiseModel(args.noise_model),
        n_starts=args.n_starts,
    )

    if alg_cfg.n_starts > 1:
        result = multi_start_run(alg_cfg, pupil, psf_resized)
    else:
        retriever = AlgorithmRegistry.create(alg_cfg, pupil)
        result = retriever.run(psf_resized)

    print(
        f"✅ {alg_name.value.upper()} — "
        f"{result.n_iterations} iter, "
        f"Strehl={result.strehl_ratio:.4f}, "
        f"RMS={result.rms_phase_rad:.4f} rad, "
        f"{result.elapsed_seconds:.2f}s"
    )

    # Save JSON summary
    config.output_dir.mkdir(parents=True, exist_ok=True)
    summary = {
        "algorithm": result.algorithm.value,
        "n_iterations": result.n_iterations,
        "converged": result.converged,
        "strehl_ratio": result.strehl_ratio,
        "rms_phase_rad": result.rms_phase_rad,
        "elapsed_seconds": result.elapsed_seconds,
        "timestamp": result.timestamp.isoformat(),
    }
    out_path = config.output_dir / f"result_{alg_name.value}.json"
    out_path.write_text(json.dumps(summary, indent=2))
    print(f"📁 {out_path}")


# ── compare ───────────────────────────────────────────────────────────────

def _cmd_compare(args: argparse.Namespace) -> None:
    """Run all algorithms and print a comparison table."""
    from src.algorithms.multi_start import multi_start_run
    from src.algorithms.registry import AlgorithmRegistry
    from src.data.loader import load_psf_from_fits, prepare_psf_for_retrieval
    from src.models.config import (
        AlgorithmConfig, AlgorithmName, BetaSchedule, NoiseModel, default_hst_config,
    )
    from src.models.optics import PSFData
    from src.optics.pupils import build_pupil

    config = default_hst_config()
    config.output_dir = Path(args.output_dir)

    if args.fits:
        fits_path = Path(args.fits)
    else:
        from src.data.downloader import list_cached_fits
        cached = list_cached_fits(config.data.data_dir)
        if not cached:
            logger.error("No FITS files found. Run 'phase-retrieval download' first.")
            sys.exit(1)
        fits_path = cached[0]

    pupil = build_pupil(config.pupil)
    psf_data = load_psf_from_fits(fits_path, config.data, config.pupil)
    psf_image = prepare_psf_for_retrieval(psf_data, config.pupil.grid_size)
    psf_resized = PSFData(
        image=psf_image,
        pixel_scale_arcsec=psf_data.pixel_scale_arcsec,
        wavelength_m=psf_data.wavelength_m,
        filter_name=psf_data.filter_name,
        telescope=psf_data.telescope,
        obs_id=psf_data.obs_id,
    )

    algorithms = [
        AlgorithmName.ERROR_REDUCTION,
        AlgorithmName.GERCHBERG_SAXTON,
        AlgorithmName.HYBRID_INPUT_OUTPUT,
        AlgorithmName.RAAR,
        AlgorithmName.WIRTINGER_FLOW,
        AlgorithmName.DOUGLAS_RACHFORD,
        AlgorithmName.ADMM,
    ]

    print(f"{'Algorithm':>6s}  {'Iter':>5s}  {'Strehl':>8s}  {'RMS (rad)':>10s}  {'Time':>7s}")
    print("-" * 50)

    for alg_name in algorithms:
        n_iter = args.iterations
        alg_cfg = AlgorithmConfig(
            name=alg_name,
            max_iterations=n_iter,
            beta=args.beta,
            beta_schedule=BetaSchedule(args.beta_schedule),
            random_seed=args.seed,
            momentum=args.momentum,
            tv_weight=args.tv_weight,
            noise_model=NoiseModel(args.noise_model),
            n_starts=args.n_starts,
        )
        if alg_cfg.n_starts > 1:
            result = multi_start_run(alg_cfg, pupil, psf_resized)
        else:
            result = AlgorithmRegistry.create(alg_cfg, pupil).run(psf_resized)
        print(
            f"{alg_name.value.upper():>6s}  "
            f"{result.n_iterations:5d}  "
            f"{result.strehl_ratio:8.4f}  "
            f"{result.rms_phase_rad:10.4f}  "
            f"{result.elapsed_seconds:6.2f}s"
        )


# ── download ──────────────────────────────────────────────────────────────

def _cmd_download(args: argparse.Namespace) -> None:
    """Download observation data from MAST."""
    from src.data.downloader import available_presets, download_preset

    data_dir = Path(args.data_dir)

    if args.list:
        presets = available_presets()
        print("Available observation presets:")
        for key, desc in presets.items():
            print(f"  {key:30s}  {desc}")
        return

    key = args.preset
    print(f"⬇️  Downloading preset '{key}' → {data_dir}")
    paths = download_preset(key, data_dir)
    for p in paths:
        print(f"  ✅ {p}")


def _add_common_algo_args(parser: argparse.ArgumentParser) -> None:
    """Add shared algorithm arguments to a subcommand parser."""
    parser.add_argument("-n", "--iterations", type=int, default=500, help="Max iterations")
    parser.add_argument("--beta", type=float, default=0.9, help="HIO/RAAR/DR feedback β")
    parser.add_argument("--beta-schedule", default="constant",
                        choices=["constant", "linear", "cosine"],
                        help="Adaptive β schedule")
    parser.add_argument("--seed", type=int, default=42, help="Random seed")
    parser.add_argument("--fits", type=str, default=None, help="Path to FITS file")
    parser.add_argument("-o", "--output-dir", default="outputs", help="Output directory")
    parser.add_argument("--momentum", type=float, default=0.0,
                        help="Nesterov/heavy-ball momentum (0=off)")
    parser.add_argument("--tv-weight", type=float, default=0.0,
                        help="Total-variation regularization weight (0=off)")
    parser.add_argument("--noise-model", default="gaussian",
                        choices=["gaussian", "poisson"],
                        help="Noise model for focal-plane projection")
    parser.add_argument("--n-starts", type=int, default=1,
                        help="Multi-start: number of random restarts")


# ── main ──────────────────────────────────────────────────────────────────

def main(argv: list[str] | None = None) -> None:
    """CLI entry point."""
    from src import __version__

    parser = argparse.ArgumentParser(
        prog="phase-retrieval",
        description="Phase retrieval for astronomical wavefront sensing.",
    )
    parser.add_argument("-V", "--version", action="version", version=f"%(prog)s {__version__}")
    parser.add_argument("-v", "--verbose", action="store_true", help="Enable debug logging")
    sub = parser.add_subparsers(dest="command")

    # --- run ---
    p_run = sub.add_parser("run", help="Run a single algorithm on a FITS file")
    p_run.add_argument("-a", "--algorithm", default="hio",
                       help="Algorithm key (er, gs, hio, raar, wf, dr, admm)")
    _add_common_algo_args(p_run)
    p_run.set_defaults(func=_cmd_run)

    # --- compare ---
    p_cmp = sub.add_parser("compare", help="Compare all algorithms on the same data")
    _add_common_algo_args(p_cmp)
    p_cmp.set_defaults(func=_cmd_compare)

    # --- download ---
    p_dl = sub.add_parser("download", help="Download observation data from MAST")
    p_dl.add_argument("-p", "--preset", default="hst-wfc3-uvis-f606w", help="Observation preset key")
    p_dl.add_argument("-d", "--data-dir", default="data", help="Download directory")
    p_dl.add_argument("-l", "--list", action="store_true", help="List available presets")
    p_dl.set_defaults(func=_cmd_download)

    args = parser.parse_args(argv)
    _configure_logging(args.verbose)

    if args.command is None:
        parser.print_help()
        sys.exit(0)

    args.func(args)


if __name__ == "__main__":
    main()
