"""Command-line interface for the phase-retrieval pipeline.

Usage
-----
    phase-retrieval run   [--algorithm hio] [--iterations 500] [--fits PATH]
    phase-retrieval compare [--fits PATH] [--save]
    phase-retrieval download [--preset hst-wfc3-uvis-f606w]
"""

from __future__ import annotations

import argparse
import importlib.util
import json
import logging
import sys
from pathlib import Path
from typing import Any

logger = logging.getLogger(__name__)


# ---------------------------------------------------------------------------
# Logging configuration
# ---------------------------------------------------------------------------


class _JsonFormatter(logging.Formatter):
    """Emit log records as newline-delimited JSON objects."""

    def format(self, record: logging.LogRecord) -> str:
        payload = {
            "ts": self.formatTime(record, self.datefmt),
            "level": record.levelname,
            "logger": record.name,
            "msg": record.getMessage(),
        }
        return json.dumps(payload)


def _configure_logging(verbose: bool = False, log_format: str = "text") -> None:
    level = logging.DEBUG if verbose else logging.INFO
    if log_format == "json":
        handler = logging.StreamHandler(sys.stdout)
        handler.setFormatter(_JsonFormatter())
        logging.basicConfig(level=level, handlers=[handler])
    else:
        logging.basicConfig(
            level=level,
            format="%(asctime)s  %(name)-28s  %(levelname)-8s  %(message)s",
            stream=sys.stdout,
        )


# ---------------------------------------------------------------------------
# Shared helpers
# ---------------------------------------------------------------------------


def _sync_pupil_to_image(config: Any, image_shape: tuple[int, int]) -> Any:
    """Keep the pupil grid consistent with the prepared PSF image shape."""
    if len(image_shape) != 2 or image_shape[0] != image_shape[1]:
        raise ValueError(f"Prepared PSF must be square, got shape {image_shape}")
    image_size = image_shape[0]
    if image_size != config.pupil.grid_size:
        logger.warning(
            "PSF grid %dx%d != pupil grid %d; rebuilding pupil.",
            image_size,
            image_size,
            config.pupil.grid_size,
        )
        config.pupil = config.pupil.model_copy(update={"grid_size": image_size})
    return config


def _has_torch() -> bool:
    return importlib.util.find_spec("torch") is not None


def _algorithm_choices() -> list[str]:
    """Return the registered algorithm keys for argparse validation."""
    from src.algorithms.registry import AlgorithmRegistry

    return AlgorithmRegistry.available()


def _preset_choices() -> list[str]:
    """Return curated download preset keys for argparse validation."""
    from src.data.downloader import available_presets

    return sorted(available_presets())


def _benchmark_case_choices() -> list[str]:
    """Return built-in benchmark case keys for argparse/help text."""
    from src.benchmark import available_benchmark_cases

    return sorted(available_benchmark_cases())


def _parse_algorithm_csv(value: str) -> list[str]:
    """Parse a comma-separated algorithm list."""
    return [item.strip() for item in value.split(",") if item.strip()]


def _parse_case_csv(value: str) -> list[str]:
    """Parse a comma-separated benchmark case list."""
    return [item.strip() for item in value.split(",") if item.strip()]


def _load_psf_and_pupil(args: argparse.Namespace, config: Any) -> Any:
    """Shared PSF loading + pupil building logic for `run` and `compare`.

    Returns
    -------
    psf_resized : PSFData
        PSF data resized to the algorithm grid.
    pupil : PupilModel
        Built pupil matching the grid.
    config : PipelineConfig
        Possibly updated config (grid_size may be adjusted).
    """
    from src.data.loader import load_psf_from_fits, prepare_psf_for_retrieval
    from src.models.optics import PSFData
    from src.optics.pupils import build_pupil

    if args.fits:
        fits_path = Path(args.fits)
    else:
        from src.data.downloader import list_cached_fits

        cached = list_cached_fits(config.data.data_dir)
        if not cached:
            logger.error("No FITS files found. Run 'phase-retrieval download' first.")
            sys.exit(1)
        fits_path = cached[0]

    psf_data = load_psf_from_fits(fits_path, config.data, config.pupil)
    calibration = psf_data.metadata.get("calibration", {})
    suggested_pupil = calibration.get("suggested_pupil", {})
    if suggested_pupil:
        config.pupil = config.pupil.model_copy(
            update={
                "telescope": suggested_pupil.get("telescope", config.pupil.telescope),
                "pixel_scale_arcsec": suggested_pupil.get(
                    "pixel_scale_arcsec", config.pupil.pixel_scale_arcsec
                ),
                "wavelength_m": suggested_pupil.get("wavelength_m", config.pupil.wavelength_m),
            }
        )
        config.data = config.data.model_copy(
            update={
                "detector": psf_data.metadata.get("detector", config.data.detector),
                "filter_name": psf_data.filter_name,
            }
        )
    psf_image = prepare_psf_for_retrieval(psf_data, config.pupil.grid_size)
    config = _sync_pupil_to_image(config, (psf_image.shape[0], psf_image.shape[1]))
    pupil = build_pupil(config.pupil)
    psf_resized = PSFData(
        image=psf_image,
        pixel_scale_arcsec=psf_data.pixel_scale_arcsec,
        wavelength_m=psf_data.wavelength_m,
        filter_name=psf_data.filter_name,
        telescope=psf_data.telescope,
        obs_id=psf_data.obs_id,
        metadata={
            **psf_data.metadata,
            "prepared_grid_size": int(psf_image.shape[0]),
        },
    )
    return psf_resized, pupil, config


def _save_compare_plots(
    *,
    psf_data: Any,
    pupil: Any,
    results: dict[str, Any],
    output_dir: Path,
) -> dict[str, str]:
    """Save standard comparison plots for a multi-algorithm run."""
    import matplotlib

    matplotlib.use("Agg")
    import matplotlib.pyplot as plt

    from src.visualization.plots import (
        plot_algorithm_comparison,
        plot_algorithm_dashboard,
        plot_strehl_rms_bar,
        save_figure,
    )

    support = pupil.amplitude > 0
    output_dir.mkdir(parents=True, exist_ok=True)

    artifacts = {
        "algorithm_comparison_plot": str(output_dir / "algorithm_comparison.png"),
        "algorithm_dashboard_plot": str(output_dir / "algorithm_dashboard.png"),
        "strehl_rms_plot": str(output_dir / "strehl_rms_comparison.png"),
    }

    fig = plot_algorithm_comparison(results, support)
    save_figure(fig, artifacts["algorithm_comparison_plot"])
    plt.close(fig)

    fig = plot_algorithm_dashboard(psf_data, results, support, pupil)
    save_figure(fig, artifacts["algorithm_dashboard_plot"])
    plt.close(fig)

    fig = plot_strehl_rms_bar(results)
    save_figure(fig, artifacts["strehl_rms_plot"])
    plt.close(fig)

    return artifacts


# ── run ───────────────────────────────────────────────────────────────────


def _cmd_run(args: argparse.Namespace) -> None:
    """Run a single phase-retrieval algorithm on a FITS file."""
    from src.algorithms.multi_start import multi_start_run
    from src.algorithms.registry import AlgorithmRegistry
    from src.metrics.quality import (
        compute_encircled_energy_error,
        compute_radial_profile_error,
        compute_ssim,
        summarise_convergence,
        zernike_decomposition,
    )
    from src.models.config import (
        AlgorithmConfig,
        AlgorithmName,
        BetaSchedule,
        NoiseModel,
        default_hst_config,
    )
    from src.reporting import build_evaluation_payload, write_evaluation_report
    from src.uncertainty import run_uncertainty_analysis
    from src.validation import compare_against_reference

    config = default_hst_config()
    config.output_dir = Path(args.output_dir)
    config.uncertainty_samples = getattr(args, "uncertainty_samples", 0)

    psf_resized, pupil, config = _load_psf_and_pupil(args, config)

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

    support = pupil.amplitude > 0
    zernike_coeffs = zernike_decomposition(result.recovered_phase, support)
    ssim = compute_ssim(psf_resized.image, result.reconstructed_psf)
    radial_profile_error = compute_radial_profile_error(
        psf_resized.image,
        result.reconstructed_psf,
    )
    encircled_energy_error = compute_encircled_energy_error(
        psf_resized.image,
        result.reconstructed_psf,
    )
    convergence = summarise_convergence(result.cost_history)
    uncertainty_summary: dict[str, Any] = {}
    if config.uncertainty_samples > 0:
        uncertainty = run_uncertainty_analysis(
            psf_data=psf_resized,
            pupil=pupil,
            algorithm_config=alg_cfg,
            n_samples=config.uncertainty_samples,
            shift_sigma_pixels=config.uncertainty_shift_sigma_pixels,
            background_sigma_fraction=config.uncertainty_background_sigma_fraction,
            noise_sigma_fraction=config.uncertainty_noise_sigma_fraction,
            seed=config.uncertainty_seed,
        )
        uncertainty_summary = {
            "n_samples": uncertainty.n_samples,
            "summary": uncertainty.summary,
            "sample_records": uncertainty.sample_records,
        }

    summary = {
        "algorithm": result.algorithm.value,
        "n_iterations": result.n_iterations,
        "converged": result.converged,
        "strehl_ratio": result.strehl_ratio,
        "rms_phase_rad": result.rms_phase_rad,
        "ssim": ssim,
        "radial_profile_error": radial_profile_error,
        "encircled_energy_error": encircled_energy_error,
        "elapsed_seconds": result.elapsed_seconds,
        "timestamp": result.timestamp.isoformat(),
    }
    reference_validation = compare_against_reference(
        observed_psf=psf_resized.image,
        reconstructed_psf=result.reconstructed_psf,
        pixel_scale_arcsec=psf_resized.pixel_scale_arcsec,
        telescope=psf_resized.telescope,
        detector=str(psf_resized.metadata.get("detector", "")),
        filter_name=psf_resized.filter_name,
    )

    output_format = getattr(args, "output_format", "text")
    quiet = getattr(args, "quiet", False)

    if output_format == "json":
        print(json.dumps(summary, indent=2))
    elif not quiet:
        print(
            f"✅ {alg_name.value.upper()} — "
            f"{result.n_iterations} iter, "
            f"Strehl={result.strehl_ratio:.4f}, "
            f"RMS={result.rms_phase_rad:.4f} rad, "
            f"{result.elapsed_seconds:.2f}s"
        )

    # Save JSON summary
    config.output_dir.mkdir(parents=True, exist_ok=True)
    out_path = config.output_dir / f"result_{alg_name.value}.json"
    out_path.write_text(json.dumps(summary, indent=2))
    if uncertainty_summary:
        (config.output_dir / f"uncertainty_{alg_name.value}.json").write_text(
            json.dumps(uncertainty_summary, indent=2)
        )
    if reference_validation:
        (config.output_dir / f"reference_validation_{alg_name.value}.json").write_text(
            json.dumps(reference_validation, indent=2)
        )
    evaluation_payload = build_evaluation_payload(
        psf_metadata={**psf_resized.metadata, "obs_id": psf_resized.obs_id},
        algorithm_name=alg_name.value,
        algorithm_config={
            "max_iterations": alg_cfg.max_iterations,
            "beta": alg_cfg.beta,
            "beta_schedule": alg_cfg.beta_schedule.value,
            "momentum": alg_cfg.momentum,
            "tv_weight": alg_cfg.tv_weight,
            "noise_model": alg_cfg.noise_model.value,
            "n_starts": alg_cfg.n_starts,
            "random_seed": alg_cfg.random_seed,
        },
        pupil_summary={
            "grid_size": pupil.grid_size,
            "support_pixels": int((pupil.amplitude > 0).sum()),
            "approximate_model": True,
            "wavelength_m": pupil.wavelength_m,
            "bandwidth_fraction": pupil.bandwidth_fraction,
            "spectral_samples": pupil.spectral_samples,
            "spectral_weighting": pupil.spectral_weighting,
            "field_defocus_waves": pupil.field_defocus_waves,
            "detector_sigma_pixels": pupil.detector_sigma_pixels,
            "jitter_sigma_pixels": pupil.jitter_sigma_pixels,
            "pixel_integration_width": pupil.pixel_integration_width,
        },
        metrics={
            **summary,
            "convergence": convergence,
            "uncertainty": uncertainty_summary,
            "reference_validation": reference_validation,
        },
        zernike_coefficients=zernike_coeffs,
    )
    report_paths = write_evaluation_report(
        evaluation_payload,
        config.output_dir,
        stem=f"evaluation_{alg_name.value}",
    )
    if not quiet and output_format != "json":
        print(f"📁 {out_path}")
        print(f"📁 {report_paths['json']}")
        print(f"📁 {report_paths['markdown']}")


# ── compare ───────────────────────────────────────────────────────────────


def _cmd_compare(args: argparse.Namespace) -> None:
    """Run all algorithms and print a comparison table."""
    from src.algorithms.multi_start import multi_start_run
    from src.algorithms.registry import AlgorithmRegistry
    from src.metrics.quality import (
        compute_encircled_energy_error,
        compute_radial_profile_error,
        compute_ssim,
    )
    from src.models.config import (
        AlgorithmConfig,
        AlgorithmName,
        BetaSchedule,
        NoiseModel,
        default_hst_config,
    )
    from src.reporting import build_comparison_payload, write_comparison_report
    from src.validation import compare_against_reference

    config = default_hst_config()
    config.output_dir = Path(args.output_dir)

    psf_resized, pupil, config = _load_psf_and_pupil(args, config)

    algorithms = [
        AlgorithmName.ERROR_REDUCTION,
        AlgorithmName.GERCHBERG_SAXTON,
        AlgorithmName.HYBRID_INPUT_OUTPUT,
        AlgorithmName.RAAR,
        AlgorithmName.WIRTINGER_FLOW,
        AlgorithmName.DOUGLAS_RACHFORD,
        AlgorithmName.ADMM,
    ]
    if _has_torch():
        algorithms.append(AlgorithmName.PINN)

    quiet = getattr(args, "quiet", False)
    save = getattr(args, "save", False)

    if not quiet:
        print(f"{'Algorithm':>6s}  {'Iter':>5s}  {'Strehl':>8s}  {'RMS (rad)':>10s}  {'Time':>7s}")
        print("-" * 50)

    results_summary = []
    plot_results: dict[str, Any] = {}
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

        summary: dict[str, Any] = {
            "algorithm": result.algorithm.value,
            "n_iterations": result.n_iterations,
            "converged": result.converged,
            "strehl_ratio": result.strehl_ratio,
            "rms_phase_rad": result.rms_phase_rad,
            "ssim": compute_ssim(psf_resized.image, result.reconstructed_psf),
            "radial_profile_error": compute_radial_profile_error(
                psf_resized.image,
                result.reconstructed_psf,
            ),
            "encircled_energy_error": compute_encircled_energy_error(
                psf_resized.image,
                result.reconstructed_psf,
            ),
            "elapsed_seconds": result.elapsed_seconds,
            "timestamp": result.timestamp.isoformat(),
        }
        summary["reference_validation"] = compare_against_reference(
            observed_psf=psf_resized.image,
            reconstructed_psf=result.reconstructed_psf,
            pixel_scale_arcsec=psf_resized.pixel_scale_arcsec,
            telescope=psf_resized.telescope,
            detector=str(psf_resized.metadata.get("detector", "")),
            filter_name=psf_resized.filter_name,
        )
        results_summary.append(summary)
        plot_results[alg_name.value.upper()] = result

        if not quiet:
            print(
                f"{alg_name.value.upper():>6s}  "
                f"{result.n_iterations:5d}  "
                f"{result.strehl_ratio:8.4f}  "
                f"{result.rms_phase_rad:10.4f}  "
                f"{result.elapsed_seconds:6.2f}s"
            )

        if save:
            config.output_dir.mkdir(parents=True, exist_ok=True)
            out_path = config.output_dir / f"result_{alg_name.value}.json"
            out_path.write_text(json.dumps(summary, indent=2))
            if not quiet:
                print(f"  📁 {out_path}")

    if save and results_summary:
        plot_artifacts = _save_compare_plots(
            psf_data=psf_resized,
            pupil=pupil,
            results=plot_results,
            output_dir=config.output_dir,
        )
        comparison_payload = build_comparison_payload(
            source_metadata={**psf_resized.metadata, "obs_id": psf_resized.obs_id},
            summaries=results_summary,
            artifacts=plot_artifacts,
        )
        report_paths = write_comparison_report(comparison_payload, config.output_dir)
        if not quiet:
            print(f"📁 {report_paths['json']}")
            print(f"📁 {report_paths['markdown']}")
            for path in plot_artifacts.values():
                print(f"📁 {path}")


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


# ── benchmark ─────────────────────────────────────────────────────────────


def _cmd_benchmark(args: argparse.Namespace) -> None:
    """Run the deterministic synthetic benchmark suite and export reports."""
    from src.benchmark import available_benchmark_cases, run_benchmark
    from src.models.config import AlgorithmName

    case_map = available_benchmark_cases()
    algorithms = [AlgorithmName(key) for key in _parse_algorithm_csv(args.algorithms)]
    case_keys = _parse_case_csv(args.cases)
    unknown_cases = [key for key in case_keys if key not in case_map]
    if unknown_cases:
        raise SystemExit(f"Unknown benchmark case(s): {', '.join(unknown_cases)}")

    summary = run_benchmark(
        algorithms=algorithms,
        cases=[case_map[key] for key in case_keys],
        max_iterations=args.iterations,
        beta=args.beta,
        random_seed=args.seed,
        output_dir=Path(args.output_dir),
    )

    if getattr(args, "quiet", False):
        return

    print(f"{'Rank':>4s}  {'Algorithm':>10s}  {'Score':>8s}  {'SSIM':>8s}  {'PhaseErr':>10s}")
    print("-" * 52)
    for idx, row in enumerate(summary.aggregate, start=1):
        print(
            f"{idx:4d}  {row['algorithm']:>10s}  {row['mean_score']:8.4f}  "
            f"{row['mean_ssim']:8.4f}  {row['mean_phase_rms_error_rad']:10.4f}"
        )
    print(f"📁 {Path(args.output_dir) / 'benchmark_results.json'}")
    print(f"📁 {Path(args.output_dir) / 'benchmark_summary.csv'}")
    print(f"📁 {Path(args.output_dir) / 'benchmark_report.md'}")
    print(f"📁 {Path(args.output_dir) / 'benchmark_study.json'}")
    print(f"📁 {Path(args.output_dir) / 'benchmark_study.csv'}")
    print(f"📁 {Path(args.output_dir) / 'benchmark_leaderboard.png'}")
    print(f"📁 {Path(args.output_dir) / 'benchmark_case_heatmap.png'}")


def _add_common_algo_args(parser: argparse.ArgumentParser) -> None:
    """Add shared algorithm arguments to a subcommand parser."""
    parser.add_argument("-n", "--iterations", type=int, default=500, help="Max iterations")
    parser.add_argument("--beta", type=float, default=0.9, help="HIO/RAAR/DR feedback β")
    parser.add_argument(
        "--beta-schedule",
        default="constant",
        choices=["constant", "linear", "cosine"],
        help="Adaptive β schedule",
    )
    parser.add_argument("--seed", type=int, default=42, help="Random seed")
    parser.add_argument("--fits", type=str, default=None, help="Path to FITS file")
    parser.add_argument("-o", "--output-dir", default="outputs", help="Output directory")
    parser.add_argument(
        "--momentum", type=float, default=0.0, help="Nesterov/heavy-ball momentum (0=off)"
    )
    parser.add_argument(
        "--tv-weight", type=float, default=0.0, help="Total-variation regularization weight (0=off)"
    )
    parser.add_argument(
        "--noise-model",
        default="gaussian",
        choices=["gaussian", "poisson"],
        help="Noise model for focal-plane projection",
    )
    parser.add_argument(
        "--n-starts", type=int, default=1, help="Multi-start: number of random restarts"
    )
    parser.add_argument(
        "--uncertainty-samples",
        type=int,
        default=0,
        help="Perturbation runs for uncertainty estimation (0=off)",
    )
    parser.add_argument(
        "--quiet", "-q", action="store_true", help="Suppress progress output (useful for scripting)"
    )


# ── cryst (crystallography) ───────────────────────────────────────────────


def _cmd_cryst(args: argparse.Namespace) -> None:
    """Run crystallographic phase retrieval on a CIF file."""
    from src.data.crystallography import (
        available_cod_presets,
        download_cod_preset,
        parse_cif,
        run_crystallography_retrieval,
        simulate_diffraction,
    )

    cif_arg = args.cif
    quiet = getattr(args, "quiet", False)

    # Check if it's a preset key
    presets = available_cod_presets()
    if cif_arg in presets:
        if not quiet:
            print(f"⬇️  Downloading COD preset '{cif_arg}'...")
        cif_path = download_cod_preset(cif_arg, Path("data"))
    else:
        cif_path = Path(cif_arg)
        if not cif_path.exists():
            logger.error("CIF file not found: %s", cif_path)
            sys.exit(1)

    # Parse and simulate
    crystal = parse_cif(cif_path)
    if not quiet:
        print(f"🔬 {crystal.formula} — {crystal.space_group}")
        print(f"   a={crystal.a:.2f} b={crystal.b:.2f} c={crystal.c:.2f} Å")

    diffraction = simulate_diffraction(crystal, grid_size=args.grid_size)

    # Run phase retrieval
    result = run_crystallography_retrieval(
        diffraction,
        algorithm_name=args.algorithm,
        max_iterations=args.iterations,
        beta=args.beta,
        random_seed=args.seed,
    )

    if not quiet:
        print(
            f"✅ {args.algorithm.upper()} — "
            f"{result.n_iterations} iter, "
            f"R-factor={result.r_factor:.4f}, "
            f"{result.elapsed_seconds:.2f}s"
        )

    # Save results
    out_dir = Path(args.output_dir)
    out_dir.mkdir(parents=True, exist_ok=True)
    summary = {
        "algorithm": result.algorithm.value,
        "formula": crystal.formula,
        "space_group": crystal.space_group,
        "r_factor": result.r_factor,
        "n_iterations": result.n_iterations,
        "converged": result.converged,
        "elapsed_seconds": result.elapsed_seconds,
    }
    out_path = out_dir / f"cryst_result_{args.algorithm}.json"
    out_path.write_text(json.dumps(summary, indent=2))
    if not quiet:
        print(f"📁 {out_path}")


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
    parser.add_argument(
        "--log-format",
        default="text",
        choices=["text", "json"],
        help="Log output format: human-readable text or machine-readable JSON",
    )
    sub = parser.add_subparsers(dest="command")

    # --- run ---
    p_run = sub.add_parser("run", help="Run a single algorithm on a FITS file")
    p_run.add_argument(
        "-a",
        "--algorithm",
        default="hio",
        choices=_algorithm_choices(),
        metavar="{" + ",".join(_algorithm_choices()) + "}",
        help="Algorithm key (er, gs, hio, raar, wf, dr, admm, pinn)",
    )
    p_run.add_argument(
        "--output-format",
        default="text",
        choices=["text", "json"],
        help="Summary output format: text (default) or json (machine-readable stdout)",
    )
    _add_common_algo_args(p_run)
    p_run.set_defaults(func=_cmd_run)

    # --- compare ---
    p_cmp = sub.add_parser("compare", help="Compare all algorithms on the same data")
    p_cmp.add_argument(
        "--save",
        action="store_true",
        default=True,
        help="Save per-algorithm JSON result files to output_dir (default: on)",
    )
    p_cmp.add_argument(
        "--no-save",
        dest="save",
        action="store_false",
        help="Disable saving per-algorithm JSON result files",
    )
    _add_common_algo_args(p_cmp)
    p_cmp.set_defaults(func=_cmd_compare)

    # --- download ---
    p_dl = sub.add_parser("download", help="Download observation data from MAST")
    p_dl.add_argument(
        "-p",
        "--preset",
        default="hst-wfc3-uvis-f606w",
        choices=_preset_choices(),
        metavar="PRESET",
        help="Observation preset key",
    )
    p_dl.add_argument("-d", "--data-dir", default="data", help="Download directory")
    p_dl.add_argument("-l", "--list", action="store_true", help="List available presets")
    p_dl.set_defaults(func=_cmd_download)

    # --- benchmark ---
    p_bench = sub.add_parser(
        "benchmark",
        help="Run deterministic synthetic benchmarks and export JSON/CSV/Markdown reports",
    )
    p_bench.add_argument(
        "--algorithms",
        default="er,gs,hio,raar,wf,dr,admm,fista,sparse_pr",
        help="Comma-separated algorithm keys to benchmark",
    )
    p_bench.add_argument(
        "--cases",
        default=",".join(_benchmark_case_choices()),
        help="Comma-separated benchmark case keys",
    )
    p_bench.add_argument("-n", "--iterations", type=int, default=40, help="Max iterations")
    p_bench.add_argument("--beta", type=float, default=0.9, help="Feedback β")
    p_bench.add_argument("--seed", type=int, default=42, help="Random seed")
    p_bench.add_argument("-o", "--output-dir", default="outputs/benchmark", help="Report directory")
    p_bench.add_argument("--quiet", "-q", action="store_true", help="Suppress stdout table")
    p_bench.set_defaults(func=_cmd_benchmark)

    # --- cryst (crystallography) ---
    p_cryst = sub.add_parser("cryst", help="Run crystallographic phase retrieval on a CIF file")
    p_cryst.add_argument(
        "cif",
        type=str,
        help="Path to CIF file or COD preset key (nacl, quartz, diamond, ...)",
    )
    p_cryst.add_argument(
        "-a",
        "--algorithm",
        default="hio",
        choices=_algorithm_choices(),
        help="Algorithm key",
    )
    p_cryst.add_argument("-n", "--iterations", type=int, default=500, help="Max iterations")
    p_cryst.add_argument("--beta", type=float, default=0.9, help="Feedback β")
    p_cryst.add_argument("--grid-size", type=int, default=128, help="Grid size")
    p_cryst.add_argument("--seed", type=int, default=42, help="Random seed")
    p_cryst.add_argument("-o", "--output-dir", default="outputs", help="Output directory")
    p_cryst.add_argument("--quiet", "-q", action="store_true", help="Suppress progress output")
    p_cryst.set_defaults(func=_cmd_cryst)

    args = parser.parse_args(argv)
    _configure_logging(args.verbose, getattr(args, "log_format", "text"))

    if args.command is None:
        parser.print_help()
        sys.exit(0)

    args.func(args)


if __name__ == "__main__":  # pragma: no cover
    main()
