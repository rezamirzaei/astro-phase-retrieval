"""Reporting helpers for single-run evaluations and multi-algorithm comparisons."""

from __future__ import annotations

import json
from datetime import UTC, datetime
from pathlib import Path
from typing import Any


def _bucket(value: float, *, good: float, acceptable: float, higher_is_better: bool = True) -> str:
    """Return a qualitative bucket for a scalar metric."""
    if higher_is_better:
        if value >= good:
            return "strong"
        if value >= acceptable:
            return "moderate"
        return "weak"
    if value <= good:
        return "strong"
    if value <= acceptable:
        return "moderate"
    return "weak"


def _top_zernike_terms(coefficients: dict[int, float], n_terms: int = 5) -> list[dict[str, float | int]]:
    """Return the strongest Zernike terms by absolute magnitude."""
    ranked = sorted(coefficients.items(), key=lambda item: abs(item[1]), reverse=True)
    return [
        {"noll_index": int(index), "coefficient_rad": float(value)}
        for index, value in ranked[:n_terms]
    ]


def build_evaluation_payload(
    *,
    psf_metadata: dict[str, Any],
    algorithm_name: str,
    algorithm_config: dict[str, Any],
    pupil_summary: dict[str, Any],
    metrics: dict[str, Any],
    zernike_coefficients: dict[int, float],
) -> dict[str, Any]:
    """Build a machine-readable single-run evaluation payload."""
    source_kind = str(psf_metadata.get("source_kind", "unknown"))
    data_regime = "real" if source_kind == "fits" else ("synthetic" if source_kind == "npy" else source_kind)

    ssim = float(metrics.get("ssim", 0.0))
    radial_error = float(metrics.get("radial_profile_error", 0.0))
    ee_error = float(metrics.get("encircled_energy_error", 0.0))
    strehl = float(metrics.get("strehl_ratio", 0.0))
    rms_phase = float(metrics.get("rms_phase_rad", 0.0))

    interpretation = {
        "ssim": _bucket(ssim, good=0.98, acceptable=0.93, higher_is_better=True),
        "radial_profile_error": _bucket(radial_error, good=0.05, acceptable=0.15, higher_is_better=False),
        "encircled_energy_error": _bucket(ee_error, good=0.01, acceptable=0.03, higher_is_better=False),
        "strehl_ratio": _bucket(strehl, good=0.8, acceptable=0.5, higher_is_better=True),
        "rms_phase_rad": _bucket(rms_phase, good=0.2, acceptable=0.5, higher_is_better=False),
    }

    return {
        "report_type": "single_run_evaluation",
        "generated_at": datetime.now(UTC).isoformat(),
        "data_regime": data_regime,
        "algorithm": algorithm_name,
        "algorithm_config": algorithm_config,
        "source": psf_metadata,
        "pupil": pupil_summary,
        "metrics": metrics,
        "interpretation": interpretation,
        "top_zernike_terms": _top_zernike_terms(zernike_coefficients),
    }


def render_evaluation_markdown(payload: dict[str, Any]) -> str:
    """Render a paper-style Markdown summary for a single run."""
    metrics = payload["metrics"]
    source = payload["source"]
    interpretation = payload["interpretation"]

    lines = [
        "# Real-Data Evaluation Report" if payload["data_regime"] == "real" else "# Evaluation Report",
        "",
        "## Abstract",
        "",
        f"This report summarises one `{payload['algorithm']}` reconstruction on a "
        f"`{payload['data_regime']}` dataset. The evaluation combines image-fidelity, "
        "radial-profile, encircled-energy, and convergence diagnostics to provide a "
        "reproducible quality snapshot.",
        "",
        "## Data",
        "",
        f"- Source kind: `{source.get('source_kind', 'unknown')}`",
        f"- Source file: `{source.get('source_filename', 'unknown')}`",
        f"- Observation ID: `{source.get('header', {}).get('ROOTNAME', source.get('obs_id', 'unknown'))}`",
        f"- Filter: `{source.get('header', {}).get('FILTER', source.get('header', {}).get('FILTER2', 'unknown'))}`",
        f"- Preprocessing: {', '.join(source.get('preprocessing', [])) or 'not recorded'}",
        "",
        "## Method",
        "",
        f"- Algorithm: `{payload['algorithm']}`",
        f"- Max iterations: `{payload['algorithm_config'].get('max_iterations', 'unknown')}`",
        f"- Noise model: `{payload['algorithm_config'].get('noise_model', 'unknown')}`",
        f"- Momentum: `{payload['algorithm_config'].get('momentum', 'unknown')}`",
        "",
        "## Quantitative Results",
        "",
        f"- Strehl ratio: **{metrics.get('strehl_ratio', 0.0):.4f}** ({interpretation['strehl_ratio']})",
        f"- RMS phase error: **{metrics.get('rms_phase_rad', 0.0):.4f} rad** ({interpretation['rms_phase_rad']})",
        f"- SSIM: **{metrics.get('ssim', 0.0):.4f}** ({interpretation['ssim']})",
        f"- Radial-profile error: **{metrics.get('radial_profile_error', 0.0):.4f}** ({interpretation['radial_profile_error']})",
        f"- Encircled-energy error: **{metrics.get('encircled_energy_error', 0.0):.4f}** ({interpretation['encircled_energy_error']})",
        f"- Iterations: **{metrics.get('n_iterations', 0)}**",
        f"- Elapsed time: **{metrics.get('elapsed_seconds', 0.0):.3f} s**",
        "",
        "## Dominant Zernike Terms",
        "",
        "| Noll index | Coefficient (rad) |",
        "|-----------:|------------------:|",
    ]
    for term in payload["top_zernike_terms"]:
        lines.append(f"| {term['noll_index']} | {term['coefficient_rad']:.4f} |")

    lines.extend(
        [
            "",
            "## Reproducibility",
            "",
            "All quantitative values in this report are mirrored in machine-readable JSON "
            "artifacts, allowing the result to be compared across commits, datasets, and algorithms.",
            "",
        ]
    )
    return "\n".join(lines)


def write_evaluation_report(payload: dict[str, Any], output_dir: Path, stem: str = "evaluation_report") -> dict[str, Path]:
    """Persist single-run evaluation reports as JSON and Markdown."""
    output_dir.mkdir(parents=True, exist_ok=True)
    json_path = output_dir / f"{stem}.json"
    md_path = output_dir / f"{stem}.md"
    json_path.write_text(json.dumps(payload, indent=2), encoding="utf-8")
    md_path.write_text(render_evaluation_markdown(payload), encoding="utf-8")
    return {"json": json_path, "markdown": md_path}


def build_comparison_payload(
    *,
    source_metadata: dict[str, Any],
    summaries: list[dict[str, Any]],
    artifacts: dict[str, str] | None = None,
) -> dict[str, Any]:
    """Build a machine-readable comparison payload for one dataset across algorithms."""
    ranked = sorted(
        summaries,
        key=lambda row: (
            -float(row.get("ssim", 0.0)),
            float(row.get("radial_profile_error", 0.0)),
            float(row.get("encircled_energy_error", 0.0)),
        ),
    )
    return {
        "report_type": "comparison_report",
        "generated_at": datetime.now(UTC).isoformat(),
        "source": source_metadata,
        "ranked_results": ranked,
        "best_algorithm": ranked[0]["algorithm"] if ranked else None,
        "artifacts": artifacts or {},
    }


def render_comparison_markdown(payload: dict[str, Any]) -> str:
    """Render a paper-style comparison report in Markdown."""
    lines = [
        "# Multi-Algorithm Evaluation Report",
        "",
        "## Abstract",
        "",
        "This report compares multiple retrieval algorithms on a single observation using "
        "shared fidelity and convergence metrics.",
        "",
        "## Data",
        "",
        f"- Source file: `{payload['source'].get('source_filename', 'unknown')}`",
        f"- Source kind: `{payload['source'].get('source_kind', 'unknown')}`",
        f"- Best algorithm by ranking: `{payload.get('best_algorithm', 'n/a')}`",
        "",
        "## Results",
        "",
        "| Rank | Algorithm | SSIM | Strehl | RMS phase (rad) | Radial err | EE err | Time (s) |",
        "|-----:|-----------|-----:|-------:|----------------:|-----------:|-------:|---------:|",
    ]
    for idx, row in enumerate(payload["ranked_results"], start=1):
        lines.append(
            f"| {idx} | {row['algorithm']} | {row['ssim']:.4f} | {row['strehl_ratio']:.4f} | "
            f"{row['rms_phase_rad']:.4f} | {row['radial_profile_error']:.4f} | "
            f"{row['encircled_energy_error']:.4f} | {row['elapsed_seconds']:.3f} |"
        )
    if payload.get("artifacts"):
        lines.extend([
            "",
            "## Comparison Plots",
            "",
        ])
        for key, value in payload["artifacts"].items():
            lines.append(f"- `{key}`: `{value}`")
    lines.append("")
    return "\n".join(lines)


def write_comparison_report(payload: dict[str, Any], output_dir: Path, stem: str = "comparison_report") -> dict[str, Path]:
    """Persist comparison reports as JSON and Markdown."""
    output_dir.mkdir(parents=True, exist_ok=True)
    json_path = output_dir / f"{stem}.json"
    md_path = output_dir / f"{stem}.md"
    json_path.write_text(json.dumps(payload, indent=2), encoding="utf-8")
    md_path.write_text(render_comparison_markdown(payload), encoding="utf-8")
    return {"json": json_path, "markdown": md_path}


