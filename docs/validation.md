# Validation

## Abstract

This project validates phase-retrieval performance using two complementary
regimes:

1. **Deterministic synthetic benchmarks** with known ground-truth phase, used for
   regression testing and cross-algorithm comparison.
2. **Real-data evaluation reports** on FITS observations, used to quantify how
   well reconstructed PSFs reproduce measured data when ground truth is not
   directly available.

The goal is not to claim perfect scientific completeness, but to make every run
more reproducible, inspectable, and comparable across commits.

## Validation Questions

The validation workflow is designed to answer four practical questions:

1. **Does the algorithm reduce the focal-plane error consistently?**
2. **Does the reconstructed PSF match the measured PSF visually and
   quantitatively?**
3. **Do different algorithms rank similarly across clean and noisy scenarios?**
4. **Can a real FITS-based run be audited later from saved artifacts alone?**

## Synthetic Benchmark Design

The built-in benchmark suite uses deterministic synthetic PSFs with controlled
aberration level, pupil geometry, and noise model. The default cases span:

- low-aberration, noiseless circular pupils
- moderate-aberration HST-like pupils
- photon-limited HST-like data
- high-aberration, noisy HST-like data

Each benchmark exports:

- `benchmark_results.json` — per-case/per-algorithm raw measurements
- `benchmark_summary.csv` — aggregate ranking table
- `benchmark_study.json` / `benchmark_study.csv` — convergence, robustness,
  and failure-mode summaries across case families
- `benchmark_report.md` — human-readable summary
- `benchmark_leaderboard.png` — ranked aggregate score chart
- `benchmark_case_heatmap.png` — per-case comparison heatmap

These artifacts are intended for CI, regression checks, and paper-style figures.

## Real-Data Evaluation Workflow

For FITS-based runs, the pipeline records provenance and quantitative quality
indicators even when no direct phase ground truth is available.

Saved artifacts include:

- `provenance.json` — source filename, checksum, selected header values,
  preprocessing history, pupil summary, and algorithm configuration
- `metrics.json` — SSIM, radial-profile error, encircled-energy error,
  convergence summary, Zernike coefficients, and optional reference-baseline
  comparisons
- `reference_validation.json` — curated external HST/JWST baseline comparison
  when a supported detector/filter match is available
- `evaluation_report.json` / `evaluation_report.md` — paper-style narrative
  summary of one run

For CLI-driven single runs, the equivalent per-algorithm files are written as
`evaluation_<algorithm>.json` and `evaluation_<algorithm>.md`.

For CLI `compare` runs with saving enabled, the workflow also writes
multi-algorithm visual artifacts:

- `algorithm_comparison.png`
- `algorithm_dashboard.png`
- `strehl_rms_comparison.png`

## Metrics

The validation workflow deliberately mixes image-space and optics-aware metrics:

- **SSIM** — structural similarity between observed and reconstructed PSFs
- **Radial-profile error** — mismatch in azimuthally averaged PSF structure
- **Encircled-energy error** — mismatch in cumulative energy concentration
- **Strehl ratio** — peak quality relative to the diffraction-limited PSF
- **RMS phase error** — compact summary of recovered wavefront strength
- **Convergence summary** — initial/final cost, improvement ratio, monotonicity
- **Zernike decomposition** — interpretable wavefront-mode summary
- **Reference FWHM / encircled-energy comparison** — deviation from curated
  STScI HST/JWST instrument-baseline values when a supported match exists

No single metric is treated as definitive; the intended use is triangulation.

## External Reference Baselines

For selected real-data configurations, the repository now compares observed and
reconstructed PSFs against curated STScI handbook or JWST documentation values.
The current curated set is intentionally narrow and evidence-based:

- HST WFC3/UVIS F606W
- HST ACS/WFC F814W
- JWST NIRCam F200W
- JWST NIRCam F356W

These checks are meant to answer a modest question: **is the PSF grossly
inconsistent with trusted instrument-scale behavior?** They are not proof of
uniquely correct phase recovery.

## Interpreting Results

A strong reconstruction typically combines:

- high SSIM
- low radial-profile error
- low encircled-energy error
- stable convergence with non-trivial cost improvement
- plausible dominant Zernike terms
- reasonable agreement with curated external baseline values when available

For real observations, this should be interpreted as **measurement-consistent**
reconstruction quality, not direct proof of uniquely correct phase recovery.

## Reproducibility Checklist

A result is considered reproducible when the following artifacts are preserved:

- the source file or preset key
- `config.json`
- `provenance.json`
- `metrics.json`
- the evaluation report (`.json` + `.md`)
- `reference_validation.json` when present
- benchmark plots/tables for algorithm-comparison studies

## Limitations

This validation layer improves traceability and comparison, but it does **not**
replace external scientific validation against mission pipelines, laboratory
wavefront truth, or specialized crystallography packages. It is best viewed as a
transparent and reproducible internal validation framework.

