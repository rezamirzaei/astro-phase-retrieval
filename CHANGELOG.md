# Changelog

All notable changes to this project will be documented in this file.

The format is based on [Keep a Changelog](https://keepachangelog.com/),
and this project adheres to [Semantic Versioning](https://semver.org/).

## [3.0.0] — 2026-04-14

### 🔬 Scientific Correctness

- **FISTA: adaptive √t Lipschitz scheduling** — The intensity loss
  f(g) = Σ(|Fg|² − y)² is quartic (non-smooth gradient), so a fixed
  Lipschitz constant diverges or stagnates.  FISTA now uses L_k = L₀·√k,
  yielding a decaying step-size η_k ∝ 1/√k that matches subgradient
  convergence theory for non-smooth problems (Nesterov 2004, §3.2.3).

- **Global phase sign ambiguity in benchmarks** — `_phase_rms_error` now
  evaluates both φ and −φ and takes min(error_plus, error_minus), correctly
  handling the inherent sign ambiguity of intensity-only measurements.

- **Zernike decomposition: proper least-squares** — Replaced the naïve
  inner-product projection (which assumes orthogonality) with a Gram-matrix
  solve N·c = b.  This produces correct coefficients for annular pupils
  (e.g. HST with secondary obstruction) where Zernike polynomials are not
  orthogonal over the masked domain.

- **Strehl ratio > 1.0 diagnostic** — Instead of silently clamping to 1.0,
  the system now emits `warnings.warn()` with the actual value when
  Strehl > 1.0, flagging normalisation mismatches or noise amplification.
  The returned value is still clamped for backward compatibility.

### 🏗️ Architecture: DRY Forward Model

- **`PupilModel.forward_model_kwargs()`** — New helper method returns the
  8 spectral/polychromatic keyword arguments as a dict.  Replaced all ~8
  copy-pasted kwarg blocks across `base.py`, `pinn.py`, `phase_diversity.py`,
  and `synthetic.py` with `**pupil.forward_model_kwargs()`.

### 🌐 Web API — Major Upgrade (v3.0)

#### New Features

- **Request-ID middleware** (`web/middleware.py`) — Every request gets a
  `X-Request-ID` header (UUID-4 or client-supplied).  Stored in a
  `ContextVar` for structured log correlation.

- **Request logging middleware** — Logs method, path, status code, latency,
  request ID, and client IP for every request (except health probes).

- **Background job queue** (`web/services/job_queue.py`) — Thread-pool-backed
  job executor with states `queued → running → completed | failed`.
  Publish/subscribe progress events for real-time streaming.

- **WebSocket progress streaming** (`/api/ws/jobs/{job_id}`) — Real-time
  iteration-level progress for running jobs.  JWT-authenticated via
  `?token=` query parameter.

- **Job management endpoints** (`/api/v1/jobs/`, `/api/v1/jobs/{id}`) —
  Poll status & progress of background jobs.

- **Readiness probe** (`GET /api/readiness`) — Checks database connectivity
  (`SELECT 1`) and disk write access.  Returns per-subsystem status for
  Kubernetes / load-balancer integration.

- **File upload** (`POST /api/data/upload`) — Upload custom FITS/NPY files
  (up to 100 MB).  Validates extension and streams to disk with size guard.

- **Batch export** (`POST /api/results/export-batch`) — Download a single
  ZIP containing outputs from multiple jobs.

- **Paginated responses** — `GET /api/results/` and `GET /api/data/fits`
  now return `PaginatedResponse[T]` with `{items, total, skip, limit}`.

- **OpenAPI enrichment** — Tag descriptions for all 10 endpoint groups,
  global 401/422 response documentation.

#### Improvements

- **CORS hardening** — `expose_headers` now includes `X-Request-ID` and
  `Content-Disposition` for download/tracing in browsers.

- **Connection pooling** — `pool_pre_ping=True` for automatic stale
  connection detection.  PostgreSQL gets `pool_size=5`, `max_overflow=10`,
  `pool_recycle=1800`.

- **Graceful shutdown** — Lifespan context drains the job thread-pool
  (configurable `shutdown_timeout_seconds`) and disposes the DB engine.

- **Health endpoint enriched** — Returns `{status, version, uptime_seconds}`.

- **API version bumped to 3.0.0**.

### Configuration

- `PR_SHUTDOWN_TIMEOUT_SECONDS` (default: 30) — max wait for running jobs.
- `PR_UPLOAD_MAX_BYTES` (default: 100 MB) — file upload size limit.

## [2.3.1] — 2026-04-13

### Fixed — Production Bugs

- **Dockerfile**: Replaced stale `python-jose[cryptography]` and `passlib`
  with `PyJWT` and `bcrypt` to match the v2.3.0 dependency migration.
  Added missing `astroquery` dependency. **This was a real container build bug.**
- **Job concurrency semaphore**: The `get_job_semaphore()` (introduced in
  v2.2.0) was declared but never wired up. Algorithm and crystallography
  routers now acquire the semaphore via `async with get_job_semaphore():`
  before running heavy compute jobs — the `max_concurrent_jobs` setting
  is now actually enforced. Moved semaphore to `web/concurrency.py` to
  avoid a circular import.

### Added

- **FISTA & Sparse PR in explain endpoint**: The `/api/explain/algorithms`
  endpoint was missing the two algorithms added in v2.2.0 (FISTA and
  Sparse Phase Retrieval). Now lists all 10 algorithms.

### Improved — Documentation

- **web/README.md**: Updated auth tech stack (python-jose → PyJWT,
  PBKDF2-SHA256 → bcrypt). Added all 15 crystallography endpoints,
  `/api/auth/refresh`, and `/api/version` to the endpoints table.
  Updated feature list to mention crystallography, security features,
  and 9+ algorithms.
- **docs/cli.md**: Added `cryst` subcommand documentation (crystallographic
  phase retrieval from CIF files or COD presets). Updated algorithm key
  list to include `fista` and `sparse_pr`.
- **docs/index.rst**: Updated project tagline to mention X-ray
  crystallography. Re-added crystallography API page to toctree.

### Improved — Code Quality

- **DRY test fixtures**: Extracted duplicate `db_session` and `client`
  fixtures from `test_web.py` and `test_crystallography_web.py` into
  `tests/conftest.py`. Both test files now use the shared fixtures.
- **CI workflow**: Added `web/` to ruff lint scope, `[web]` extras to
  test install, `--cov=web` to coverage scope, and `[web]` to security
  audit install.
- **Makefile**: Added `--cov=web` to `test` and `coverage` targets.
- **Pre-commit**: Bumped `ruff-pre-commit` from v0.4.4 → v0.8.6.

### All 415 tests pass ✅

## [2.3.0] — 2026-04-13

### Security Hardening

- **Password hashing**: Replaced hand-rolled PBKDF2-SHA256 (timing-attack
  vulnerable via `==` comparison, manual salt management) with **bcrypt**
  (constant-time verification, automatic salting).
- **JWT library**: Replaced unmaintained **python-jose** with **PyJWT ≥ 2.8**
  (actively maintained, no known CVEs). Updated `pyproject.toml` and mypy
  overrides accordingly.
- **Refresh-token flow**: Short-lived access tokens (15 min) + long-lived
  refresh tokens (7 days) with ``"type"`` claim validation. New
  `POST /api/auth/refresh` endpoint. The `Token` response schema now includes
  a `refresh_token` field.
- **Token type validation**: `get_current_user` dependency now rejects refresh
  tokens used as bearer tokens — only ``type=access`` JWTs are accepted.
- **Login rate limiting**: In-memory per-IP rate limiter (5 attempts / 60 s)
  on the login endpoint. Returns HTTP 429 when threshold is exceeded.
- **Security headers middleware**: Every response now includes
  `X-Content-Type-Options: nosniff`, `X-Frame-Options: DENY`,
  `X-XSS-Protection: 1; mode=block`, `Referrer-Policy: strict-origin-when-cross-origin`,
  `Permissions-Policy`, and `Strict-Transport-Security`. Auth responses also
  get `Cache-Control: no-store`.
- **Audit logging**: Structured log messages on user registration, login
  success/failure, token refresh, password verification failure, and JWT
  decode errors.

### Improved

- **Access token TTL**: Reduced from 24 hours to 15 minutes (compensated by
  refresh-token flow).
- **Config**: Added `refresh_token_expire_days` setting (default: 7 days),
  configurable via `PR_REFRESH_TOKEN_EXPIRE_DAYS` environment variable.
- **API version**: Bumped to 2.3.0.

### Tests

| New / updated test | Covers |
|--------------------|--------|
| `test_login_success` | Now asserts `refresh_token` field in response |
| `test_refresh_token` | Refresh endpoint returns new token pair |
| `test_refresh_with_access_token_rejected` | Access token rejected as refresh token |
| `test_refresh_token_rejected_as_bearer` | Refresh token rejected as bearer |
| `test_security_headers` | Hardening headers present on all responses |
| `test_auth_cache_control` | `Cache-Control: no-store` on auth responses |
| `test_login_rate_limit` | 429 after 5 failed attempts |

## [2.2.0] — 2026-04-10

### Added — New Algorithms

- **FISTA (Fast Iterative Shrinkage-Thresholding Algorithm)**: Proximal-gradient
  phase retrieval with Nesterov acceleration (Beck & Teboulle 2009). Supports
  pluggable regularisers (TV, L1-wavelet). Key `fista`.
- **Sparse Phase Retrieval (ThWF)**: Thresholded Wirtinger Flow for sparsity-
  promoting phase recovery (Cai, Li & Ma 2016). Supports hard and soft
  thresholding with adaptive decay. Key `sparse_pr`.

### Added — Architecture & Reusability

- **Pipeline Orchestrator** (`src/pipeline.py`): Composable, reusable pipeline
  that unifies data loading → algorithm execution → metric computation →
  visualization → persistence. Both CLI and web service can delegate to a
  single `RetrievalPipeline.run_from_psf()` call.
- **Synthetic Data Generator** (`src/data/synthetic.py`): Configurable synthetic
  PSF generation with Zernike-composed aberrations, Poisson photon noise,
  Gaussian read noise, and multiple telescope geometries.
- **Shared NumpyModel base** (`src/models/_base.py`): Eliminated duplicate
  `_NumpyModel` classes from `optics.py` and `crystallography.py` — both now
  inherit from a single shared `NumpyModel` base.

### Added — Configuration & Validation

- **`Regulariser` enum**: `none`, `tv`, `l1_wavelet` — type-safe regulariser
  selection for FISTA and sparse PR.
- **`er_finish_fraction`** field: Configurable ER-finish fraction (was hardcoded
  `_ER_FRACTION = 0.1` in HIO, RAAR, DR). Now set via config, eliminating
  per-subclass boilerplate.
- **`sw_sigma_start` / `sw_sigma_end`** fields: Configurable Shrink-Wrap σ
  annealing schedule (exponential decay).
- **`sparsity_threshold`** field: Thresholding level for sparse PR.
- **`proximal_weight` / `fista_lipschitz`** fields: FISTA-specific parameters.
- **Cross-field `model_validator`** on `AlgorithmConfig`: warns when `admm_rho`
  is set for non-ADMM algorithms, when `regulariser` is set for non-FISTA
  algorithms, and when `sw_sigma_start < sw_sigma_end`.

### Improved — Algorithms

- **Shrink-Wrap**: Configurable σ annealing (exponential decay from
  `sw_sigma_start` → `sw_sigma_end`), connectivity filtering via
  `scipy.ndimage.label` to remove small isolated support islands.
- **ER-finish**: Extracted duplicated `_ER_FRACTION` class constants from
  HIO, RAAR, DR into a unified `config.er_finish_fraction` field.
- **`multi_start.py`**: Replaced bare `assert` with proper `RuntimeError`
  guard (not stripped by `-O`).

### Added — CLI

- **`phase-retrieval cryst`** subcommand: Run crystallographic phase retrieval
  from the command line. Accepts CIF file paths or COD preset keys (e.g.
  `phase-retrieval cryst nacl`).

### Improved — Web & MLOps

- **Rate-limiting semaphore**: Global `asyncio.Semaphore` keyed to
  `settings.max_concurrent_jobs` prevents resource exhaustion from parallel
  algorithm runs.
- **FISTA/Sparse PR defaults** added to algorithm service.
- **Compare endpoint** now includes FISTA and Sparse PR in the default set.
- API version bumped to 2.2.0.

### Improved — Build & Tooling

- **Makefile**: Added `web-dev`, `docker-up`, `audit`, and `benchmark` targets.
  `lint` and `format` now include `web/`. `install` now includes `[web]` extra.
- **pyproject.toml**: Bumped version to 2.2.0, added FISTA/sparse keywords.

### Improved — Documentation

- **Sphinx docs**: Added FISTA and Sparse PR to `docs/api/algorithms.rst`.
- **CHANGELOG.md**: Added this entry.

### Tests — 389 pass, 0 fail

| New test class | Tests |
|----------------|-------|
| `TestFISTA` | FISTA convergence, TV regulariser, L1 regulariser, result fields, support |
| `TestSparsePR` | Sparse PR runs, soft threshold, result shape |
| `TestSyntheticDataGenerator` | Basic gen, Poisson noise, read noise, normalisation, RMS target, reproducibility |
| `TestPipelineOrchestrator` | Pipeline run, output saving |
| `TestCrossFieldValidation` | admm_rho warning, regulariser warning, sigma warning |
| `TestRegistryCompleteness` | All algorithms registered |
| `TestERFinishFraction` | Custom ER fraction, zero ER fraction |
| `TestNumpyModelBase` | Shared base import, PSFData works |

## [2.0.2] — 2026-04-07

### Fixed

- **CI branch target**: Changed trigger from `main` → `master` to match the
  actual repository default branch — CI now runs on push and PR.
- **Dead code**: Removed unused bare expression `ca * x + sa * y` in
  `pupils.py:_spider_mask` (leftover debug line with no side effect).
- **SECURITY.md**: Replaced placeholder `security@example.com` with GitHub
  private vulnerability reporting URL.

### Improved — Test Coverage

- Test count: **136 → 206** tests, all passing.
- Overall coverage: **90% → 98%** (90% CI gate enforced).
- `data/downloader.py`: **21% → 100%** — added comprehensive mock-based tests
  for `search_and_download`, `download_preset`, and `download_all_presets`
  covering curated/general queries, skycell filtering, exposure-time selection,
  product fallback, and error paths.
- `data/loader.py`: **62% → 100%** — added FITS I/O tests using `astropy.io.fits`,
  full `load_psf_from_fits` pipeline test, `_header_filter` / `_header_wavelength`
  edge cases, and `prepare_psf_for_retrieval` same-size copy path.
- `cli.py`: **84% → 99%** — added version flag, verbose logging, auto-discovered
  FITS, multi-start, download preset mock, `_has_torch` failure path, and
  `_sync_pupil_to_image` error path tests.
- `metrics/quality.py`: **89% → 95%** — added empty support, zero pupil, explicit
  data_range, zero-image SSIM fallback, and default max_sep tests.
- `models/optics.py`: **98% → 100%** — added `PSFPair` shape-mismatch validation
  test.
- `optics/pupils.py`: **97% → 100%** — added `_spider_mask` no-spider and
  zero-width tests, unknown telescope error test.
- `optics/propagator.py`: **96% → 100%** — added `psf_to_pupil`, `add_defocus`
  tests.
- Added `test_main_module.py` — exercises `python -m src` via subprocess.

### Improved — CI/CD

- Added **Python 3.13** to the test matrix (3.11, 3.12, 3.13).
- Bumped coverage gate from `--cov-fail-under=85` to `--cov-fail-under=90`.
- Added **GitHub Pages deployment** for Sphinx docs (via `deploy-pages` action).

## [2.0.1] — 2026-04-07

### Fixed — Critical Bug Fixes

- **`raar.py`**: Fixed corrupt file — floating code at top of file, undefined
  `cost` variable in return statement, double `@staticmethod` decorator, broken
  `_er_step` method that referenced `self` in a static method and used
  undefined variables (`g_new`, `g_prime`).
- **`douglas_rachford.py`**: Fixed identical corruption pattern — floating code,
  double `@staticmethod`, broken `_er_step` with undefined variables and `self`
  references in static method, dangling `)`.
- **`phase_diversity.py`**: Fixed scrambled code order — variables used before
  definition (`G1` before `g1`, `G2` before `g2`), code inside parameter lists,
  and broken single-image `_iterate` fallback method.
- **`registry.py`**: Fixed syntax error — `PINN` entry was outside the dict
  literal, and `RAAR` import was missing entirely.
- **`algorithms/__init__.py`**: Added missing `RAAR` import (was referenced in
  `__all__` but never imported).

### Improved — Code Quality

- **Zero lint warnings**: Fixed all 94 ruff warnings (import sorting, unused
  imports, line length, blind exception catches).
- **Dead code removed**: Deleted `_noll_to_nm` and `_noll_to_nm_standard` from
  `zernike.py` — both were intermediate functions that unconditionally
  redirected to `_noll_lookup` without contributing any logic.
- **DRY refactor**: Extracted duplicate `_er_step` method from `RAAR` and
  `DouglasRachford` into `PhaseRetriever` base class (with defensive `+1e-30`
  epsilon standardised across both).

### Improved — Test Coverage

- Test count: **104 → 136** tests, all passing.
- Overall coverage: **72% → 90%** (85% CI gate enforced).
- `visualization/plots.py`: **38% → 99%** — added smoke tests for all 20+
  plot functions, `save_figure`, `_azimuthal_average`, `set_style`, plus edge
  cases (single-algorithm views, linear scale, no-zernike summary).
- `zernike.py`: **55% → 98%** — tests for Noll indices beyond lookup table,
  `radial_polynomial` zero-return branch, `zernike_basis` with `start_j=1`.
- `test_data.py`: expanded with `list_cached_fits` with actual files, filter
  wavelength spot checks, curated-obs structure validation, unknown-preset
  `KeyError` test, and preset-set equality check.
- `test_models.py`: replaced blind `Exception` catches with `ValidationError`.

### Improved — Documentation & CI

- **Sphinx docs site**: Added `docs/` with `conf.py` (furo theme, napoleon,
  myst-parser), API reference pages for all 6 modules, and Makefile.
- **`pyproject.toml`**: Added `[docs]` optional-dependency group, bumped
  version to 2.0.1, added Python 3.13 classifier.
- **CI workflow**: Added `docs` build job (`sphinx-build -W`), added
  `--cov-fail-under=85` to test job.
- **SECURITY.md**: Added responsible-disclosure policy.
- **`.github/dependabot.yml`**: Automated weekly dependency updates for pip
  and GitHub Actions ecosystems.
- **CONTRIBUTING.md**: Added "Adding a New Algorithm" guide, type-checking
  section, and updated repository URLs.
- **CHANGELOG.md**: Added this entry documenting all fixes.

## [2.0.0] — 2026-03-30

### Added — State-of-the-Art Algorithms

- **Wirtinger Flow (WF)**: Gradient-descent phase retrieval with spectral
  initialization (Candès, Li & Soltanolkotabi 2015). Converges linearly to the
  global optimum. Key `wf`.
- **Douglas-Rachford (DR)**: Proximal splitting algorithm with provably better
  fixed-point convergence than HIO (Bauschke, Combettes & Luke 2002). Key `dr`.
- **ADMM**: Alternating Direction Method of Multipliers with Fourier/support
  splitting and dual variable (Chang & Marchesini 2018). Key `admm`.

### Added — Algorithm Enhancements

- **Nesterov/heavy-ball momentum** acceleration — applies to all algorithms
  transparently via `--momentum` flag.
- **Adaptive β scheduling**: constant, linear ramp-down, or cosine annealing
  via `--beta-schedule` flag.
- **Total-variation (TV) regularization** via Chambolle proximal operator for
  noise-robust phase recovery. `--tv-weight` flag.
- **Poisson noise model**: maximum-likelihood focal-plane projection for
  photon-limited observations. `--noise-model poisson` flag.
- **Multi-start optimization**: run from N random seeds, return the best result.
  `--n-starts` flag.

### Added — Quality Metrics

- **MTF** (`compute_mtf`): radially averaged Modulation Transfer Function.
- **SSIM** (`compute_ssim`): Structural Similarity Index between observed and
  reconstructed PSFs (uses scikit-image).
- **Phase Structure Function** (`compute_phase_structure_function`): standard
  wavefront diagnostic D_φ(r).

### Changed

- All existing algorithms (ER, GS, HIO, RAAR) now use noise-model-aware
  Fourier projection and adaptive β from the enhanced base class.
- CLI `compare` command now runs all 7 algorithms (was 4).
- CLI gained `--beta-schedule`, `--momentum`, `--tv-weight`, `--noise-model`,
  and `--n-starts` flags for run and compare subcommands.

## [1.0.0] — 2026-03-30

### Added

- **Algorithms**: Error Reduction (ER), Gerchberg–Saxton (GS), Hybrid Input-Output
  (HIO), Relaxed Averaged Alternating Reflections (RAAR), and Phase Diversity.
- **Real data pipeline**: automated download of calibrated HST/WFC3 and ACS
  observations from the MAST archive via `astroquery`.
- **Pupil models**: HST (annular + 4 spider vanes), JWST (hexagonal segments +
  3 struts), and generic circular aperture.
- **Zernike decomposition**: Noll-ordered polynomials up to j = 37 for wavefront
  analysis.
- **Quality metrics**: Strehl ratio, RMS wavefront error, convergence tracking.
- **Visualization**: 15+ publication-quality plot types (summary dashboards,
  radial profiles, 3-D wavefront surfaces, encircled energy, polar Zernike maps).
- **CLI**: `phase-retrieval run`, `compare`, and `download` subcommands.
- **Pydantic models**: fully validated configuration and result containers.
- **Test suite**: 40+ pytest tests on synthetic data (no network required).
- **CI**: GitHub Actions workflow with linting (Ruff) and testing (Python 3.11/3.12).
- **PEP 561**: `py.typed` marker for downstream type checking.
