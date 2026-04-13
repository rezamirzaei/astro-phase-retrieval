# Command-Line Interface

`phase-retrieval` ships a fully-featured CLI. Install the package and run:

```bash
phase-retrieval --help
```

---

## Global flags

| Flag | Description |
|------|-------------|
| `-V` / `--version` | Print version and exit |
| `-v` / `--verbose` | Enable DEBUG-level logging |
| `--log-format {text,json}` | Switch between human-readable and JSON-structured log output |

---

## `run` — single algorithm

Run one phase-retrieval algorithm on a FITS file and save a JSON result summary.

```bash
# Auto-discover a cached FITS file and run RAAR for 500 iterations
phase-retrieval run --algorithm raar --iterations 500

# Explicit FITS path, cosine β schedule, TV regularization
phase-retrieval run \
    --algorithm raar \
    --fits data/mastDownload/HST/mystar.fits \
    --iterations 1000 \
    --beta-schedule cosine \
    --tv-weight 0.005 \
    --output-dir results/

# Quiet mode — no stdout noise, machine-readable JSON summary to stdout
phase-retrieval run --algorithm hio --fits my.fits --output-format json --quiet
```

### Options

| Option | Default | Description |
|--------|---------|-------------|
| `-a` / `--algorithm` | `hio` | Algorithm key: `er`, `gs`, `hio`, `raar`, `wf`, `dr`, `admm`, `fista`, `sparse_pr`, `pinn` |
| `-n` / `--iterations` | `500` | Maximum number of iterations |
| `--beta` | `0.9` | Feedback parameter β (HIO / RAAR / DR) |
| `--beta-schedule` | `constant` | β schedule: `constant`, `linear`, `cosine` |
| `--momentum` | `0.0` | Heavy-ball momentum coefficient (`0` = off) |
| `--tv-weight` | `0.0` | Total-variation regularization weight (`0` = off) |
| `--noise-model` | `gaussian` | Focal-plane noise model: `gaussian` or `poisson` |
| `--n-starts` | `1` | Multi-start random restarts (best result kept) |
| `--fits` | auto | Path to calibrated FITS file |
| `-o` / `--output-dir` | `outputs` | Directory for JSON result file |
| `--output-format` | `text` | Summary format: `text` or `json` (stdout) |
| `-q` / `--quiet` | off | Suppress progress output |
| `--seed` | `42` | Random seed for reproducibility |

---

## `compare` — algorithm comparison

Run all available algorithms on the same data and print a comparison table.

```bash
phase-retrieval compare --fits data/my.fits --iterations 200 --output-dir results/
```

Use `--no-save` to skip writing per-algorithm JSON files:

```bash
phase-retrieval compare --fits my.fits --iterations 100 --no-save
```

Comparison runs now save the usual per-algorithm JSON summaries, while full
pipeline runs also emit richer `metrics.json` and `provenance.json` manifests.

---

## `benchmark` — deterministic synthetic validation

Run a repeatable synthetic benchmark suite and export reports in JSON, CSV, and
Markdown formats.

```bash
# Quick benchmark on one clean synthetic case
phase-retrieval benchmark --algorithms er,hio,raar --cases clean-low --iterations 20

# Full benchmark report set
phase-retrieval benchmark \
    --algorithms er,gs,hio,raar,wf,dr,admm,fista,sparse_pr \
    --cases clean-low,clean-hst,poisson-hst,noisy-high \
    --iterations 40 \
    --output-dir outputs/benchmark
```

### Options

| Option | Default | Description |
|--------|---------|-------------|
| `--algorithms` | `er,gs,hio,raar,wf,dr,admm,fista,sparse_pr` | Comma-separated algorithm keys |
| `--cases` | all built-in cases | Comma-separated benchmark case keys (`clean-low`, `clean-hst`, `poisson-hst`, `noisy-high`) |
| `-n` / `--iterations` | `40` | Maximum iterations per run |
| `--beta` | `0.9` | Feedback parameter β |
| `--seed` | `42` | Random seed for algorithm reproducibility |
| `-o` / `--output-dir` | `outputs/benchmark` | Directory for `benchmark_results.json`, `benchmark_summary.csv`, `benchmark_report.md` |
| `-q` / `--quiet` | off | Suppress stdout ranking table |

---

## `download` — fetch real HST/JWST data

Download a curated observation from the MAST archive (requires internet):

```bash
# List available presets
phase-retrieval download --list

# Download HST WFC3/UVIS F606W white-dwarf standard star
phase-retrieval download --preset hst-wfc3-uvis-f606w --data-dir data/
```

---

## `cryst` — crystallographic phase retrieval

Run phase retrieval on X-ray crystallography data from CIF files.

```bash
# Run on a local CIF file
phase-retrieval cryst data/crystallography/test_nacl.cif --algorithm hio

# Use a COD preset key (downloads from the Crystallography Open Database)
phase-retrieval cryst nacl
```

### Options

| Option | Default | Description |
|--------|---------|-------------|
| `cif` | — | Positional: path to a CIF file or COD preset key |
| `-a` / `--algorithm` | `hio` | Algorithm key (see `run` options) |
| `-n` / `--iterations` | `500` | Maximum number of iterations |
| `--beta` | `0.9` | Feedback parameter β |
| `--grid-size` | `128` | Fourier grid size |
| `-o` / `--output-dir` | `outputs` | Directory for results |

The first positional argument is treated as a COD preset key (e.g. `nacl`,
`quartz`, `silicon`, `diamond`).

---

## Exit codes

| Code | Meaning |
|------|---------|
| `0`  | Success |
| `1`  | Runtime error (e.g. no FITS file found) |
| `2`  | Argument parsing error |

