# Developer Scripts

Standalone utility scripts for development, diagnostics, and notebook generation.
These are **not** part of the installable package.

| Script | Purpose |
|--------|---------|
| `main.py` | Full HST wavefront-sensing demo (PyCharm / Jupyter cell format) |
| `_diag.py` | Deep diagnostic — run all algorithms and generate comparison figures |
| `_gen_notebook.py` | Regenerate `notebooks/phase_retrieval_hst.ipynb` |
| `_gen_notebook_cryst.py` | Regenerate `notebooks/phase_retrieval_crystallography.ipynb` |
| `_test_synth.py` | Quick RAAR sanity-check on a synthetic problem |

## Usage

Run from the **project root** so that `src/` imports resolve correctly:

```bash
python scripts/main.py
python scripts/_diag.py
python scripts/_gen_notebook.py
python scripts/_gen_notebook_cryst.py
python scripts/_test_synth.py
```

