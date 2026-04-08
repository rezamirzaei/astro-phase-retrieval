# Sphinx configuration for phase-retrieval documentation

from __future__ import annotations

import os
import sys
from importlib.metadata import PackageNotFoundError, version as _pkg_version

# -- Path setup ----------------------------------------------------------------
sys.path.insert(0, os.path.abspath(".."))

# -- Project information -------------------------------------------------------
project = "phase-retrieval"
author = "Reza Mirzaeifard"

try:
    release = _pkg_version("phase-retrieval")
except PackageNotFoundError:
    # Running docs build from a checkout without the package installed
    import tomllib
    from pathlib import Path

    _pyproject = Path(__file__).resolve().parent.parent / "pyproject.toml"
    with _pyproject.open("rb") as _fh:
        release = tomllib.load(_fh)["project"]["version"]

# -- General configuration -----------------------------------------------------
extensions = [
    "sphinx.ext.autodoc",
    "sphinx.ext.autosummary",
    "sphinx.ext.napoleon",
    "sphinx.ext.viewcode",
    "sphinx.ext.intersphinx",
    "myst_parser",
]

autosummary_generate = True
autosummary_imported_members = True

templates_path = ["_templates"]
exclude_patterns = ["_build", "Thumbs.db", ".DS_Store"]

# -- Napoleon (NumPy / Google docstrings) --------------------------------------
napoleon_google_docstring = False
napoleon_numpy_docstring = True
napoleon_include_init_with_doc = True

# -- Autodoc settings ----------------------------------------------------------
autodoc_member_order = "bysource"
autodoc_typehints = "description"

# -- Intersphinx ---------------------------------------------------------------
intersphinx_mapping = {
    "python": ("https://docs.python.org/3", None),
    "numpy": ("https://numpy.org/doc/stable/", None),
    "scipy": ("https://docs.scipy.org/doc/scipy/", None),
    "matplotlib": ("https://matplotlib.org/stable/", None),
}

# -- MyST (Markdown support) ---------------------------------------------------
source_suffix = {
    ".rst": "restructuredtext",
    ".md": "markdown",
}

# -- HTML output ---------------------------------------------------------------
html_theme = "furo"
html_title = "phase-retrieval"
html_static_path = ["_static"]


