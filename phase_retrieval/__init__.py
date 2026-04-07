"""Canonical import package for phase-retrieval.

This package mirrors the in-repository ``src`` package so installed users can rely on
``import phase_retrieval`` and ``python -m phase_retrieval``.
"""

from __future__ import annotations

from importlib import import_module
from typing import cast

_src = import_module("src")

__version__ = _src.__version__
__all__ = list(cast(list[str], getattr(_src, "__all__", ["__version__"])))
__path__ = list(cast(list[str], _src.__path__))
