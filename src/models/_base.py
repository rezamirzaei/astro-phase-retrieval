"""Shared Pydantic base models for numpy-aware containers.

This module provides the single ``_NumpyModel`` base class used by
:mod:`src.models.optics` and :mod:`src.models.crystallography` to
avoid duplicating the ``arbitrary_types_allowed`` configuration.
"""

from __future__ import annotations

from pydantic import BaseModel, ConfigDict


class NumpyModel(BaseModel):
    """Base model that permits arbitrary types (numpy arrays).

    All domain models that contain ``np.ndarray`` fields should inherit
    from this class instead of ``BaseModel`` directly.
    """

    model_config = ConfigDict(arbitrary_types_allowed=True)
