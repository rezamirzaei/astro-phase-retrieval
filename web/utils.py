"""Shared security utilities — filename sanitisation, path containment checks."""

from __future__ import annotations

import re
from pathlib import Path

from fastapi import HTTPException


def sanitize_filename(filename: str) -> str:
    """Strip directory components and dangerous characters from *filename*.

    Prevents path-traversal attacks when a user-supplied filename is used
    to construct a filesystem path.  The result is a flat basename with
    only alphanumeric, dash, underscore, and dot characters.

    Raises ``HTTPException(422)`` if the result would be empty.
    """
    # Take only the final component (strip any directory traversal)
    name = Path(filename).name

    # Remove any remaining path separators or null bytes
    name = name.replace("\x00", "").replace("/", "").replace("\\", "")

    # Strip leading dots to prevent hidden files / traversal
    name = name.lstrip(".")

    # Allow only safe characters
    name = re.sub(r"[^\w.\-]", "_", name)

    if not name:
        raise HTTPException(status_code=422, detail="Invalid filename")

    return name


def assert_path_within(child: Path, parent: Path) -> Path:
    """Resolve *child* and verify it is contained within *parent*.

    Returns the resolved path on success; raises ``HTTPException(400)``
    if the resolved path escapes the expected directory.
    """
    resolved = child.resolve()
    parent_resolved = parent.resolve()
    if not str(resolved).startswith(str(parent_resolved) + "/") and resolved != parent_resolved:
        raise HTTPException(
            status_code=400,
            detail="Path traversal detected — access denied",
        )
    return resolved

