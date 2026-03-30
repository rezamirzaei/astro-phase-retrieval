"""Regression checks for the generated notebook content."""

from __future__ import annotations

import json
from pathlib import Path


NOTEBOOK_PATH = Path(__file__).resolve().parent.parent / "notebooks" / "phase_retrieval_hst.ipynb"


def _notebook_source() -> str:
    notebook = json.loads(NOTEBOOK_PATH.read_text())
    parts: list[str] = []
    for cell in notebook["cells"]:
        parts.extend(cell.get("source", []))
        parts.append("\n")
    return "".join(parts)


def test_notebook_autoruns_pinn_when_torch_is_available() -> None:
    text = _notebook_source()
    assert "RUN_PINN = TORCH_AVAILABLE" in text
    assert "pinn_result = None" in text
    assert "pinn_result = AlgorithmRegistry.create(pinn_cfg, pupil).run(psf_data_resized)" in text
    assert 'comparison_results["PINN"] = pinn_result' in text


def test_notebook_does_not_contain_stale_manual_pinn_values() -> None:
    text = _notebook_source()
    assert "RUN_PINN = False  # Set to True to benchmark the optional PINN solver when PyTorch is available" not in text
    assert "pinn_result = True" not in text


