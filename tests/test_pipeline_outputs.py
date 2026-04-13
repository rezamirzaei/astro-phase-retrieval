"""Tests for pipeline output manifests and richer metrics exports."""

from __future__ import annotations

import json

from src.models.config import default_hst_config
from src.models.optics import PSFData
from src.pipeline import RetrievalPipeline


class TestPipelineOutputs:
    def test_run_from_psf_writes_metrics_and_provenance(self, tmp_path, pupil, psf_data) -> None:
        config = default_hst_config()
        config.pupil = config.pupil.model_copy(update={"grid_size": pupil.grid_size})
        config.algorithm = config.algorithm.model_copy(update={"max_iterations": 5})
        config.output_dir = tmp_path / "pipeline_out"

        psf_with_metadata = PSFData(
            image=psf_data.image,
            pixel_scale_arcsec=psf_data.pixel_scale_arcsec,
            wavelength_m=psf_data.wavelength_m,
            filter_name=psf_data.filter_name,
            telescope=psf_data.telescope,
            obs_id=psf_data.obs_id,
            metadata={"source_kind": "synthetic-test", "case": "unit"},
        )

        result = RetrievalPipeline(config).run_from_psf(psf_with_metadata, pupil)
        assert result.ssim > 0.0
        assert result.radial_profile_error >= 0.0
        assert result.encircled_energy_error >= 0.0
        assert "improvement_ratio" in result.convergence_summary

        out_dir = config.output_dir
        for name in ("config.json", "result.json", "metrics.json", "provenance.json"):
            assert (out_dir / name).exists(), f"Missing output file: {name}"

        metrics = json.loads((out_dir / "metrics.json").read_text())
        assert "ssim" in metrics
        assert "radial_profile_error" in metrics
        assert "encircled_energy_error" in metrics
        assert "convergence" in metrics

        provenance = json.loads((out_dir / "provenance.json").read_text())
        assert provenance["psf"]["source_kind"] == "synthetic-test"
        assert provenance["algorithm"]["name"] == config.algorithm.name.value

