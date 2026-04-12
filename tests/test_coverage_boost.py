"""Tests to boost coverage for src/cli.py (_cmd_cryst), src/pipeline.py (run_from_file),
and src/data/crystallography.py (edge cases).

Targets:
  src/cli.py                 lines 354-417 (_cmd_cryst)
  src/pipeline.py            lines 99, 149-204 (run_from_file, multi_start branch)
  src/data/crystallography.py lines 162-179 (urllib fallback), 284-285 (unparseable value),
                             349-350 (unquoted string), 369 (no _atom_site_label),
                             388 (missing fract columns), 393 (comment/underscore lines),
                             396 (too few columns), 415-416 (ValueError in atom parse)
"""

from __future__ import annotations

import json
import textwrap
from pathlib import Path
from unittest.mock import patch

import numpy as np
import pytest

from src.data.crystallography import (
    _download_url_to_file,
    _extract_cif_string,
    _extract_cif_value,
    _parse_atom_sites,
    parse_cif,
)

# =====================================================================
# src/cli.py  _cmd_cryst  (lines 354-417)
# =====================================================================


class TestCmdCryst:
    """Tests for the crystallography CLI subcommand."""

    def _make_cif(self, path: Path) -> None:
        """Write a minimal valid CIF file."""
        path.write_text(
            "data_test\n"
            "_cell_length_a 5.64\n"
            "_cell_length_b 5.64\n"
            "_cell_length_c 5.64\n"
            "_cell_angle_alpha 90.0\n"
            "_cell_angle_beta 90.0\n"
            "_cell_angle_gamma 90.0\n"
            "_symmetry_space_group_name_H-M 'F m -3 m'\n"
            "_chemical_formula_sum 'Cl Na'\n"
            "loop_\n"
            "_atom_site_label\n"
            "_atom_site_type_symbol\n"
            "_atom_site_fract_x\n"
            "_atom_site_fract_y\n"
            "_atom_site_fract_z\n"
            "_atom_site_occupancy\n"
            "Na1 Na 0.0000 0.0000 0.0000 1.000\n"
            "Cl1 Cl 0.5000 0.5000 0.5000 1.000\n"
        )

    def test_cryst_with_cif_file(self, tmp_path: Path, capsys: pytest.CaptureFixture[str]) -> None:
        """Run cryst subcommand with a direct CIF file path."""
        from src.cli import main

        cif = tmp_path / "test.cif"
        self._make_cif(cif)
        out_dir = tmp_path / "cryst_out"

        main(
            [
                "cryst",
                str(cif),
                "-a",
                "er",
                "-n",
                "5",
                "--grid-size",
                "32",
                "-o",
                str(out_dir),
            ]
        )

        captured = capsys.readouterr()
        assert "ER" in captured.out
        assert "R-factor" in captured.out

        result_file = out_dir / "cryst_result_er.json"
        assert result_file.exists()
        data = json.loads(result_file.read_text())
        assert data["algorithm"] == "er"
        assert "r_factor" in data

    def test_cryst_with_preset_key(
        self, tmp_path: Path, capsys: pytest.CaptureFixture[str]
    ) -> None:
        """Run cryst subcommand with a preset key (mock download)."""
        from src.cli import main

        cif = tmp_path / "crystallography" / "1000041.cif"
        cif.parent.mkdir(parents=True, exist_ok=True)
        self._make_cif(cif)

        with patch(
            "src.data.crystallography.download_cod_preset",
            return_value=cif,
        ):
            main(
                [
                    "cryst",
                    "nacl",
                    "-a",
                    "er",
                    "-n",
                    "5",
                    "--grid-size",
                    "32",
                    "-o",
                    str(tmp_path / "out"),
                ]
            )

        captured = capsys.readouterr()
        assert "Downloading" in captured.out or "ER" in captured.out

    def test_cryst_quiet_flag(self, tmp_path: Path, capsys: pytest.CaptureFixture[str]) -> None:
        """--quiet suppresses output but still writes JSON."""
        from src.cli import main

        cif = tmp_path / "test.cif"
        self._make_cif(cif)
        out_dir = tmp_path / "quiet_out"

        main(
            [
                "cryst",
                str(cif),
                "-a",
                "er",
                "-n",
                "3",
                "--grid-size",
                "32",
                "-o",
                str(out_dir),
                "--quiet",
            ]
        )

        captured = capsys.readouterr()
        assert "✅" not in captured.out
        assert (out_dir / "cryst_result_er.json").exists()

    def test_cryst_missing_file_exits(self, tmp_path: Path) -> None:
        """Non-existent CIF path should exit with code 1."""
        from src.cli import main

        with pytest.raises(SystemExit) as exc_info:
            main(
                [
                    "cryst",
                    "/no/such/file.cif",
                    "-a",
                    "er",
                    "-n",
                    "3",
                    "-o",
                    str(tmp_path / "out"),
                ]
            )
        assert exc_info.value.code == 1


# =====================================================================
# src/pipeline.py  run_from_file  (lines 149-204) + multi_start (line 99)
# =====================================================================


class TestPipelineRunFromFile:
    """Tests for RetrievalPipeline.run_from_file."""

    def test_run_from_npy_matching_grid(self, tmp_path: Path) -> None:
        """Load a .npy file whose grid already matches the config grid size."""
        from src.models.config import PipelineConfig
        from src.pipeline import RetrievalPipeline

        config = PipelineConfig()
        grid = config.pupil.grid_size  # default = 256

        # Create a synthetic PSF that matches the default grid
        rng = np.random.default_rng(42)
        psf = rng.random((grid, grid))
        psf /= psf.sum()
        npy_path = tmp_path / "test.npy"
        np.save(str(npy_path), psf)

        config.algorithm.max_iterations = 3
        config.algorithm.random_seed = 42
        pipeline = RetrievalPipeline(config)
        result = pipeline.run_from_file(npy_path, output_dir=tmp_path / "out")

        assert result.result.n_iterations >= 1
        assert result.ssim >= 0.0

    def test_run_from_npy_needs_resize(self, tmp_path: Path) -> None:
        """Load a .npy whose grid differs from the config → triggers resize path."""
        from src.models.config import PipelineConfig
        from src.pipeline import RetrievalPipeline

        config = PipelineConfig()
        config.pupil.grid_size = 64

        # Create a PSF at a DIFFERENT size (32x32)
        rng = np.random.default_rng(42)
        psf = rng.random((32, 32))
        psf /= psf.sum()
        npy_path = tmp_path / "small.npy"
        np.save(str(npy_path), psf)

        config.algorithm.max_iterations = 3
        config.algorithm.random_seed = 42
        pipeline = RetrievalPipeline(config)
        result = pipeline.run_from_file(npy_path, output_dir=tmp_path / "out")

        assert result.result.n_iterations >= 1
        # The resized image should be 64x64
        assert result.psf_data.image.shape == (64, 64)

    def test_run_from_file_fits_path(self, tmp_path: Path, psf_data) -> None:
        """Exercise the FITS loading branch via mock."""
        from src.models.config import PipelineConfig
        from src.pipeline import RetrievalPipeline

        config = PipelineConfig()
        config.pupil.grid_size = 64
        config.algorithm.max_iterations = 3
        config.algorithm.random_seed = 42

        fake_fits = tmp_path / "test.fits"
        fake_fits.touch()

        with (
            patch("src.data.loader.load_psf_from_fits", return_value=psf_data),
            patch("src.data.loader.prepare_psf_for_retrieval", return_value=psf_data.image),
        ):
            pipeline = RetrievalPipeline(config)
            result = pipeline.run_from_file(fake_fits, output_dir=tmp_path / "out")

        assert result.result.n_iterations >= 1

    def test_run_from_psf_multi_start(self, pupil, psf_data) -> None:
        """Exercise the n_starts > 1 branch in run_from_psf (line 99)."""
        from src.models.config import AlgorithmConfig, AlgorithmName, PipelineConfig
        from src.pipeline import RetrievalPipeline

        config = PipelineConfig()
        config.pupil.grid_size = 64

        alg_cfg = AlgorithmConfig(
            name=AlgorithmName.ERROR_REDUCTION,
            max_iterations=5,
            random_seed=42,
            n_starts=2,
        )

        pipeline = RetrievalPipeline(config)
        result = pipeline.run_from_psf(psf_data, pupil, alg_cfg)

        assert result.result.n_iterations >= 1


# =====================================================================
# src/data/crystallography.py — edge cases
# =====================================================================


class TestCrystallographyEdgeCases:
    """Cover uncovered lines in crystallography.py."""

    def test_download_urllib_fallback(self, tmp_path: Path) -> None:
        """Lines 162-179: when httpx import fails, fall back to urllib."""
        dest = tmp_path / "test.cif"

        # Make httpx import fail inside _download_url_to_file
        import builtins

        real_import = builtins.__import__

        def mock_import(name, *args, **kwargs):
            if name == "httpx":
                raise ImportError("mocked")
            return real_import(name, *args, **kwargs)

        with (
            patch("builtins.__import__", side_effect=mock_import),
            patch("urllib.request.urlretrieve") as mock_retrieve,
        ):
            _download_url_to_file("http://example.com/test.cif", dest)
            mock_retrieve.assert_called_once_with("http://example.com/test.cif", str(dest))

    def test_extract_cif_value_unparseable(self) -> None:
        """Lines 284-285 (333-334): non-numeric CIF value returns default."""
        text = "_cell_length_a NOTANUMBER\n"
        result = _extract_cif_value(text, "_cell_length_a", 5.0)
        assert result == 5.0

    def test_extract_cif_value_missing(self) -> None:
        """Tag not present in text → returns default."""
        result = _extract_cif_value("_other_tag 1.0\n", "_cell_length_a", 5.0)
        assert result == 5.0

    def test_extract_cif_string_unquoted(self) -> None:
        """Lines 349-350: unquoted string value."""
        text = "_symmetry_space_group_name_H-M Fm-3m\n"
        result = _extract_cif_string(text, "_symmetry_space_group_name_H-M", "P 1")
        assert result == "Fm-3m"

    def test_extract_cif_string_missing(self) -> None:
        """Tag not found → returns default."""
        result = _extract_cif_string("_other stuff\n", "_chemical_formula_sum", "unknown")
        assert result == "unknown"

    def test_parse_atom_sites_no_label_column(self) -> None:
        """Line 369: loop without _atom_site_label is skipped."""
        text = textwrap.dedent("""\
            loop_
            _atom_site_fract_x
            _atom_site_fract_y
            _atom_site_fract_z
            0.0 0.0 0.0
        """)
        atoms = _parse_atom_sites(text)
        assert atoms == []

    def test_parse_atom_sites_missing_fract_columns(self) -> None:
        """Line 388: loop has _atom_site_label but no fract columns → skipped."""
        text = textwrap.dedent("""\
            loop_
            _atom_site_label
            _atom_site_type_symbol
            Na1 Na
        """)
        atoms = _parse_atom_sites(text)
        assert atoms == []

    def test_parse_atom_sites_underscore_and_empty_lines(self) -> None:
        """Lines 392-393: underscore and empty lines in data block are skipped."""
        # The regex data-block capture excludes '#' lines, but '_' and blank
        # lines *are* captured and should be filtered by the loop body.
        text = textwrap.dedent("""\
            loop_
            _atom_site_label
            _atom_site_type_symbol
            _atom_site_fract_x
            _atom_site_fract_y
            _atom_site_fract_z
            Na1 Na 0.0 0.0 0.0
            _extra_tag value
            Cl1 Cl 0.5 0.5 0.5
        """)
        atoms = _parse_atom_sites(text)
        # _extra_tag line is skipped; Na1 + Cl1 parsed
        assert len(atoms) == 2
        assert atoms[0].label == "Na1"
        assert atoms[1].label == "Cl1"

    def test_parse_atom_sites_too_few_columns(self) -> None:
        """Line 396: data line with too few columns is skipped."""
        text = textwrap.dedent("""\
            loop_
            _atom_site_label
            _atom_site_type_symbol
            _atom_site_fract_x
            _atom_site_fract_y
            _atom_site_fract_z
            Na1 Na
            Cl1 Cl 0.5 0.5 0.5
        """)
        atoms = _parse_atom_sites(text)
        assert len(atoms) == 1
        assert atoms[0].label == "Cl1"

    def test_parse_atom_sites_value_error_in_coordinates(self) -> None:
        """Lines 415-416: invalid float in coordinate → line is skipped."""
        text = textwrap.dedent("""\
            loop_
            _atom_site_label
            _atom_site_type_symbol
            _atom_site_fract_x
            _atom_site_fract_y
            _atom_site_fract_z
            Na1 Na BAD 0.0 0.0
            Cl1 Cl 0.5 0.5 0.5
        """)
        atoms = _parse_atom_sites(text)
        assert len(atoms) == 1
        assert atoms[0].label == "Cl1"

    def test_parse_cif_fallback_formula(self, tmp_path: Path) -> None:
        """Cover the _chemical_formula_structural fallback."""
        cif = tmp_path / "test.cif"
        cif.write_text(
            "data_test\n"
            "_cell_length_a 5.0\n"
            "_cell_length_b 5.0\n"
            "_cell_length_c 5.0\n"
            "_chemical_formula_structural 'Na Cl'\n"
            "loop_\n"
            "_atom_site_label\n"
            "_atom_site_fract_x\n"
            "_atom_site_fract_y\n"
            "_atom_site_fract_z\n"
            "Na1 0.0 0.0 0.0\n"
        )
        crystal = parse_cif(cif)
        assert crystal.formula == "Na Cl"

    def test_parse_cif_alt_space_group(self, tmp_path: Path) -> None:
        """Cover the _space_group_name_H-M_alt fallback."""
        cif = tmp_path / "test.cif"
        cif.write_text(
            "data_test\n"
            "_cell_length_a 5.0\n"
            "_cell_length_b 5.0\n"
            "_cell_length_c 5.0\n"
            "_space_group_name_H-M_alt 'P 21/c'\n"
            "loop_\n"
            "_atom_site_label\n"
            "_atom_site_fract_x\n"
            "_atom_site_fract_y\n"
            "_atom_site_fract_z\n"
            "Na1 0.0 0.0 0.0\n"
        )
        crystal = parse_cif(cif)
        assert crystal.space_group == "P 21/c"
