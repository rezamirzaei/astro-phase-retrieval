"""Tests for crystallography data: CIF parsing, diffraction simulation, COD download."""

from __future__ import annotations

from pathlib import Path
from unittest.mock import MagicMock, patch

import pytest

from src.data.crystallography import (
    _CURATED_COD,
    _atomic_scattering_factor,
    _extract_cif_string,
    _extract_cif_value,
    _parse_atom_sites,
    available_cod_presets,
    download_cif,
    download_cod_preset,
    list_cached_cif,
    parse_cif,
    run_crystallography_retrieval,
    simulate_diffraction,
)
from src.models.crystallography import (
    AtomSite,
    CrystallographyResult,
    CrystalStructure,
    DiffractionPattern,
)

# ---------------------------------------------------------------------------
# Sample CIF content for testing (inline, no file I/O needed in setup)
# ---------------------------------------------------------------------------

_SAMPLE_CIF = """\
data_1000041
_cell_length_a     5.6400(1)
_cell_length_b     5.6400(1)
_cell_length_c     5.6400(1)
_cell_angle_alpha  90.0
_cell_angle_beta   90.0
_cell_angle_gamma  90.0
_symmetry_space_group_name_H-M  'F m -3 m'
_chemical_formula_sum  'Cl Na'
loop_
_atom_site_label
_atom_site_type_symbol
_atom_site_fract_x
_atom_site_fract_y
_atom_site_fract_z
_atom_site_occupancy
Na1 Na 0.0000 0.0000 0.0000 1.000
Cl1 Cl 0.5000 0.5000 0.5000 1.000
"""


class TestAtomicScatteringFactor:
    def test_known_element(self) -> None:
        assert _atomic_scattering_factor("Na") == 11.0
        assert _atomic_scattering_factor("Cl") == 17.0

    def test_unknown_element_defaults(self) -> None:
        assert _atomic_scattering_factor("Xx") == 6.0

    def test_strips_numbers(self) -> None:
        assert _atomic_scattering_factor("Na1") == 11.0


class TestCIFHelpers:
    def test_extract_value(self) -> None:
        assert _extract_cif_value(_SAMPLE_CIF, "_cell_length_a", 0.0) == pytest.approx(5.64)

    def test_extract_value_missing(self) -> None:
        assert _extract_cif_value(_SAMPLE_CIF, "_nonexistent", 99.0) == 99.0

    def test_extract_string(self) -> None:
        sg = _extract_cif_string(_SAMPLE_CIF, "_symmetry_space_group_name_H-M", "P 1")
        assert "F m -3 m" in sg

    def test_extract_string_missing(self) -> None:
        assert _extract_cif_string(_SAMPLE_CIF, "_nonexistent", "default") == "default"


class TestParseAtomSites:
    def test_parses_nacl(self) -> None:
        atoms = _parse_atom_sites(_SAMPLE_CIF)
        assert len(atoms) == 2
        labels = {a.label for a in atoms}
        assert "Na1" in labels
        assert "Cl1" in labels

    def test_empty_text(self) -> None:
        atoms = _parse_atom_sites("")
        assert atoms == []


class TestParseCIF:
    def test_parses_file(self, tmp_path: Path) -> None:
        cif_path = tmp_path / "1000041.cif"
        cif_path.write_text(_SAMPLE_CIF)
        crystal = parse_cif(cif_path)
        assert crystal.a == pytest.approx(5.64)
        assert crystal.b == pytest.approx(5.64)
        assert len(crystal.atoms) == 2
        assert "F m -3 m" in crystal.space_group
        assert crystal.cod_id == "1000041"

    def test_minimal_cif(self, tmp_path: Path) -> None:
        """CIF with only cell params, no atom sites."""
        cif_path = tmp_path / "minimal.cif"
        cif_path.write_text(
            "data_test\n_cell_length_a 4.0\n_cell_length_b 4.0\n_cell_length_c 4.0\n"
        )
        crystal = parse_cif(cif_path)
        assert crystal.a == pytest.approx(4.0)
        assert crystal.atoms == []

    def test_alternative_space_group(self, tmp_path: Path) -> None:
        """CIF with _space_group_name_H-M_alt instead of _symmetry_..."""
        cif_path = tmp_path / "alt.cif"
        cif_path.write_text(
            "data_alt\n"
            "_cell_length_a 4.0\n_cell_length_b 4.0\n_cell_length_c 4.0\n"
            "_space_group_name_H-M_alt 'P m -3 m'\n"
        )
        crystal = parse_cif(cif_path)
        assert "P m -3 m" in crystal.space_group

    def test_structural_formula_fallback(self, tmp_path: Path) -> None:
        """CIF with _chemical_formula_structural but no _chemical_formula_sum."""
        cif_path = tmp_path / "structural.cif"
        cif_path.write_text(
            "data_s\n"
            "_cell_length_a 3.0\n_cell_length_b 3.0\n_cell_length_c 3.0\n"
            "_chemical_formula_structural 'CaTiO3'\n"
        )
        crystal = parse_cif(cif_path)
        assert crystal.formula == "CaTiO3"

    def test_extract_cif_value_bad_float(self) -> None:
        """Non-numeric value for a numeric field returns default."""
        assert _extract_cif_value("_cell_length_a notanumber\n", "_cell_length_a", 99.0) == 99.0


class TestSimulateDiffraction:
    def test_basic_simulation(self) -> None:
        crystal = CrystalStructure(
            a=5.64, b=5.64, c=5.64,
            atoms=[
                AtomSite(label="Na1", symbol="Na", x=0.0, y=0.0, z=0.0),
                AtomSite(label="Cl1", symbol="Cl", x=0.5, y=0.5, z=0.5),
            ],
        )
        pattern = simulate_diffraction(crystal, grid_size=64)
        assert isinstance(pattern, DiffractionPattern)
        assert pattern.image.shape == (64, 64)
        assert pattern.image.sum() == pytest.approx(1.0, abs=1e-10)

    def test_empty_atoms_generates_synthetic(self) -> None:
        crystal = CrystalStructure(a=5.0, b=5.0, c=5.0, atoms=[])
        pattern = simulate_diffraction(crystal, grid_size=64)
        assert pattern.image.shape == (64, 64)
        assert pattern.image.sum() > 0

    def test_different_grid_sizes(self) -> None:
        crystal = CrystalStructure(
            a=5.0, b=5.0, c=5.0,
            atoms=[AtomSite(label="C1", symbol="C", x=0.25, y=0.25, z=0.0)],
        )
        for size in [64, 128]:
            pattern = simulate_diffraction(crystal, grid_size=size)
            assert pattern.image.shape == (size, size)


class TestAvailableCODPresets:
    def test_returns_dict(self) -> None:
        presets = available_cod_presets()
        assert isinstance(presets, dict)
        assert len(presets) > 0
        assert "nacl" in presets

    def test_matches_curated(self) -> None:
        presets = available_cod_presets()
        assert set(presets.keys()) == set(_CURATED_COD.keys())

    def test_curated_structure(self) -> None:
        for _key, entry in _CURATED_COD.items():
            assert "cod_id" in entry
            assert "formula" in entry
            assert "description" in entry


class TestListCachedCIF:
    def test_empty_dir(self, tmp_path: Path) -> None:
        assert list_cached_cif(tmp_path) == []

    def test_with_files(self, tmp_path: Path) -> None:
        cif_dir = tmp_path / "crystallography"
        cif_dir.mkdir()
        (cif_dir / "1000041.cif").write_text("data")
        (cif_dir / "1000042.cif").write_text("data")
        result = list_cached_cif(tmp_path)
        assert len(result) == 2


class TestDownloadCIF:
    @patch("src.data.crystallography.urllib.request.urlretrieve")
    def test_download_success(self, mock_retrieve: MagicMock, tmp_path: Path) -> None:
        def _fake_retrieve(url: str, path: str) -> None:
            Path(path).write_text("data_test\n_cell_length_a 5.0\n")

        mock_retrieve.side_effect = _fake_retrieve
        path = download_cif("1000041", tmp_path)
        assert path.exists()
        assert path.suffix == ".cif"
        mock_retrieve.assert_called_once()

    @patch("src.data.crystallography.urllib.request.urlretrieve")
    def test_download_cached(self, mock_retrieve: MagicMock, tmp_path: Path) -> None:
        cif_dir = tmp_path / "crystallography"
        cif_dir.mkdir(parents=True)
        (cif_dir / "1000041.cif").write_text("cached")
        path = download_cif("1000041", tmp_path)
        assert path.exists()
        mock_retrieve.assert_not_called()

    @patch("src.data.crystallography.urllib.request.urlretrieve")
    def test_download_failure(self, mock_retrieve: MagicMock, tmp_path: Path) -> None:
        mock_retrieve.side_effect = Exception("network error")
        with pytest.raises(RuntimeError, match="Failed to download"):
            download_cif("999999", tmp_path)

    @patch("src.data.crystallography.urllib.request.urlretrieve")
    def test_download_failure_cleans_partial(
        self, mock_retrieve: MagicMock, tmp_path: Path
    ) -> None:
        """Partial files should be cleaned up on failure."""
        cif_dir = tmp_path / "crystallography"
        cif_dir.mkdir(parents=True)
        partial = cif_dir / "partial.cif"

        def _fail(url: str, path: str) -> None:
            Path(path).write_text("partial data")
            raise OSError("network interrupted")

        mock_retrieve.side_effect = _fail
        with pytest.raises(RuntimeError, match="Failed to download"):
            download_cif("partial", tmp_path)
        assert not partial.exists()


class TestDownloadCODPreset:
    @patch("src.data.crystallography.download_cif")
    def test_known_preset(self, mock_dl: MagicMock, tmp_path: Path) -> None:
        mock_dl.return_value = tmp_path / "1000041.cif"
        path = download_cod_preset("nacl", tmp_path)
        mock_dl.assert_called_once_with("1000041", tmp_path)
        assert path == tmp_path / "1000041.cif"

    def test_unknown_preset(self, tmp_path: Path) -> None:
        with pytest.raises(KeyError, match="Unknown COD preset"):
            download_cod_preset("nonexistent", tmp_path)


class TestRunCrystallographyRetrieval:
    def test_runs_successfully(self) -> None:
        crystal = CrystalStructure(
            a=5.64, b=5.64, c=5.64,
            atoms=[
                AtomSite(label="Na1", symbol="Na", x=0.0, y=0.0, z=0.0),
                AtomSite(label="Cl1", symbol="Cl", x=0.5, y=0.5, z=0.5),
            ],
        )
        pattern = simulate_diffraction(crystal, grid_size=64)
        result = run_crystallography_retrieval(
            pattern,
            algorithm_name="er",
            max_iterations=10,
            random_seed=42,
        )
        assert isinstance(result, CrystallographyResult)
        assert result.n_iterations >= 1
        assert result.recovered_phase.shape == (64, 64)
        assert result.electron_density.shape == (64, 64)
        assert 0.0 <= result.r_factor <= 1.0

    def test_different_algorithms(self) -> None:
        crystal = CrystalStructure(
            a=5.0, b=5.0, c=5.0,
            atoms=[AtomSite(label="C1", symbol="C", x=0.0, y=0.0, z=0.0)],
        )
        pattern = simulate_diffraction(crystal, grid_size=64)
        for alg in ["er", "hio", "raar"]:
            result = run_crystallography_retrieval(
                pattern, algorithm_name=alg, max_iterations=5
            )
            assert result.algorithm.value == alg


