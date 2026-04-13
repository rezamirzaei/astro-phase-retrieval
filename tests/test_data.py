"""Tests for data loading helpers (no network, no real FITS files)."""

from __future__ import annotations

from pathlib import Path
from unittest.mock import patch

import numpy as np
import pytest
from astropy.io import fits as pyfits
from astropy.table import Table

from src.data.downloader import (
    _CURATED_OBS,
    FILTER_WAVELENGTH_M,
    available_presets,
    download_all_presets,
    download_preset,
    list_cached_fits,
    search_and_download,
)
from src.data.loader import (
    _header_filter,
    _header_wavelength,
    extract_psf_cutout,
    find_brightest_source,
    load_fits_image,
    load_psf_from_fits,
    normalise_psf,
    prepare_psf_for_retrieval,
    subtract_background,
)
from src.models.config import DataConfig, PupilConfig, TelescopeType
from src.models.optics import PSFData


class TestBrightestSource:
    def test_finds_injected_peak(self) -> None:
        img = np.zeros((200, 200), dtype=np.float64)
        img[120, 130] = 100.0
        r, c = find_brightest_source(img, border=50)
        assert r == 120
        assert c == 130

    def test_excludes_border(self) -> None:
        img = np.zeros((200, 200), dtype=np.float64)
        img[10, 10] = 999.0  # in border
        img[120, 130] = 50.0  # outside border
        r, c = find_brightest_source(img, border=50)
        assert r == 120
        assert c == 130


class TestExtractCutout:
    def test_correct_shape(self) -> None:
        img = np.random.default_rng(0).random((256, 256))
        cutout = extract_psf_cutout(img, (128, 128), 32)
        assert cutout.shape == (64, 64)

    def test_edge_padding(self) -> None:
        img = np.ones((100, 100))
        cutout = extract_psf_cutout(img, (5, 5), 32)
        assert cutout.shape == (64, 64)


class TestSubtractBackground:
    def test_non_negative_result(self) -> None:
        img = np.random.default_rng(0).random((64, 64)) + 10.0
        result = subtract_background(img)
        assert np.all(result >= 0)


class TestNormalisePSF:
    def test_sums_to_one(self) -> None:
        img = np.random.default_rng(0).random((64, 64))
        normed = normalise_psf(img)
        assert normed.sum() == pytest.approx(1.0)

    def test_zero_input(self) -> None:
        img = np.zeros((64, 64))
        normed = normalise_psf(img)
        assert normed.sum() == 0.0


class TestPreparePSF:
    def test_padding(self) -> None:
        small = np.random.default_rng(0).random((32, 32))
        small /= small.sum()
        psf = PSFData(
            image=small,
            pixel_scale_arcsec=0.04,
            wavelength_m=606e-9,
            filter_name="TEST",
            telescope="test",
        )
        result = prepare_psf_for_retrieval(psf, 64)
        assert result.shape == (64, 64)
        assert result.sum() == pytest.approx(1.0, abs=1e-10)

    def test_cropping(self) -> None:
        big = np.random.default_rng(0).random((128, 128))
        big /= big.sum()
        psf = PSFData(
            image=big,
            pixel_scale_arcsec=0.04,
            wavelength_m=606e-9,
            filter_name="TEST",
            telescope="test",
        )
        result = prepare_psf_for_retrieval(psf, 64)
        assert result.shape == (64, 64)

    def test_same_size_returns_copy(self) -> None:
        img = np.random.default_rng(0).random((64, 64))
        img /= img.sum()
        psf = PSFData(
            image=img,
            pixel_scale_arcsec=0.04,
            wavelength_m=606e-9,
            filter_name="TEST",
            telescope="test",
        )
        result = prepare_psf_for_retrieval(psf, 64)
        assert result.shape == (64, 64)
        np.testing.assert_array_equal(result, img)
        assert result is not img  # must be a copy


class TestDownloaderHelpers:
    def test_available_presets(self) -> None:
        presets = available_presets()
        assert isinstance(presets, dict)
        assert len(presets) > 0
        assert "hst-wfc3-uvis-f606w" in presets

    def test_list_cached_fits_empty_dir(self, tmp_path) -> None:
        assert list_cached_fits(tmp_path) == []

    def test_list_cached_fits_nonexistent(self, tmp_path) -> None:
        assert list_cached_fits(tmp_path / "nope") == []

    def test_list_cached_fits_with_files(self, tmp_path) -> None:
        (tmp_path / "a.fits").write_bytes(b"fake")
        (tmp_path / "sub").mkdir()
        (tmp_path / "sub" / "b.fits").write_bytes(b"fake")
        (tmp_path / "c.txt").write_bytes(b"nope")
        result = list_cached_fits(tmp_path)
        assert len(result) == 2
        assert all(p.suffix == ".fits" for p in result)

    def test_filter_wavelength_spot_check(self) -> None:
        assert FILTER_WAVELENGTH_M["F606W"] == pytest.approx(606e-9)
        assert FILTER_WAVELENGTH_M["F814W"] == pytest.approx(814e-9)
        assert FILTER_WAVELENGTH_M["F275W"] == pytest.approx(275e-9)

    def test_curated_obs_structure(self) -> None:
        for _key, entry in _CURATED_OBS.items():
            assert "target_name" in entry
            assert "instrument_name" in entry
            assert "filters" in entry
            assert "obs_collection" in entry
            assert "description" in entry

    def test_download_preset_unknown_key(self, tmp_path) -> None:
        with pytest.raises(KeyError, match="Unknown preset"):
            download_preset("no-such-key", tmp_path)

    def test_available_presets_matches_curated(self) -> None:
        presets = available_presets()
        assert set(presets.keys()) == set(_CURATED_OBS.keys())


# ---------------------------------------------------------------------------
# Mock-based tests for downloader (no network)
# ---------------------------------------------------------------------------


def _make_obs_table(n: int = 3) -> Table:
    """Build a fake MAST observation table."""
    return Table(
        {
            "obs_id": [f"obs-{i}" for i in range(n)],
            "target_name": ["GRW+70D5824"] * n,
            "t_exptime": [float(5 + i * 5) for i in range(n)],
        }
    )


def _make_product_table(n: int = 2) -> Table:
    return Table(
        {
            "productFilename": [f"file_{i}_flt.fits" for i in range(n)],
            "obs_id": [f"obs-{i}" for i in range(n)],
        }
    )


def _make_manifest(tmp_path: Path, n: int = 1) -> Table:
    paths = []
    for i in range(n):
        p = tmp_path / f"download_{i}.fits"
        p.write_bytes(b"fake")
        paths.append(str(p))
    return Table({"Local Path": paths, "Status": ["COMPLETE"] * n})


class TestSearchAndDownloadMocked:
    """Tests for search_and_download with mocked MAST calls."""

    @patch("src.data.downloader.Observations")
    def test_curated_preset_succeeds(self, mock_obs, tmp_path) -> None:
        obs_table = _make_obs_table()
        mock_obs.query_criteria.return_value = obs_table
        mock_obs.get_product_list.return_value = _make_product_table()
        mock_obs.filter_products.return_value = _make_product_table(1)
        mock_obs.download_products.return_value = _make_manifest(tmp_path)

        cfg = DataConfig(
            data_dir=tmp_path,
            detector="WFC3/UVIS",
            filter_name="F606W",
        )
        paths = search_and_download(cfg)
        assert len(paths) == 1
        mock_obs.query_criteria.assert_called_once()

    @patch("src.data.downloader.Observations")
    def test_general_query_fallback(self, mock_obs, tmp_path) -> None:
        obs_table = _make_obs_table()
        mock_obs.query_criteria.return_value = obs_table
        mock_obs.get_product_list.return_value = _make_product_table()
        mock_obs.filter_products.return_value = _make_product_table(1)
        mock_obs.download_products.return_value = _make_manifest(tmp_path)

        cfg = DataConfig(
            data_dir=tmp_path,
            detector="WFC3/UVIS",
            filter_name="F555W",  # not a curated key
        )
        paths = search_and_download(cfg)
        assert len(paths) == 1

    @patch("src.data.downloader.Observations")
    def test_no_results_raises(self, mock_obs, tmp_path) -> None:
        mock_obs.query_criteria.return_value = Table(
            {"obs_id": [], "target_name": [], "t_exptime": []}
        )
        cfg = DataConfig(data_dir=tmp_path, detector="WFC3/UVIS", filter_name="F606W")
        with pytest.raises(RuntimeError, match="No observations found"):
            search_and_download(cfg)

    @patch("src.data.downloader.Observations")
    def test_all_skycell_raises(self, mock_obs, tmp_path) -> None:
        obs_table = Table(
            {
                "obs_id": ["hst_skycell_1", "hst_skycell_2"],
                "target_name": ["STAR"] * 2,
                "t_exptime": [10.0, 20.0],
            }
        )
        mock_obs.query_criteria.return_value = obs_table
        cfg = DataConfig(data_dir=tmp_path, detector="WFC3/UVIS", filter_name="F606W")
        with pytest.raises(RuntimeError, match="skycell"):
            search_and_download(cfg)

    @patch("src.data.downloader.Observations")
    def test_no_filtered_products_falls_back(self, mock_obs, tmp_path) -> None:
        """When FLT/FLC filter returns empty, it should fall back to any FITS."""
        obs_table = _make_obs_table(1)
        mock_obs.query_criteria.return_value = obs_table
        mock_obs.get_product_list.return_value = _make_product_table()
        # First filter_products call (FLT/FLC) returns empty, second returns data
        mock_obs.filter_products.side_effect = [
            Table({"productFilename": [], "obs_id": []}),
            _make_product_table(1),
        ]
        mock_obs.download_products.return_value = _make_manifest(tmp_path)

        cfg = DataConfig(data_dir=tmp_path, detector="WFC3/UVIS", filter_name="F606W")
        paths = search_and_download(cfg)
        assert len(paths) == 1

    @patch("src.data.downloader.Observations")
    def test_no_products_at_all_raises(self, mock_obs, tmp_path) -> None:
        obs_table = _make_obs_table(1)
        mock_obs.query_criteria.return_value = obs_table
        mock_obs.get_product_list.return_value = _make_product_table()
        mock_obs.filter_products.return_value = Table({"productFilename": [], "obs_id": []})

        cfg = DataConfig(data_dir=tmp_path, detector="WFC3/UVIS", filter_name="F606W")
        with pytest.raises(RuntimeError, match="No suitable FITS products"):
            search_and_download(cfg)

    @patch("src.data.downloader.Observations")
    def test_moderate_exposure_preferred(self, mock_obs, tmp_path) -> None:
        """Observations with t_exptime in [1, 30]s should be preferred."""
        obs_table = Table(
            {
                "obs_id": ["long", "good", "short"],
                "target_name": ["STAR"] * 3,
                "t_exptime": [100.0, 10.0, 0.1],
            }
        )
        mock_obs.query_criteria.return_value = obs_table
        mock_obs.get_product_list.return_value = _make_product_table(1)
        mock_obs.filter_products.return_value = _make_product_table(1)
        mock_obs.download_products.return_value = _make_manifest(tmp_path)

        cfg = DataConfig(data_dir=tmp_path, detector="WFC3/UVIS", filter_name="F606W")
        paths = search_and_download(cfg)
        assert len(paths) == 1

    @patch("src.data.downloader.Observations")
    def test_no_moderate_exposure_falls_back(self, mock_obs, tmp_path) -> None:
        """When no observation has t_exptime in [1, 30]s, use the first."""
        obs_table = Table(
            {
                "obs_id": ["vlong", "vlong2"],
                "target_name": ["STAR"] * 2,
                "t_exptime": [100.0, 200.0],
            }
        )
        mock_obs.query_criteria.return_value = obs_table
        mock_obs.get_product_list.return_value = _make_product_table(1)
        mock_obs.filter_products.return_value = _make_product_table(1)
        mock_obs.download_products.return_value = _make_manifest(tmp_path)

        cfg = DataConfig(data_dir=tmp_path, detector="WFC3/UVIS", filter_name="F606W")
        paths = search_and_download(cfg)
        assert len(paths) == 1


class TestDownloadPresetMocked:
    @patch("src.data.downloader.Observations")
    def test_success(self, mock_obs, tmp_path) -> None:
        obs_table = _make_obs_table()
        mock_obs.query_criteria.return_value = obs_table
        mock_obs.get_product_list.return_value = _make_product_table()
        mock_obs.filter_products.return_value = _make_product_table(1)
        mock_obs.download_products.return_value = _make_manifest(tmp_path)

        paths = download_preset("hst-wfc3-uvis-f606w", tmp_path)
        assert len(paths) == 1

    @patch("src.data.downloader.Observations")
    def test_empty_results(self, mock_obs, tmp_path) -> None:
        mock_obs.query_criteria.return_value = Table(
            {"obs_id": [], "target_name": [], "t_exptime": []}
        )
        paths = download_preset("hst-wfc3-uvis-f606w", tmp_path)
        assert paths == []

    @patch("src.data.downloader.Observations")
    def test_all_skycell_returns_empty(self, mock_obs, tmp_path) -> None:
        obs_table = Table(
            {
                "obs_id": ["hst_skycell_1"],
                "target_name": ["STAR"],
                "t_exptime": [10.0],
            }
        )
        mock_obs.query_criteria.return_value = obs_table
        paths = download_preset("hst-wfc3-uvis-f606w", tmp_path)
        assert paths == []

    @patch("src.data.downloader.Observations")
    def test_no_products_returns_empty(self, mock_obs, tmp_path) -> None:
        obs_table = _make_obs_table(1)
        mock_obs.query_criteria.return_value = obs_table
        mock_obs.get_product_list.return_value = _make_product_table()
        mock_obs.filter_products.return_value = Table({"productFilename": [], "obs_id": []})
        paths = download_preset("hst-wfc3-uvis-f606w", tmp_path)
        assert paths == []

    @patch("src.data.downloader.Observations")
    def test_no_flt_falls_back(self, mock_obs, tmp_path) -> None:
        obs_table = _make_obs_table(1)
        mock_obs.query_criteria.return_value = obs_table
        mock_obs.get_product_list.return_value = _make_product_table()
        mock_obs.filter_products.side_effect = [
            Table({"productFilename": [], "obs_id": []}),
            _make_product_table(1),
        ]
        mock_obs.download_products.return_value = _make_manifest(tmp_path)
        paths = download_preset("hst-wfc3-uvis-f606w", tmp_path)
        assert len(paths) == 1

    @patch("src.data.downloader.Observations")
    def test_no_moderate_exposure_in_preset(self, mock_obs, tmp_path) -> None:
        """download_preset should fall back to first obs when none are moderate."""
        obs_table = Table(
            {
                "obs_id": ["vlong"],
                "target_name": ["STAR"],
                "t_exptime": [500.0],
            }
        )
        mock_obs.query_criteria.return_value = obs_table
        mock_obs.get_product_list.return_value = _make_product_table(1)
        mock_obs.filter_products.return_value = _make_product_table(1)
        mock_obs.download_products.return_value = _make_manifest(tmp_path)
        paths = download_preset("hst-wfc3-uvis-f606w", tmp_path)
        assert len(paths) == 1


class TestDownloadAllPresetsMocked:
    @patch("src.data.downloader.download_preset")
    def test_downloads_all(self, mock_dl, tmp_path) -> None:
        mock_dl.return_value = [tmp_path / "f.fits"]
        result = download_all_presets(tmp_path, keys=["hst-wfc3-uvis-f606w"])
        assert "hst-wfc3-uvis-f606w" in result

    @patch("src.data.downloader.download_preset")
    def test_handles_failure_gracefully(self, mock_dl, tmp_path) -> None:
        mock_dl.side_effect = RuntimeError("network error")
        result = download_all_presets(tmp_path, keys=["hst-wfc3-uvis-f606w"])
        assert result == {}

    @patch("src.data.downloader.download_preset")
    def test_skips_empty_results(self, mock_dl, tmp_path) -> None:
        mock_dl.return_value = []
        result = download_all_presets(tmp_path, keys=["hst-wfc3-uvis-f606w"])
        assert result == {}

    @patch("src.data.downloader.download_preset")
    def test_default_keys_uses_all(self, mock_dl, tmp_path) -> None:
        mock_dl.return_value = [tmp_path / "f.fits"]
        result = download_all_presets(tmp_path)
        assert len(result) == len(_CURATED_OBS)


# ---------------------------------------------------------------------------
# FITS loader tests (no real FITS files — create in-memory)
# ---------------------------------------------------------------------------


def _create_test_fits(path: Path, *, add_header_keys: bool = True) -> None:
    """Write a minimal valid FITS file."""
    primary = pyfits.PrimaryHDU()
    if add_header_keys:
        primary.header["FILTER"] = "F606W"
        primary.header["INSTRUME"] = "WFC3"
        primary.header["DETECTOR"] = "UVIS"
        primary.header["ROOTNAME"] = "test_obs"
        primary.header["TARGNAME"] = "GRW+70D5824"
        primary.header["EXPTIME"] = 10.0
    img = np.random.default_rng(0).random((200, 200)).astype(np.float64)
    img[100, 100] = 1000.0  # bright star
    sci = pyfits.ImageHDU(data=img, name="SCI")
    hdul = pyfits.HDUList([primary, sci])
    hdul.writeto(path, overwrite=True)


class TestLoadFitsImage:
    def test_reads_science_extension(self, tmp_path) -> None:
        fpath = tmp_path / "test.fits"
        _create_test_fits(fpath)
        data, header = load_fits_image(fpath, ext=1)
        assert data.ndim == 2
        assert data.shape == (200, 200)
        assert data.dtype == np.float64

    def test_reads_with_string_extension(self, tmp_path) -> None:
        fpath = tmp_path / "test.fits"
        _create_test_fits(fpath)
        data, header = load_fits_image(fpath, ext="SCI")
        assert data.shape == (200, 200)

    def test_header_merges_primary(self, tmp_path) -> None:
        fpath = tmp_path / "test.fits"
        _create_test_fits(fpath)
        _, header = load_fits_image(fpath)
        assert header["FILTER"] == "F606W"
        assert header["INSTRUME"] == "WFC3"
        assert header["ROOTNAME"] == "test_obs"

    def test_fallback_to_first_image_ext(self, tmp_path) -> None:
        """When requested extension doesn't exist, fall back to first 2-D HDU."""
        fpath = tmp_path / "test.fits"
        _create_test_fits(fpath)
        data, _ = load_fits_image(fpath, ext=99)
        assert data.ndim == 2

    def test_no_image_extension_raises(self, tmp_path) -> None:
        fpath = tmp_path / "empty.fits"
        primary = pyfits.PrimaryHDU()
        pyfits.HDUList([primary]).writeto(fpath, overwrite=True)
        with pytest.raises(ValueError, match="No 2-D image"):
            load_fits_image(fpath, ext=99)

    def test_stat_oserror_falls_back_gracefully(self, tmp_path) -> None:
        """OSError from stat() should be silently absorbed (size treated as 0)."""
        fpath = tmp_path / "test.fits"
        _create_test_fits(fpath)
        with patch("pathlib.Path.stat", side_effect=OSError("permission denied")):
            # Should NOT raise — falls back to size=0 which is under the limit
            data, _ = load_fits_image(fpath, ext=1)
        assert data.ndim == 2

    def test_too_large_file_raises(self, tmp_path) -> None:
        """Files larger than 2 GiB should raise ValueError before opening."""
        fpath = tmp_path / "huge.fits"
        _create_test_fits(fpath)
        from unittest.mock import MagicMock

        fake_stat = MagicMock()
        fake_stat.st_size = 3 * 1024**3  # 3 GiB
        with (
            patch("pathlib.Path.stat", return_value=fake_stat),
            pytest.raises(ValueError, match="too large"),
        ):
            load_fits_image(fpath)


class TestLoadPSFFromFits:
    def test_full_pipeline(self, tmp_path) -> None:
        fpath = tmp_path / "test.fits"
        _create_test_fits(fpath)
        data_cfg = DataConfig(data_dir=tmp_path, cutout_size=64)
        pupil_cfg = PupilConfig(
            telescope=TelescopeType.GENERIC_CIRCULAR,
            grid_size=64,
        )
        psf = load_psf_from_fits(fpath, data_cfg, pupil_cfg)
        assert isinstance(psf, PSFData)
        assert psf.image.shape == (64, 64)
        assert psf.image.sum() == pytest.approx(1.0, abs=1e-10)
        assert psf.filter_name == "F606W"
        assert psf.telescope == "generic_circular"

    def test_wavelength_from_header(self, tmp_path) -> None:
        fpath = tmp_path / "test.fits"
        _create_test_fits(fpath)
        data_cfg = DataConfig(data_dir=tmp_path, cutout_size=64)
        pupil_cfg = PupilConfig(grid_size=64)
        psf = load_psf_from_fits(fpath, data_cfg, pupil_cfg)
        assert psf.wavelength_m == pytest.approx(606e-9)

    def test_provenance_metadata_added(self, tmp_path) -> None:
        fpath = tmp_path / "test.fits"
        _create_test_fits(fpath)
        data_cfg = DataConfig(data_dir=tmp_path, cutout_size=64)
        pupil_cfg = PupilConfig(grid_size=64)
        psf = load_psf_from_fits(fpath, data_cfg, pupil_cfg)
        assert psf.metadata["source_kind"] == "fits"
        assert psf.metadata["source_filename"] == "test.fits"
        assert "source_sha256" in psf.metadata
        assert psf.metadata["prepared_shape"] == [64, 64]
        assert "header" in psf.metadata


class TestHeaderFilter:
    def test_returns_filter_key(self) -> None:
        assert _header_filter({"FILTER": "F814W"}, "fallback") == "F814W"

    def test_skips_clear(self) -> None:
        header = {"FILTER": "CLEAR1L", "FILTER2": "F438W"}
        assert _header_filter(header, "fallback") == "F438W"

    def test_fallback(self) -> None:
        assert _header_filter({}, "F555W") == "F555W"

    def test_filter1_used(self) -> None:
        assert _header_filter({"FILTER1": "F275W"}, "fb") == "F275W"


class TestHeaderWavelength:
    def test_known_filter(self) -> None:
        assert _header_wavelength({"FILTER": "F814W"}, 500e-9) == pytest.approx(814e-9)

    def test_unknown_filter_uses_fallback(self) -> None:
        assert _header_wavelength({"FILTER": "UNKNOWN"}, 500e-9) == pytest.approx(500e-9)

    def test_missing_filter_uses_fallback(self) -> None:
        assert _header_wavelength({}, 500e-9) == pytest.approx(500e-9)
