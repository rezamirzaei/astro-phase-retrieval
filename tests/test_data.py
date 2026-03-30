"""Tests for data loading helpers (no network, no real FITS files)."""

from __future__ import annotations

import numpy as np
import pytest

from src.data.downloader import available_presets, list_cached_fits
from src.data.loader import (
    extract_psf_cutout,
    find_brightest_source,
    normalise_psf,
    subtract_background,
    prepare_psf_for_retrieval,
)
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
