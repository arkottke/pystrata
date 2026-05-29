"""Tests for constitutive model curve fitting against various nonlinear soil types.

Verifies that MKZ and HH model fits accurately reproduce the target modulus reduction
and damping curves for several empirical soil models.
"""

from __future__ import annotations

import numpy as np
import pytest

import pystrata.site as site
from pystrata.constitutive import calc_damping, calc_mod_reduc
from pystrata.curve_fitting import fit_layer_hh, fit_layer_mkz, fit_profile

# Strain range for evaluation
STRAINS = np.logspace(-6, -1.5, 100)

# Tolerance: max RMS error in G/Gmax fit
RTOL_MOD_REDUC = 0.05  # 5%
# Tolerance: max absolute error in damping (decimal)
ATOL_DAMPING = 0.03  # 3% damping


def _make_layer(soil_type, thickness=10.0, shear_vel=200.0):
    """Create a Layer with the given soil type."""
    return site.Layer(soil_type, thickness, shear_vel)


def _check_mod_reduc_fit(layer, params):
    """Assert the fitted params reproduce the target G/Gmax curve."""
    st = layer.soil_type
    if st.mod_reduc is None or not callable(st.mod_reduc):
        return
    target = st.mod_reduc(STRAINS)
    predicted = calc_mod_reduc(STRAINS, params)
    rms = np.sqrt(np.mean((predicted - target) ** 2))
    assert rms < RTOL_MOD_REDUC, (
        f"G/Gmax RMS error {rms:.4f} exceeds {RTOL_MOD_REDUC} "
        f"for {st.name or type(st).__name__}"
    )


def _check_damping_fit(layer, params):
    """Assert the fitted params reproduce the target damping curve reasonably."""
    st = layer.soil_type
    if not callable(getattr(st, "damping", None)):
        return
    target = st.damping(STRAINS)
    predicted = calc_damping(STRAINS, params)
    # Compare Masing-only damping (subtract min damping offset from target)
    target_adjusted = target - target[0]
    mae = np.mean(np.abs(predicted - target_adjusted))
    assert mae < ATOL_DAMPING, (
        f"Damping MAE {mae:.4f} exceeds {ATOL_DAMPING} "
        f"for {st.name or type(st).__name__}"
    )


# ---------------------------------------------------------------------------
# Soil type fixtures
# ---------------------------------------------------------------------------


@pytest.fixture
def darendeli_soft():
    """Soft clay: low Vs, high PI, low stress."""
    return _make_layer(
        site.DarendeliSoilType(unit_wt=17.0, plas_index=30, ocr=1.0, stress_mean=25),
        shear_vel=150,
    )


@pytest.fixture
def darendeli_stiff():
    """Stiff clay: higher Vs, moderate PI, high stress."""
    return _make_layer(
        site.DarendeliSoilType(unit_wt=20.0, plas_index=10, ocr=2.0, stress_mean=200),
        shear_vel=350,
    )


@pytest.fixture
def menq_gravel():
    """Gravel with Menq model."""
    return _make_layer(
        site.MenqSoilType(unit_wt=21.0, coef_unif=15, diam_mean=10, stress_mean=100),
        shear_vel=300,
    )


@pytest.fixture
def kishida_organic():
    """Highly organic soil with Kishida model."""
    return _make_layer(
        site.KishidaSoilType(unit_wt=14.0, stress_vert=50, organic_content=30),
        shear_vel=100,
    )


@pytest.fixture
def rollins_gravel():
    """Gravel with Rollins et al.

    model.
    """
    return _make_layer(
        site.RollinsEtAlSoilType(unit_wt=22.0, stress_mean=150, coef_unif=20),
        shear_vel=400,
    )


# ---------------------------------------------------------------------------
# MKZ fitting tests
# ---------------------------------------------------------------------------


class TestFitMKZ:
    """Test MKZ fitting against various soil types."""

    def test_darendeli_soft(self, darendeli_soft):
        params = fit_layer_mkz(darendeli_soft, seed=42)
        _check_mod_reduc_fit(darendeli_soft, params)

    def test_darendeli_stiff(self, darendeli_stiff):
        params = fit_layer_mkz(darendeli_stiff, seed=42)
        _check_mod_reduc_fit(darendeli_stiff, params)

    def test_menq_gravel(self, menq_gravel):
        params = fit_layer_mkz(menq_gravel, seed=42)
        _check_mod_reduc_fit(menq_gravel, params)

    def test_kishida_organic(self, kishida_organic):
        params = fit_layer_mkz(kishida_organic, seed=42)
        _check_mod_reduc_fit(kishida_organic, params)

    def test_rollins_gravel(self, rollins_gravel):
        params = fit_layer_mkz(rollins_gravel, seed=42)
        _check_mod_reduc_fit(rollins_gravel, params)

    def test_linear_layer(self):
        """Linear layer should return default MKZ params without error."""
        layer = _make_layer(
            site.SoilType("Rock", unit_wt=24.0, mod_reduc=None, damping=0.01),
            shear_vel=700,
        )
        params = fit_layer_mkz(layer)
        assert params.shear_mod == layer.initial_shear_mod


# ---------------------------------------------------------------------------
# HH fitting tests
# ---------------------------------------------------------------------------


class TestFitHH:
    """Test HH fitting against various soil types."""

    def test_darendeli_soft(self, darendeli_soft):
        params = fit_layer_hh(darendeli_soft, depth=5.0, seed=42)
        _check_mod_reduc_fit(darendeli_soft, params)
        _check_damping_fit(darendeli_soft, params)

    def test_darendeli_stiff(self, darendeli_stiff):
        params = fit_layer_hh(darendeli_stiff, depth=15.0, seed=42)
        _check_mod_reduc_fit(darendeli_stiff, params)
        _check_damping_fit(darendeli_stiff, params)

    def test_menq_gravel(self, menq_gravel):
        params = fit_layer_hh(menq_gravel, depth=10.0, seed=42)
        _check_mod_reduc_fit(menq_gravel, params)
        _check_damping_fit(menq_gravel, params)

    def test_kishida_organic(self, kishida_organic):
        params = fit_layer_hh(kishida_organic, depth=3.0, seed=42)
        _check_mod_reduc_fit(kishida_organic, params)
        _check_damping_fit(kishida_organic, params)

    def test_rollins_gravel(self, rollins_gravel):
        params = fit_layer_hh(rollins_gravel, depth=12.0, seed=42)
        _check_mod_reduc_fit(rollins_gravel, params)
        _check_damping_fit(rollins_gravel, params)

    def test_linear_layer(self):
        """Linear layer should return default HH params without error."""
        layer = _make_layer(
            site.SoilType("Rock", unit_wt=24.0, mod_reduc=None, damping=0.01),
            shear_vel=700,
        )
        params = fit_layer_hh(layer, depth=20.0)
        assert params.shear_mod == layer.initial_shear_mod


# ---------------------------------------------------------------------------
# fit_profile tests
# ---------------------------------------------------------------------------


class TestFitProfile:
    """Test fit_profile on a multi-layer profile."""

    @pytest.fixture
    def two_layer_profile(self):
        return site.Profile(
            [
                site.Layer(
                    site.DarendeliSoilType(
                        unit_wt=18.0, plas_index=20, ocr=1.0, stress_mean=50
                    ),
                    thickness=10,
                    shear_vel=200,
                ),
                site.Layer(
                    site.DarendeliSoilType(
                        unit_wt=19.0, plas_index=15, ocr=1.0, stress_mean=150
                    ),
                    thickness=10,
                    shear_vel=350,
                ),
                site.Layer(
                    site.SoilType("Rock", unit_wt=24.0, mod_reduc=None, damping=0.01),
                    thickness=0,
                    shear_vel=700,
                ),
            ]
        )

    def test_mkz_returns_correct_count(self, two_layer_profile):
        params = fit_profile(two_layer_profile, model="mkz", seed=42)
        assert len(params) == 2  # excludes halfspace

    def test_hh_returns_correct_count(self, two_layer_profile):
        params = fit_profile(two_layer_profile, model="hh", seed=42)
        assert len(params) == 2

    def test_mkz_fit_quality(self, two_layer_profile):
        params = fit_profile(two_layer_profile, model="mkz", seed=42)
        for i, layer in enumerate(two_layer_profile[:-1]):
            _check_mod_reduc_fit(layer, params[i])

    def test_hh_fit_quality(self, two_layer_profile):
        params = fit_profile(two_layer_profile, model="hh", seed=42)
        for i, layer in enumerate(two_layer_profile[:-1]):
            _check_mod_reduc_fit(layer, params[i])
            _check_damping_fit(layer, params[i])

    def test_discretized_profile(self, two_layer_profile):
        """Fit should work on auto-discretized profiles."""
        disc = two_layer_profile.auto_discretize(max_freq=50)
        params = fit_profile(disc, model="mkz", seed=42)
        assert len(params) == len(disc) - 1

    def test_invalid_model_raises(self, two_layer_profile):
        with pytest.raises(ValueError, match="Unknown model"):
            fit_profile(two_layer_profile, model="invalid")
