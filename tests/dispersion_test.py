import numpy as np
import pytest
from numpy.testing import assert_allclose

import pystrata.site as site
import pystrata.tools as tools
import pystrata.variation as variation

# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------


def _make_profile(poissons_ratio=0.25):
    """Build a simple layered profile with Poisson's ratio set on every layer."""
    return site.Profile(
        [
            site.Layer(
                site.SoilType("Sand", 18.0, None, 0.02),
                5,
                150,
                poissons_ratio=poissons_ratio,
            ),
            site.Layer(
                site.SoilType("Sand", 18.0, None, 0.02),
                5,
                200,
                poissons_ratio=poissons_ratio,
            ),
            site.Layer(
                site.SoilType("Sand", 19.0, None, 0.02),
                10,
                300,
                poissons_ratio=poissons_ratio,
            ),
            site.Layer(
                site.SoilType("Sand", 20.0, None, 0.01),
                20,
                500,
                poissons_ratio=poissons_ratio,
            ),
            site.Layer(
                site.SoilType("Rock", 24.0, None, 0.01),
                0,
                800,
                poissons_ratio=poissons_ratio,
            ),
        ]
    )


# ---------------------------------------------------------------------------
# Layer / Profile property tests
# ---------------------------------------------------------------------------


class TestLayerPoissonsRatio:
    def test_default_none(self):
        layer = site.Layer(site.SoilType("", 18.0, None, 0.02), 5, 200)
        assert layer.poissons_ratio is None
        assert layer.comp_vel is None

    def test_comp_vel_from_poissons_ratio(self):
        nu = 0.25
        vs = 200.0
        layer = site.Layer(
            site.SoilType("", 18.0, None, 0.02), 5, vs, poissons_ratio=nu
        )
        expected_vp = vs * np.sqrt(2 * (1 - nu) / (1 - 2 * nu))
        assert_allclose(layer.comp_vel, expected_vp)

    def test_copy_preserves_poissons_ratio(self):
        layer = site.Layer(
            site.SoilType("", 18.0, None, 0.02), 5, 200, poissons_ratio=0.3
        )
        copied = layer.copy()
        assert copied.poissons_ratio == 0.3
        assert_allclose(copied.comp_vel, layer.comp_vel)


class TestProfileDispersionArrays:
    def test_comp_vel_array(self):
        profile = _make_profile(poissons_ratio=0.25)
        comp_vel = profile.comp_vel
        assert comp_vel.shape == (len(profile),)
        assert all(v is not None for v in comp_vel)

    def test_poissons_ratio_array(self):
        profile = _make_profile(poissons_ratio=0.3)
        pr = profile.poissons_ratio
        assert_allclose(pr, 0.3)


# ---------------------------------------------------------------------------
# tools.calc_poissons_ratio
# ---------------------------------------------------------------------------


class TestCalcPoissonsRatio:
    def test_scalar(self):
        vs = 200.0
        nu_in = 0.25
        vp = vs * np.sqrt(2 * (1 - nu_in) / (1 - 2 * nu_in))
        nu_out = tools.calc_poissons_ratio(vs, vp)
        assert_allclose(nu_out, nu_in)

    def test_array(self):
        vs = np.array([150.0, 200.0, 300.0])
        nu_in = np.array([0.20, 0.25, 0.35])
        vp = vs * np.sqrt(2 * (1 - nu_in) / (1 - 2 * nu_in))
        nu_out = tools.calc_poissons_ratio(vs, vp)
        assert_allclose(nu_out, nu_in)


# ---------------------------------------------------------------------------
# Profile.calc_dispersion
# ---------------------------------------------------------------------------

disba = pytest.importorskip("disba")


class TestCalcDispersion:
    def test_raises_without_poissons_ratio(self):
        profile = site.Profile(
            [
                site.Layer(site.SoilType("", 18.0, None, 0.02), 10, 200),
                site.Layer(site.SoilType("", 24.0, None, 0.01), 0, 800),
            ]
        )
        with pytest.raises(ValueError, match="poissons_ratio"):
            profile.calc_dispersion(np.array([1.0, 5.0, 10.0]))

    def test_returns_reasonable_velocities(self):
        profile = _make_profile()
        freqs = np.logspace(-0.5, 1.5, 50)
        vel = profile.calc_dispersion(freqs)
        # Rayleigh phase velocity should be positive and in the range of
        # the profile's shear velocities (roughly 0.9× Vs_min to Vs_max)
        assert vel.shape == freqs.shape
        assert np.all(vel > 0)
        vs_min = min(layer.initial_shear_vel for layer in profile)
        vs_max = max(layer.initial_shear_vel for layer in profile)
        assert np.all(vel >= 0.5 * vs_min)
        assert np.all(vel <= 1.5 * vs_max)

    def test_group_velocity(self):
        profile = _make_profile()
        freqs = np.logspace(-0.5, 1, 30)
        vel_phase = profile.calc_dispersion(freqs, dc_type="phase")
        vel_group = profile.calc_dispersion(freqs, dc_type="group")
        assert vel_group.shape == vel_phase.shape
        assert np.all(vel_group > 0)

    def test_invalid_dc_type(self):
        profile = _make_profile()
        with pytest.raises(ValueError, match="dc_type"):
            profile.calc_dispersion(np.array([1.0]), dc_type="invalid")


# ---------------------------------------------------------------------------
# DispersionCheck + iter_varied_profiles
# ---------------------------------------------------------------------------


class TestDispersionCheck:
    def test_z_score_filtering(self):
        """Generate varied profiles and verify the z-score constraint holds."""
        profile = _make_profile(poissons_ratio=0.25)
        freqs = np.logspace(-0.5, 1.5, 30)

        # Compute the "target" dispersion from the base profile
        target = profile.calc_dispersion(freqs)
        ln_std = 0.1 * np.ones_like(freqs)
        max_z = 2.0

        check = variation.DispersionCheck(freqs, target, ln_std, max_z_score=max_z)

        var_velocity = variation.ToroVelocityVariation.generic_model("USGS C")
        count = 5
        for p in variation.iter_varied_profiles(
            profile,
            count,
            var_velocity=var_velocity,
            check=check,
        ):
            vel = p.calc_dispersion(freqs)
            z = np.log(vel / target) / ln_std
            assert np.max(np.abs(z)) <= max_z + 1e-10

    def test_check_accepts_base_profile(self):
        """The base profile itself should always pass the check."""
        profile = _make_profile()
        freqs = np.logspace(-0.5, 1.5, 20)
        target = profile.calc_dispersion(freqs)
        ln_std = 0.1 * np.ones_like(freqs)

        check = variation.DispersionCheck(freqs, target, ln_std, max_z_score=1.0)
        assert check(profile) is True

    def test_iter_without_check_unchanged(self):
        """Passing check=None should behave identically to the original API."""
        profile = _make_profile()
        var_velocity = variation.ToroVelocityVariation.generic_model("USGS C")
        profiles = list(
            variation.iter_varied_profiles(
                profile, 3, var_velocity=var_velocity, check=None
            )
        )
        assert len(profiles) == 3
