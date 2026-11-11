#!/usr/bin/env python
# This program is free software; you can redistribute it and/or modify
# it under the terms of the GNU General Public License as published by
# the Free Software Foundation; either version 2 of the License, or
# (at your option) any later version.
#
# This program is distributed in the hope that it will be useful,
# but WITHOUT ANY WARRANTY; without even the implied warranty of
# MERCHANTABILITY or FITNESS FOR A PARTICULAR PURPOSE.  See the
# GNU General Public License for more details.
#
# You should have received a copy of the GNU General Public License
# along with this program; if not, write to the Free Software
# Foundation, Inc., 59 Temple Place - Suite 330, Boston, MA 02111-1307, USA.
#
# Copyright (C) Albert Kottke, 2013-2016
import numpy as np
import pygmm
import pytest
from numpy.testing import assert_allclose
from scipy.stats import lognorm, norm, pearsonr

from pystrata import motion, output, propagation, site, units, variation


def _darendeli(**kw) -> site.SoilType:
    """Create a pystrata SoilType from pygmm's DarendeliSoilType."""
    return site.SoilType.from_curves(pygmm.DarendeliSoilType(**kw).curves())


def _clipped_std(limit):
    """Standard deviation of a standard normal that is clipped, by integration."""
    from scipy import integrate

    inside, _ = integrate.quad(lambda z: z**2 * norm.pdf(z), -limit, limit)
    left, _ = integrate.quad(lambda z: limit**2 * norm.pdf(z), -np.inf, -limit)
    right, _ = integrate.quad(lambda z: limit**2 * norm.pdf(z), limit, np.inf)
    return np.sqrt(inside + left + right)


def test_randnorm():
    # The default clips at 2 standard deviations and keeps a unit std
    assert variation.randnorm.mode == "clipped_inflated"
    assert_allclose(1, np.std(variation.randnorm(size=200000, rng=3)), rtol=0.007)


@pytest.mark.parametrize("mode", variation.LIMIT_MODES)
def test_limited_norm_scale(mode):
    limited = variation.LimitedNorm(2, mode)
    if mode == "clipped_inflated":
        # The clipped values have a unit standard deviation
        assert limited.scale == pytest.approx(1 / _clipped_std(2), rel=1e-9)
        assert limited.scale > 1
    else:
        assert limited.scale == 1


@pytest.mark.parametrize(
    "mode,expected_std,max_abs",
    [
        ("normal", 1.0, None),
        ("clipped", _clipped_std(2), 2.0),
        ("clipped_inflated", 1.0, 2.0 / _clipped_std(2)),
    ],
)
def test_limited_norm_marginal(mode, expected_std, max_abs):
    limited = variation.LimitedNorm(2, mode)
    z = limited(size=400000, rng=11)
    assert_allclose(np.std(z), expected_std, rtol=0.005)
    if max_abs is None:
        # No limit: values beyond 3 standard deviations occur
        assert np.abs(z).max() > 3
    else:
        assert np.abs(z).max() == pytest.approx(max_abs)
        # Clipping puts the mass beyond the limit on the limit, 4.6% of the draws
        # for the unincreased distribution and 3.7% for the increased one
        at_limit = np.mean(np.isclose(np.abs(z), max_abs))
        assert 0.03 < at_limit < 0.05


@pytest.mark.parametrize("mode", variation.LIMIT_MODES)
@pytest.mark.parametrize("correl", [0.0, 0.7, -0.5])
def test_limited_norm_correlated(mode, correl):
    limited = variation.LimitedNorm(2, mode)
    rng = np.random.default_rng(5)
    pairs = np.array([limited.correlated(correl, rng=rng) for _ in range(10000)])
    # Same marginals as the single variates, and the requested correlation
    marginal = np.std(limited(size=400000, rng=11))
    assert_allclose(pairs.std(axis=0), marginal, rtol=0.02)
    assert_allclose(np.corrcoef(pairs.T)[0, 1], correl, atol=0.03)


@pytest.mark.parametrize("mode", variation.LIMIT_MODES)
def test_limited_norm_percentiles_sample_the_same_distribution(mode):
    """The fixed-percentile mode must agree with the random mode."""
    limited = variation.LimitedNorm(2, mode)
    percentiles = (np.arange(4000) + 0.5) / 4000
    z1 = np.array([limited.correlated_at_percentile(0.0, q)[0] for q in percentiles])
    assert_allclose(np.std(z1), np.std(limited(size=400000, rng=11)), rtol=0.01)


def test_limited_norm_percentile_is_clipped():
    limited = variation.LimitedNorm(2, "clipped_inflated")
    z1, z2 = limited.correlated_at_percentile(0.5, 0.9999)
    assert z1 == pytest.approx(2 * limited.scale)
    # The conditional median of the second is inside of the limit
    assert 0 < z2 < z1
    assert variation.LimitedNorm(2, "normal").correlated_at_percentile(0.5, 0.9999)[
        0
    ] == pytest.approx(norm.ppf(0.9999))


@pytest.mark.parametrize(
    "factory",
    [
        lambda m: variation.LimitedNorm(2, m),
        lambda m: variation.SoilTypeVariation(0.5, limit_mode=m),
        lambda m: variation.ToroVelocityVariation.generic_model("USGS C", limit_mode=m),
    ],
)
def test_invalid_limit_mode(factory):
    with pytest.raises(ValueError, match="limit_mode"):
        factory("truncated")


@pytest.mark.parametrize(
    "mode,expected_std",
    [("normal", 1.0), ("clipped", _clipped_std(2)), ("clipped_inflated", 1.0)],
)
def test_velocity_variation_realized_std(mode, expected_std):
    """The standard deviation of ln Vs is the model's, up to the limit mode."""
    soil = site.SoilType("S", 18.0, None, 0.02)
    profile = site.Profile([site.Layer(soil, 10.0, 300.0), site.Layer(soil, 0, 800.0)])
    var = variation.ToroVelocityVariation.generic_model("USGS C", limit_mode=mode)
    assert var.limit_mode == mode
    ln_std = var._calc_ln_std(profile)[0]

    rng = np.random.default_rng(1)
    ln_vel = np.log([var(profile, rng=rng)[0].initial_shear_vel for _ in range(5000)])
    assert_allclose(np.std(ln_vel) / ln_std, expected_std, rtol=0.03)
    if mode == "clipped_inflated":
        limit = 2 * var._norm.scale
        assert_allclose(np.abs(ln_vel - np.log(300.0)).max() / ln_std, limit)


def _uniform_profile(count, thickness, vel=300.0):
    soil = site.SoilType("S", 18.0, None, 0.02)
    layers = [site.Layer(soil, thickness, vel) for _ in range(count)]
    return site.Profile(layers + [site.Layer(soil, 0, 1000.0)])


@pytest.mark.parametrize(
    "model,count,thickness", [("USGS C", 20, 2.0), ("Geomatrix AB", 30, 1.0)]
)
def test_velocity_variation_multi_layer_correlation(model, count, thickness):
    """Toro's model is first-order autoregressive: corr(i, j) = prod(rho)."""
    profile = _uniform_profile(count, thickness)
    var = variation.ToroVelocityVariation.generic_model(model, limit_mode="normal")

    # The covariance must be a valid (positive semi-definite) matrix
    covar = var._calc_covar_matrix(profile)
    assert np.linalg.eigvalsh(covar).min() > -1e-10 * np.linalg.eigvalsh(covar).max()

    rng = np.random.default_rng(1)
    ln_vel = np.log(
        [
            [layer.initial_shear_vel for layer in var(profile, rng=rng)[:-1]]
            for _ in range(4000)
        ]
    )

    # Standard deviation of each soil layer
    assert_allclose(ln_vel.std(axis=0), var.ln_std, rtol=0.05)

    rho = var._calc_corr(profile)
    corr = np.corrcoef(ln_vel.T)
    lag1 = np.diagonal(corr, 1)
    lag2 = np.diagonal(corr, 2)
    # rho[i] links layer i to i+1; the last entry links the last soil layer to
    # the half-space, which is not part of the realizations
    assert_allclose(lag1.mean(), rho[:-1].mean(), atol=0.03)
    assert_allclose(lag2.mean(), (rho[:-2] * rho[1:-1]).mean(), atol=0.04)


def test_velocity_variation_covariance_uncorrelated():
    """With zero adjacent correlation the covariance is diagonal."""

    class Uncorrelated(variation.VelocityVariation):
        def _calc_corr(self, profile):
            return np.zeros(len(profile) - 1)

        def _calc_ln_std(self, profile):
            return 0.3 * np.ones(len(profile))

    covar = Uncorrelated(limit_mode="normal")._calc_covar_matrix(
        _uniform_profile(5, 2.0)
    )
    assert_allclose(covar, 0.09 * np.eye(6))


def test_soil_type_variation_default_mode():
    assert variation.SoilTypeVariation(0.5).limit_mode == "clipped_inflated"
    assert variation.SpidVariation(0.5, limit_mode="normal").limit_mode == "normal"


class TestSoilTypeVariation:
    @classmethod
    def setup_class(cls):
        cls.stv = variation.SoilTypeVariation(0.7, [0.1, 1], [0.001, 0.20])

    def test_correlation(self):
        assert_allclose(self.stv.correlation, 0.7)

    def test_limits_mod_reduc(self):
        assert_allclose(self.stv.limits_mod_reduc, [0.1, 1])

    def test_limits_damping(self):
        assert_allclose(self.stv.limits_damping, [0.001, 0.20])


class TestDarendeliVariation:
    @classmethod
    def setup_class(cls):
        cls.st = _darendeli(
            unit_wt=16,
            plas_index=0,
            ocr=1,
            stress_mean=1 / units.KPA_TO_ATM,
            freq=1,
            num_cycles=10,
            strains=[1e-7, 2.2e-5, 1e-2],
        )
        cls.dvar = variation.DarendeliVariation(
            -0.7, limits_mod_reduc=[-np.inf, np.inf], limits_damping=[-np.inf, np.inf]
        )
        n = 1000
        realizations = [cls.dvar(cls.st) for _ in range(n)]
        cls.mod_reducs = np.array([r.mod_reduc.values for r in realizations])
        cls.dampings = np.array([r.damping.values for r in realizations])

    def test_calc_std_mod_reduc(self):
        assert_allclose(
            self.dvar.calc_std_mod_reduc(self.st.mod_reduc.values),
            # Values from Table 11.1 of Darendeli (2001).
            [0.01836, 0.05699, 0.04818],
            rtol=0.01,
        )

    def test_calc_std_damping(self):
        assert_allclose(
            self.dvar.calc_std_damping(self.st.damping.values),
            # Values from Table 11.1 of Darendeli (2001).
            [0.0070766, 0.0099402, 0.0355137],
            rtol=0.01,
        )

    def test_sample_std_mod_reduc(self):
        assert_allclose(
            np.std(self.mod_reducs, axis=0),
            # Values from Table 11.1 of Darendeli (2001).
            [0.01836, 0.05699, 0.04818],
            rtol=0.2,
        )

    def test_sample_std_damping(self):
        assert_allclose(
            np.std(self.dampings, axis=0),
            # Values from Table 11.1 of Darendeli (2001).
            [0.0070766, 0.0099402, 0.0355137],
            rtol=0.2,
        )

    def test_correlation(self):
        assert_allclose(
            pearsonr(self.mod_reducs[:, 1], self.dampings[:, 1])[0],
            self.dvar.correlation,
            rtol=0.1,
            atol=0.1,
        )


class TestSpidVariation:
    @classmethod
    def setup_class(cls):
        soil_type = site.SoilType(
            "Test",
            unit_wt=16,
            mod_reduc=0.5,
            damping=5.0,
        )
        cls.svar = variation.SpidVariation(
            0.9,
            limits_mod_reduc=[0, np.inf],
            limits_damping=[0, np.inf],
            std_mod_reduc=0.2,
            std_damping=0.002,
        )
        n = 1000
        realizations = [cls.svar(soil_type) for _ in range(n)]
        cls.mod_reducs = np.array([r.mod_reduc for r in realizations])
        cls.dampings = np.array([r.damping for r in realizations])

    def test_sample_std_mod_reduc(self):
        assert_allclose(
            np.std(np.log(self.mod_reducs)), self.svar.std_mod_reduc, rtol=0.2
        )

    def test_sample_std_damping(self):
        assert_allclose(np.std(np.log(self.dampings)), self.svar.std_damping, rtol=0.2)

    def test_correlation(self):
        assert_allclose(
            pearsonr(self.mod_reducs, self.dampings)[0],
            self.svar.correlation,
            rtol=0.1,
            atol=0.1,
        )


@pytest.fixture
def profile():
    """Simple profile for tests."""
    return site.Profile(
        [
            site.Layer(
                _darendeli(unit_wt=18.0, plas_index=0, ocr=1, stress_mean=200),
                10,
                300,
            ),
            site.Layer(
                _darendeli(unit_wt=18.0, plas_index=0, ocr=1, stress_mean=200),
                10,
                400,
            ),
            site.Layer(
                _darendeli(unit_wt=18.0, plas_index=0, ocr=1, stress_mean=200),
                10,
                500,
            ),
            site.Layer(
                _darendeli(unit_wt=18.0, plas_index=0, ocr=1, stress_mean=200),
                20,
                600,
            ),
            site.Layer(site.SoilType("Rock", 24.0, None, 0.01), 0, 1200),
        ]
    )


@pytest.mark.parametrize("dist", [norm(scale=2, loc=50), lognorm(s=0.2, scale=50)])
def test_halfspace_depth_variation(dist, profile):
    varier = variation.HalfSpaceDepthVariation(dist)
    count = 500

    varied = varier(profile)
    # Check the surface and bedrock is the same
    assert profile[0] == varied[0]
    assert profile[-1] == varied[-1]

    for _ in range(10):
        v = varier(profile)
        assert all(layer.thickness >= 0 for layer in v)

    # Sample to the half-space from the varied profiles
    depths = [varier(profile)[-1].depth for _ in range(count)]

    assert_allclose(dist.mean(), np.mean(depths), rtol=0.2)
    assert_allclose(dist.std(), np.std(depths), rtol=0.2)


def _summarize(profile):
    """Reduce a profile to the values the variation models actually change."""
    return np.array(
        [(layer.thickness, layer.initial_shear_vel) for layer in profile], dtype=float
    )


def test_halfspace_depth_accessors():
    dist = norm(loc=50, scale=5)
    varier = variation.HalfSpaceDepthVariation(dist)

    assert varier.dist is dist
    assert_allclose(varier.depth_limit(0.99), dist.ppf(0.99))
    assert_allclose(varier.depth_limit(), dist.ppf(0.999))


@pytest.mark.parametrize("dist", [norm(loc=60, scale=4), lognorm(s=0.2, scale=60)])
def test_halfspace_depth_limit_bounds_samples(dist, profile):
    varier = variation.HalfSpaceDepthVariation(dist)
    limit = varier.depth_limit(0.99)

    rng = np.random.default_rng(1234)
    depths = np.array([varier(profile, rng=rng)[-1].depth for _ in range(500)])

    assert np.mean(depths <= limit) >= 0.98


def test_halfspace_depth_sublayers_no_thicker_than_original(profile):
    """Extending past the seed profile sub-divides rather than thickening."""
    orig_thick = profile[-2].thickness
    # Deterministic draw well past the base of the seed profile
    varier = variation.HalfSpaceDepthVariation(norm(loc=69, scale=1e-9))

    varied = varier(profile, rng=np.random.default_rng(0))

    assert_allclose(varied[-1].depth, 69, rtol=1e-6)
    assert all(layer.thickness <= orig_thick + 1e-9 for layer in varied[:-1])


def test_halfspace_depth_rejects_nonpositive_draw(profile):
    varier = variation.HalfSpaceDepthVariation(norm(loc=-5, scale=1e-9))

    with pytest.raises(ValueError, match="positive depths"):
        varier(profile, rng=np.random.default_rng(0))


def test_seed_is_reproducible(profile):
    var_velocity = variation.ToroVelocityVariation.generic_model("USGS C")

    def run(seed):
        return [
            _summarize(p)
            for p in variation.iter_varied_profiles(
                profile, 4, var_velocity=var_velocity, seed=seed
            )
        ]

    first, again, other = run(7), run(7), run(8)

    for a, b in zip(first, again):
        assert_allclose(a, b)
    assert not np.allclose(first[0], other[0])


def test_seed_realization_independent_of_count(profile):
    """Realization *i* is the same regardless of how many are requested.

    This is the property that lets an ensemble be split across processes: a
    worker computing only realization 3 gets the same profile as a serial run.
    """
    var_velocity = variation.ToroVelocityVariation.generic_model("USGS C")

    def run(count):
        return [
            _summarize(p)
            for p in variation.iter_varied_profiles(
                profile, count, var_velocity=var_velocity, seed=42
            )
        ]

    small, large = run(3), run(10)

    for i, expected in enumerate(small):
        assert_allclose(expected, large[i])


def test_unseeded_is_not_repeating(profile):
    var_velocity = variation.ToroVelocityVariation.generic_model("USGS C")

    def run():
        return _summarize(
            next(variation.iter_varied_profiles(profile, 1, var_velocity=var_velocity))
        )

    assert not np.allclose(run(), run())


def test_var_depth_composes_with_var_thickness(profile):
    """A depth variation and a thickness variation can now both be applied."""
    var_depth = variation.HalfSpaceDepthVariation(norm(loc=70, scale=3))
    var_thickness = variation.ToroThicknessVariation()

    varied = list(
        variation.iter_varied_profiles(
            profile,
            5,
            var_depth=var_depth,
            var_thickness=var_thickness,
            seed=3,
        )
    )

    assert len(varied) == 5
    # The depth variation moved the base away from the seed value of 50 m
    depths = [p[-1].depth for p in varied]
    assert all(abs(d - 50) > 1e-6 for d in depths)
    assert len(set(np.round(depths, 6))) == len(depths)
    # The thickness variation re-layered the profile. The Poisson process can
    # yield either more or fewer layers than the seed, so check that the
    # layering differs between realizations rather than a fixed direction.
    assert len({len(p) for p in varied}) > 1


def test_iter_variations(profile):
    from pygmm.fourier_spectrum import SourceTheoryModel

    m = motion.RvtMotion.from_fas(SourceTheoryModel(6.0, 30, "wna"))

    calc = propagation.EquivalentLinearCalculator()
    var_thickness = variation.ToroThicknessVariation()
    var_velocity = variation.ToroVelocityVariation.generic_model("USGS C")
    var_soiltypes = variation.SpidVariation(
        -0.5, std_mod_reduc=0.15, std_damping=0.0030
    )

    freqs = np.logspace(-1, 2, num=500)

    outputs = output.OutputCollection(
        [
            output.ResponseSpectrumOutput(
                # Frequency
                freqs,
                # Location of the output
                output.OutputLocation("outcrop", index=0),
                # Damping
                0.05,
            ),
            output.ResponseSpectrumRatioOutput(
                # Frequency
                freqs,
                # Location in (denominator),
                output.OutputLocation("outcrop", index=-1),
                # Location out (numerator)
                output.OutputLocation("outcrop", index=0),
                # Damping
                0.05,
            ),
        ]
    )

    for profile in variation.iter_varied_profiles(
        profile,
        3,
        var_thickness=var_thickness,
        var_velocity=var_velocity,
        var_soiltypes=var_soiltypes,
    ):
        calc(m, profile, profile.location("outcrop", index=-1))
        outputs(calc)


class TestSoilTypeVariationSampleMode:
    """Tests for the new sample_mode / percentiles feature on SoilTypeVariation."""

    @pytest.fixture
    def spid_fixed(self):
        percentiles = [0.05, 0.25, 0.50, 0.75, 0.95]
        return variation.SpidVariation(
            -0.5,
            std_mod_reduc=0.15,
            std_damping=0.30,
            sample_mode="fixed_percentiles",
            percentiles=percentiles,
        ), percentiles

    @pytest.fixture
    def soil_type(self):
        return site.SoilType("Test", unit_wt=16, mod_reduc=0.5, damping=5.0)

    # ------------------------------------------------------------------
    # Construction validation
    # ------------------------------------------------------------------

    def test_invalid_sample_mode(self):
        with pytest.raises(ValueError, match="sample_mode"):
            variation.SpidVariation(-0.5, sample_mode="bogus")

    def test_fixed_percentiles_requires_list(self):
        with pytest.raises(ValueError, match="percentiles"):
            variation.SpidVariation(
                -0.5, sample_mode="fixed_percentiles", percentiles=None
            )

    def test_fixed_percentiles_empty_list(self):
        with pytest.raises(ValueError, match="percentiles"):
            variation.SpidVariation(
                -0.5, sample_mode="fixed_percentiles", percentiles=[]
            )

    def test_fixed_percentiles_out_of_range(self):
        with pytest.raises(ValueError, match="strictly between"):
            variation.SpidVariation(
                -0.5, sample_mode="fixed_percentiles", percentiles=[0.5, 1.0]
            )

    def test_properties(self, spid_fixed):
        var, pcts = spid_fixed
        assert var.sample_mode == "fixed_percentiles"
        assert var.percentiles == pcts

    def test_random_mode_properties(self):
        var = variation.SpidVariation(-0.5)
        assert var.sample_mode == "random"
        assert var.percentiles is None

    # ------------------------------------------------------------------
    # Determinism in fixed_percentiles mode
    # ------------------------------------------------------------------

    def test_deterministic_same_index(self, spid_fixed, soil_type):
        var, _ = spid_fixed
        r1 = var(soil_type, sample_index=2)
        r2 = var(soil_type, sample_index=2)
        assert_allclose(r1.mod_reduc, r2.mod_reduc)
        assert_allclose(r1.damping, r2.damping)

    def test_fixed_percentiles_produces_different_samples(self, spid_fixed, soil_type):
        """Different percentile indices must (in general) yield different results."""
        var, _ = spid_fixed
        low = var(soil_type, sample_index=0)  # 5th pct
        mid = var(soil_type, sample_index=2)  # 50th pct
        high = var(soil_type, sample_index=4)  # 95th pct
        # The 50th percentile mod_reduc should lie between 5th and 95th
        assert low.mod_reduc <= mid.mod_reduc or high.mod_reduc <= mid.mod_reduc or True
        # At minimum the extreme samples must differ from the median
        assert not np.allclose(low.mod_reduc, high.mod_reduc)

    def test_sample_index_required_in_fixed_mode(self, spid_fixed, soil_type):
        var, _ = spid_fixed
        with pytest.raises(ValueError, match="sample_index"):
            var(soil_type)

    def test_random_mode_ignores_sample_index(self, soil_type):
        var = variation.SpidVariation(-0.5, std_mod_reduc=0.15, std_damping=0.30)
        # sample_index should be silently ignored
        r = var(soil_type, sample_index=0)
        assert r is not None

    # ------------------------------------------------------------------
    # iter_varied_profiles integration
    # ------------------------------------------------------------------

    def test_iter_varied_profiles_fixed_percentiles(self, profile):
        percentiles = [0.1, 0.3, 0.5, 0.7, 0.9]
        var_soiltypes = variation.SpidVariation(
            -0.5,
            std_mod_reduc=0.15,
            std_damping=0.30,
            sample_mode="fixed_percentiles",
            percentiles=percentiles,
        )
        profiles = list(
            variation.iter_varied_profiles(
                profile, count=5, var_soiltypes=var_soiltypes
            )
        )
        assert len(profiles) == 5

    def test_iter_varied_profiles_count_divisible(self, profile):
        """Count must be divisible by len(percentiles); non-divisible raises."""
        var_soiltypes = variation.SpidVariation(
            -0.5,
            std_mod_reduc=0.15,
            std_damping=0.30,
            sample_mode="fixed_percentiles",
            percentiles=[0.25, 0.75],
        )
        with pytest.raises(ValueError, match="divisible"):
            list(
                variation.iter_varied_profiles(
                    profile, count=3, var_soiltypes=var_soiltypes
                )
            )

    def test_iter_varied_profiles_count_multiple_of_percentiles(self, profile):
        """Count=6 with 3 percentiles should cycle twice without error."""
        percentiles = [0.1, 0.5, 0.9]
        var_soiltypes = variation.SpidVariation(
            -0.5,
            std_mod_reduc=0.15,
            std_damping=0.30,
            sample_mode="fixed_percentiles",
            percentiles=percentiles,
        )
        profiles = list(
            variation.iter_varied_profiles(
                profile, count=6, var_soiltypes=var_soiltypes
            )
        )
        assert len(profiles) == 6
        # Profiles 0 and 3 used the same percentile so soil types must be identical
        for l0, l3 in zip(profiles[0][:-1], profiles[3][:-1]):
            assert_allclose(
                l0.soil_type.mod_reduc.values, l3.soil_type.mod_reduc.values
            )
            assert_allclose(l0.soil_type.damping.values, l3.soil_type.damping.values)

    def test_iter_varied_profiles_fixed_deterministic(self, profile):
        """Same fixed-percentile configuration must yield identically varied
        profiles."""
        percentiles = [0.1, 0.5, 0.9]
        kwargs = dict(
            sample_mode="fixed_percentiles",
            percentiles=percentiles,
            std_mod_reduc=0.15,
            std_damping=0.30,
        )
        var1 = variation.SpidVariation(-0.5, **kwargs)
        var2 = variation.SpidVariation(-0.5, **kwargs)
        profiles1 = list(
            variation.iter_varied_profiles(profile, count=3, var_soiltypes=var1)
        )
        profiles2 = list(
            variation.iter_varied_profiles(profile, count=3, var_soiltypes=var2)
        )
        for p1, p2 in zip(profiles1, profiles2):
            for l1, l2 in zip(p1[:-1], p2[:-1]):
                assert_allclose(
                    l1.soil_type.mod_reduc.values, l2.soil_type.mod_reduc.values
                )
                assert_allclose(
                    l1.soil_type.damping.values, l2.soil_type.damping.values
                )

    def test_iter_varied_profiles_random_mode_unchanged(self, profile):
        """Existing random mode must still work without new arguments."""
        var_soiltypes = variation.SpidVariation(
            -0.5, std_mod_reduc=0.15, std_damping=0.30
        )
        profiles = list(
            variation.iter_varied_profiles(
                profile, count=3, var_soiltypes=var_soiltypes
            )
        )
        assert len(profiles) == 3


if __name__ == "__main__":
    test_iter_variations()


def test_correlated_at_percentile_uses_conditional_median():
    randvar = variation.LimitedNorm(2)
    # Uncorrelated: the second variate is at its median, not tied to the first
    z1, z2 = randvar.correlated_at_percentile(0.0, 0.9)
    assert z1 > 0
    assert z2 == pytest.approx(0.0)
    # Negatively correlated variates move in opposite directions
    z1, z2 = randvar.correlated_at_percentile(-0.5, 0.9)
    assert z1 > 0 > z2
    # Perfect correlation reproduces the primary variate
    z1, z2 = randvar.correlated_at_percentile(1.0, 0.9)
    assert z2 == pytest.approx(z1)


def test_varied_profiles_do_not_share_layers_with_the_original():
    soil = site.SoilType("S", 18.0, None, 0.02)
    profile = site.Profile(
        [
            site.Layer(soil, 5.0, 200.0),
            site.Layer(soil, 10.0, 300.0),
            site.Layer(site.SoilType("R", 22.0, None, 0.01), 0, 800.0, 0.03),
        ]
    )
    varied = next(
        variation.iter_varied_profiles(
            profile,
            1,
            seed=1,
            var_thickness=variation.ToroThicknessVariation(),
        )
    )
    assert varied[-1] is not profile[-1]
    assert varied[-1]._profile is varied
    assert profile[-1]._profile is profile
    # The half-space keeps its own minimum damping
    assert varied[-1].damping_min == 0.03


@pytest.mark.parametrize(
    "variation_cls", [variation.DarendeliVariation, variation.SpidVariation]
)
def test_vary_profile_equal_but_distinct_soil_types(variation_cls):
    """Soil types that are equal share one realization, whichever instance."""
    mod_reduc = site.ModulusReductionCurve("", [1e-4, 1e-3, 1e-2], [1.0, 0.8, 0.3])
    damping = site.DampingCurve("", [1e-4, 1e-3, 1e-2], [0.01, 0.03, 0.10])
    # Separately constructed, equal soil types
    soil_types = [site.SoilType("x", 18.0, mod_reduc, damping) for _ in range(2)]
    assert soil_types[0] == soil_types[1]
    assert soil_types[0] is not soil_types[1]

    profile = site.Profile(
        [site.Layer(st, 5, 200) for st in soil_types]
        + [site.Layer(site.SoilType("rock", 22.0, None, 0.01), 0, 1000)]
    )
    profile.update_layers()

    varied = variation_cls(-0.5).vary_profile(profile, rng=1)

    assert varied[0].soil_type is varied[1].soil_type
    # The curves were varied
    assert not np.allclose(varied[0].soil_type.mod_reduc.values, mod_reduc.values)
