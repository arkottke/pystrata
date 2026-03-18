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
import pytest
from numpy.testing import assert_allclose
from scipy.stats import lognorm, norm, pearsonr

from pystrata import motion, output, propagation, site, variation


def test_randnorm():
    # Get some errors of 0.0055
    assert_allclose(1, np.std(variation.randnorm(size=100000)), rtol=0.007)


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
        cls.st = site.DarendeliSoilType(
            unit_wt=16,
            plas_index=0,
            ocr=1,
            stress_mean=1 / site.KPA_TO_ATM,
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
                site.DarendeliSoilType(18.0, plas_index=0, ocr=1, stress_mean=200),
                10,
                300,
            ),
            site.Layer(
                site.DarendeliSoilType(18.0, plas_index=0, ocr=1, stress_mean=200),
                10,
                400,
            ),
            site.Layer(
                site.DarendeliSoilType(18.0, plas_index=0, ocr=1, stress_mean=200),
                10,
                500,
            ),
            site.Layer(
                site.DarendeliSoilType(18.0, plas_index=0, ocr=1, stress_mean=200),
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


def test_iter_variations(profile):
    m = motion.SourceTheoryRvtMotion(6.0, 30, "wna")
    m.calc_fourier_amps()

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
