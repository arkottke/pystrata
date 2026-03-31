"""Tests for pint unit conversion decorator."""

import numpy as np
import pint
import pytest
from numpy.testing import assert_allclose

from pystrata import motion, output, site, tools, ureg, variation


class TestConvertUnitsDecorator:
    """Test the core @convert_units decorator behavior."""

    def test_plain_float_passthrough(self):
        """Plain floats pass through unchanged."""
        st = site.SoilType("test", unit_wt=18.0)
        assert st.unit_wt == 18.0

    def test_pint_same_units(self):
        """Quantity in expected units extracts magnitude."""
        st = site.SoilType("test", unit_wt=18.0 * ureg.kilonewton / ureg.meter**3)
        assert st.unit_wt == 18.0

    def test_pint_converted_units(self):
        """Quantity in different units is converted."""
        # 1 kip/ft³ ≈ 157.087... kN/m³
        wt = 1.0 * ureg.force_pound / ureg.foot**3
        st = site.SoilType("test", unit_wt=wt)
        assert_allclose(st.unit_wt, wt.to("kilonewton / meter**3").magnitude, rtol=1e-6)

    def test_incompatible_units_raises(self):
        """Incompatible units raise DimensionalityError."""
        with pytest.raises(pint.DimensionalityError):
            site.SoilType("test", unit_wt=5.0 * ureg.meter)

    def test_none_passthrough(self):
        """None values are not converted."""
        layer = site.Layer(site.SoilType("", 18.0), 10.0, 200.0, damping_min=None)
        assert layer.damping_min == site.SoilType("", 18.0).damping_min


class TestLayerUnits:
    """Test Layer constructor unit conversion."""

    def test_thickness_meters(self):
        st = site.SoilType("test", 18.0)
        layer = site.Layer(st, 5.0 * ureg.meter, 200.0)
        assert layer.thickness == 5.0

    def test_thickness_feet(self):
        st = site.SoilType("test", 18.0)
        layer = site.Layer(st, 5.0 * ureg.feet, 200.0)
        assert_allclose(layer.thickness, 1.524, rtol=1e-6)

    def test_shear_vel_mps(self):
        st = site.SoilType("test", 18.0)
        layer = site.Layer(st, 10.0, 200.0 * ureg.meter / ureg.second)
        assert layer.initial_shear_vel == 200.0

    def test_shear_vel_fps(self):
        st = site.SoilType("test", 18.0)
        vel_fps = 656.168 * ureg.feet / ureg.second
        layer = site.Layer(st, 10.0, vel_fps)
        assert_allclose(layer.initial_shear_vel, vel_fps.to("m/s").magnitude, rtol=1e-4)

    def test_plain_floats_unchanged(self):
        st = site.SoilType("test", 18.0)
        layer = site.Layer(st, 10.0, 200.0)
        assert layer.thickness == 10.0
        assert layer.initial_shear_vel == 200.0


class TestDarendeliSoilTypeUnits:
    """Test DarendeliSoilType with pint units."""

    def test_stress_mean_atm(self):
        """1 atmosphere ≈ 101.325 kPa."""
        st = site.DarendeliSoilType(stress_mean=1.0 * ureg.atmosphere)
        assert_allclose(st._stress_mean, 101.325, rtol=1e-4)

    def test_stress_mean_kpa(self):
        st = site.DarendeliSoilType(stress_mean=200.0 * ureg.kilopascal)
        assert st._stress_mean == 200.0

    def test_plain_float_default(self):
        st = site.DarendeliSoilType()
        assert st._stress_mean == 101.3


class TestMenqSoilTypeUnits:
    """Test MenqSoilType with pint units."""

    def test_diam_mean_cm(self):
        """1 cm = 10 mm."""
        st = site.MenqSoilType(diam_mean=1.0 * ureg.centimeter)
        assert_allclose(st._diam_mean, 10.0, rtol=1e-10)

    def test_stress_mean_psi(self):
        stress_psi = 14.696 * ureg.psi
        st = site.MenqSoilType(stress_mean=stress_psi)
        assert_allclose(
            st._stress_mean, stress_psi.to("kilopascal").magnitude, rtol=1e-4
        )


class TestProfileUnits:
    """Test Profile constructor unit conversion."""

    def test_wt_depth_meters(self):
        p = site.Profile(wt_depth=5.0 * ureg.meter)
        assert p.wt_depth == 5.0

    def test_wt_depth_feet(self):
        p = site.Profile(wt_depth=10.0 * ureg.feet)
        assert_allclose(p.wt_depth, 3.048, rtol=1e-4)


class TestTimeSeriesMotionUnits:
    """Test TimeSeriesMotion unit conversion."""

    def test_time_step_seconds(self):
        accels = np.array([0.0, 0.1, -0.05, 0.0])
        m = motion.TimeSeriesMotion("", "", 0.01 * ureg.second, accels)
        assert m.time_step == 0.01

    def test_time_step_milliseconds(self):
        accels = np.array([0.0, 0.1, -0.05, 0.0])
        m = motion.TimeSeriesMotion("", "", 10.0 * ureg.millisecond, accels)
        assert_allclose(m.time_step, 0.01, rtol=1e-10)

    def test_plain_float(self):
        accels = np.array([0.0, 0.1, -0.05, 0.0])
        m = motion.TimeSeriesMotion("", "", 0.01, accels)
        assert m.time_step == 0.01


class TestOutputLocationUnits:
    """Test OutputLocation unit conversion."""

    def test_depth_meters(self):
        loc = output.OutputLocation("within", depth=5.0 * ureg.meter)
        assert loc.depth == 5.0

    def test_depth_feet(self):
        loc = output.OutputLocation("within", depth=10.0 * ureg.feet)
        assert_allclose(loc.depth, 3.048, rtol=1e-4)

    def test_depth_none(self):
        loc = output.OutputLocation("within", index=0)
        assert loc.depth is None


class TestDispersionCheckUnits:
    """Test DispersionCheck unit conversion."""

    def test_target_velocity_fps(self):
        freqs = [1.0, 2.0, 5.0]
        target_fps = np.array([500.0, 400.0, 300.0]) * ureg.feet / ureg.second
        dc = variation.DispersionCheck(
            freqs, target_fps, ln_std=[0.1, 0.1, 0.1], max_z_score=2.0
        )
        assert_allclose(dc.target, target_fps.to("meter/second").magnitude, rtol=1e-6)

    def test_plain_array(self):
        freqs = [1.0, 2.0, 5.0]
        target = [200.0, 150.0, 100.0]
        dc = variation.DispersionCheck(
            freqs, target, ln_std=[0.1, 0.1, 0.1], max_z_score=2.0
        )
        assert_allclose(dc.target, target)


class TestCalcPoissonsRatioUnits:
    """Test calc_poissons_ratio with pint units."""

    def test_fps_input(self):
        vs = 1000.0 * ureg.feet / ureg.second
        vp = 2000.0 * ureg.feet / ureg.second
        result = tools.calc_poissons_ratio(vs, vp)
        # Same as plain m/s — ratio is dimensionless, so units cancel
        expected = tools.calc_poissons_ratio(
            vs.to("m/s").magnitude, vp.to("m/s").magnitude
        )
        assert_allclose(result, expected, rtol=1e-10)

    def test_plain_float(self):
        result = tools.calc_poissons_ratio(200.0, 400.0)
        r2 = (400.0 / 200.0) ** 2
        expected = (r2 - 2) / (2 * (r2 - 1))
        assert_allclose(result, expected)


class TestKishidaSoilTypeUnits:
    """Test KishidaSoilType stress_vert conversion."""

    def test_stress_vert_atm(self):
        st = site.KishidaSoilType(stress_vert=1.0 * ureg.atmosphere)
        assert_allclose(st._stress_vert, 101.325, rtol=1e-4)


class TestRollinsEtAlSoilTypeUnits:
    """Test RollinsEtAlSoilType stress_mean conversion."""

    def test_stress_mean_psi(self):
        stress = 14.696 * ureg.psi
        st = site.RollinsEtAlSoilType(stress_mean=stress)
        assert_allclose(st._stress_mean, stress.to("kilopascal").magnitude, rtol=1e-4)


class TestAlemuEtAlSoilTypeUnits:
    """Test AlemuEtAlSoilType stress_mean conversion."""

    def test_stress_mean_bar(self):
        stress = 1.0 * ureg.bar
        st = site.AlemuEtAlSoilType(stress_mean=stress)
        assert_allclose(st._stress_mean, 100.0, rtol=1e-4)


class TestTwoParamModifiedHyperbolicUnits:
    """Test TwoParamModifiedHyperbolicSoilType unit conversion."""

    def test_stress_mean_atm(self):
        st = site.TwoParamModifiedHyperbolicSoilType(stress_mean=1.0 * ureg.atmosphere)
        assert_allclose(st._stress_mean, 101.325, rtol=1e-4)
