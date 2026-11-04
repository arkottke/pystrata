#!/usr/bin/env python
"""Test site module."""

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
# Copyright (C) Albert Kottke, 2013-2015

import numpy as np
import pytest
from numpy.testing import assert_allclose

from pystrata import site


def test_published_nonlinear_curve():
    site.NonlinearCurve.from_published("Vucetic & Dobry (91), PI=0", "damping")


def test_published_soiltype_same():
    model_mr = "Vucetic & Dobry (91), PI=0"
    model_d = "Vucetic & Dobry (91), PI=15"

    st = site.SoilType.from_published("test", 14, model_mr, model_d)
    assert st.mod_reduc.name == model_mr
    assert st.damping.name == model_d


@pytest.fixture
def nlp():
    """Create an example NonlinearCurve for testing interpolation."""
    # Use explicit limits to allow full range for interpolation testing
    return site.ModulusReductionCurve("", [0.01, 1], [0.0, 1.0], limits=(0, 1))


@pytest.mark.parametrize(
    "strain,expected",
    [
        (0.001, 0.0),
        (2.0, 1.0),
        (0.1, 0.5),
    ],
)
def test_nlp(nlp, strain, expected):
    """Test NonlinearCurve interpolation."""
    assert_allclose(nlp(strain), expected)


@pytest.mark.parametrize("strains", [0.1, [0.1, 10]])
def test_nlp_update(nlp, strains):
    """Test if strains are saved."""
    nlp.strains = strains
    assert_allclose(nlp.strains, strains)


def test_iterative_value():
    """Test the iterative value and relative error."""
    iv = site.IterativeValue(11)
    value = 10
    iv.value = value
    assert_allclose(iv.value, value)
    assert_allclose(iv.relative_error, 0.1)


def test_soil_type_linear():
    """Test the soil type update process on a linear material."""
    damping = 1.0
    layer = site.Layer(site.SoilType("", 18.0, None, damping), 2.0, 500.0)
    layer.strain = 0.1

    assert_allclose(layer.shear_mod, layer.initial_shear_mod)
    assert_allclose(layer.damping, damping)


def test_soil_type_iterative():
    """Test the soil type update process on a nonlinear curve."""
    mod_reduc = site.ModulusReductionCurve("", [0.0001, 0.01], [1, 0])
    damping = site.DampingCurve("", [0.0001, 0.01], [0, 0.10])

    st = site.SoilType("", 18.0, mod_reduc, damping)
    layer = site.Layer(st, 2.0, 500.0)

    strain = 0.001
    layer.strain = strain

    assert_allclose(layer.strain, strain)
    assert_allclose(layer.shear_mod, 0.5 * layer.initial_shear_mod)
    assert_allclose(layer.damping, 0.05)


@pytest.mark.parametrize("depth,expected", [(10, 300), (20, 400), (30, 490.909)])
def test_time_average_vel(depth, expected):
    """Test time averaged shear-wave velocity."""
    st = site.SoilType(unit_wt=17)
    p = site.Profile(
        [
            site.Layer(st, 10, 300),
            site.Layer(st, 10, 600),
            site.Layer(st, 0, 900),
        ]
    )
    assert_allclose(p.time_average_vel(depth), expected, atol=0.001)


def test_simplified_rayleigh_vel():
    """Test simplified Rayleigh wave velocity."""
    # Example from Urzua et al. (2017). Table 1 in Appendix A
    layers = [
        (8, 828, 105),
        (5, 726, 133),
        (7, 1039, 120),
        (8, 825, 120),
        (5, 951, 137),
        (65, 1270, 125),
        (24, 1065, 127),
        (16, 1205, 119),
        (9, 1071, 138),
        (7, 1633, 135),
        (21, 1223, 138),
        (25, 2777, 140),
    ]
    p = site.Profile(
        [
            site.Layer(site.SoilType(unit_wt=unit_wt), thick, vs)
            for thick, vs, unit_wt in layers
        ]
    )

    assert_allclose(
        p.simplified_rayliegh_vel(),
        1349.076,
        atol=0.001,
    )


def create_soil_types():
    """Generate all soil types for interpolation testing."""
    soil_types = []

    # Base SoilType with custom NonlinearCurves
    strains = np.logspace(-6, -1.5, num=20)
    mr = site.ModulusReductionCurve("test_mr", strains, np.linspace(1, 0.1, 20))
    d = site.DampingCurve("test_d", strains, np.linspace(0.01, 0.15, 20))
    soil_types.append(("SoilType", site.SoilType("test", 18.0, mr, d)))

    return soil_types


@pytest.mark.parametrize("name,soil_type", create_soil_types())
@pytest.mark.parametrize("curve_name", ["mod_reduc", "damping"])
def test_nonlinear_curve_interpolation_matches_underlying(name, soil_type, curve_name):
    """Test that interpolating at underlying strain values returns original values.

    This test ensures that limits are properly applied and do not clip values that are
    within the expected range of the nonlinear curves. The interpolated values at the
    underlying strains should match the original curve values.
    """
    curve = getattr(soil_type, curve_name)

    if curve is None or isinstance(curve, (int, float)):
        pytest.skip(f"{name} has no {curve_name} curve")

    strains = curve.strains
    expected = curve.values

    # Interpolate at the same strains used to construct the curve
    actual = curve(strains)

    # Values should match closely - any significant difference indicates
    # limits are improperly clipping values
    assert_allclose(
        actual,
        expected,
        rtol=1e-5,
        atol=1e-7,
        err_msg=f"{name}.{curve_name} interpolation does not match underlying curve",
    )


@pytest.mark.parametrize("name,soil_type", create_soil_types())
@pytest.mark.parametrize("curve_name", ["mod_reduc", "damping"])
def test_nonlinear_curve_limits_not_too_restrictive(name, soil_type, curve_name):
    """Test that curve limits do not clip values within the underlying curve range.

    This test detects issues where default limits improperly clip values at the extremes
    of the curve (e.g., low strain mod_reduc near 1.0 or high strain damping values).
    """
    curve = getattr(soil_type, curve_name)

    if curve is None or isinstance(curve, (int, float)):
        pytest.skip(f"{name} has no {curve_name} curve")

    values = curve.values.ravel() if curve.values.ndim > 1 else curve.values

    min_val, max_val = curve._limits

    # Values at curve extremes should not be clipped by limits
    assert values.min() >= min_val, (
        f"{name}.{curve_name}: min value {values.min()} below limit {min_val}"
    )
    assert values.max() <= max_val, (
        f"{name}.{curve_name}: max value {values.max()} above limit {max_val}"
    )


def test_depth_power_law_mid_depth():
    st = site.SoilType("", 20.0, None, 0.02)
    layers = site.layers_from_depth_power_law(st, [2.0, 4.0], a=100.0, b=0.5)
    assert_allclose([layer.initial_shear_vel for layer in layers], [100.0, 200.0])
    assert_allclose([layer.thickness for layer in layers], [2.0, 4.0])

    # Offset datum
    layers = site.layers_from_depth_power_law(st, [2.0], a=10.0, b=1.0, depth_top=9.0)
    assert_allclose(layers[0].initial_shear_vel, 100.0)


def test_depth_power_law_requires_positive_depth():
    st = site.SoilType("", 20.0, None, 0.02)
    with pytest.raises(ValueError):
        site.layers_from_depth_power_law(st, [2.0], a=100.0, b=0.5, depth_top=-1.0)


def test_stress_power_law_values():
    st = site.SoilType("", 20.0, None, 0.02)
    layers = site.layers_from_stress_power_law(
        st, [2.0, 2.0], a=200.0, b=0.25, wt_depth=1.0, k0=0.5
    )
    sigma_v = np.array([20.0, 60.0])
    pore = 9.80665 * np.array([0.0, 2.0])
    sigma_m = (sigma_v - pore) * 2 / 3
    expected = 200.0 * (sigma_m / site.PA) ** 0.25
    assert_allclose([layer.initial_shear_vel for layer in layers], expected, rtol=1e-4)


def test_stress_power_law_chaining():
    """A second segment with an initial stress continues the first."""
    st = site.SoilType("", 19.0, None, 0.02)
    one = site.layers_from_stress_power_law(st, [3.0] * 4, 150.0, 0.3, wt_depth=4.0)
    first = site.layers_from_stress_power_law(st, [3.0] * 2, 150.0, 0.3, wt_depth=4.0)
    second = site.layers_from_stress_power_law(
        st, [3.0] * 2, 150.0, 0.3, wt_depth=4.0 - 6.0, stress_vert_top=19.0 * 6.0
    )
    assert_allclose(
        [layer.initial_shear_vel for layer in first + second],
        [layer.initial_shear_vel for layer in one],
    )


def test_stress_power_law_matches_profile_stress():
    """Mean effective stress is consistent with Profile / Layer stresses."""
    st = site.SoilType("", 19.0, None, 0.02)
    layers = site.layers_from_stress_power_law(st, [2.0] * 5, 150.0, 0.3, wt_depth=3.0)
    profile = site.Profile(layers + [site.Layer(st, 1.0, 600.0)], wt_depth=3.0)
    for layer in profile[:-1]:
        stress = layer.stress_mean(layer.thickness / 2, effective=True)
        assert_allclose(
            layer.initial_shear_vel, 150.0 * (stress / site.PA) ** 0.3, rtol=1e-12
        )


def test_stress_power_law_negative_stress():
    st = site.SoilType("", 19.0, None, 0.02)
    with pytest.raises(ValueError):
        # Fully submerged by a water table far above with no overburden
        site.layers_from_stress_power_law(st, [2.0], 150.0, 0.3, wt_depth=-100.0)


def test_layer_copy_preserves_type():
    class MyLayer(site.Layer):
        pass

    st = site.SoilType("", 19.0, None, 0.02)
    assert type(MyLayer(st, 1.0, 100.0).copy()) is MyLayer


def test_iterative_value_error_is_two_sided():
    value = site.IterativeValue(100.0)
    value.value = 150.0
    # A rise is as unconverged as a fall
    assert value.relative_error == pytest.approx(1 / 3)
    value.value = 100.0
    assert value.relative_error == pytest.approx(0.5)


def test_layer_copy_keeps_initial_shear_vel():
    layer = site.Layer(site.SoilType("S", 18.0, None, 0.02), 5.0, 200.0)
    layer._shear_mod.value = 0.5 * layer.initial_shear_mod
    softened = layer.shear_vel
    assert softened < 200.0

    copied = layer.copy()
    assert copied.initial_shear_vel == 200.0
    assert copied.shear_vel == 200.0
