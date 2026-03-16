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
import json

import numpy as np
import pytest
import scipy.constants
from numpy.testing import assert_allclose

from pystrata import site

from . import FPATH_DATA


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


@pytest.fixture
def soil_type_darendeli():
    """Create an example DarendeliSoilType."""
    stress_mean = 0.25 / site.KPA_TO_ATM
    return site.DarendeliSoilType(
        plas_index=30,
        ocr=1.0,
        stress_mean=stress_mean,
        freq=1,
        num_cycles=10,
        strains=[1e-7, 2.2e-5, 1e-2],
    )


@pytest.mark.parametrize(
    "attr,expected",
    [
        ("mod_reduc", [1.0, 0.936, 0.050]),
        ("damping", [0.01778, 0.02476, 0.21542]),
    ],
)
def test_darendeli(soil_type_darendeli, attr, expected):
    """Test calculated values of the DarendeliSoilType."""
    # Reference values taken from Tables 10.13 and 10.14 of the Darendeli
    # dissertation.
    actual = getattr(soil_type_darendeli, attr).values.tolist()
    assert_allclose(actual, expected, rtol=0.01)


# ---------------------------------------------------------------------------
# RollinsEtAlSoilType tests
# Reference values computed analytically from Eqs. (1), (5), (8) in
# Rollins et al. (2020), J. Geotech. Geoenviron. Eng., 146(9): 04020076.
# G/Gmax = 1 / (1 + (γ/γ_ref)^0.84)
# Eq. 5 (no Cu):  γ_ref [%] = 0.0039 * σ'₀^0.42
# Eq. 8 (with Cu): γ_ref [%] = 0.0046 * Cu^(-0.197) * σ'₀^0.52
# ---------------------------------------------------------------------------

_ROLLINS_TEST_STRAINS = [1e-6, 1e-4, 1e-3, 1e-2]

# (stress_mean_kPa, coef_unif_or_None, expected_strain_ref_decimal,
#  expected_mod_reduc_at_strains, expected_damping_min)
_ROLLINS_CASES = [
    (
        # Eq. 5: no Cu, σ'₀ = 100 kPa
        100.0,
        None,
        2.698141e-04,
        [0.991, 0.6971, 0.2497, 0.0459],
        0.01,
    ),
    (
        # Eq. 8: Cu = 7, σ'₀ = 100 kPa
        100.0,
        7.0,
        3.437744e-04,
        [0.9926, 0.7383, 0.2897, 0.0557],
        0.01,
    ),
    (
        # Eq. 8: Cu = 7, σ'₀ = 25 kPa  (lower confining pressure)
        25.0,
        7.0,
        1.671869e-04,
        [0.9866, 0.6063, 0.1821, 0.0312],
        0.01,
    ),
    (
        # Eq. 8: Cu = 7, σ'₀ = 400 kPa  (higher confining pressure)
        400.0,
        7.0,
        7.068784e-04,
        [0.996, 0.8379, 0.4277, 0.0975],
        0.01,
    ),
]


@pytest.mark.parametrize(
    "stress_mean,coef_unif,expected_strain_ref,expected_mr,expected_dmin",
    _ROLLINS_CASES,
)
def test_rollins_mod_reduc(
    stress_mean, coef_unif, expected_strain_ref, expected_mr, expected_dmin
):
    """G/Gmax backbone matches Eqs. (1), (5) and (8) of Rollins et al. (2020)."""
    st = site.RollinsEtAlSoilType(
        unit_wt=20.0,
        stress_mean=stress_mean,
        coef_unif=coef_unif,
        num_cycles=10,
        strains=_ROLLINS_TEST_STRAINS,
    )
    # Reference strain matches the paper equation
    assert_allclose(st.strain_ref, expected_strain_ref, rtol=1e-4)
    # G/Gmax at reference strain is exactly 0.5 by definition
    assert_allclose(
        1.0 / (1.0 + (st.strain_ref / st.strain_ref) ** st.curvature),
        0.5,
        rtol=1e-10,
    )
    # G/Gmax values match analytical calculation
    assert_allclose(st.mod_reduc.values, expected_mr, rtol=0.01)
    # Curvature is fixed at 0.84
    assert st.curvature == 0.84


@pytest.mark.parametrize(
    "stress_mean,coef_unif,expected_strain_ref,expected_mr,expected_dmin",
    _ROLLINS_CASES,
)
def test_rollins_damping_min(
    stress_mean, coef_unif, expected_strain_ref, expected_mr, expected_dmin
):
    """Minimum damping defaults to 1 % and is approached at small strains."""
    st = site.RollinsEtAlSoilType(
        unit_wt=20.0,
        stress_mean=stress_mean,
        coef_unif=coef_unif,
        num_cycles=10,
        strains=_ROLLINS_TEST_STRAINS,
    )
    assert_allclose(st.damping.values[0], expected_dmin, rtol=0.10)
    # Damping must increase monotonically (or stay the same) with strain
    assert np.all(np.diff(st.damping.values) >= 0)


@pytest.mark.parametrize(
    "num_cycles,expected_b",
    [
        (1, 0.530000),
        (10, 0.516875),
        (30, 0.510613),
    ],
)
def test_rollins_masing_scaling(num_cycles, expected_b):
    """Masing scaling factor matches Eq. (14): b = 0.53 - 0.0057*ln(N)."""
    st = site.RollinsEtAlSoilType(unit_wt=20.0, stress_mean=100.0, num_cycles=num_cycles)
    assert_allclose(st.masing_scaling, expected_b, rtol=1e-4)


def test_rollins_damping_custom_min():
    """User-supplied damping_min overrides the 1 % default."""
    st = site.RollinsEtAlSoilType(
        unit_wt=20.0, stress_mean=100.0, damping_min=0.02,
        strains=[1e-6, 1e-4, 1e-3, 1e-2],
    )
    # At very small strains damping should be close to d_min = 2 %
    assert_allclose(st.damping.values[0], 0.02, rtol=0.05)


def iter_wang_stokoe_cases():
    # Ranges for the test cases
    ranges = {
        "stress_mean": (50, 1000),
        "plas_index": (0, 100),
        "ocr": (1, 10),
        "void_ratio": (0.3, 1.5),
        "coef_unif": (1, 40),
        "diam_50": (0.1, 20),
        "fines_cont": (5, 40),
        "water_cont": (0, 35),
    }
    param_names = list(ranges.keys())
    print(param_names)

    for _ in range(20):
        params = {
            k: np.random.uniform(*ranges[k])
            # stress_mean is required
            for k in param_names[: np.random.randint(1, len(param_names))]
        }
        for soil_group in site.WangSoilType.FACTORS:
            yield soil_group, params


@pytest.mark.parametrize("soil_type,params", iter_wang_stokoe_cases())
def test_wang_stokoe(soil_type, params):
    """Test random parameters and check within reasonable ranges"""
    st = site.WangSoilType(soil_type, **params)

    damping_min = st.damping.values[0]
    assert damping_min > 0
    assert damping_min < 0.15

    assert np.max(st.mod_reduc.values) <= 1
    assert np.min(st.mod_reduc.values) >= 0

    ref_strain = np.interp(0.5, st.mod_reduc.values[::-1], st.mod_reduc.strains[::-1])

    assert ref_strain > 1e-4
    assert ref_strain < 1e-1


def test_iterative_value():
    """Test the iterative value and relative error."""
    iv = site.IterativeValue(11)
    value = 10
    iv.value = value
    assert_allclose(iv.value, value)
    assert_allclose(iv.relative_error, 10.0)


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


with (FPATH_DATA / "kishida_2009.json").open() as fp:
    kishida_cases = json.load(fp)
    for i in range(len(kishida_cases)):
        kishida_cases[i]["strains"] = np.array(kishida_cases[i]["strains"]) / 100
        kishida_cases[i]["dampings"] = np.array(kishida_cases[i]["dampings"]) / 100


def format_kishida_case_id(case):
    """Create an ID for the Kishida test cases."""
    fmt = "({stress_vert:.1f} kN/m², OC={organic_content:.0f} %)"
    return fmt.format(**case)


@pytest.mark.parametrize("case", kishida_cases, ids=format_kishida_case_id)
def test_kishida_unit_wt(case):
    """Test calculation of Unit Wt. by KishidaSoilType."""
    st = site.KishidaSoilType(
        "test",
        unit_wt=None,
        stress_vert=case["stress_vert"],
        organic_content=case["organic_content"],
        strains=case["strains"],
    )
    assert_allclose(st.unit_wt, scipy.constants.g * case["density"], rtol=0.005)


@pytest.mark.parametrize("case", kishida_cases, ids=format_kishida_case_id)
@pytest.mark.parametrize(
    "curve,attr,key",
    [
        ("mod_reduc", "strains", "strains"),
        ("mod_reduc", "values", "mod_reducs"),
        ("damping", "strains", "strains"),
        ("damping", "values", "dampings"),
    ],
)
def test_kishida_nlc(case, curve, attr, key):
    """Test properties calculated by KishidaSoilType."""
    st = site.KishidaSoilType(
        "test",
        unit_wt=None,
        stress_vert=case["stress_vert"],
        organic_content=case["organic_content"],
        strains=case["strains"],
    )
    # Decimal damping used inside pyStrata
    scale = 100 if key == "dampings" else 1
    scale = 1
    assert_allclose(
        scale * getattr(getattr(st, curve), attr), case[key], rtol=0.005, atol=0.0005
    )


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

    # DarendeliSoilType
    soil_types.append((
        "DarendeliSoilType",
        site.DarendeliSoilType(
            plas_index=30, ocr=1.0, stress_mean=100, freq=1, num_cycles=10
        ),
    ))

    # MenqSoilType
    soil_types.append((
        "MenqSoilType",
        site.MenqSoilType(coef_unif=10, diam_mean=5, stress_mean=100, num_cycles=10),
    ))

    # TwoParamModifiedHyperbolicSoilType
    soil_types.append((
        "TwoParamModifiedHyperbolicSoilType",
        site.TwoParamModifiedHyperbolicSoilType(stress_mean=100),
    ))

    # WangSoilType - all soil groups
    for soil_group in site.WangSoilType.FACTORS:
        soil_types.append((
            f"WangSoilType_{soil_group}",
            site.WangSoilType(soil_group, stress_mean=200, void_ratio=0.6),
        ))

    # KishidaSoilType
    soil_types.append((
        "KishidaSoilType",
        site.KishidaSoilType(stress_vert=100, organic_content=20),
    ))

    # RollinsEtAlSoilType - without and with uniformity coefficient
    soil_types.append((
        "RollinsEtAlSoilType_no_cu",
        site.RollinsEtAlSoilType(unit_wt=20.0, stress_mean=100.0),
    ))
    soil_types.append((
        "RollinsEtAlSoilType_cu7",
        site.RollinsEtAlSoilType(unit_wt=20.0, stress_mean=100.0, coef_unif=7.0),
    ))

    return soil_types


@pytest.mark.parametrize("name,soil_type", create_soil_types())
@pytest.mark.parametrize("curve_name", ["mod_reduc", "damping"])
def test_nonlinear_curve_interpolation_matches_underlying(name, soil_type, curve_name):
    """Test that interpolating at underlying strain values returns original values.

    This test ensures that limits are properly applied and do not clip values
    that are within the expected range of the nonlinear curves. The interpolated
    values at the underlying strains should match the original curve values.
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

    This test detects issues where default limits improperly clip values at
    the extremes of the curve (e.g., low strain mod_reduc near 1.0 or high
    strain damping values).
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
