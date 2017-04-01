#!/usr/bin/env python
# encoding: utf-8

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

import os
import json

import pytest
import scipy.constants

from numpy.testing import assert_allclose

from pysra import site


@pytest.fixture
def nlp():
    # Simple nonlinear property
    return site.NonlinearProperty('', [0.01, 1], [0., 1.])


def test_nlp_lowerbound(nlp):
    assert_allclose(nlp(0.001), 0.)


def test_nlp_upperbound(nlp):
    assert_allclose(nlp(2.), 1.)


def test_nlp_midpoint(nlp):
    assert_allclose(nlp(0.1), 0.5)


def test_nlp_update(nlp):
    new_values = [0, 2]
    nlp.values = new_values
    assert_allclose(new_values, nlp.values)

    new_strains = [0.1, 10]
    nlp.strains = new_strains
    assert_allclose(new_strains, nlp.strains)

    assert_allclose(nlp(1.), 1.)


@pytest.fixture
def soil_type_darendeli():
    mean_stress = 0.25 / site.KPA_TO_ATM
    return site.DarendeliSoilType(
        plas_index=30,
        ocr=1.0,
        mean_stress=mean_stress,
        freq=1,
        num_cycles=10,
        strains=[1E-5, 2.2E-3, 1E-0], )


@pytest.mark.parametrize('attr,expected', [
    ('mod_reduc', [1.0, 0.936, 0.050]),
    ('damping', [0.01778, 0.02476, 0.21542]),
])
def test_darendeli(soil_type_darendeli, attr, expected):
    # Reference values taken from Tables 10.13 and 10.14 of the Darendeli
    # dissertation.
    actual = getattr(soil_type_darendeli, attr).values.tolist()
    assert_allclose(actual, expected, rtol=0.01)


def test_iterative_value():
    """Test the iterative value and relative error."""
    iv = site.IterativeValue(11)
    value = 10
    iv.value = value
    assert_allclose(iv.value, value)
    assert_allclose(iv.relative_error, 10.)


def test_soil_type_linear():
    """Test the soil type update process on a linear material."""
    damping = 1.0
    layer = site.Layer(site.SoilType('', 18.0, None, damping), 2., 500.)
    layer.strain = 0.1

    assert_allclose(layer.shear_mod.value, layer.initial_shear_mod)
    assert_allclose(layer.damping.value, damping)


def test_soil_type_iterative():
    """Test the soil type update process on a nonlinear property."""
    mod_reduc = site.NonlinearProperty('', [0.01, 1.], [1, 0])
    damping = site.NonlinearProperty('', [0.01, 1.], [0, 10])

    st = site.SoilType('', 18.0, mod_reduc, damping)
    layer = site.Layer(st, 2., 500.)

    strain = 0.1
    layer.strain = strain

    assert_allclose(layer.strain.value, strain)
    assert_allclose(layer.shear_mod.value, 0.5 * layer.initial_shear_mod)
    assert_allclose(layer.damping.value, 5.0)


with open(
        os.path.join(os.path.dirname(__file__), 'data',
                     'kishida_2009.json')) as fp:
    kishida_cases = json.load(fp)


def format_kishida_case_id(case):
    fmt = "({mean_stress:.1f} kN/m², OC={organic_content:.0f} %)"
    return fmt.format(**case)


@pytest.mark.parametrize('case', kishida_cases, ids=format_kishida_case_id)
def test_kishida_unit_wt(case):
    st = site.KishidaSoilType(
        'test',
        unit_wt=None,
        mean_stress=case['mean_stress'],
        organic_content=case['organic_content'],
        strains=case['strains'])
    assert_allclose(
        st.unit_wt, scipy.constants.g * case['density'], rtol=0.005)


@pytest.mark.parametrize('case', kishida_cases, ids=format_kishida_case_id)
@pytest.mark.parametrize('curve,attr,key', [
    ('mod_reduc', 'strains', 'strains'),
    ('mod_reduc', 'values', 'mod_reducs'),
    ('damping', 'strains', 'strains'),
    ('damping', 'values', 'dampings'),
])
def test_kishida_nlc(case, curve, attr, key):
    st = site.KishidaSoilType(
        'test',
        unit_wt=None,
        mean_stress=case['mean_stress'],
        organic_content=case['organic_content'],
        strains=case['strains'])
    assert_allclose(
        getattr(getattr(st, curve), attr), case[key], rtol=0.005, atol=0.0005)
