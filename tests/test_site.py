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
# Copyright (C) Albert Kottke, 2013-2015

import pytest

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


class TestDarendeli:
    @classmethod
    def setup_class(cls):
        kwds = dict(
            plas_index=30, ocr=1.0, mean_stress=0.25, freq=1, num_cycles=10,
            strains=[1E-5, 2.2E-3, 1E-0],
        )
        cls.mod_reduc = site.DarendeliNonlinearProperty(
            **kwds, param='mod_reduc')
        cls.damping = site.DarendeliNonlinearProperty(
            **kwds, param='damping')
        return cls


    @pytest.mark.parametrize(
        'attr,expected',
        [
            ('mod_reduc', [1.0, 0.936, 0.050]),
            ('damping', [0.01778, 0.02476, 0.21542]),
        ]
    )
    def test_values(self, attr, expected):
        # Reference values taken from Tables 10.13 and 10.14 of the Darendeli
        # dissertation.
        actual = getattr(self, attr).values.tolist()
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
    l = site.Layer(site.SoilType('', 18.0, None, damping), 2., 500.)
    l.strain = 0.1

    assert_allclose(l.shear_mod.value, l.initial_shear_mod)
    assert_allclose(l.damping.value, damping)


def test_soil_type_iterative():
    """Test the soil type update process on a nonlinear property."""
    mod_reduc = site.NonlinearProperty('', [0.01, 1.], [1, 0])
    damping = site.NonlinearProperty('', [0.01, 1.], [0, 10])

    st = site.SoilType('', 18.0, mod_reduc, damping)
    l = site.Layer(st, 2., 500.)

    strain = 0.1
    l.strain = strain

    assert_allclose(l.strain.value, strain)
    assert_allclose(l.shear_mod.value, 0.5 * l.initial_shear_mod)
    assert_allclose(l.damping.value, 5.0)


