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

from pysra import site, variation


def test_darendeli_variation_mod_reduc():
    nlp = site.DarendeliNonlinearProperty(
        0, 1, 1, freq=1, num_cycles=10,
        strains=[1E-5, 2.2E-3, 1E0],
        param='mod_reduc'
    )
    print(nlp.values)
    darendeli_var = variation.DarendeliVariation(0)
    actual = darendeli_var._calc_std_mod_reduc(nlp.values)
    # Values from Table 11.1 of Darendeli dissertation.
    expected = [0.01836, 0.05699, 0.04818]
    assert_allclose(actual, expected, rtol=0.01)


def test_darendeli_variation_damping():
    nlp = site.DarendeliNonlinearProperty(
        0, 1, 1, freq=1, num_cycles=10,
        strains=[1E-5, 2.2E-3, 1E0],
        param='damping'
    )
    darendeli_var = variation.DarendeliVariation(0)
    actual = darendeli_var._calc_std_damping(nlp.values)
    # Values from Table 11.1 of Darendeli dissertation.
    expected = [0.70766, 0.99402, 3.55137]
    assert_allclose(actual, expected, rtol=0.01)
