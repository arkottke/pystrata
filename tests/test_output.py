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

import numpy as np

from numpy.testing import assert_allclose

import pysra


def test_add_refs():
    output = pysra.output.Output()
    refs = [1.1, 2, 3]
    output._add_refs(refs)
    assert_allclose(refs, output.refs)


def test_add_refs_same():
    output = pysra.output.Output()
    # Force float arrays
    a = [1.1, 2, 3]
    b = [1.1, 2, 3]

    output._add_refs(a)
    output._add_refs(b)

    assert np.ndim(output.refs) == 1
    assert_allclose(output.refs, a)


def test_add_refs_diff():
    output = pysra.output.Output()
    # Force float arrays
    a = [1.1, 2, 3]
    b = [1.1, 2, 3, 4, 5]

    output._add_refs(a)
    output._add_refs(b)

    assert np.ndim(output.refs) == 2
    assert len(output.refs) == len(b)
    assert_allclose(output.refs[:, 0], a + 2 * [np.nan])
    assert_allclose(output.refs[:, 1], b)
