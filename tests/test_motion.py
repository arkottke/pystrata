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


import os

import nose
from numpy.testing import assert_almost_equal, assert_equal

from pysra import motion


def ts_setup():
    '''Setup the default time series for testing.'''
    global ts
    ts = motion.TimeSeriesMotion.load_at2_file(
            os.path.join(os.path.dirname(__file__), 'data', 'NIS090.AT2'))

def ts_teardown():
    global ts
    del ts

@nose.with_setup(ts_setup, ts_teardown)
def test_ts_load_at2_file():
    global ts

    assert_equal(ts.accels.size, 4096)
    assert_almost_equal(ts.time_step, 0.01)

    assert_almost_equal(ts.accels[0], 0.233833E-06)
    assert_almost_equal(ts.accels[-1], 0.496963E-04)

@nose.with_setup(ts_setup, ts_teardown)
def test_ts_freqs():
    global ts

    freqs = ts.freqs

    assert_almost_equal(freqs[0], 0)
    assert_almost_equal(freqs[-1], 50.)

    assert_equal(ts.freqs.size, ts.fourier_amps.size)
