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


import pathlib

import numpy as np
import pytest

from numpy.testing import assert_allclose, assert_equal

from pysra import motion


@pytest.fixture
def tsm():
    """Create a default time series for testing."""
    return motion.TimeSeriesMotion.load_at2_file(
        pathlib.Path(__file__).parent / 'data/NIS090.AT2'
    )


def test_ts_load_at2_file(tsm):
    """Test loading of an AT2 file."""
    assert_equal(tsm.accels.size, 4096)
    assert_allclose(tsm.time_step, 0.01)
    assert_allclose(tsm.accels[0], 0.233833E-06)
    assert_allclose(tsm.accels[-1], 0.496963E-04)


def test_ts_times(tsm):
    """Test times."""
    assert_allclose(
        [tsm.times[0], tsm.times[1], tsm.times[-1]],
        [0, tsm.time_step, tsm.time_step * (len(tsm.accels) - 1)],
    )


def test_ts_freqs(tsm):
    """Test calculation of a time series frequencies."""
    freqs = tsm.freqs
    assert_equal(tsm.freqs.size, tsm.fourier_amps.size)
    assert_allclose(freqs[0], 0)
    assert_allclose(freqs[-1], 50.)


def test_ts_fft(tsm):
    """Test FFT with no transfer function."""
    assert_allclose(
        tsm.accels,
        tsm.calc_time_series(),
    )


def test_ts_fft_with_tf(tsm):
    """Test FFT with a transfer function."""
    assert_allclose(
        tsm.accels,
        tsm.calc_time_series(2 * np.ones_like(tsm.freqs)) / 2,
    )


@pytest.mark.parametrize('fname', ['2516b_a.smc'])
def test_ts_load_smc_file(fname):
    tsm = motion.TimeSeriesMotion.load_smc_file(
        pathlib.Path(__file__).parent / 'data' / fname
    )

    assert tsm.description == 'VA: Reston; Fire Station #25; 360'

    assert_allclose(
        tsm.time_step,
        1 / 200.
    )

    assert_allclose(
        [tsm.accels[0], tsm.accels[1], tsm.accels[-1]],
        [2.3489E-2, -1.6646E-2, 3.4990E-3],
        rtol=1E-4,
    )
