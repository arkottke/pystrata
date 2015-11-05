#!/usr/bin/env python
# encoding: utf-8

import os

import nose
from numpy.testing import assert_almost_equal, assert_equal

from pysra import motion


def ts_setup():
    '''Setup the default time series for testing.'''
    global ts
    ts = motion.TimeSeriesMotion.load_at2_file(
            os.path.join(os.path.dirname(__file__), 'NIS090.AT2'))

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
