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
from numpy.testing import assert_allclose, assert_equal

from pystrata import motion

from . import FPATH_DATA


@pytest.fixture
def tsm():
    """Create a default time series for testing."""
    return motion.TimeSeriesMotion.load_at2_file(FPATH_DATA / "NIS090.AT2")


def test_ts_load_at2_file(tsm):
    """Test loading of an AT2 file."""
    assert_equal(tsm.accels.size, 4096)
    assert_allclose(tsm.time_step, 0.01)
    assert_allclose(tsm.accels[0], 0.233833e-06)
    assert_allclose(tsm.accels[-1], 0.496963e-04)


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
    assert_allclose(freqs[-1], 50.0)


def test_ts_max(tsm):
    """Check that maximum is consistent with the time domain maximum (PGA)."""
    assert_allclose(tsm.pga, tsm.calc_peak())


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


def test_ts_fa_normalize(tsm):
    """Test the normalization of the Fourier amplitudes with Parsevla's theorem."""
    assert_allclose(
        np.trapezoid(tsm.accels**2, dx=tsm.time_step),
        2 * np.trapezoid(np.abs(tsm.fourier_amps) ** 2, x=tsm.freqs),
    )


@pytest.mark.parametrize("fname", ["2516b_a.smc"])
def test_ts_load_smc_file(fname):
    tsm = motion.TimeSeriesMotion.load_smc_file(FPATH_DATA / fname)
    assert tsm.description == "VA: Reston; Fire Station #25; 360"

    assert_allclose(tsm.time_step, 1 / 200.0)

    assert_allclose(
        [tsm.accels[0], tsm.accels[1], tsm.accels[-1]],
        [2.3489e-2, -1.6646e-2, 3.4990e-3],
        rtol=1e-4,
    )


@pytest.mark.parametrize(
    "header,time_step",
    [
        ("NPTS=  6000, DT= .0100 SEC", 0.01),
        ("NPTS=  16396, DT= 0.005 SEC", 0.005),
        ("NPTS=6000, DT=0.0100 SEC", 0.01),
        ("   4096   0.0100   NPTS, DT", 0.01),
    ],
)
def test_load_at2_file_header_formats(tmp_path, header, time_step):
    fpath = tmp_path / "test.AT2"
    fpath.write_text(
        "PEER NGA\nEVENT\nACCELERATION TIME SERIES IN UNITS OF G\n"
        f"{header}\n 0.1 0.2 0.3\n"
    )
    ts = motion.TimeSeriesMotion.load_at2_file(fpath)
    assert_allclose(ts.time_step, time_step)
    assert_allclose(ts.accels, [0.1, 0.2, 0.3])


def test_rvt_from_fas_returns_the_pystrata_class():
    class Fas:
        freqs = np.logspace(-1, 2, 50)
        fourier_amps = 0.01 / (1 + (freqs / 5) ** 2)
        duration = 10.0

    rvt = motion.RvtMotion.from_fas(Fas())
    assert type(rvt) is motion.RvtMotion
    assert isinstance(rvt, motion.Motion)


def test_sdof_tf_is_the_pyrvt_one(tsm):
    import pyrvt

    assert_allclose(
        tsm._calc_sdof_tf(2.0, 0.03),
        pyrvt.motions.calc_sdof_tf(tsm.freqs, 2.0, 0.03),
    )
