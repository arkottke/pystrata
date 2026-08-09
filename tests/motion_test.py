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


@pytest.mark.parametrize(
    "line",
    [
        # Values preceding their labels
        "  4096    0.0100    NPTS, DT",
        "  0.0100    4096    DT, NPTS",
        "  4096    0.0100    NPTS DT",
        # Values following their labels
        "NPTS=   4096, DT=   .0100 SEC,",
        "DT=   .0100 SEC, NPTS=   4096,",
        "NPTS=   4096 DT=   .0100 SEC",
        "NPTS   4096   DT   .0100",
        # Unlabeled
        "  4096    0.0100",
    ],
)
def test_parse_at2_header(line):
    """Test parsing of the NPTS and DT header line variations."""
    npts, time_step = motion._parse_at2_header(line)
    assert_equal(npts, 4096)
    assert_allclose(time_step, 0.01)


def test_parse_at2_header_invalid():
    """Test that an unparsable header line is reported."""
    with pytest.raises(ValueError):
        motion._parse_at2_header("NPTS, DT")


@pytest.mark.parametrize(
    "header",
    ["  8    0.0100    NPTS, DT", "NPTS=      8, DT=   .0100 SEC,"],
)
def test_ts_load_at2_file_headers(tmp_path, header):
    """Test loading of an AT2 file using each header layout."""
    accels = [0.1, -0.2, 0.3, -0.4, 0.5, -0.6, 0.7, -0.8]
    fpath = tmp_path / "test.AT2"
    fpath.write_text(
        "PEER NGA STRONG MOTION DATABASE RECORD\n"
        "Imperial Valley-02, 5/19/1940, El Centro Array #9, 270\n"
        "ACCELERATION TIME SERIES IN UNITS OF G\n"
        f"{header}\n" + "  ".join(f"{a:.6E}" for a in accels) + "\n"
    )

    tsm = motion.TimeSeriesMotion.load_at2_file(fpath)
    assert_equal(
        tsm.description, "Imperial Valley-02, 5/19/1940, El Centro Array #9, 270"
    )
    assert_allclose(tsm.time_step, 0.01)
    assert_allclose(tsm.accels, accels)


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
