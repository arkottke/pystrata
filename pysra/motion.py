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

import enum

import numpy as np

import pyrvt

# Gravity in m/sec²
# Source: http://physics.nist.gov/cgi-bin/cuu/Value?gn
GRAVITY = 9.80665


class WaveField(enum.Enum):
    outcrop = 0
    within = 1
    incoming_only = 2


class Motion(object):
    def __init__(self, freqs=None):
        object.__init__(self)

        self._freqs = np.array([] if freqs is None else freqs)
        self._pga = None
        self._pgv = None

    @property
    def freqs(self):
        return self._freqs

    @property
    def angular_freqs(self):
        return 2 * np.pi * self.freqs

    @property
    def pgv(self):
        """Peak-ground velocity [cm/sec]."""
        if self._pgv is None:
            tf = 1 / (self.angular_freqs * 1j)
            tf[0] = 0.
            self._pgv = GRAVITY * 100 * self.calc_peak(tf)

        return self._pgv

    @property
    def pga(self):
        if self._pga is None:
            self._pga = self.calc_peak()
        return self._pga


class TimeSeriesMotion(Motion):
    def __init__(self, filename, description, time_step, accels,
                 fa_length=None):
        Motion.__init__(self)

        self._filename = filename
        self._description = description
        self._time_step = time_step
        self._accels = np.asarray(accels)

        self._calc_fourier_spectrum(fa_length)

    @property
    def accels(self):
        return self._accels

    @property
    def filename(self):
        return self._filename

    @property
    def description(self):
        return self._description

    @property
    def time_step(self):
        return self._time_step

    @property
    def freqs(self):
        """Return the frequencies."""
        if self._freqs is None:
            self._calc_fourier_spectrum()

        return self._freqs

    @property
    def fourier_amps(self):
        """Return the frequencies."""
        if self._fourier_amps is None:
            self._calc_fourier_spectrum()

        return self._fourier_amps

    def calc_time_series(self, tf=None):
        if tf is None:
            ts = np.fft.irfft(self._fourier_amps)
        else:
            ts = np.fft.irfft(tf * self._fourier_amps)
        return ts

    def calc_peak(self, tf=None, **kwargs):
        ts = self.calc_time_series(tf)
        return np.abs(ts).max()

    def calc_osc_accels(self, osc_freqs, osc_damping=0.05, tf=None):
        """Compute the pseudo-acceleration spectral response of an oscillator
        with a specific frequency and damping.

        Parameters
        ----------
        osc_freq : float
            Frequency of the oscillator (Hz).
        osc_damping : float
            Fractional damping of the oscillator (dec). For example, 0.05 for a
            damping ratio of 5%.
        tf : array_like, optional
            Transfer function to be applied to motion prior calculation of the
            oscillator response.

        Returns
        -------
        spec_accels : :class:`numpy.ndarray`
            Peak pseudo-spectral acceleration of the oscillator
        """
        if tf is None:
            tf = np.ones_like(self.freqs)
        else:
            tf = np.asarray(tf).astype(complex)

        resp = np.array(
            [self.calc_peak(tf * self._calc_sdof_tf(of, osc_damping))
             for of in osc_freqs]
        )
        return resp

    def _calc_fourier_spectrum(self, fa_length=None):
        """Compute the Fourier Amplitude Spectrum of the time series."""

        if fa_length is None:
            # Use the next power of 2 for the length
            n = 1
            while n < self.accels.size:
                n <<= 1
        else:
            n = fa_length

        self._fourier_amps = np.fft.rfft(self._accels, n)

        freq_step = 1. / (2 * self._time_step * (n / 2))
        self._freqs = freq_step * np.arange(1 + n / 2)

    def _calc_sdof_tf(self, osc_freq, damping=0.05):
        """Compute the transfer function for a single-degree-of-freedom
        oscillator.

        Parameters
        ----------
        osc_freq : float
            natural frequency of the oscillator [Hz]
        damping : float, optional
            damping ratio of the oscillator in decimal. Default value is
            0.05, or 5%.

        Returns
        -------
        tf : :class:`numpy.ndarray`
            Complex-valued transfer function with length equal to `self.freq`.
        """
        return (-osc_freq ** 2. /
                (np.square(self.freqs) - np.square(osc_freq) -
                    2.j * damping * osc_freq * self.freqs))

    @classmethod
    def load_at2_file(cls, filename):
        """Read an AT2 formatted time series.

        Parameters
        ----------
        filename: str
            Filename to open.

        """
        with open(filename) as fp:
            next(fp)
            description = next(fp).strip()
            next(fp)
            parts = next(fp).split()
            time_step = float(parts[1])

            accels = [float(p) for l in fp for p in l.split()]

        return cls(filename, description, time_step, accels)


class RvtMotion(pyrvt.motions.RvtMotion, Motion):
    def __init__(self, osc_freqs, osc_accels_target, duration=None,
                 peak_calculator=None, calc_kwds=None):
        Motion.__init__(self)
        pyrvt.motions.RvtMotion.__init__(
            self, osc_freqs, osc_accels_target, duration=duration,
            peak_calculator=peak_calculator,
            calc_kwds=calc_kwds)


class CompatibleRvtMotion(pyrvt.motions.CompatibleRvtMotion, Motion):
    def __init__(self, osc_freqs, osc_accels_target, duration=None,
                 osc_damping=0.05, event_kwds=None, window_len=None,
                 peak_calculator=None, calc_kwds=None):
        Motion.__init__(self)
        pyrvt.motions.CompatibleRvtMotion.__init__(
            self, osc_freqs, osc_accels_target, duration=duration,
            osc_damping=osc_damping, event_kwds=event_kwds,
            window_len=window_len, peak_calculator=peak_calculator,
            calc_kwds=calc_kwds)


class SourceTheoryRvtMotion(pyrvt.motions.SourceTheoryMotion, Motion):
    def __init__(self, magnitude, distance, region, stress_drop=None,
                 depth=8, peak_calculator=None, calc_kwds=None):
        Motion.__init__(self)
        pyrvt.motions.SourceTheoryMotion.__init__(
            self, magnitude, distance, region, stress_drop, depth,
            peak_calculator=peak_calculator, calc_kwds=calc_kwds)
