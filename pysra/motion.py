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

import numpy as np

import pyrvt


class Motion(object):
    def __init__(self, freqs=None):
        self._freqs = np.array([] if freqs is None else freqs)

        self._pgv = None

    def _compute_oscillator_transfer_function(self, osc_freq, damping=0.05):
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
        Complex-valued transfer function with length equal to self.freq
        """
        return (-osc_freq ** 2. /
                (np.square(self.freqs) - np.square(osc_freq) -
                    2.j * damping * osc_freq * self.freqs))

    @property
    def freqs(self):
        return self._freqs

    @property
    def angular_freqs(self):
        return 2 * np.pi * self.freqs

    @property
    def pgv(self):
        if self._pgv is None:
            self._pgv = self.compute_peak(1 / (self.angular_freqs * 1j))

        return self.pgv

    def compute_peak(self, transfer_func=None, **kwargs):
        raise NotImplementedError


class TimeSeriesMotion(Motion):
    def __init__(self, filename, description, time_step, accels):
        super(Motion, self).__init__()

        self._filename = filename
        self._description = description
        self._time_step = time_step
        self._accels = np.asarray(accels)

        self._calc_fourier_spectrum()

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

    def compute_peak(self, trans_func=None, **kwargs):
        return np.fft.irfft(trans_func * self._fourier_amps)

    def _calc_fourier_spectrum(self):
        """Compute the Fourier Amplitude Spectrum of the time series."""

        # Use the next power of 2 for the length
        n = 1
        while n < self.accels.size:
            n <<= 1

        self._fourier_amps = np.fft.rfft(self._accels, n)

        freq_step = 1. / (2 * self._time_step * (n / 2))
        self._freqs = freq_step * np.arange(1 + n / 2)

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


class CompatibleRvtMotion(pyrvt.motions.CompatibleRvtMotion, Motion):
    """A :class:`~.motion.CompatibleRvtMotion` object is used to compute a
    Fourier amplitude spectrum that is compatible with a target response
    spectrum.

    Parameters
    ----------
    osc_freqs : :class:`numpy.array`
        Frequencies of the oscillator response [Hz].

    osc_accels_target : :class:`numpy.array`
        Spectral acceleration of the oscillator at the specified frequencies
        [g].

    duration : float or None, default: None
        Duration of the ground motion [sec]. If ``None``, then the duration is
        computed using

    osc_damping : float, default: 0.05
        Damping ratio of the oscillator [dec].

    event_kwds : dict or ``None``, default: ``None``
        Keywords passed to :class:`~.motions.SourceTheoryMotion` and used
        to compute the duration of the motion. Only *duration* or
        *event_kwds* should be specified.

    window_len : int or ``None``, default: ``None``
        Window length used for smoothing the computed Fourier amplitude
        spectrum. If ``None``, then no smoothing is applied. The smoothing is
        applied as a moving average with a width of ``window_len``.

    peak_calculator : str or :class:`~.peak_calculators.Calculator`, default: ``None``
        Peak calculator to use. If ``None``, then the default peak
        calculator is used. The peak calculator may either be specified by a
        :class:`~.peak_calculators.Calculator` object, or by the initials of
        the calculator.

    calc_kwds : dict or ``None``, default: ``None``
        Keywords to be passed during the creation the peak calculator. These
        keywords are only required for some peak calculators.

    """
    def __init__(self, osc_freqs, osc_accels_target, duration=None,
                 osc_damping=0.05, event_kwds=None, window_len=None,
                 peak_calculator=None, calc_kwds=None):
        super(CompatibleRvtMotion, self).__init__(
            osc_freqs, osc_accels_target, duration=duration,
            osc_damping=osc_damping, event_kwds=event_kwds,
            window_len=window_len, peak_calculator=peak_calculator,
            calc_kwds=calc_kwds)

    def compute_peak(self, transfer_func=None, osc_freq=None,
                     osc_damping=None):
        return CompatibleRvtMotion.compute_peak(
            self, transfer_func, osc_freq, osc_damping)


class SourceTheoryRvtMotion(pyrvt.motions.SourceTheoryMotion, Motion):
    """Single-corner source theory model with default parameters from [C03]_.

    Parameters
    ----------
    magnitude : float
        Moment magnitude of the event

    distance : float
        Epicentral distance [km]

    region : {'cena', 'wna'}, str
        Region for the parameters. Either 'cena' for Central and Eastern
        North America, or 'wna' for Western North America.

    stress_drop : float or None, default: ``None``
        Stress drop of the event [bars]. If ``None``, then the default value is
        used. For *region* = 'cena', the default value is computed by the
        [AB11]_ model, while for *region* = 'wna' the default value is 100
        bars.

    depth : float, default: 8
        Hypocenter depth [km]. The *depth* is combined with the
        *distance* to compute the hypocentral distance.

    peak_calculator : str or :class:`~.peak_calculators.Calculator`, default: ``None``
        Peak calculator to use. If ``None``, then the default peak
        calculator is used. The peak calculator may either be specified by a
        :class:`~.peak_calculators.Calculator` object, or by the initials of
        the calculator.

    calc_kwds : dict or ``None``, default: ``None``
        Keywords to be passed during the creation the peak calculator. These
        keywords are only required for some peak calculators.

    """

    def __init__(self, magnitude, distance, region, stress_drop=None,
                 depth=8, peak_calculator=None, calc_kwds=None):
        super(SourceTheoryRvtMotion, self).__init__(
            magnitude, distance, region, stress_drop, depth,
            peak_calculator=peak_calculator, calc_kwds=calc_kwds)

    def compute_peak(self, transfer_func=None, osc_freq=None,
                     osc_damping=None):
        return CompatibleRvtMotion.compute_peak(
            self, transfer_func, osc_freq, osc_damping)