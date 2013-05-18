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
# Copyright (C) Albert Kottke, 2013

import numpy as np

from scipy import Inf
from scipy.integrate import quad


class Motion(object):
    def _compute_oscillator_transfer_function(self, osc_freq, damping=0.05):
        '''Compute the transfer function for a single-degree-of-freedom oscillator.

        Parameters
        ----------
            osc_freq : float
                natural frequency of the oscillator [Hz]
            damping : float, optional
                damping ratio of the oscillator in decimal. Default value is 0.05, or 5%.

        Returns
        -------
        Complex-valued transfer function with length equal to self.freq
        '''
        return (-osc_freq ** 2. /
                (np.square(self.freq) - np.square(osc_freq)
                    - 2.j * damping * osc_freq * self.freq))


class TimeSeriesMotion(Motion):
    def compute_peak(self, fourier_amp=None):
        pass

    def __init__(self, filename, description, time_step, accels):
        self._filename = None
        self._description = description
        self._time_step = time_step
        self._accels = np.asarray(accels)

        self._freqs = None
        self._fourier_amps = None

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
        '''Return the frequencies.'''
        if self._freqs is None:
            self._calc_fourier_spectrum()
        
        return self._freqs

    @property
    def fourier_amps(self):
        '''Return the frequencies.'''
        if self._fourier_amps is None:
            self._calc_fourier_spectrum()

        return self._fourier_amps

    def _calc_fourier_spectrum(self):
        '''Compute the Fourier Amplitude Spectrum of the time series.'''

        # Use the next power of 2 for the length
        n = 1
        while n < self.accels.size:
            n <<= 1

        self._fourier_amps = np.fft.rfft(self._accels, n)

        freq_step = 1. / (2 * self._time_step * (n / 2))
        self._freqs = freq_step * np.arange(1 + n / 2)

    @classmethod
    def load_at2_file(cls, filename):
        '''Read an AT2 formatted time series.'''
        with open(filename) as fp:
            next(fp)
            description = next(fp).strip()
            next(fp)
            parts = next(fp).split()
            time_step = float(parts[1])

            accels = [float(p) for l in fp for p in l.split()]

        return cls(filename, description, time_step, accels)

class RvtMotion(Motion):
    def __init__(self, freq=None, fourier_amp=None, duration=None):
        self.freq = freq
        self.fourier_amp = fourier_amp
        self.duration = duration

    def compute_osc_resp(self, osc_freq, damping=0.05):
        '''Compute the response of an oscillator with a specific frequency and
        damping.

        Parameters
        ----------
        osc_freq : array_like
            natural frequency of the oscillator
        damping : float (optional)
            damping of the oscillator in decimal

        Returns:
        psa : float
            peak psuedo spectral acceleration of the oscillator
        '''

        def compute_spec_accel(fn):
            # Compute the transfer function
            h = np.abs(self._compute_oscillator_transfer_function(fn, damping))

            fourier_amp = self.fourier_amp * h
            duration_rms = self._compute_duration_rms(
                fn, damping, method='liu_pezeshk',
                fa_sqr=np.square(fourier_amp))

            return self.compute_peak(fourier_amp, duration_rms)

        return np.array(map(compute_spec_accel, osc_freq))

    def compute_peak(self, fourier_amp=None, duration=None):
        '''Compute the expected peak response in the time domain.

        Parameters
        ----------
        fourier_amp : array_like, optional
            Fourier amplitude spectra at frequencies of self.freq

        duration : float, optional
            root-mean-squared duration. If no value is given, the ground motion
            duration is used.

        Returns:
        --------
        peak : float
            peak response in the time domain
        '''
        if fourier_amp is None:
            fourier_amp = self.fourier_amp

        if duration is None:
            duration = self.duration

        fa_sqr = np.square(fourier_amp)
        m0 = self._compute_moment(fa_sqr, 0)
        m2 = self._compute_moment(fa_sqr, 2)
        m4 = self._compute_moment(fa_sqr, 4)

        bandWidth = np.sqrt((m2 * m2) / (m0 * m4))
        numExtrema = max(2., np.sqrt(m4 / m2) * self.duration / np.pi)

        # Compute the peak factor by the indefinite integral
        peakFactor = np.sqrt(2.) * quad(
            lambda z: 1. - (1. - bandWidth * np.exp(-z * z)) ** numExtrema,
            0, Inf)[0]

        return np.sqrt(m0 / duration) * peakFactor

    def _compute_moment(self, fa_sqr, order=0):
        '''Compute the n-th moment.

        Parameters
        ----------
            fa_sqr : array_like
                Squared Fourier amplitude spectrum according to frequencies of
                self.freq
            order : int, optional
                the order of the moment. Default is 0

        Returns
        -------
        out : float
            the moment of the Fourier amplitude spectrum
        '''
        return 2. * np.trapz(
            np.power(2 * np.pi * self.freq, order) * fa_sqr, self.freq)

    def _compute_duration_rms(self, osc_freq, damping=0.05,
                              method='liu_pezeshk', fa_sqr=None):
        '''Compute the oscillator duration correction using the Liu and
        Pezeshk correction.

        The duration

        Parameters
        ----------
            osc_freq : float
                Frequency of the oscillator in Hz
            damping : float
                Damping of the oscillator in decimal.
            method : str, optional
                Method used to compute the oscillator duration. Default is Liu and Pezeshk.
            fa_sqr : array_like, optional
                Squared Fourier amplitude spectrum (FAS). If none is provided
                then the FAS of the motion is used.


        Returns
        -------
            The root-mean-squared duration of the ground motion.
        '''
        if method == 'liu_pezeshk':
            if fa_sqr is None:
                fa_sqr = np.square(self.fourier_amp)

            m0 = self._compute_moment(fa_sqr, 0)
            m1 = self._compute_moment(fa_sqr, 1)
            m2 = self._compute_moment(fa_sqr, 2)

            power = 2.0
            bar = np.sqrt(2. * np.pi * (1. - m1 ** 2. / (m0 * m2)))
        elif method == 'boore_joyner':
            power = 3.0
            bar = 1.0 / 3.0
        else:
            raise NotImplementedError

        osc_freq = np.asarray(osc_freq)

        foo = np.power(self.duration * osc_freq, power)

        duration_osc = 1. / (2. * np.pi * damping * osc_freq)
        duration_rms = self.duration + duration_osc * (foo / (foo + bar))

        return duration_rms
