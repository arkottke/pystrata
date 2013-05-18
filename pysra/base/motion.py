#!/usr/bin/env python
# encoding: utf-8

import numpy as np

class Motion:
    def __init__(self):
        pass


class TimeSeriesMotion(Motion):
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
    def __init__(self):
        pass
