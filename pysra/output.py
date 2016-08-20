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

import collections

import numpy as np
import scipy.integrate

from .motion import TimeSeriesMotion, WaveField, GRAVITY


class OutputCollection(collections.UserList):
    def __init__(self, *outputs):
        super().__init__(outputs)

    def __call__(self, calc):
        # Save results
        for o in self.data:
            o(calc)


def append_arrays(many, single):
    """Append an array to another padding with NaNs for constant length.

    Parameters
    ----------
    many : array_like of rank (j, k)
        values appended to a copy of this array. This may be a 1-D or 2-D
        array.
    single : array_like of rank l
        values to append. This should be a 1-D array.

    Returns
    -------
    append : :class:`numpy.ndarray`
        2-D array with rank (j + 1, max(k, l)) with missing values padded
        with :class:`numpy.nan`
    """
    assert np.ndim(single) == 1

    # Check if the values need to be padded to for equal length
    diff = single.shape[0] - many.shape[0]
    if diff < 0:
        single = np.pad(single, (0, -diff), 'constant', constant_values=np.nan)
    elif diff > 0:
        many = np.pad(many, ((0, diff),), 'constant', constant_values=np.nan)
    else:
        # No padding needed
        pass
    return np.c_[many, single]


class Output(object):
    def __init__(self, refs=None):
        self._refs = refs if refs is None else np.asarray(refs)
        self._values = None

    def __call__(self, calc):
        raise NotImplementedError

    @property
    def refs(self):
        return self._refs

    @property
    def values(self):
        return self._values

    def _add_refs(self, refs):
        refs = np.asarray(refs)
        if self._refs is None:
            self._refs = refs
        elif len(refs) == len(self._refs) and np.allclose(refs, self._refs):
            # Same values
            pass
        else:
            # Different values
            self._refs = append_arrays(self._refs, refs)

    def _add_values(self, values):
        values = np.asarray(values)
        if self._values is None:
            self._values = values
        else:
            self._values = append_arrays(self._values, values)


class OutputLocation(object):
    def __init__(self, wave_field, depth=None, index=None):
        self._depth = depth
        self._index = index
        if not isinstance(wave_field, WaveField):
            wave_field = WaveField[wave_field]
        self._wave_field = wave_field

    @property
    def depth(self):
        return self._depth

    @property
    def index(self):
        return self._index

    @property
    def wave_field(self):
        return self._wave_field

    def __call__(self, profile):
        """Lookup the location with the profile."""
        return profile.location(
            self.wave_field, depth=self.depth, index=self.index)


class LocationBasedOutput(Output):
    def __init__(self, ref, location):
        super().__init__(ref)
        self._location = location

    @property
    def location(self):
        return self._location

    def __call__(self, calc):
        raise NotImplementedError

    def _get_location(self, calc):
        """Locate location within the profile."""
        return self._location(calc.profile)


class TimeSeriesOutput(LocationBasedOutput):
    xlabel = 'Time (sec)'
    ylabel = NotImplemented

    def __init__(self, location):
        super().__init__(None, location)

    @property
    def times(self):
        return self.refs

    def __call__(self, calc):
        if not isinstance(calc.motion, TimeSeriesMotion):
            raise NotImplementedError
        # Compute the response
        loc = self._get_location(calc)
        tf = self._get_trans_func(calc, loc)
        values = calc.motion.calc_time_series(tf)
        values = self._modify_values(calc, loc, values)
        self._add_values(values)
        # Add the reference
        refs = calc.motion.time_step * np.arange(len(values))
        self._add_refs(refs)

    def _get_trans_func(self, calc, location):
        raise NotImplementedError

    def _modify_values(self, calc, location, values):
        return values


class AccelerationTSOutput(TimeSeriesOutput):
    ylabel = 'Acceleration (g)'

    def _get_trans_func(self, calc, location):
        return calc.calc_accel_tf(calc.loc_input, location)


class AriasIntensityTSOutput(AccelerationTSOutput):
    ylabel = 'Arias Intensity (m/s)'

    def _modify_values(self, calc, location, values):
        time_step = calc.motion.time_step
        values = scipy.integrate.cumtrapz(values ** 2, dx=time_step)
        values *= GRAVITY * np.pi / 2
        return values


class StrainTSOutput(TimeSeriesOutput):
    ylabel = 'Shear Strain (%)'

    def __init__(self, location):
        super().__init__(location)
        assert self.location.wave_field == WaveField.within

    def _get_trans_func(self, calc, location):
        return calc.calc_strain_tf(calc.loc_input, location)

    def _modify_values(self, calc, location, values):
        # Convert to percent
        return 100. * values


class StressTSOutput(TimeSeriesOutput):
    ylabel = 'Stress Ratio (τ/σ`ᵥ)'

    def __init__(self, location, damped=False):
        super().__init__(location)
        self._damped = damped
        assert self.location.wave_field == WaveField.within

    @property
    def damped(self):
        return self._damped

    def _get_trans_func(self, calc, location):
        tf = calc.calc_stress_tf(calc.loc_input, location, self.damped)
        # Correct by effective stress at depth
        tf /= location.vert_stress(effective=True)
        return tf


class ResponseSpectrumOutput(LocationBasedOutput):
    xlabel = 'Frequency (Hz)'
    # fixme: Include damping?
    ylabel = 'Spectral Accel. (g)'

    def __init__(self, freqs, location, osc_damping):
        super().__init__(freqs, location)
        self._osc_damping = osc_damping

    @property
    def freqs(self):
        return self._refs

    @property
    def periods(self):
        return 1. / np.asarray(self._refs)

    @property
    def osc_damping(self):
        return self._osc_damping

    def __call__(self, calc):
        loc = self._get_location(calc)
        tf = calc.calc_accel_tf(calc.loc_input, loc)
        ars = calc.motion.calc_osc_accels(self.freqs, self.osc_damping, tf)
        self._add_values(ars)


class RatioBasedOutput(Output):
    def __init__(self, refs, location_in, location_out):
        super().__init__(refs)
        self._location_in = location_in
        self._location_out = location_out

    @property
    def location_in(self):
        return self._location_in

    @property
    def location_out(self):
        return self._location_out

    def __call__(self, calc):
        raise NotImplementedError

    def _get_locations(self, calc):
        """Locate locations within the profile."""
        return (self._location_in(calc.profile),
                self._location_out(calc.profile))


class AccelTransferFunctionOutput(RatioBasedOutput):
    def __call__(self, calc):
        # Locate position within the profile
        loc_in, loc_out = self._get_locations(calc)
        # Compute the response
        tf = calc.calc_accel_tf(loc_in, loc_out)
        self._add_values(tf)
        self._add_refs(calc.motion.freqs)

    @property
    def freqs(self):
        return self._refs


class ResponseSpectrumRatioOutput(RatioBasedOutput):
    def __init__(self, freqs, location_in, location_out, osc_damping):
        super().__init__(freqs, location_in, location_out)
        self._osc_damping = osc_damping

    @property
    def freqs(self):
        return self._refs

    @property
    def periods(self):
        return 1. / np.asarray(self._refs)

    @property
    def osc_damping(self):
        return self._osc_damping

    def __call__(self, calc):
        loc_in, loc_out = self._get_locations(calc)
        in_ars = calc.motion.calc_osc_accels(
            self.freqs, self.osc_damping,
            calc.calc_accel_tf(calc.loc_input, loc_in)
        )
        out_ars = calc.motion.calc_osc_accels(
            self.freqs, self.osc_damping,
            calc.calc_accel_tf(calc.loc_input, loc_out)
        )
        ratio = out_ars / in_ars
        self._add_values(ratio)
