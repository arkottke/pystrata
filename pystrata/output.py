# -*- coding: utf-8 -*-
# The MIT License (MIT)
#
# Copyright (c) 2016-2021 Albert Kottke
#
# Permission is hereby granted, free of charge, to any person obtaining a copy
# of this software and associated documentation files (the "Software"), to deal
# in the Software without restriction, including without limitation the rights
# to use, copy, modify, merge, publish, distribute, sublicense, and/or sell
# copies of the Software, and to permit persons to whom the Software is
# furnished to do so, subject to the following conditions:
#
# The above copyright notice and this permission notice shall be included in
# all copies or substantial portions of the Software.
#
# THE SOFTWARE IS PROVIDED "AS IS", WITHOUT WARRANTY OF ANY KIND, EXPRESS OR
# IMPLIED, INCLUDING BUT NOT LIMITED TO THE WARRANTIES OF MERCHANTABILITY,
# FITNESS FOR A PARTICULAR PURPOSE AND NONINFRINGEMENT. IN NO EVENT SHALL THE
# AUTHORS OR COPYRIGHT HOLDERS BE LIABLE FOR ANY CLAIM, DAMAGES OR OTHER
# LIABILITY, WHETHER IN AN ACTION OF CONTRACT, TORT OR OTHERWISE, ARISING FROM,
# OUT OF OR IN CONNECTION WITH THE SOFTWARE OR THE USE OR OTHER DEALINGS IN THE
# SOFTWARE.
import collections

import matplotlib.pyplot as plt
import numpy as np
import scipy.integrate
from matplotlib.colors import LogNorm
from matplotlib.colors import TwoSlopeNorm
from scipy.interpolate import interp1d

try:
    import pandas as pd
except ImportError:
    pd = None

import pykooh

from .motion import TimeSeriesMotion, WaveField, GRAVITY


def plot_amplification_evolv(
    calc,
    metric="accel_tf",
    depths=None,
    freqs=None,
    normalized=False,
    wave_field_out="within",
    diverging_cmap=True,
    include_vs_profile=False,
    ax=None,
    **kwds,
):
    # Default plotting kwds. Combine both set of plot keywords and prefer the provided
    kwds = {
        "cmap": "RdBu" if diverging_cmap else "magma_r",
        "shading": "gouraud",
        "norm": TwoSlopeNorm(vmin=0, vcenter=1) if diverging_cmap else LogNorm(),
    } | kwds

    if freqs is None:
        freqs = np.logspace(-1, 2, num=301)

    osc_damping = 0.05 if "osc_damping" not in kwds else kwds["osc_damping"]

    ln_freqs = np.log(freqs)
    ln_freqs_mot = np.log(calc.motion.freqs)

    def get_amp(metric, depth):
        loc_output = calc.profile.location(wave_field_out, depth=depth)
        if metric == "accel_tf":
            y = np.abs(calc.calc_accel_tf(calc.loc_input, loc_output))
            # Interpolate the specific frequencies
            y = np.interp(ln_freqs, ln_freqs_mot, y)
        elif metric == "site_amp":
            if get_amp.in_ars is None:
                get_amp.in_ars = calc.motion.calc_osc_accels(freqs, osc_damping)

            out_ars = calc.motion.calc_osc_accels(
                freqs, osc_damping, calc.calc_accel_tf(calc.loc_input, loc_output)
            )
            y = out_ars / get_amp.in_ars
        else:
            raise NotImplementedError

        return y

    # Initialize static variable
    get_amp.in_ars = None

    if depths is None:
        depths = np.linspace(0, calc.profile[-1].depth)

    if ax is None:
        fig, ax = plt.subplots()

    amps = np.array([get_amp(metric, d) for d in depths])
    if normalized:
        amps /= amps[-1, :]

    cf = ax.pcolormesh(freqs, depths, amps, **kwds)

    cb = plt.colorbar(cf, ax=ax)
    cb.set_label("|TF|" if metric == "accel_tf" else "Site Ampl.")

    ax.set(
        xlabel="Frequency (Hz)",
        xscale="log",
        ylabel="Depth (m)",
        yscale="linear",
        ylim=(depths[0], 0),
    )

    return ax


class OutputCollection(collections.abc.Collection):
    def __init__(self, outputs):
        super().__init__()
        self.outputs = outputs

    def __iter__(self):
        return iter(self.outputs)

    def __contains__(self, value):
        return value in self.outputs

    def __len__(self):
        return len(self.outputs)

    def __getitem__(self, key):
        return self.outputs[key]

    def __call__(self, calc, name=None):
        # Save results
        for o in self:
            o(calc, name=name)

    def reset(self):
        for o in self:
            o.reset()


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
        single = np.pad(single, (0, -diff), "constant", constant_values=np.nan)
    elif diff > 0:
        # Need different padding based on if many is 1d or 2d.
        padding = ((0, diff), (0, 0)) if len(many.shape) > 1 else (0, diff)
        many = np.pad(many, padding, "constant", constant_values=np.nan)
    else:
        # No padding needed
        pass
    return np.c_[many, single]


class Output(object):
    _const_ref = False

    xscale = "log"
    yscale = "log"
    drawstyle = "default"

    def __init__(self, refs=None):
        self._refs = np.asarray([] if refs is None else refs)
        self._values = None
        self._names = []

    def __call__(self, calc, name=None):
        if name is None:
            if self.values is None:
                i = 1
            elif len(self.values.shape) == 1:
                i = 2
            else:
                i = self.values.shape[1] + 1
            name = "r%d" % i
        self._names.append(name)

    @property
    def refs(self):
        return self._refs

    @property
    def values(self):
        return self._values

    @property
    def names(self):
        return self._names

    def reset(self):
        self._values = None
        self._names = []
        if not self._const_ref:
            self._refs = np.array([])

    def iter_results(self):
        shared_ref = len(self.refs.shape) == 1
        for i, name in enumerate(self.names):
            refs = self.refs if shared_ref else self.refs[:, i]
            values = self.values if len(self.values.shape) == 1 else self.values[:, i]
            yield name, refs, values

    def _add_refs(self, refs):
        refs = np.asarray(refs)
        if len(self._refs) == 0:
            self._refs = np.array(refs)
        else:
            self._refs = append_arrays(self._refs, refs)

    def _add_values(self, values):
        values = np.asarray(values)
        if self._values is None:
            self._values = values
        else:
            self._values = append_arrays(self._values, values)

    def calc_stats(self, as_dataframe=False):
        ln_values = np.log(self.values)
        median = np.exp(np.nanmean(ln_values, axis=1))
        ln_std = np.nanstd(ln_values, axis=1)

        stats = {"ref": self.refs, "median": median, "ln_std": ln_std}
        if as_dataframe and pd:
            stats = pd.DataFrame(stats).set_index("ref")
            stats.index.name = self.ref_name

        return stats

    def to_dataframe(self):
        if not pd:
            raise RuntimeError("Install `pandas` library.")

        if isinstance(self.names[0], tuple):
            columns = pd.MultiIndex.from_tuples(self.names)
        else:
            columns = self.names

        df = pd.DataFrame(self.values, index=self.refs, columns=columns)

        return df

    @staticmethod
    def _get_xy(refs, values):
        return refs, values

    def plot(self, ax=None, style="indiv"):
        assert style in ["stats", "indiv"]

        if ax is None:
            fig, ax = plt.subplots()

        if style == "stats" and len(self.values.shape) > 1 and self.values.shape[1] < 3:
            raise RuntimeError("Unable to plot stats for less than 3 values.")

        if style == "stats":
            kwds = {"color": "C0", "alpha": 0.6, "lw": 0.8, "drawstyle": self.drawstyle}
        elif style == "indiv":
            kwds = {"lw": 1.0, "drawstyle": self.drawstyle}
        else:
            raise NotImplementedError("Valid options are: stats, indiv.")

        # Add the data
        x, y = self._get_xy(self.refs, self.values)
        lines = ax.plot(x, y, **kwds)

        if style == "stats":
            lines[0].set_label("Realization")
        else:
            for layer, name in zip(lines, self.names):
                layer.set_label(name)

        if style == "stats":
            stats = self.calc_stats()

            ax.plot(
                *self._get_xy(stats["ref"], stats["median"]),
                color="C1",
                lw=2,
                label="Median",
            )

        ax.set(
            xlabel=self.xlabel,
            xscale=self.xscale,
            ylabel=self.ylabel,
            yscale=self.yscale,
        )

        if len(lines) > 1:
            ax.legend()

        return ax


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
        return profile.location(self.wave_field, depth=self.depth, index=self.index)


class LocationBasedOutput(Output):
    def __init__(self, ref, location):
        super().__init__(ref)
        self._location = location

    @property
    def location(self):
        return self._location

    def __call__(self, calc, name=None):
        raise NotImplementedError

    def _get_location(self, calc):
        """Locate location within the profile."""
        return self._location(calc.profile)


class TimeSeriesOutput(LocationBasedOutput):
    xlabel = "Time (sec)"
    xscale = "linear"
    ylabel = NotImplemented
    yscale = "linear"

    ref_name = "time"

    def __init__(self, location):
        super().__init__(None, location)

    @property
    def times(self):
        return self.refs

    def __call__(self, calc, name=None):
        if not isinstance(calc.motion, TimeSeriesMotion):
            raise NotImplementedError
        Output.__call__(self, calc, name)
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

    def to_dataframe(self):
        raise NotImplementedError


class AccelerationTSOutput(TimeSeriesOutput):
    ylabel = "Acceleration (g)"

    def _get_trans_func(self, calc, location):
        return calc.calc_accel_tf(calc.loc_input, location)


class AriasIntensityTSOutput(AccelerationTSOutput):
    ylabel = "Arias Intensity (m/s)"

    def _modify_values(self, calc, location, values):
        time_step = calc.motion.time_step
        values = scipy.integrate.cumtrapz(values**2, dx=time_step)
        values *= GRAVITY * np.pi / 2
        return values


class StrainTSOutput(TimeSeriesOutput):
    def __init__(self, location, in_percent=False):
        super().__init__(location)
        self._in_percent = in_percent
        assert self.location.wave_field == WaveField.within

    def _get_trans_func(self, calc, location):
        return calc.calc_strain_tf(calc.loc_input, location)

    def _modify_values(self, calc, location, values):
        if self._in_percent:
            # Convert to percent
            values *= 100.0
        return values

    @property
    def ylabel(self):
        suffix = "(%)" if self._in_percent else "(dec)"
        return "Shear Strain " + suffix


class StressTSOutput(TimeSeriesOutput):
    def __init__(self, location, damped=False, normalized=False):
        super().__init__(location)
        self._damped = damped
        self._normalized = normalized
        assert self.location.wave_field == WaveField.within

    @property
    def damped(self):
        return self._damped

    @property
    def ylabel(self):
        if self._normalized:
            ylabel = "Stress Ratio (τ/σ`ᵥ)"
        else:
            ylabel = "Stress (τ)"

        return ylabel

    def _get_trans_func(self, calc, location):
        tf = calc.calc_stress_tf(calc.loc_input, location, self.damped)

        if self._normalized:
            # Correct by effective stress at depth
            tf /= location.stress_vert(effective=True)

        return tf


class FourierAmplitudeSpectrumOutput(LocationBasedOutput):
    _const_ref = True
    xlabel = "Frequency (Hz)"
    ylabel = "Fourier Ampl. (cm/s)"

    ref_name = "freq"

    def __init__(self, freqs, location, ko_bandwidth=30):
        super().__init__(freqs, location)
        self._ko_bandwidth = ko_bandwidth

    @property
    def freqs(self):
        return self._refs

    @property
    def ko_bandwidth(self):
        return self._ko_bandwidth

    def __call__(self, calc, name=None):
        Output.__call__(self, calc, name)
        loc = self._get_location(calc)
        tf = calc.calc_accel_tf(calc.loc_input, loc)

        if self._ko_bandwidth is None:
            tf = np.interp(self.freqs, calc.motion.freqs, tf)
        else:
            tf = pykooh.smooth(
                self.freqs,
                calc.motion.freqs,
                np.abs(tf * calc.motion.fourier_amps),
                self.ko_bandwidth,
            )

        self._add_values(tf)


class ResponseSpectrumOutput(LocationBasedOutput):
    _const_ref = True
    xlabel = "Frequency (Hz)"

    ref_name = "freq"

    def __init__(self, freqs, location, osc_damping):
        super().__init__(freqs, location)
        self._osc_damping = osc_damping

    @property
    def freqs(self):
        return self._refs

    @property
    def periods(self):
        return 1.0 / np.asarray(self._refs)

    @property
    def osc_damping(self):
        return self._osc_damping

    @property
    def ylabel(self):
        return f"{100 * self.osc_damping:g}%-Damped, Spec. Accel. (g)"

    def __call__(self, calc, name=None):
        Output.__call__(self, calc, name)
        loc = self._get_location(calc)
        tf = calc.calc_accel_tf(calc.loc_input, loc)
        ars = calc.motion.calc_osc_accels(self.freqs, self.osc_damping, tf)
        self._add_values(ars)


class RatioBasedOutput(Output):
    _const_ref = True

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

    def __call__(self, calc, name=None):
        raise NotImplementedError

    def _get_locations(self, calc):
        """Locate locations within the profile."""
        return (self._location_in(calc.profile), self._location_out(calc.profile))


class AccelTransferFunctionOutput(RatioBasedOutput):
    xlabel = "Frequency (Hz)"
    ylabel = "Accel. Transfer Func."

    ref_name = "freq"

    def __init__(
        self, refs, location_in, location_out, ko_bandwidth=None, absolute=True
    ):
        super().__init__(refs, location_in, location_out)
        self._ko_bandwidth = ko_bandwidth
        self._absolute = absolute

    def __call__(self, calc, name=None):
        Output.__call__(self, calc, name)
        # Locate position within the profile
        loc_in, loc_out = self._get_locations(calc)
        # Compute the response
        if self._absolute:
            tf = np.abs(calc.calc_accel_tf(loc_in, loc_out))
        else:
            tf = calc.calc_accel_tf(loc_in, loc_out)

        if self._ko_bandwidth is None:
            tf = np.interp(self.freqs, calc.motion.freqs, tf)
        else:
            tf = pykooh.smooth(self.freqs, calc.motion.freqs, tf, self._ko_bandwidth)

        self._add_values(tf)

    @property
    def freqs(self):
        return self._refs


class ResponseSpectrumRatioOutput(RatioBasedOutput):
    xlabel = "Frequency (Hz)"

    ref_name = "freq"

    def __init__(self, freqs, location_in, location_out, osc_damping):
        super().__init__(freqs, location_in, location_out)
        self._osc_damping = osc_damping

    @property
    def freqs(self):
        return self._refs

    @property
    def periods(self):
        return 1.0 / np.asarray(self._refs)

    @property
    def osc_damping(self):
        return self._osc_damping

    @property
    def ylabel(self):
        return f"{100 * self.osc_damping:g}%-Damped, Resp. Spectral Ratio"

    def __call__(self, calc, name=None):
        Output.__call__(self, calc, name)
        loc_in, loc_out = self._get_locations(calc)
        in_ars = calc.motion.calc_osc_accels(
            self.freqs, self.osc_damping, calc.calc_accel_tf(calc.loc_input, loc_in)
        )
        out_ars = calc.motion.calc_osc_accels(
            self.freqs, self.osc_damping, calc.calc_accel_tf(calc.loc_input, loc_out)
        )
        ratio = out_ars / in_ars
        self._add_values(ratio)


class ProfileBasedOutput(Output):
    ylabel = "Depth (m)"
    yscale = "linear"
    drawstyle = "steps-post"

    ref_name = "depth"

    def __init__(self):
        super().__init__()

    def __call__(self, calc, name=None):
        Output.__call__(self, calc, name)
        depths = np.r_[0, calc.profile.depth_mid[:-1]]
        self._add_refs(depths)

    def calc_stats(self, as_dataframe=False, ref=None):
        if ref is None:
            ref = np.linspace(0, np.nanmax(self.refs) * 1.05, num=512)

        n = self.values.shape[1]
        with np.errstate(divide="ignore", invalid="ignore"):
            # Ignore zeros in the data
            ln_values = np.array([self._ln_interp(i, ref) for i in range(n)]).T
            median = np.exp(np.nanmean(ln_values, axis=1))
            ln_std = np.nanstd(ln_values, axis=1)

        stats = {"ref": ref, "median": median, "ln_std": ln_std}
        if as_dataframe and pd:
            stats = pd.DataFrame(stats).set_index("ref")
            stats.index.name = self.ref_name

        return stats

    @staticmethod
    def _get_xy(refs, values):
        return values, refs

    def plot(self, ax=None, style="stats"):
        ax = Output.plot(self, ax, style)
        ax.invert_yaxis()
        return ax

    def _ln_interp(self, i, ref):
        """Interpolate the values in log-y space."""

        _ref = self.refs[:, i]
        # Only select points with valid entries
        mask = np.isfinite(_ref)
        _ref = _ref[mask]
        _ln_values = np.log(self.values[mask, i])

        if np.any(mask):
            f = interp1d(
                _ref,
                _ln_values,
                kind="next",
                fill_value=(_ln_values[0], _ln_values[-1]),
                bounds_error=False,
            )
            _ln_interped = f(ref)
        else:
            nans = np.empty_like(ref)
            nans[:] = np.nan
            _ln_interped = np.array(nans)

        return _ln_interped

    def to_dataframe(self, ref=None):
        if not pd:
            raise RuntimeError("Install `pandas` library.")

        if ref is None:
            ref = np.linspace(0, np.nanmax(self.refs))

        if isinstance(self.names[0], tuple):
            columns = pd.MultiIndex.from_tuples(self.names)
        else:
            columns = self.names

        # Ignore zeros in the data
        n = self.values.shape[1]
        values = np.exp(np.array([self._ln_interp(i, ref) for i in range(n)])).T

        df = pd.DataFrame(values, index=ref, columns=columns)

        return df


class MaxStrainProfile(ProfileBasedOutput):
    xlabel = "Max. Strain (dec)"

    def __init__(self):
        super().__init__()

    def __call__(self, calc, name=None):
        ProfileBasedOutput.__call__(self, calc, name)
        values = [0] + [layer.strain_max for layer in calc.profile[:-1]]
        self._add_values(values)


class DampingProfile(ProfileBasedOutput):
    xlabel = "Damping (dec)"

    def __call__(self, calc, name=None):
        Output.__call__(self, calc, name)
        # Add depth at top of layer
        self._add_refs(calc.profile.depth)

        values = [layer.damping for layer in calc.profile[:-1]]
        # Bring the first mid-layer value to the surface
        values.insert(0, values[0])
        self._add_values(values)


class ShearModReducProfile(ProfileBasedOutput):
    xlabel = "G/Gmax"

    def __call__(self, calc, name=None):
        Output.__call__(self, calc, name)
        # Add depth at top of layer
        self._add_refs(calc.profile.depth)

        values = [layer.shear_mod_reduc for layer in calc.profile[:-1]]
        # Bring the first mid-layer value to the surface
        values.insert(0, values[0])
        self._add_values(values)


class InitialVelProfile(ProfileBasedOutput):
    xlabel = "Initial Velocity (m/s)"

    def __init__(self):
        super().__init__()

    def __call__(self, calc, name=None):
        Output.__call__(self, calc, name)
        # Add depth at top of layer
        self._add_refs(calc.profile.depth)

        values = [layer.initial_shear_vel for layer in calc.profile[:-1]]
        values.insert(0, values[0])
        self._add_values(values)


class CompatVelProfile(ProfileBasedOutput):
    xlabel = "Strain-Compatible Velocity (m/s)"

    def __init__(self):
        super().__init__()

    def __call__(self, calc, name=None):
        Output.__call__(self, calc, name)
        # Add depth at top of layer
        self._add_refs(calc.profile.depth)

        values = [np.min(layer.shear_vel) for layer in calc.profile[:-1]]
        values.insert(0, values[0])
        self._add_values(values)


class CyclicStressRatioProfile(ProfileBasedOutput):
    # From Idriss and Boulanger (2008, pg. 70):
    # The 0.65 is a constant used to represent the reference stress
    # level. While being somewhat arbitrary it was selected in the
    # beginning of the development of liquefaction procedures in 1966
    # and has been in use ever since.
    _stress_level = 0.65

    def __init__(self):
        super().__init__()

    def __call__(self, calc, name=None):
        ProfileBasedOutput.__call__(self, calc, name)
        values = [
            layer.stress_shear_max / layer.stress_vert(layer.thickness / 2, True)
            for layer in calc.profile[:-1]
        ]
        # Repeat the first value for the surface
        values = self._stress_level * np.array([values[0]] + values)
        self._add_values(values)


class MaxAccelProfile(ProfileBasedOutput):
    xlabel = "Max. Accel. (g)"

    def __call__(self, calc, name=None):
        Output.__call__(self, calc, name)
        # Add depth at top of layer
        depths = calc.profile.depth
        values = [self._calc_accel(calc, depth) for depth in depths]
        self._add_refs(depths)
        self._add_values(values)

    def _calc_accel(self, calc, depth):
        return calc.motion.calc_peak(
            calc.calc_accel_tf(
                calc.loc_input, calc.profile.location("within", depth=depth)
            )
        )
