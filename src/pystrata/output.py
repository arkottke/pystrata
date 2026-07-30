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

from __future__ import annotations

import collections
import warnings

import matplotlib.pyplot as plt
import numpy as np
import numpy.typing as npt
import scipy.integrate
import xarray as xr
from matplotlib.colors import LogNorm, TwoSlopeNorm
from scipy.interpolate import interp1d

try:
    import pandas as pd
except ImportError:
    pd = None

import pykooh

from .motion import TimeSeriesMotion, WaveField
from .units import GRAVITY, convert_units


def plot_amplification_evolv(
    calc,
    metric: str = "accel_tf",
    depths: npt.ArrayLike | None = None,
    freqs: npt.ArrayLike | None = None,
    normalized: bool = False,
    wave_field_out: str = "within",
    diverging_cmap: bool = True,
    include_vs_profile: bool = False,
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
    def __init__(self, outputs: list[Output]) -> None:
        super().__init__()
        self.outputs = outputs

    def __iter__(self):
        return iter(self.outputs)

    def __contains__(self, value) -> bool:
        return value in self.outputs

    def __len__(self) -> int:
        return len(self.outputs)

    def __getitem__(self, key) -> Output:
        return self.outputs[key]

    def __call__(self, calc, name=None, index: int | None = None) -> None:
        # Save results
        for o in self:
            o(calc, name=name, index=index)

    def reset(self) -> None:
        for o in self:
            o.reset()

    def reserve(self, count: int) -> None:
        """Pre-allocate storage for *count* realizations in each output."""
        for o in self:
            o.reserve(count)

    def extend(self, other: OutputCollection) -> None:
        """Append the results of *other*, output by output."""
        if len(other) != len(self):
            raise ValueError(
                f"Cannot extend {len(self)} outputs with {len(other)} outputs."
            )
        for mine, theirs in zip(self, other):
            mine.extend(theirs)


def append_arrays(many: np.ndarray, single: npt.ArrayLike) -> np.ndarray:
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


def stack_columns(columns: list[np.ndarray]) -> np.ndarray:
    """Stack 1-D arrays as columns, padding short ones with NaN.

    Parameters
    ----------
    columns : list of :class:`numpy.ndarray`
        1-D arrays, which need not share a length.

    Returns
    -------
    :class:`numpy.ndarray`
        2-D array of shape ``(max length, len(columns))``.
    """
    length = max(len(c) for c in columns)
    dtype = np.result_type(*[c.dtype for c in columns], np.float64)

    if all(len(c) == length for c in columns):
        return np.column_stack(columns).astype(dtype, copy=False)

    stacked = np.full((length, len(columns)), np.nan, dtype=dtype)
    for i, column in enumerate(columns):
        stacked[: len(column), i] = column

    return stacked


class Output:
    _const_ref = False

    xscale = "log"
    yscale = "log"
    drawstyle = "default"

    #: dtype of the stored values. Overridden per instance by outputs that
    #: store complex values.
    _dtype = float

    def __init__(self, refs: npt.ArrayLike | None = None) -> None:
        self._refs = np.asarray([] if refs is None else refs)
        self._names: list = []

        # Results accumulate as a list of columns and are stacked on demand.
        # Stacking on every call is quadratic in the number of realizations.
        self._ref_cols: list[np.ndarray] = []
        self._value_cols: list[np.ndarray] = []
        self._refs_cache: np.ndarray | None = None
        self._values_cache: np.ndarray | None = None

        # Populated by reserve(), which switches to assignment by index
        self._buffer: np.ndarray | None = None
        self._index: int | None = None

    def reserve(self, count: int) -> None:
        """Pre-allocate storage for *count* realizations.

        Results are then written by index rather than appended, so they may be
        collected out of order -- realization *i* always lands in column *i*.
        Columns that are never written remain NaN.

        This requires constant references, since the shape must be known up
        front. Calling it is optional; without it results are appended.

        Parameters
        ----------
        count : int
            Number of realizations to allocate.
        """
        if not self._const_ref:
            raise RuntimeError(
                f"{type(self).__name__} does not have constant references, so "
                "the result shape is not known in advance and cannot be "
                "pre-allocated."
            )
        if self._value_cols or self._buffer is not None:
            raise RuntimeError("reserve() must be called before collecting results.")

        self._buffer = np.full((len(self._refs), count), np.nan, dtype=self._dtype)
        self._names = [None] * count

    def __call__(self, calc, name=None, index: int | None = None) -> None:
        if index is None:
            index = (
                len(self._value_cols) if self._buffer is None else self._next_index()
            )
        self._index = index

        if name is None:
            name = "r%d" % (index + 1)

        if self._buffer is None:
            self._names.append(name)
        else:
            self._names[index] = name

    def _next_index(self) -> int:
        """Index of the first unwritten column of a pre-allocated buffer."""
        written = [i for i, n in enumerate(self._names) if n is not None]
        return (max(written) + 1) if written else 0

    @property
    def refs(self) -> np.ndarray:
        if self._ref_cols:
            if self._refs_cache is None:
                self._refs_cache = (
                    self._ref_cols[0]
                    if len(self._ref_cols) == 1
                    else stack_columns(self._ref_cols)
                )
            return self._refs_cache
        return self._refs

    @property
    def values(self) -> np.ndarray | None:
        if self._buffer is not None:
            return self._buffer
        if not self._value_cols:
            return None
        if self._values_cache is None:
            self._values_cache = (
                self._value_cols[0]
                if len(self._value_cols) == 1
                else stack_columns(self._value_cols)
            )
        return self._values_cache

    @property
    def names(self) -> list:
        return self._names

    def reset(self) -> None:
        self._names = []
        self._value_cols = []
        self._values_cache = None
        self._index = None
        if self._buffer is not None:
            self._buffer[:] = np.nan
            self._names = [None] * self._buffer.shape[1]
        if not self._const_ref:
            self._refs = np.array([])
            self._ref_cols = []
            self._refs_cache = None

    def extend(self, other: Output) -> None:
        """Append the results of *other* to this output.

        Used to combine results computed separately, such as chunks of an
        ensemble evaluated in different processes.

        Parameters
        ----------
        other : Output
            Output of the same type holding additional realizations.
        """
        if type(other) is not type(self):
            raise TypeError(
                f"Cannot extend {type(self).__name__} with {type(other).__name__}."
            )
        if self._buffer is not None or other._buffer is not None:
            raise RuntimeError(
                "extend() is not supported for pre-allocated outputs; write "
                "into the reserved columns by index instead."
            )

        self._value_cols.extend(other._value_cols)
        self._ref_cols.extend(other._ref_cols)
        self._names.extend(other._names)
        self._values_cache = None
        self._refs_cache = None

    def iter_results(self):
        shared_ref = len(self.refs.shape) == 1
        for i, name in enumerate(self.names):
            refs = self.refs if shared_ref else self.refs[:, i]
            values = self.values if len(self.values.shape) == 1 else self.values[:, i]
            yield name, refs, values

    def _add_refs(self, refs: npt.ArrayLike) -> None:
        self._ref_cols.append(np.asarray(refs))
        self._refs_cache = None

    def _add_values(self, values: npt.ArrayLike) -> None:
        values = np.asarray(values)
        if self._buffer is None:
            self._value_cols.append(values)
            self._values_cache = None
        else:
            self._buffer[:, self._index] = values

    def calc_stats(self, as_dataframe: bool = False):
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

    def _to_xarray_flat(self) -> xr.DataArray | xr.Dataset:
        """Results keyed by realization, without a logic tree.

        Constant references become a coordinate. Varying references are stored
        as a 2-D data variable instead: aligning on them would outer-join every
        distinct value, producing an array that is almost entirely empty.
        """
        ref_name = getattr(self, "ref_name", "ref")
        values = self.values
        if values is None:
            raise ValueError("No results have been collected.")
        if values.ndim == 1:
            values = values[:, None]

        realizations = [
            name if isinstance(name, str) else str(name) for name in self.names
        ]

        if self.refs.ndim == 1:
            return xr.DataArray(
                values,
                dims=(ref_name, "realization"),
                coords={ref_name: self.refs, "realization": realizations},
                name=getattr(self, "ylabel", None),
            )

        # A positional dimension, so that the varying references stay a data
        # variable. Naming the dimension after them would make xarray promote
        # them to a coordinate and align on their values.
        return xr.Dataset(
            {
                "value": (("index", "realization"), values),
                ref_name: (("index", "realization"), self.refs),
            },
            coords={"realization": realizations},
        )

    def to_xarray(self, tree=None) -> xr.DataArray | xr.Dataset:
        """Convert output results into a labeled array.

        With no *tree*, results are keyed by realization. Given a logic tree,
        results are reshaped into one dimension per node; the stored ``names``
        must then be :class:`~pystrata.logic_tree.Branch` objects (i.e. the
        output was called with ``output(calc, name=branch)``).

        Parameters
        ----------
        tree : LogicTree, optional
            A rectangular logic tree (no ``requires``/``excludes`` conditions).

        Returns
        -------
        xr.DataArray or xr.Dataset
            With *tree*, a DataArray with dimensions
            ``(ref_name, node1, node2, ...)``. Without one, a DataArray with
            dimensions ``(ref_name, realization)``, or a Dataset when the
            references vary between realizations.

        Raises
        ------
        ValueError
            If the tree is not rectangular or names are not Branch objects.
        """
        from .logic_tree import Branch

        if tree is None:
            return self._to_xarray_flat()

        if not tree.is_rectangular:
            raise ValueError(
                "to_xarray only supports fully-crossed (rectangular) logic trees. "
                "The provided tree has conditional requires/excludes."
            )

        if not self.names or not isinstance(self.names[0], Branch):
            raise ValueError(
                "Output.names must contain Branch objects. "
                "Call the output with name=branch."
            )

        ref_name = getattr(self, "ref_name", "ref")
        node_names = [n.name for n in tree.nodes]
        shape = [len(self.refs)] + [len(n) for n in tree.nodes]

        data = np.full(shape, np.nan)

        for i, branch in enumerate(self.names):
            idx = []
            for node in tree.nodes:
                bval = branch.value(node.name)
                for j, opt in enumerate(node.options):
                    if (
                        isinstance(opt, float) and np.isclose(opt, bval)
                    ) or opt == bval:
                        idx.append(j)
                        break
            col = self.values[:, i] if self.values.ndim > 1 else self.values
            data[(slice(None), *idx)] = col

        coords = {ref_name: self.refs}
        for node in tree.nodes:
            coords[node.name] = list(node.options)

        dims = [ref_name] + node_names

        return xr.DataArray(data, dims=dims, coords=coords)

    @staticmethod
    def _get_xy(refs, values):
        return refs, values

    def plot(self, ax=None, style: str = "indiv"):
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


class OutputLocation:
    @convert_units(depth="meter")
    def __init__(
        self,
        wave_field: str | WaveField,
        depth: float | None = None,
        index: int | None = None,
    ) -> None:
        self._depth = depth
        self._index = index
        self._wave_field = WaveField(wave_field)

    @property
    def depth(self) -> float | None:
        return self._depth

    @property
    def index(self) -> int | None:
        return self._index

    @property
    def wave_field(self) -> WaveField:
        return self._wave_field

    def __call__(self, profile):
        """Lookup the location with the profile."""
        return profile.location(self.wave_field, depth=self.depth, index=self.index)


class LocationBasedOutput(Output):
    def __init__(self, ref: npt.ArrayLike | None, location: OutputLocation) -> None:
        super().__init__(ref)
        self._location = location

    @property
    def location(self) -> OutputLocation:
        return self._location

    def __call__(self, calc, name=None, index: int | None = None):
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

    def __init__(self, location: OutputLocation) -> None:
        super().__init__(None, location)

    @property
    def times(self) -> np.ndarray:
        return self.refs

    def __call__(self, calc, name=None, index: int | None = None) -> None:
        if not isinstance(calc.motion, TimeSeriesMotion):
            raise NotImplementedError
        Output.__call__(self, calc, name, index)
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

    def _modify_values(self, calc, location, values: np.ndarray) -> np.ndarray:
        return values

    def to_dataframe(self):
        raise NotImplementedError


class AccelerationTSOutput(TimeSeriesOutput):
    ylabel = "Acceleration (g)"

    def _get_trans_func(self, calc, location) -> np.ndarray:
        return calc.calc_accel_tf(calc.loc_input, location)


class AriasIntensityTSOutput(AccelerationTSOutput):
    ylabel = "Arias Intensity (m/s)"

    def _modify_values(self, calc, location, values: np.ndarray) -> np.ndarray:
        time_step = calc.motion.time_step
        values = scipy.integrate.cumulative_trapezoid(values**2, dx=time_step)
        values *= GRAVITY * np.pi / 2
        return values


class StrainTSOutput(TimeSeriesOutput):
    def __init__(self, location: OutputLocation, in_percent: bool = False) -> None:
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
    def __init__(
        self,
        location: OutputLocation,
        damped: bool = False,
        normalized: bool = False,
    ) -> None:
        super().__init__(location)
        self._damped = damped
        self._normalized = normalized
        assert self.location.wave_field == WaveField.within

    @property
    def damped(self) -> bool:
        return self._damped

    @property
    def ylabel(self) -> str:
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

    @convert_units(freqs="hertz")
    def __init__(
        self,
        freqs: npt.ArrayLike,
        location: OutputLocation,
        ko_bandwidth: float | None = None,
    ) -> None:
        super().__init__(freqs, location)
        self._ko_bandwidth = ko_bandwidth

    @property
    def freqs(self) -> np.ndarray:
        return self._refs

    @property
    def ko_bandwidth(self) -> float:
        return self._ko_bandwidth

    def __call__(self, calc, name=None, index: int | None = None) -> None:
        Output.__call__(self, calc, name, index)
        loc = self._get_location(calc)
        tf = calc.calc_accel_tf(calc.loc_input, loc)

        # Only return the absolute value
        fas = np.abs(tf * calc.motion.fourier_amps)

        # Interpolate to the specified frequencies
        if self._ko_bandwidth is None:
            fas = np.interp(self.freqs, calc.motion.freqs, fas)
        else:
            fas = pykooh.smooth(
                self.freqs,
                calc.motion.freqs,
                fas,
                self.ko_bandwidth,
            )

        self._add_values(fas)


class ResponseSpectrumOutput(LocationBasedOutput):
    _const_ref = True
    xlabel = "Frequency (Hz)"

    ref_name = "freq"

    @convert_units(freqs="hertz")
    def __init__(
        self,
        freqs: npt.ArrayLike,
        location: OutputLocation,
        osc_damping: float,
    ) -> None:
        super().__init__(freqs, location)
        self._osc_damping = osc_damping

    @property
    def freqs(self) -> np.ndarray:
        return self._refs

    @property
    def periods(self) -> np.ndarray:
        return 1.0 / np.asarray(self._refs)

    @property
    def osc_damping(self) -> float:
        return self._osc_damping

    @property
    def ylabel(self) -> str:
        return f"{100 * self.osc_damping:g}%-Damped, Spec. Accel. (g)"

    def __call__(self, calc, name=None, index: int | None = None) -> None:
        Output.__call__(self, calc, name, index)
        loc = self._get_location(calc)
        tf = calc.calc_accel_tf(calc.loc_input, loc)
        ars = calc.motion.calc_osc_accels(self.freqs, self.osc_damping, tf)
        self._add_values(ars)


class RatioBasedOutput(Output):
    _const_ref = True

    def __init__(
        self,
        refs: npt.ArrayLike,
        location_in: OutputLocation,
        location_out: OutputLocation,
    ) -> None:
        super().__init__(refs)
        self._location_in = location_in
        self._location_out = location_out

    @property
    def location_in(self) -> OutputLocation:
        return self._location_in

    @property
    def location_out(self) -> OutputLocation:
        return self._location_out

    def __call__(self, calc, name=None, index: int | None = None):
        raise NotImplementedError

    def _get_locations(self, calc):
        """Locate locations within the profile."""
        return (self._location_in(calc.profile), self._location_out(calc.profile))


class AccelTransferFunctionOutput(RatioBasedOutput):
    xlabel = "Frequency (Hz)"
    ylabel = "Accel. Transfer Func."

    ref_name = "freq"

    @convert_units(refs="hertz")
    def __init__(
        self,
        refs: npt.ArrayLike,
        location_in: OutputLocation,
        location_out: OutputLocation,
        ko_bandwidth: float | None = None,
        absolute: bool = True,
    ) -> None:
        super().__init__(refs, location_in, location_out)
        self._ko_bandwidth = ko_bandwidth
        self._absolute = absolute
        # The unmodified transfer function is complex. Assigning it into a
        # real-valued buffer would silently discard the imaginary part.
        self._dtype = float if absolute else complex

    def __call__(self, calc, name=None, index: int | None = None) -> None:
        Output.__call__(self, calc, name, index)
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
    def freqs(self) -> np.ndarray:
        return self._refs


class ResponseSpectrumRatioOutput(RatioBasedOutput):
    xlabel = "Frequency (Hz)"

    ref_name = "freq"

    @convert_units(freqs="hertz")
    def __init__(
        self,
        freqs: npt.ArrayLike,
        location_in: OutputLocation,
        location_out: OutputLocation,
        osc_damping: float,
    ) -> None:
        super().__init__(freqs, location_in, location_out)
        self._osc_damping = osc_damping

    @property
    def freqs(self) -> np.ndarray:
        return self._refs

    @property
    def periods(self) -> np.ndarray:
        return 1.0 / np.asarray(self._refs)

    @property
    def osc_damping(self) -> float:
        return self._osc_damping

    @property
    def ylabel(self) -> str:
        return f"{100 * self.osc_damping:g}%-Damped, Resp. Spectral Ratio"

    def __call__(self, calc, name=None, index: int | None = None) -> None:
        Output.__call__(self, calc, name, index)
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
    """Base class for outputs reported as a function of depth.

    By default each realization stores its own layer depths, which differ
    whenever the layering varies. Passing *depths* instead resamples every
    realization onto a fixed grid, so results are directly comparable and the
    stored array has a constant shape. Use
    :meth:`~pystrata.site.Profile.depth_grid` to build a suitable grid.

    Resampling is lossy: layer interfaces are snapped to the next grid node and
    the original depths are not recoverable.

    Parameters
    ----------
    depths : array_like, optional
        Fixed, monotonically increasing depths [m] onto which each realization
        is resampled. When ``None`` (default) the layer depths of each
        realization are stored as-is.
    fill_below : {'nan', 'hold'}, optional
        Value reported below the base of a realization's soil column.
        ``'nan'`` excludes those depths from statistics, and is the default
        when *depths* is given. ``'hold'`` repeats the deepest value, and is
        the default otherwise.
    """

    ylabel = "Depth (m)"
    yscale = "linear"
    drawstyle = "steps-post"

    ref_name = "depth"

    #: Interpolation used when resampling. Layer properties are piecewise
    #: constant, so they use the value of the layer the depth falls within.
    _interp_kind = "next"

    #: Number of points in the default reporting grid.
    _stats_count = 512

    def __init__(
        self,
        depths: npt.ArrayLike | None = None,
        fill_below: str | None = None,
    ) -> None:
        super().__init__(depths)
        self._const_ref = depths is not None

        if self._const_ref:
            _depths = self._refs
            if _depths.ndim != 1 or _depths.size < 2:
                raise ValueError(
                    "depths must be a 1-D array of at least two values; got shape "
                    f"{_depths.shape}."
                )
            if not np.all(np.diff(_depths) > 0):
                raise ValueError("depths must be strictly increasing.")

        if fill_below is None:
            fill_below = "nan" if self._const_ref else "hold"
        if fill_below not in ("nan", "hold"):
            raise ValueError(f"fill_below must be 'nan' or 'hold', not {fill_below!r}.")
        self._fill_below = fill_below
        self._warned_extent = False

    @property
    def depths(self) -> np.ndarray:
        """Depths at which the values are reported."""
        return self._refs

    @property
    def fill_below(self) -> str:
        return self._fill_below

    def __call__(self, calc, name=None, index: int | None = None) -> None:
        Output.__call__(self, calc, name, index)
        depths, values = self._calc_profile(calc)

        if self._const_ref:
            self._check_extent(calc)
            self._add_values(self._resample(depths, values))
        else:
            self._add_refs(depths)
            self._add_values(values)

    def reset(self) -> None:
        super().reset()
        self._warned_extent = False

    def _calc_profile(self, calc) -> tuple[np.ndarray, np.ndarray]:
        """Return the depths and values of a single realization."""
        raise NotImplementedError

    def _check_extent(self, calc) -> None:
        """Warn once if a realization extends below the fixed grid."""
        base = calc.profile[-2].depth_base
        if base > self._refs[-1] and not self._warned_extent:
            self._warned_extent = True
            warnings.warn(
                "A realization extends below the fixed depth grid; results "
                f"below {self._refs[-1]:.1f} m are discarded. Rebuild the grid "
                "with a larger max_depth.",
                stacklevel=3,
            )

    def _resample(
        self,
        depths: npt.ArrayLike,
        values: npt.ArrayLike,
        grid: npt.ArrayLike | None = None,
    ) -> np.ndarray:
        """Resample one realization onto *grid*.

        Interpolation is performed in linear space. For ``kind='next'`` no
        arithmetic is done between nodes, so this is identical to interpolating
        the logarithm, while avoiding the round trip through ``log(0)`` for
        outputs that report zero at the surface.
        """
        grid = self._refs if grid is None else np.asarray(grid, dtype=float)
        depths = np.asarray(depths, dtype=float)
        values = np.asarray(values, dtype=float)

        below = values[-1] if self._fill_below == "hold" else np.nan
        f = interp1d(
            depths,
            values,
            kind=self._interp_kind,
            fill_value=(values[0], below),
            bounds_error=False,
        )
        return f(grid)

    def _resample_stored(self, i: int, grid: npt.ArrayLike) -> np.ndarray:
        """Resample stored realization *i*, dropping any NaN padding."""
        refs = self.refs[:, i] if self.refs.ndim > 1 else self.refs
        values = self.values[:, i] if self.values.ndim > 1 else self.values

        mask = np.isfinite(refs)
        if not np.any(mask):
            return np.full(np.shape(grid), np.nan)

        return self._resample(refs[mask], values[mask], grid)

    def _default_ref(self) -> np.ndarray:
        """The grid used by :meth:`calc_stats` and :meth:`to_dataframe`."""
        if self._const_ref:
            return self._refs

        # With NaN fill the padded tail would be empty by construction, so the
        # margin only manufactures rows with no data behind them.
        margin = 1.0 if self._fill_below == "nan" else 1.05
        return np.linspace(0, np.nanmax(self.refs) * margin, num=self._stats_count)

    def calc_stats(self, as_dataframe: bool = False, ref: npt.ArrayLike | None = None):
        if ref is None:
            ref = self._default_ref()
            resampled = self._const_ref
        else:
            ref = np.asarray(ref, dtype=float)
            resampled = False

        with (
            np.errstate(divide="ignore", invalid="ignore"),
            warnings.catch_warnings(),
        ):
            # Depths below every realization have no data, which numpy reports
            # as an empty slice. A count of zero already conveys this.
            warnings.filterwarnings("ignore", "Mean of empty slice", RuntimeWarning)
            warnings.filterwarnings("ignore", "Degrees of freedom", RuntimeWarning)

            # Outputs that report zero at the surface give -inf here, which
            # carries through to a median of zero.
            if resampled:
                values = self.values
                ln_values = np.log(values if values.ndim > 1 else values[:, None])
            else:
                n = self.values.shape[1] if self.values.ndim > 1 else 1
                ln_values = np.array(
                    [self._resample_stored(i, ref) for i in range(n)]
                ).T
                ln_values = np.log(ln_values)

            median = np.exp(np.nanmean(ln_values, axis=1))
            ln_std = np.nanstd(ln_values, axis=1)

        stats = {
            "ref": ref,
            "median": median,
            "ln_std": ln_std,
            # Realizations contributing at each depth. Varies with depth when
            # fill_below='nan' and the profile depth is randomized.
            "count": np.sum(np.isfinite(ln_values), axis=1),
        }
        if as_dataframe and pd:
            stats = pd.DataFrame(stats).set_index("ref")
            stats.index.name = self.ref_name

        return stats

    @staticmethod
    def _get_xy(refs, values):
        return values, refs

    def plot(self, ax=None, style: str = "stats"):
        ax = Output.plot(self, ax, style)
        ax.invert_yaxis()
        return ax

    def to_dataframe(self, ref: npt.ArrayLike | None = None):
        if not pd:
            raise RuntimeError("Install `pandas` library.")

        if ref is None:
            ref = self._default_ref()
            resampled = self._const_ref
        else:
            ref = np.asarray(ref, dtype=float)
            resampled = False

        if isinstance(self.names[0], tuple):
            columns = pd.MultiIndex.from_tuples(self.names)
        else:
            columns = self.names

        if resampled:
            values = self.values
            values = values if values.ndim > 1 else values[:, None]
        else:
            n = self.values.shape[1] if self.values.ndim > 1 else 1
            values = np.array([self._resample_stored(i, ref) for i in range(n)]).T

        return pd.DataFrame(values, index=ref, columns=columns)


class _LayerTopProfile(ProfileBasedOutput):
    """Profile output reported at the top of each layer.

    The surface takes the value of the first layer, so the reported depths are the layer
    tops and the deepest is the base of the soil column.
    """

    def _layer_values(self, calc) -> list:
        raise NotImplementedError

    def _calc_profile(self, calc):
        depths = np.asarray(calc.profile.depth, dtype=float)
        values = self._layer_values(calc)
        # Bring the first mid-layer value to the surface
        return depths, np.r_[values[0], values]


class _LayerMidProfile(ProfileBasedOutput):
    """Profile output reported at the mid-depth of each layer.

    The surface takes the value of the first layer. A final point is reported at the
    base of the soil column so that the deepest layer is represented over its full
    thickness rather than only to its mid-depth.
    """

    def _layer_values(self, calc) -> list:
        raise NotImplementedError

    def _surface_value(self, values):
        return values[0]

    def _calc_profile(self, calc):
        depths = np.r_[
            0, calc.profile.depth_mid[:-1], calc.profile[-2].depth_base
        ].astype(float)
        values = self._layer_values(calc)
        return depths, np.r_[self._surface_value(values), values, values[-1]]


class MaxStrainProfile(_LayerMidProfile):
    xlabel = "Max. Strain (dec)"

    def _surface_value(self, values):
        # No strain at the free surface
        return 0.0

    def _layer_values(self, calc):
        return [layer.strain_max for layer in calc.profile[:-1]]


class DampingProfile(_LayerTopProfile):
    xlabel = "Damping (dec)"

    def _layer_values(self, calc):
        return [layer.damping for layer in calc.profile[:-1]]


class ShearModReducProfile(_LayerTopProfile):
    xlabel = "G/Gmax"

    def _layer_values(self, calc):
        return [layer.shear_mod_reduc for layer in calc.profile[:-1]]


class InitialVelProfile(_LayerTopProfile):
    xlabel = "Initial Velocity (m/s)"

    def _layer_values(self, calc):
        return [layer.initial_shear_vel for layer in calc.profile[:-1]]


class CompatVelProfile(_LayerTopProfile):
    xlabel = "Strain-Compatible Velocity (m/s)"

    def _layer_values(self, calc):
        return [np.min(layer.shear_vel) for layer in calc.profile[:-1]]


class CyclicStressRatioProfile(_LayerMidProfile):
    xlabel = "Cyclic Stress Ratio"

    # From Idriss and Boulanger (2008, pg. 70):
    # The 0.65 is a constant used to represent the reference stress
    # level. While being somewhat arbitrary it was selected in the
    # beginning of the development of liquefaction procedures in 1966
    # and has been in use ever since.
    _stress_level = 0.65

    def _layer_values(self, calc):
        return [
            self._stress_level
            * layer.stress_shear_max
            / layer.stress_vert(layer.thickness / 2, True)
            for layer in calc.profile[:-1]
        ]


class MaxAccelProfile(_LayerTopProfile):
    xlabel = "Max. Accel. (g)"

    # Acceleration is a continuous field rather than a layer property, so it is
    # interpolated between the depths at which it was computed.
    _interp_kind = "linear"

    def _calc_profile(self, calc):
        depths = np.asarray(calc.profile.depth, dtype=float)
        values = np.array([self._calc_accel(calc, depth) for depth in depths])
        return depths, values

    def _calc_accel(self, calc, depth):
        return calc.motion.calc_peak(
            calc.calc_accel_tf(
                calc.loc_input, calc.profile.location("within", depth=depth)
            )
        )
