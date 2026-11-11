# The MIT License (MIT)
#
# Copyright (c) 2016-2018 Albert Kottke
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
"""Classes used to define input motions."""

from __future__ import annotations

import enum
import logging
import re
import warnings

import numpy as np
import numpy.typing as npt
import pyrvt
from scipy import stats

from .units import GRAVITY, convert_units

logger = logging.getLogger(__name__)


# Integers and floats, including values without a leading digit (e.g., ".0100")
# and Fortran style exponents (e.g., "1.0D-2").
_RE_NUMBER = re.compile(r"[+-]?(?:\d+\.?\d*|\.\d+)(?:[eEdD][+-]?\d+)?")


def _to_float(text: str) -> float:
    """Convert a string to a float, permitting a Fortran style exponent."""
    return float(text.replace("D", "E").replace("d", "e"))


def _parse_at2_header(line: str) -> tuple[int, float]:
    """Parse the point count and time step from the header of an AT2 file.

    Both of the PEER NGA layouts are supported::

        4096    0.0100    NPTS, DT
        NPTS=   5346, DT=   .0100 SEC,

    as are variations that reverse the order of the two values, or that omit the
    commas separating them.

    Parameters
    ----------
    line : str
        Fourth line of an AT2 file.

    Returns
    -------
    npts : int
        Number of points in the time series.
    time_step : float
        Time step of the time series [s].

    Raises
    ------
    ValueError
        If the point count and the time step cannot be found.
    """
    # Values that follow their label, e.g., "NPTS= 5346" or "DT .0100". Each value
    # is located by its own label, so their order does not matter.
    found = {}
    for key in ("NPTS", "DT"):
        match = re.search(
            r"\b" + key + r"\b\s*[=:]?\s*(" + _RE_NUMBER.pattern + ")",
            line,
            re.IGNORECASE,
        )
        if match:
            found[key] = _to_float(match.group(1))

    if len(found) < 2:
        # Values that precede their labels, e.g., "4096  0.0100  NPTS, DT"
        values = [_to_float(v) for v in _RE_NUMBER.findall(line)]
        if len(values) < 2:
            raise ValueError(f"Unable to parse NPTS and DT from AT2 header: {line!r}")

        values = values[:2]
        upper = line.upper()
        pos = {key: upper.find(key) for key in ("NPTS", "DT")}
        if all(p >= 0 for p in pos.values()):
            # Pair the values with the labels by order of appearance
            keys = sorted(pos, key=lambda key: pos[key])
        else:
            # Unlabeled, so rely on the magnitude: the time step is the smaller
            keys = ["DT", "NPTS"] if values[0] < values[1] else ["NPTS", "DT"]

        found = dict(zip(keys, values, strict=True))

    return int(found["NPTS"]), found["DT"]


def fit_attenuation(
    freqs: npt.ArrayLike,
    amps: npt.ArrayLike,
    min_freq: float,
    max_freq: float | None = None,
) -> tuple[float, float, np.ndarray, np.ndarray]:
    r"""Compute the site attenuation (:math:`\kappa`) from a log-linear fit.

    The Fourier amplitudes are fit with
    :math:`a(f) = A_0 \exp(-\pi \kappa f)` for the frequencies in a range
    (Anderson and Hough, 1984), which is what ``calc_attenuation`` of the motions
    of pyRVT does. Amplitudes that are not finite and positive are not used.

    Parameters
    ----------
    freqs : array_like
        Frequencies [Hz].
    amps : array_like
        Fourier amplitudes at `freqs`. Complex amplitudes are reduced to their
        magnitude.
    min_freq : float
        Minimum frequency of the fit [Hz].
    max_freq : float, optional
        Maximum frequency of the fit [Hz]. The maximum frequency is used if
        ``None``.

    Returns
    -------
    atten : float
        Attenuation, :math:`\kappa` [s].
    r_sqr : float
        Squared correlation coefficient of the fit.
    freqs : np.ndarray
        Frequencies of the fit [Hz].
    fitted : np.ndarray
        Fitted amplitudes at `freqs`.

    Raises
    ------
    ValueError
        If fewer than two amplitudes are usable in the frequency range.
    """
    freqs = np.asarray(freqs, dtype=float)
    amps = np.abs(np.asarray(amps))
    max_freq = freqs[-1] if max_freq is None else max_freq
    mask = (min_freq <= freqs) & (freqs <= max_freq) & np.isfinite(amps) & (amps > 0)
    if mask.sum() < 2:
        raise ValueError(
            f"Fewer than two positive amplitudes between {min_freq} and {max_freq} Hz."
        )
    fit = stats.linregress(freqs[mask], np.log(amps[mask]))
    return (
        -fit.slope / np.pi,
        fit.rvalue**2,
        freqs[mask],
        np.exp(fit.intercept + fit.slope * freqs[mask]),
    )


class WaveField(enum.Enum):
    """Wave field at a location in the soil column.

    ``outcrop`` is twice the up-going wave, ``within`` the sum of the up- and
    down-going waves, and ``up_going`` and ``down_going`` the individual waves.
    """

    outcrop = 0
    within = 1
    up_going = 2
    down_going = 3

    @classmethod
    def _missing_(cls, value):
        # Resolve string names and shorthand aliases (e.g. "2A", "A+B", "A",
        # "B") to their canonical members. Called by ``WaveField(value)`` when
        # ``value`` is not one of the member values.
        if isinstance(value, str):
            key = value.strip()
            if key in cls._member_map_:
                return cls._member_map_[key]
            alias = _WAVE_FIELD_ALIASES.get(key)
            if alias is not None:
                return cls._member_map_[alias]
        return None


#: Shorthand aliases for the wave-field conditions. Each maps to the name of a
#: canonical :class:`WaveField` member:
#:
#: - ``"2A"`` -- outcrop (twice the up-going wave)
#: - ``"A+B"`` -- within (sum of up- and down-going waves)
#: - ``"A"`` -- up_going (up-going wave only)
#: - ``"B"`` -- down_going (down-going wave only)
#:
#: ``"incoming_only"`` is retained as an alias for ``up_going`` for backwards
#: compatibility.
_WAVE_FIELD_ALIASES = {
    "2A": "outcrop",
    "A+B": "within",
    "A": "up_going",
    "incoming_only": "up_going",
    "upgoing": "up_going",
    "B": "down_going",
    "downgoing": "down_going",
}


class Motion:
    """Base class for input motions defined on a frequency grid.

    Caches the peak ground acceleration and velocity, the Arias intensity and
    the cumulative absolute velocity, which subclasses compute with the
    ``calc_*`` methods.
    """

    def __init__(self, freqs: npt.ArrayLike | None = None) -> None:
        object.__init__(self)

        self._freqs = np.array([] if freqs is None else freqs)
        self._pga: float | None = None
        self._pgv: float | None = None
        self._arias_intensity: float | None = None
        self._cav: float | None = None

    @property
    def freqs(self) -> np.ndarray:
        """Frequencies [Hz] of the motion."""
        return self._freqs

    @property
    def angular_freqs(self) -> np.ndarray:
        """Angular frequencies [rad/s] of the motion."""
        return 2 * np.pi * self.freqs

    @property
    def pgv(self) -> float:
        """Peak ground velocity [cm/sec]."""
        if self._pgv is None:
            self._pgv = self.calc_pgv()
        return self._pgv

    @property
    def pga(self) -> float:
        """Peak ground acceleration [g]."""
        if self._pga is None:
            self._pga = self.calc_pga()
        return self._pga

    @property
    def arias_intensity(self) -> float:
        """Arias intensity [m/s]."""
        if self._arias_intensity is None:
            self._arias_intensity = self.calc_arias_intensity()
        return self._arias_intensity

    @property
    def cav(self) -> float:
        """Cumulative absolute velocity [m/s]."""
        if self._cav is None:
            self._cav = self.calc_cav()
        return self._cav

    def calc_peak(self, tf: npt.ArrayLike | None = None, **kwargs) -> float:
        """Compute the peak of the motion; implemented by the subclasses."""
        raise NotImplementedError


class TimeSeriesMotion(Motion):
    """Time-series motion for time series based site response analysis."""

    @convert_units(time_step="second", accels="standard_gravity")
    def __init__(
        self,
        filename: str,
        description: str,
        time_step: float,
        accels: npt.ArrayLike,
        fa_length: int | None = None,
    ) -> None:
        """Initialize the class from specified acceleration values.

        The *filename* and *description* parameters are only used to help track the
        motion.

        Parameters
        ----------
        filename: str
            Source of data
        description: str
            Description to store helpful information
        time_step: float
            Time step of the accleration values
        accels: array_like
            Accelerations in units of *g*
        fa_length: optional int
            Length to use for the Fourier amplitude spectrum. If not provided, will be
            automatically computed to the next power of 2. Must be even and at
            least the number of samples; otherwise a ``ValueError`` is raised.
        """
        Motion.__init__(self)

        self._filename = filename
        self._description = description
        self._time_step = time_step
        self._accels = np.asarray(accels)

        self._calc_fourier_spectrum(fa_length)

        logger.debug(
            "TimeSeriesMotion: %d samples, dt=%.4fs, PGA=%.3fg",
            len(self._accels),
            self._time_step,
            self.pga,
        )

    @property
    def accels(self) -> np.ndarray:
        """Acceleration time series [g]."""
        return self._accels

    @property
    def filename(self) -> str:
        """Name of the file that the motion was read from."""
        return self._filename

    @property
    def description(self) -> str:
        """Description of the motion."""
        return self._description

    @property
    def time_step(self) -> float:
        """Time step of the time series [s]."""
        return self._time_step

    @property
    def times(self) -> np.ndarray:
        """Times [s] of the time series."""
        return self._time_step * np.arange(self._accels.size)

    @property
    def freqs(self) -> np.ndarray:
        """Return the frequencies."""
        if self._freqs is None:
            self._calc_fourier_spectrum()

        return self._freqs

    @property
    def fourier_amps(self) -> np.ndarray:
        """Return the frequencies."""
        if self._fourier_amps is None:
            self._calc_fourier_spectrum()

        # Normalize the Fourier amplitude by the time step
        return self.time_step * self._fourier_amps

    def calc_attenuation(
        self, min_freq: float, max_freq: float | None = None
    ) -> tuple[float, float, np.ndarray, np.ndarray]:
        r"""Compute the site attenuation (:math:`\kappa`) of the Fourier spectrum.

        This has the arguments and the results of ``calc_attenuation`` of the
        random vibration theory motions, and is computed by
        :func:`fit_attenuation`.

        Parameters
        ----------
        min_freq : float
            Minimum frequency of the fit [Hz].
        max_freq : float, optional
            Maximum frequency of the fit [Hz]. The maximum frequency of the
            spectrum is used if ``None``.

        Returns
        -------
        atten : float
            Attenuation, :math:`\kappa` [s].
        r_sqr : float
            Squared correlation coefficient of the fit.
        freqs : np.ndarray
            Frequencies of the fit [Hz].
        fitted : np.ndarray
            Fitted amplitudes at `freqs`.
        """
        return fit_attenuation(self.freqs, self.fourier_amps, min_freq, max_freq)

    def calc_time_series(self, tf: npt.ArrayLike | None = None) -> np.ndarray:
        """Compute the acceleration time series [g] with a transfer function applied.

        Parameters
        ----------
        tf : array_like, optional
            Transfer function to apply to the motion. If ``None``, none is applied.
        """
        if tf is None:
            ts = np.fft.irfft(self.fourier_amps / self.time_step)
        else:
            ts = np.fft.irfft(tf * self.fourier_amps / self.time_step)
        return ts

    def calc_pgv(self, tf: npt.ArrayLike | None = None) -> float:
        """Compute the peak ground velocity [cm/s].

        Parameters
        ----------
        tf : array_like, optional
            Transfer function to apply to the motion. If ``None``, none is applied.
        """
        tf = 1 if tf is None else np.asarray(tf)
        # Compute transfer function from acceleration to velocity
        # only over non-zero frequencies
        mask = ~np.isclose(self.angular_freqs, 0)
        tf_av = np.zeros_like(mask, dtype=complex)
        tf_av[mask] = 1 / (self.angular_freqs[mask] * 1j)
        return GRAVITY * 100 * self.calc_peak(tf_av * tf)

    def calc_pga(self, tf: npt.ArrayLike | None = None) -> float:
        """Compute the peak ground acceleration [g].

        Parameters
        ----------
        tf : array_like, optional
            Transfer function to apply to the motion. If ``None``, none is applied.
        """
        return self.calc_peak(tf)

    def calc_peak(self, tf: npt.ArrayLike | None = None, **kwargs) -> float:
        """Compute the peak absolute acceleration [g].

        Parameters
        ----------
        tf : array_like, optional
            Transfer function to apply to the motion. If ``None``, none is applied.
        """
        ts = self.calc_time_series(tf)
        return np.abs(ts).max()

    def calc_arias_intensity(self, tf: npt.ArrayLike | None = None) -> float:
        """Compute the Arias intensity [m/s].

        Parameters
        ----------
        tf : array_like, optional
            Transfer function to apply to the motion. If ``None``, none is applied.
        """
        tf = 1 if tf is None else np.asarray(tf)
        ts = self.calc_time_series(tf)
        return np.pi * GRAVITY / 2 * np.trapezoid(ts**2, dx=self.time_step)

    def calc_cav(self, tf: npt.ArrayLike | None = None) -> float:
        """Compute the cumulative absolute velocity [m/s].

        Parameters
        ----------
        tf : array_like, optional
            Transfer function to apply to the motion. If ``None``, none is applied.
        """
        tf = 1 if tf is None else np.asarray(tf)
        ts = self.calc_time_series(tf)
        return GRAVITY * np.trapezoid(np.abs(ts), dx=self.time_step)

    def calc_osc_accels(
        self,
        osc_freqs: npt.ArrayLike,
        osc_damping: float = 0.05,
        tf: npt.ArrayLike | None = None,
    ) -> np.ndarray:
        """Compute the pseudo-acceleration spectral response of an oscillator.

        The oscillator has a specific frequency and damping.

        Parameters
        ----------
        osc_freq : float
            Frequency of the oscillator [Hz].
        osc_damping : float
            Fractional damping of the oscillator [dec]. For example, 0.05 for a
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
            [
                self.calc_peak(tf * self._calc_sdof_tf(of, osc_damping))
                for of in osc_freqs
            ]
        )
        return resp

    def _calc_fourier_spectrum(self, fa_length: int | None = None) -> None:
        """Compute the Fourier Amplitude Spectrum of the time series."""
        if fa_length is None:
            # Use the next power of 2 for the length
            n = 1
            while n < self.accels.size:
                n <<= 1
        else:
            n = int(fa_length)
            if n < self._accels.size:
                raise ValueError(
                    f"fa_length ({n}) must be at least the number of samples "
                    f"in the record ({self._accels.size}); a shorter length "
                    "would truncate the record."
                )
            if n % 2:
                raise ValueError(f"fa_length ({n}) must be even.")

        self._fourier_amps = np.fft.rfft(self._accels, n)
        self._freqs = np.fft.rfftfreq(n, d=self._time_step)

    def _calc_sdof_tf(self, osc_freq: float, damping: float = 0.05) -> np.ndarray:
        """Compute the transfer function for a single-degree-of-freedom oscillator.

        The transfer function computes the pseudo-spectral acceleration. See
        :func:`pyrvt.motions.calc_sdof_tf`.

        Parameters
        ----------
        osc_freq : float
            Natural frequency of the oscillator [Hz].
        damping : float, default=0.05
            Damping ratio of the oscillator [decimal].

        Returns
        -------
        tf : :class:`numpy.ndarray`
            Complex-valued transfer function with length equal to `self.freq`.
        """
        return pyrvt.motions.calc_sdof_tf(self.freqs, osc_freq, damping)

    @classmethod
    def load_at2_file(cls, filename: str, scale: float = 1.0) -> TimeSeriesMotion:
        """Read an AT2 formatted time series.

        The fourth line of the file provides the number of points and the time
        step; both of the PEER NGA layouts are read, and a warning is issued if
        the number of points does not match the number of accelerations.

        Parameters
        ----------
        filename: str
            Filename to open.
        scale: float, default: 1.
            Scale factor to apply to the motion.
        """
        with open(filename) as fp:
            next(fp)
            description = next(fp).strip()
            next(fp)
            npts, time_step = _parse_at2_header(next(fp))

            accels = np.array([float(part) for line in fp for part in line.split()])

        if accels.size != npts:
            warnings.warn(
                f"AT2 file '{filename}' specifies NPTS={npts}, but {accels.size} "
                "accelerations were read.",
                stacklevel=2,
            )

        accels *= scale
        return cls(filename, description, time_step, accels)

    @classmethod
    def load_smc_file(cls, filename: str, scale: float = 1.0) -> TimeSeriesMotion:
        """Read an SMC formatted time series.

        Format of the time series is provided by:
            https://escweb.wr.usgs.gov/nsmp-data/smcfmt.html

        Parameters
        ----------
        filename: str
            Filename to open.
        scale: float, default: 1.
            Scale factor to apply to the motion.
        """
        from .tools import parse_fixed_width

        with open(filename) as fp:
            lines = list(fp)

        # 11 lines of strings
        lines_str = [lines.pop(0) for _ in range(11)]

        if lines_str[0].strip() != "2 CORRECTED ACCELEROGRAM":
            raise RuntimeWarning("Loading uncorrected SMC file.")

        m = re.search("station =(.+)component=(.+)", lines_str[5])
        description = "; ".join([g.strip() for g in m.groups()])

        # 6 lines of (8i10) formatted integers
        values_int = parse_fixed_width(
            48 * [(10, int)], [lines.pop(0) for _ in range(6)]
        )
        count_comment = values_int[15]
        count = values_int[16]

        # 10 lines of (5e15.7) formatted floats
        values_float = parse_fixed_width(
            50 * [(15, float)], [lines.pop(0) for _ in range(10)]
        )
        time_step = 1 / values_float[1]

        # Skip comments
        lines = lines[count_comment:]

        accels = np.array(
            parse_fixed_width(
                count
                * [
                    (10, float),
                ],
                lines,
            )
        )
        accels *= scale

        return TimeSeriesMotion(filename, description, time_step, accels)


class RvtMotion(pyrvt.motions.RvtMotion, Motion):
    """RVT motion based on a user-specified Fourier amplitude spectrum + duration.

    Wraps :class:`pyrvt.motions.RvtMotion` to add pint-based unit conversion
    and the pystrata :class:`Motion` interface. Construct directly with
    arrays, or use :meth:`from_fas` (inherited from pyrvt) to build from any
    FAS-producing object (e.g. ``pygmm.fourier_spectrum.SourceTheoryModel``).
    """

    @convert_units(fourier_amps="standard_gravity * second", duration="second")
    def __init__(
        self,
        freqs: npt.ArrayLike,
        fourier_amps: npt.ArrayLike,
        duration: float | None = None,
        peak_calculator=None,
        calc_kwds: dict | None = None,
    ) -> None:
        Motion.__init__(self)
        pyrvt.motions.RvtMotion.__init__(
            self,
            np.asarray(freqs),
            np.asarray(fourier_amps),
            duration=duration,
            peak_calculator=peak_calculator,
            calc_kwds=calc_kwds,
        )


class CompatibleRvtMotion(pyrvt.motions.CompatibleRvtMotion, Motion):
    """RVT motion based on a user specified response spectrum and duration."""

    @convert_units(osc_accels_target="standard_gravity", duration="second")
    def __init__(
        self,
        osc_freqs: npt.ArrayLike,
        osc_accels_target: npt.ArrayLike,
        duration: float | None = None,
        osc_damping: float = 0.05,
        window_len: int | None = None,
        peak_calculator=None,
        calc_kwds: dict | None = None,
    ) -> None:
        Motion.__init__(self)
        pyrvt.motions.CompatibleRvtMotion.__init__(
            self,
            osc_freqs,
            osc_accels_target,
            duration=duration,
            osc_damping=osc_damping,
            window_len=window_len,
            peak_calculator=peak_calculator,
            calc_kwds=calc_kwds,
        )
