"""Dam and joined dam-site models."""

from __future__ import annotations

import logging
import warnings

import numpy as np
import numpy.typing as npt
from scipy.optimize import brentq

from ..site import Layer, Profile, SoilType, layers_from_depth_power_law

logger = logging.getLogger(__name__)


class WedgeLayer(Layer):
    """Layer of a triangular shear-wedge dam.

    The distance from the crest (``layer.depth + depth_within``) is the radial
    coordinate of the wedge solution. Otherwise identical to :class:`Layer`.
    """


class Dam:
    """Triangular (shear-wedge) dam made of stacked layers.

    The layers are ordered from the crest down. Each layer is a homogeneous
    segment of an infinitely long wedge (plane-strain,
    :cite:p:`gazetas1992seismic`); inhomogeneity is represented by layers with
    increasing velocity.

    The cross-section is a trapezoid with a symmetric slope and an optional
    crest width. The width at a depth :math:`y` below the crest is
    :math:`b + 2 s y`, which is the width :math:`2 s r` of the wedge at a
    distance :math:`r` from its apex, :math:`r = y + r_0`, with the crest at
    :math:`r_0 = b / (2 s)`. A crest width truncates the wedge, and the crest is
    a free surface. Without a slope the crest is the apex of the wedge.

    Parameters
    ----------
    layers : list of :class:`Layer`
        Layers from the crest to the base. Plain layers are converted to
        :class:`WedgeLayer`.
    slope : float, optional
        Horizontal run per unit height of each of the two slopes, e.g., 2.5 for
        2.5H:1V. Required to draw the section, to define a failure surface and
        for a crest width.
    crest_width : float, default=0
        Width of the crest [m]. It requires `slope`.
    """

    def __init__(
        self,
        layers: list[Layer],
        slope: float | None = None,
        crest_width: float = 0.0,
    ):
        if not layers:
            raise ValueError("A dam requires at least one layer.")
        if slope is not None and not slope > 0:
            raise ValueError("The slope must be positive, e.g., 2.5 for 2.5H:1V.")
        if crest_width < 0:
            raise ValueError("The crest width cannot be negative.")
        if crest_width > 0 and slope is None:
            raise ValueError("A crest width requires a slope.")
        self.layers = [self._as_wedge(layer) for layer in layers]
        self._slope = slope
        self._crest_width = float(crest_width)

    @staticmethod
    def _as_wedge(layer: Layer) -> WedgeLayer:
        return WedgeLayer(
            layer.soil_type,
            layer.thickness,
            layer.initial_shear_vel,
            layer.damping_min,
            layer.poissons_ratio,
        )

    @classmethod
    def from_inhomogeneity(
        cls,
        soil_type: SoilType,
        height: float,
        shear_vel_avg: float,
        m: float = 0.0,
        n_layers: int = 20,
        slope: float | None = None,
        crest_width: float = 0.0,
    ) -> Dam:
        r"""Dam with a shear modulus proportional to :math:`z^m`.

        Parameters
        ----------
        soil_type : SoilType
        height : float
            Height of the dam [m].
        shear_vel_avg : float
            Area-weighted average shear-wave velocity [m/s].
        m : float, default=0
            Inhomogeneity parameter, :math:`G \propto z^m`
            (:math:`V_s \propto z^{m/2}`).
            :math:`m = 0` is a homogeneous dam; :cite:t:`gazetas1992seismic`
            consider 0 to 2/3.
        n_layers : int, default=20
            Number of layers of equal thickness.
        slope : float, optional
            Horizontal run per unit height of the slopes, see :class:`Dam`.
        crest_width : float, default=0
            Width of the crest [m], see :class:`Dam`.
        """
        # The velocity is a r^{m/2} at a distance r from the apex of the wedge,
        # and `shear_vel_avg` averages over the area of the cross-section
        crest_offset = crest_width / (2 * slope) if slope else 0.0
        r_base = crest_offset + height
        power = m / 2 + 2
        a = (
            shear_vel_avg
            * power
            * (r_base**2 - crest_offset**2)
            / (2 * (r_base**power - crest_offset**power))
        )
        thicknesses = np.full(n_layers, height / n_layers)
        layers = layers_from_depth_power_law(
            soil_type,
            thicknesses,
            a,
            m / 2,
            depth_top=crest_offset,
            layer_cls=WedgeLayer,
        )
        return cls(layers, slope=slope, crest_width=crest_width)

    @property
    def slope(self) -> float | None:
        """Horizontal run per unit height of each slope; ``None`` if not given."""
        return self._slope

    @property
    def crest_width(self) -> float:
        """Width of the crest [m]."""
        return self._crest_width

    @property
    def crest_offset(self) -> float:
        """Distance [m] from the apex of the wedge to the crest, ``b / (2 s)``."""
        return self._crest_width / (2 * self._slope) if self._slope else 0.0

    def _require_slope(self) -> float:
        if self._slope is None:
            raise ValueError("The dam geometry requires a slope.")
        return self._slope

    def width(self, depth: npt.ArrayLike) -> np.ndarray:
        """Width [m] of the cross-section at a depth [m] below the crest."""
        return self._crest_width + 2 * self._require_slope() * np.asarray(depth)

    @property
    def base_width(self) -> float:
        """Width [m] of the base of the dam."""
        return float(self.width(self.height))

    @property
    def area(self) -> float:
        """Area [m²] of the cross-section."""
        return float(0.5 * (self._crest_width + self.base_width) * self.height)

    @property
    def thicknesses(self) -> np.ndarray:
        """Thickness [m] of each layer."""
        return np.array([layer.thickness for layer in self.layers])

    @property
    def shear_vels(self) -> np.ndarray:
        """Initial shear-wave velocity [m/s] of each layer."""
        return np.array([layer.initial_shear_vel for layer in self.layers])

    @property
    def height(self) -> float:
        """Height of the dam [m]."""
        return float(self.thicknesses.sum())

    @property
    def shear_vel_avg(self) -> float:
        """Area-weighted average shear-wave velocity [m/s]."""
        # Areas are proportional to the difference of the squared distances from
        # the apex of the wedge
        r_base = self.thicknesses.cumsum() + self.crest_offset
        r_top = r_base - self.thicknesses
        weights = (r_base**2 - r_top**2) / (r_base[-1] ** 2 - r_top[0] ** 2)
        return float(np.sum(weights * self.shear_vels))

    def scaled(self, factor: float) -> Dam:
        """Copy of the dam with every shear-wave velocity multiplied by factor."""
        return Dam(
            [
                WedgeLayer(
                    layer.soil_type,
                    layer.thickness,
                    layer.initial_shear_vel * factor,
                    layer.damping_min,
                    layer.poissons_ratio,
                )
                for layer in self.layers
            ],
            slope=self._slope,
            crest_width=self._crest_width,
        )

    def fixed_base_freqs(self, n: int = 1) -> np.ndarray:
        """First ``n`` natural frequencies [Hz] with a rigid (fixed) base.

        Computed for the undamped layered dam from the roots of the displacement at the
        base.
        """
        from .propagation import _calc_wedge_waves, _top_of_layer, _wedge_state

        thick = self.thicknesses
        crest_offset = self.crest_offset
        # Distances from the apex of the wedge
        depths_base = thick.cumsum() + crest_offset
        vels = self.shear_vels
        mods = np.array([layer.initial_shear_mod for layer in self.layers])
        height = depths_base[-1]

        def base_disp(freqs):
            wn = 2 * np.pi * np.asarray(freqs)[None, :] / vels[:, None]
            coeff_a, coeff_b = _calc_wedge_waves(
                wn + 0j,
                np.repeat(mods[:, None], wn.shape[1], 1) + 0j,
                depths_base,
                crest_offset,
            )
            disp, _ = _wedge_state(
                wn[-1] + 0j,
                coeff_a[-1],
                coeff_b[-1],
                height,
                _top_of_layer(depths_base, len(depths_base) - 1, crest_offset),
            )
            return disp.real

        f_est = 2.405 * self.shear_vel_avg / (2 * np.pi * height)
        f_max = 0.5 * f_est
        roots: list[float] = []
        while len(roots) < n:
            f_max *= 2
            grid = np.linspace(f_est * 0.05, f_max, int(400 * f_max / f_est) + 1)
            vals = base_disp(grid)
            idx = np.nonzero(vals[:-1] * vals[1:] < 0)[0]
            roots = [
                brentq(lambda f: base_disp([f])[0], grid[i], grid[i + 1], xtol=1e-12)
                for i in idx
            ]
        return np.array(roots[:n])

    def fixed_base_freq(self, n: int = 1) -> float:
        """The ``n``-th natural frequency [Hz] with a rigid base."""
        return float(self.fixed_base_freqs(n)[-1])

    def tune(
        self,
        freq: float,
        site: Profile | None = None,
        tol: float = 1e-4,
        max_iter: int = 30,
    ) -> Dam:
        """Scale the velocities so the fundamental frequency equals ``freq``.

        Parameters
        ----------
        freq : float
            Target frequency [Hz].
        site : :class:`Profile`, optional
            If None, the fundamental frequency of the dam on a rigid base is
            matched. If provided, the frequency of the peak nearest ``freq`` of
            the crest to half-space-outcrop transfer function of the joined
            dam and site is matched.
        tol : float, default=1e-4
            Relative tolerance on the frequency (coupled tuning only).
        max_iter : int, default=30
            Maximum number of iterations (coupled tuning only).

        Returns
        -------
        Dam
            A new dam; the velocity ratios between layers are unchanged.
        """
        if freq <= 0:
            raise ValueError("The target frequency must be positive.")

        # Frequency is proportional to velocity: exact for a rigid base
        scale = freq / self.fixed_base_freq()
        if site is None:
            return self.scaled(scale)

        scale0 = scale
        for _ in range(max_iter):
            dam = self.scaled(scale)
            f_peak = DamProfile(dam, site).peak_freq(f_guess=freq)
            if abs(f_peak / freq - 1) < tol:
                return dam
            scale *= freq / f_peak
            if not 0.1 < scale / scale0 < 10:
                raise RuntimeError(
                    "Coupled tuning diverged; the target frequency may not be "
                    "a resonance of the dam on this site."
                )
        warnings.warn(
            f"Coupled tuning did not converge to {tol:g} in {max_iter} iterations.",
            stacklevel=2,
        )
        return self.scaled(scale)


class DamProfile(Profile):
    """Dam joined to a site profile.

    Depths are measured from the crest. The dam layers come first, followed by
    the site layers; the last site layer is the half space. The total vertical
    stress at the top of the site includes the weight of the dam.

    Parameters
    ----------
    dam : :class:`Dam`
    site : :class:`Profile`
        Site profile under the dam. The last layer is the half space.
    wt_depth : float, optional
        Depth to the water table from the crest. By default the site water
        table, taken from the base of the dam.
    """

    def __init__(self, dam: Dam, site: Profile, wt_depth: float | None = None):
        self._n_dam = len(dam.layers)
        self._site_wt_depth = site.wt_depth
        self._slope = dam.slope
        self._crest_width = dam.crest_width
        if wt_depth is None:
            wt_depth = dam.height + site.wt_depth
        layers = [layer.copy() for layer in dam.layers] + [
            layer.copy() for layer in site
        ]
        super().__init__(layers, wt_depth)

    @property
    def dam(self) -> Dam:
        """The dam layers as they are now, e.g. after the velocities are varied."""
        return Dam(
            self.layers[: self._n_dam],
            slope=self._slope,
            crest_width=self._crest_width,
        )

    @property
    def site(self) -> Profile:
        """The site layers as they are now."""
        return Profile(
            [layer.copy() for layer in self.layers[self._n_dam :]],
            wt_depth=self._site_wt_depth,
        )

    @property
    def n_dam_layers(self) -> int:
        """Number of layers of the dam; the site layers follow them."""
        return self._n_dam

    @property
    def crest_offset(self) -> float:
        """Distance [m] from the apex of the wedge to the crest."""
        return self._crest_width / (2 * self._slope) if self._slope else 0.0

    @property
    def dam_height(self) -> float:
        """Height of the dam [m]: the total thickness of the dam layers."""
        return float(sum(layer.thickness for layer in self.layers[: self._n_dam]))

    def copy(self) -> DamProfile:
        """Return a copy of the profile with new layers."""
        n = self.n_dam_layers
        dam = Dam(
            [layer.copy() for layer in self.layers[:n]],
            slope=self._slope,
            crest_width=self._crest_width,
        )
        site = Profile(
            [layer.copy() for layer in self.layers[n:]],
            wt_depth=self._site_wt_depth,
        )
        return DamProfile(dam, site, self.wt_depth)

    def transfer_function(
        self, freqs: npt.ArrayLike, depth_out: float = 0.0, damped: bool = True
    ) -> np.ndarray:
        """Linear transfer function from the half-space outcrop to a depth.

        Parameters
        ----------
        freqs : array_like
            Frequencies [Hz].
        depth_out : float, default=0
            Depth from the crest [m]; 0 is the crest.
        damped : bool, default=True
            If the damping of the layers is included. If false, the
            transfer function of the undamped profile is returned.
        """
        from .propagation import DamLinearElasticCalculator

        profile = self
        if not damped:
            profile = self.copy()
            for layer in profile:
                layer.damping_min = 0.0
            profile.reset_layers()

        calc = DamLinearElasticCalculator()
        loc_in = profile.location("outcrop", index=len(profile) - 1)
        loc_out = profile.location("within", depth=depth_out)
        return calc.transfer_function(freqs, profile, loc_in, loc_out)

    def peak_freq(
        self, f_guess: float | None = None, f_range: tuple[float, float] | None = None
    ) -> float:
        """Frequency [Hz] of the peak of the crest transfer function nearest a guess.

        The transfer function is from the half-space outcrop to the crest. A
        soft foundation can add peaks below the dam's resonance, so the peak
        closest to ``f_guess`` (in log frequency) is selected.

        Parameters
        ----------
        f_guess : float, optional
            Frequency [Hz] about which to search; the search covers 0.2 to 5
            times this value. Default is the fixed-base frequency of the dam.
        f_range : tuple, optional
            Explicit search range [Hz]; overrides the default range.
        """
        f_guess = f_guess or self.dam.fixed_base_freq()
        if f_range is None:
            f_range = (0.2 * f_guess, 5 * f_guess)
        freqs = np.geomspace(*f_range, 1500)
        amp = np.abs(self.transfer_function(freqs))
        peaks = np.nonzero((amp[1:-1] > amp[:-2]) & (amp[1:-1] >= amp[2:]))[0] + 1
        if not len(peaks):
            raise ValueError("No peak found in the frequency range.")
        i = peaks[np.argmin(np.abs(np.log(freqs[peaks] / f_guess)))]
        # Refine the peak with a local quadratic fit
        local = np.linspace(freqs[i - 1], freqs[i + 1], 41)
        amp_local = np.abs(self.transfer_function(local))
        coef = np.polyfit(local, amp_local, 2)
        return float(-coef[1] / (2 * coef[0]))
