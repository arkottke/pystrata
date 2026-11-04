"""Dam and joined dam-site models."""

from __future__ import annotations

import logging
import warnings

import numpy as np
import numpy.typing as npt
from scipy.optimize import brentq

from ..site import Layer, Profile, SoilType, layers_from_depth_power_law
from . import analytical

logger = logging.getLogger(__name__)


class WedgeLayer(Layer):
    """Layer of a triangular shear-wedge dam.

    The distance from the crest (``layer.depth + depth_within``) is the radial
    coordinate of the wedge solution. Otherwise identical to :class:`Layer`.
    """


class Dam:
    """Triangular (shear-wedge) dam made of stacked layers.

    The layers are ordered from the crest down. Each layer is a homogeneous
    segment of an infinitely long wedge (plane-strain, Gazetas and Dakoulas,
    1992); inhomogeneity is represented by layers with increasing velocity.

    Parameters
    ----------
    layers : list of :class:`Layer`
        Layers from the crest to the base. Plain layers are converted to
        :class:`WedgeLayer`.
    """

    def __init__(self, layers: list[Layer]):
        if not layers:
            raise ValueError("A dam requires at least one layer.")
        self.layers = [self._as_wedge(layer) for layer in layers]

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
    ) -> Dam:
        """Dam with a shear modulus proportional to ``z ** m``.

        Parameters
        ----------
        soil_type : SoilType
        height : float
            Height of the dam [m].
        shear_vel_avg : float
            Area-weighted average shear-wave velocity [m/s].
        m : float, default=0
            Inhomogeneity parameter, ``G ~ z ** m`` (``Vs ~ z ** (m / 2)``).
            ``m = 0`` is a homogeneous dam; Gazetas and Dakoulas consider 0 to 2/3.
        n_layers : int, default=20
            Number of layers of equal thickness.
        """
        vel_base = analytical.wedge_base_vel(shear_vel_avg, m)
        a = vel_base / height ** (m / 2)
        thicknesses = np.full(n_layers, height / n_layers)
        layers = layers_from_depth_power_law(
            soil_type, thicknesses, a, m / 2, layer_cls=WedgeLayer
        )
        return cls(layers)

    @property
    def thicknesses(self) -> np.ndarray:
        return np.array([layer.thickness for layer in self.layers])

    @property
    def shear_vels(self) -> np.ndarray:
        return np.array([layer.initial_shear_vel for layer in self.layers])

    @property
    def height(self) -> float:
        """Height of the dam [m]."""
        return float(self.thicknesses.sum())

    @property
    def shear_vel_avg(self) -> float:
        """Area-weighted average shear-wave velocity [m/s]."""
        depth_base = self.thicknesses.cumsum()
        depth_top = depth_base - self.thicknesses
        weights = (depth_base**2 - depth_top**2) / depth_base[-1] ** 2
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
            ]
        )

    def fixed_base_freqs(self, n: int = 1) -> np.ndarray:
        """First ``n`` natural frequencies [Hz] with a rigid (fixed) base.

        Computed for the undamped layered dam from the roots of the displacement at the
        base.
        """
        from .propagation import _calc_wedge_waves, _wedge_state

        thick = self.thicknesses
        depths_base = thick.cumsum()
        vels = self.shear_vels
        mods = np.array([layer.initial_shear_mod for layer in self.layers])
        height = depths_base[-1]

        def base_disp(freqs):
            wn = 2 * np.pi * np.asarray(freqs)[None, :] / vels[:, None]
            coeff_a, coeff_b = _calc_wedge_waves(
                wn + 0j, np.repeat(mods[:, None], wn.shape[1], 1) + 0j, depths_base
            )
            disp, _ = _wedge_state(
                wn[-1] + 0j,
                coeff_a[-1],
                coeff_b[-1],
                height,
                depths_base[-2] if len(depths_base) > 1 else None,
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
        self._dam = dam
        self._site = site
        if wt_depth is None:
            wt_depth = dam.height + site.wt_depth
        layers = [layer.copy() for layer in dam.layers] + [
            layer.copy() for layer in site
        ]
        super().__init__(layers, wt_depth)

    @property
    def dam(self) -> Dam:
        return self._dam

    @property
    def site(self) -> Profile:
        return self._site

    @property
    def n_dam_layers(self) -> int:
        return len(self._dam.layers)

    @property
    def dam_height(self) -> float:
        return self._dam.height

    def copy(self) -> DamProfile:
        n = self.n_dam_layers
        dam = Dam([layer.copy() for layer in self.layers[:n]])
        site = Profile(
            [layer.copy() for layer in self.layers[n:]], wt_depth=self._site.wt_depth
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
        """
        from .propagation import DamLinearElasticCalculator

        calc = DamLinearElasticCalculator()
        loc_in = self.location("outcrop", index=len(self) - 1)
        loc_out = self.location("within", depth=depth_out)
        return calc.transfer_function(freqs, self, loc_in, loc_out)

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
        f_guess = f_guess or self._dam.fixed_base_freq()
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
