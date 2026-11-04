r"""Wave propagation through a shear-wedge dam coupled to a layered site.

Within the dam (a triangular wedge of unit-thickness shear beams whose width
is proportional to the distance ``r`` from the crest) a homogeneous layer
satisfies :math:`(1/r) \, d/dr \, (r G \, du/dr) + \rho \omega^2 u = 0`. The top
layer contains the crest, where the solution must be regular, so it is
:math:`u = c J_0(k r)`.
Below it the solution is written as up- and down-going Hankel waves, each
referenced to the top of its layer :math:`r_t`,

.. math::
    u(r) = A \, H^{(1)}_0(k r) e^{-i k r_t} + B \, H^{(2)}_0(k r) e^{i k r_t},

the wedge analogue of the :math:`A e^{ikz} + B e^{-ikz}` plane waves. These are
evaluated with the exponentially scaled Hankel functions so that the
amplitudes stay bounded in damped layers. The equivalent :math:`c_J J_0 +
c_Y Y_0` form needs two large terms to cancel and loses all precision at
moderate frequencies in damped, layered dams.

Displacement and shear stress are continuous across every interface; the same
conditions couple the base of the dam to the first site layer, which is a
standard plane-wave layer. This is the dam-on-elastic-rock model of
:cite:t:`gazetas1992seismic` generalized to a stack of dam layers and a layered site.
"""

from __future__ import annotations

import logging

import numpy as np
from scipy.special import hankel1e, hankel2e, jv

from .. import propagation
from ..motion import WaveField
from ..site import Location, Profile
from ..units import GRAVITY
from .model import WedgeLayer

logger = logging.getLogger(__name__)


def _wedge_state(wave_nums, coeff_a, coeff_b, r, r_top=None):
    """Displacement and strain term of a wedge layer at distance ``r``.

    Parameters
    ----------
    wave_nums : np.ndarray
        Complex wave number of the layer.
    coeff_a, coeff_b : np.ndarray
        Up- and down-going amplitudes referenced to ``r_top``. For the layer
        containing the crest (``r_top`` is ``None``) ``coeff_a`` multiplies
        :math:`J_0` and ``coeff_b`` is unused.
    r : float
        Distance from the crest.
    r_top : float, optional
        Distance from the crest to the top of the layer.

    Returns
    -------
    disp : np.ndarray
        Displacement ``u``.
    strain : np.ndarray
        ``-(du/dr) / k``; the shear stress is ``-G k strain``.
    """
    x = wave_nums * r
    if r_top is None:
        return coeff_a * jv(0, x), coeff_a * jv(1, x)

    shift = 1j * wave_nums * (r - r_top)
    up, down = np.exp(shift), np.exp(-shift)
    disp = coeff_a * hankel1e(0, x) * up + coeff_b * hankel2e(0, x) * down
    strain = coeff_a * hankel1e(1, x) * up + coeff_b * hankel2e(1, x) * down
    return disp, strain


def _top_of_layer(depths_base, i, crest_offset):
    """Distance from the apex to the top of wedge layer ``i``.

    It is ``None`` for the layer at the apex of a wedge that is not truncated.
    """
    if i > 0:
        return depths_base[i - 1]
    return crest_offset if crest_offset > 0 else None


def _calc_wedge_waves(wave_nums, shear_mods, depths_base, crest_offset=0.0):
    """Amplitudes ``(A, B)`` of each wedge layer, see :func:`_wedge_state`.

    Parameters
    ----------
    wave_nums, shear_mods : np.ndarray, shape (n_dam, n_freqs)
        Complex wave numbers and shear moduli of the wedge layers.
    depths_base : np.ndarray, shape (n_dam,)
        Distance from the apex of the wedge to the base of each layer.
    crest_offset : float, default=0
        Distance from the apex to the crest. If it is positive the wedge is
        truncated, and the crest is a free surface.
    """
    coeff_a = np.ones_like(wave_nums, dtype=complex)
    coeff_b = np.zeros_like(wave_nums, dtype=complex)
    if crest_offset > 0:
        # No shear stress at the crest, du/dr = 0, so the up-going wave is
        # reflected into a down-going one
        x_crest = wave_nums[0] * crest_offset
        coeff_b[0] = -hankel1e(1, x_crest) / hankel2e(1, x_crest)
    for i in range(len(wave_nums) - 1):
        r = depths_base[i]
        r_top = _top_of_layer(depths_base, i, crest_offset)
        disp, strain = _wedge_state(wave_nums[i], coeff_a[i], coeff_b[i], r, r_top)
        # Continuity of shear stress -> strain term in the next layer
        strain_next = (
            shear_mods[i] * wave_nums[i] / (shear_mods[i + 1] * wave_nums[i + 1])
        ) * strain

        # Solve for the amplitudes of the next layer, which starts at ``r``
        x = wave_nums[i + 1] * r
        h1_0, h2_0 = hankel1e(0, x), hankel2e(0, x)
        h1_1, h2_1 = hankel1e(1, x), hankel2e(1, x)
        det = h1_0 * h2_1 - h2_0 * h1_1
        coeff_a[i + 1] = (disp * h2_1 - h2_0 * strain_next) / det
        coeff_b[i + 1] = (h1_0 * strain_next - h1_1 * disp) / det
    return coeff_a, coeff_b


def _calc_site_waves(wave_a, wave_b, wave_nums, shear_mods, thicknesses):
    """Plane-wave amplitudes in the site layers given those at the top.

    Same recursion as :func:`pystrata.propagation._calc_waves_python`, but the
    amplitudes of the first layer are provided rather than set to unity.
    """
    n_layers = wave_nums.shape[0]
    waves_a = np.ones_like(wave_nums, dtype=complex)
    waves_b = np.ones_like(wave_nums, dtype=complex)
    waves_a[0], waves_b[0] = wave_a, wave_b
    for i in range(n_layers - 1):
        with np.errstate(invalid="ignore"):
            cimped = (wave_nums[i] * shear_mods[i]) / (
                wave_nums[i + 1] * shear_mods[i + 1]
            )
        cterm = 1j * wave_nums[i] * thicknesses[i]
        waves_a[i + 1] = 0.5 * waves_a[i] * (1 + cimped) * np.exp(
            cterm
        ) + 0.5 * waves_b[i] * (1 - cimped) * np.exp(-cterm)
        waves_b[i + 1] = 0.5 * waves_a[i] * (1 - cimped) * np.exp(
            cterm
        ) + 0.5 * waves_b[i] * (1 + cimped) * np.exp(-cterm)
        mask = ~np.isfinite(cimped)
        waves_a[i + 1, mask] = 1.0
        waves_b[i + 1, mask] = 1.0
    return waves_a, waves_b


class _DamWavesMixin:
    """Replaces the layered plane-wave solution with the dam-on-site solution.

    Used ahead of one of the linear-elastic based calculators in the method resolution
    order so that equivalent-linear iterations, strain transfer functions and outputs
    work unchanged.
    """

    def _check_profile(self, profile):
        if not isinstance(getattr(profile, "layers", None), list) or not any(
            isinstance(layer, WedgeLayer) for layer in profile
        ):
            raise ValueError("The profile must contain dam (WedgeLayer) layers.")

    def _calc_waves(self, angular_freqs, profile):
        # Bessel functions overflow at high frequencies in strongly damped
        # layers. The response there is negligible; the non-finite values
        # are returned as a zero transfer function like in the plane-wave code.
        with np.errstate(over="ignore", invalid="ignore", divide="ignore"):
            self._calc_waves_impl(angular_freqs, profile)

    def _calc_waves_impl(self, angular_freqs, profile):
        if self._incidence_angle or self._inhomogeneity:
            raise NotImplementedError(
                "Dam calculators only support vertically propagating waves."
            )
        self._check_profile(profile)

        n_layers = len(profile)
        n_dam = sum(isinstance(layer, WedgeLayer) for layer in profile)
        if n_dam == n_layers:
            raise ValueError("The profile must contain site layers below the dam.")
        n_freqs = len(angular_freqs)

        # Avoid evaluating Bessel functions at the singular zero frequency
        zero = np.isclose(angular_freqs, 0)
        freqs_safe = np.where(zero, 1.0, angular_freqs)

        comp_vels = np.empty((n_layers, n_freqs), dtype=complex)
        comp_mods = np.empty((n_layers, n_freqs), dtype=complex)
        for i, layer in enumerate(profile):
            comp_vels[i, :] = layer.comp_shear_vel
            comp_mods[i, :] = layer.comp_shear_mod
        wave_nums = freqs_safe / comp_vels

        # Distances in the solution are from the apex of the wedge, which is above
        # the crest if the dam has a crest width
        crest_offset = getattr(profile, "crest_offset", 0.0)
        depths_base = (
            np.array([layer.depth_base for layer in profile[:n_dam]]) + crest_offset
        )
        cj, cy = _calc_wedge_waves(
            wave_nums[:n_dam], comp_mods[:n_dam], depths_base, crest_offset
        )

        # Couple the base of the dam to the top of the site
        disp, strain = _wedge_state(
            wave_nums[n_dam - 1],
            cj[-1],
            cy[-1],
            depths_base[-1],
            _top_of_layer(depths_base, n_dam - 1, crest_offset),
        )
        imped = (comp_mods[n_dam - 1] * wave_nums[n_dam - 1]) / (
            comp_mods[n_dam] * wave_nums[n_dam]
        )
        wave_a = 0.5 * (disp + 1j * imped * strain)
        wave_b = 0.5 * (disp - 1j * imped * strain)

        thicknesses = profile.thickness[n_dam:]
        site_a, site_b = _calc_site_waves(
            wave_a, wave_b, wave_nums[n_dam:], comp_mods[n_dam:], thicknesses
        )

        waves_a = np.concatenate([cj[:n_dam], site_a])
        waves_b = np.concatenate([cy[:n_dam], site_b])

        # Rigid-body response at zero frequency
        wave_nums = np.where(zero, 0.0, wave_nums)
        waves_a[:n_dam, zero] = 2.0
        waves_a[n_dam:, zero] = 1.0
        waves_b[:n_dam, zero] = 0.0
        waves_b[n_dam:, zero] = 1.0

        self._waves_a, self._waves_b, self._wave_nums = waves_a, waves_b, wave_nums
        self._zero_freq = zero
        self._crest_offset = crest_offset

    def _wedge_wave(self, loc):
        """Displacement and strain of the wedge solution at a location."""
        # The location is at a depth below the crest; the solution is in the
        # distance from the apex
        r = loc.layer.depth + loc.depth_within + self._crest_offset
        with np.errstate(over="ignore", invalid="ignore", divide="ignore"):
            return self._wedge_wave_at(loc, r)

    def _wedge_wave_at(self, loc, r):
        if loc.index > 0:
            r_top = loc.layer.depth + self._crest_offset
        else:
            r_top = self._crest_offset if self._crest_offset > 0 else None
        disp, strain = _wedge_state(
            self._wave_nums[loc.index],
            self._waves_a[loc.index],
            self._waves_b[loc.index],
            r,
            r_top,
        )
        # The scaled Hankel functions are singular at zero frequency, where the
        # dam moves as a rigid body with the site
        zero = self._zero_freq
        return np.where(zero, 2.0, disp), np.where(
            zero, 0.0, -self._wave_nums[loc.index] * strain
        )

    def wave_at_location(self, loc: Location) -> np.ndarray:
        """Displacement amplitude at a location (dam or site)."""
        if not isinstance(loc.layer, WedgeLayer):
            return super().wave_at_location(loc)
        if loc.wave_field != WaveField.within:
            raise ValueError("Only 'within' locations are defined in the dam.")
        return self._wedge_wave(loc)[0]

    def calc_strain_tf(self, lin, lout):
        """Strain transfer function from the acceleration at ``lin``."""
        ang_freqs = self.motion.angular_freqs
        if isinstance(lout.layer, WedgeLayer):
            if lout.wave_field != WaveField.within:
                raise ValueError("Only 'within' locations are defined in the dam.")
            _, strain = self._wedge_wave(lout)
        else:
            k = self._wave_nums[lout.index]
            cterm = 1j * k * lout.depth_within
            with np.errstate(over="ignore", invalid="ignore"):
                strain = (
                    1j
                    * k
                    * (
                        self._waves_a[lout.index] * np.exp(cterm)
                        - self._waves_b[lout.index] * np.exp(-cterm)
                    )
                )
        with np.errstate(over="ignore", invalid="ignore", divide="ignore"):
            denom = -(ang_freqs**2) * self.wave_at_location(lin)
            mask = ~np.isclose(ang_freqs, 0)
            tf = np.zeros_like(mask, dtype=complex)
            tf[mask] = GRAVITY * strain[mask] / denom[mask]
        return np.where(np.isfinite(tf), tf, 0)

    def calc_accel_tf(self, lin, lout):
        """Acceleration transfer function; non-finite values are returned as 0."""
        with np.errstate(over="ignore", invalid="ignore", divide="ignore"):
            return super().calc_accel_tf(lin, lout)

    def transfer_function(self, freqs, profile: Profile, loc_in, loc_out):
        """Acceleration transfer function without running a motion.

        Parameters
        ----------
        freqs : array_like
            Frequencies [Hz].
        profile : :class:`~.model.DamProfile`
        loc_in, loc_out : :class:`~pystrata.site.Location`
        """
        self._calc_waves(2 * np.pi * np.asarray(freqs, dtype=float), profile)
        return self.calc_accel_tf(loc_in, loc_out)


class DamLinearElasticCalculator(_DamWavesMixin, propagation.LinearElasticCalculator):
    """Linear-elastic response of a dam on a layered site."""

    name = "DamLE"


class DamEquivalentLinearCalculator(
    _DamWavesMixin, propagation.EquivalentLinearCalculator
):
    """Equivalent-linear response of a dam on a layered site."""

    name = "DamEQL"


class DamFrequencyDependentEqlCalculator(
    _DamWavesMixin, propagation.FrequencyDependentEqlCalculator
):
    """Frequency-dependent equivalent-linear response of a dam on a site."""

    def _estimate_strains(self):
        eql = DamEquivalentLinearCalculator(
            strain_ratio=self.strain_ratio,
            strain_limit=self.strain_limit,
            tolerance=self.tolerance,
        )
        eql(self._motion, self._profile, self._loc_input)
