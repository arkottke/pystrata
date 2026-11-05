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


def _calc_wedge_waves(
    wave_nums, shear_mods, depths_base, crest_offset=0.0, disp_jumps=None, top_amp=1.0
):
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
    disp_jumps : np.ndarray, shape (n_dam - 1, n_freqs), optional
        Jump of the displacement added crossing the interface below each layer,
        e.g., the change of the particular solution of a canyon mode.
    top_amp : float, default=1
        Amplitude of the top layer. Zero gives the solution driven only by
        ``disp_jumps``.
    """
    coeff_a = np.ones_like(wave_nums, dtype=complex)
    coeff_a[0] = top_amp
    coeff_b = np.zeros_like(wave_nums, dtype=complex)
    if crest_offset > 0:
        # No shear stress at the crest, du/dr = 0, so the up-going wave is
        # reflected into a down-going one
        x_crest = wave_nums[0] * crest_offset
        coeff_b[0] = -top_amp * hankel1e(1, x_crest) / hankel2e(1, x_crest)
    for i in range(len(wave_nums) - 1):
        r = depths_base[i]
        r_top = _top_of_layer(depths_base, i, crest_offset)
        disp, strain = _wedge_state(wave_nums[i], coeff_a[i], coeff_b[i], r, r_top)
        if disp_jumps is not None:
            disp = disp + disp_jumps[i]
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


def _mode_wave_nums(wave_nums, kappa):
    """Wave numbers ``sqrt(k^2 - kappa^2)`` of a longitudinal canyon mode.

    The branch has a non-positive imaginary part, like the wave numbers of the damped
    layers, so a mode below its cutoff is evanescent.
    """
    mode_nums = np.sqrt(wave_nums**2 - kappa**2 + 0j)
    return np.where(mode_nums.imag > 0, -mode_nums, mode_nums)


class _DamWavesMixin:
    """Replaces the layered plane-wave solution with the dam-on-site solution.

    Used ahead of one of the linear-elastic based calculators in the method resolution
    order so that equivalent-linear iterations, strain transfer functions and outputs
    work unchanged.

    If the profile has a canyon, the response is that at the position ``canyon_x``
    [m] along the canyon, which is the center of the canyon by default.
    """

    canyon_x: float | None = None
    _canyon = None

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
        canyon = getattr(profile, "canyon", None)
        self._canyon = canyon
        self._crest_offset = crest_offset
        self._zero_freq = zero
        if canyon is not None:
            self._calc_canyon_waves(
                canyon,
                wave_nums,
                comp_mods,
                depths_base,
                n_dam,
                profile.thickness[n_dam:],
            )
            return

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

    def _resolve_canyon_x(self, canyon):
        x = canyon.length / 2 if self.canyon_x is None else self.canyon_x
        if not 0 <= x <= canyon.length:
            raise ValueError("The position must be within the canyon.")
        return float(x)

    def _calc_canyon_waves(
        self, canyon, wave_nums, comp_mods, depths_base, n_dam, thicknesses
    ):
        r"""Waves of the dam in a canyon on independent columns of the site.

        The displacement is :math:`w = 1 + \sum_j \sin(\kappa_j x) V_j(r)` in units of
        the motion of the abutments, which is that of the outcrop of the half space.
        Each column of the site below the dam is a 1D profile, so that the shear stress
        at the base is linear in the displacement, :math:`\tau = Z_1 u + Z_0 u_{out}`.
        This is a Robin condition for each mode, so the modes remain independent.
        """
        crest_offset = self._crest_offset
        zero = self._zero_freq
        n_freqs = wave_nums.shape[1]
        self._canyon_x = x = self._resolve_canyon_x(canyon)

        # Site columns as a linear relation at the base of the dam: the amplitude
        # of the up-going wave in the half space for a unit up-going (p) and
        # down-going (q) wave at the top of the site
        site_nums, site_mods = wave_nums[n_dam:], comp_mods[n_dam:]
        one = np.ones(n_freqs, dtype=complex)
        null = np.zeros(n_freqs, dtype=complex)
        p = _calc_site_waves(one, null, site_nums, site_mods, thicknesses)[0][-1]
        q = _calc_site_waves(null, one, site_nums, site_mods, thicknesses)[0][-1]
        ik_g = 1j * site_nums[0] * site_mods[0]
        z_out = ik_g / (p - q)
        z_disp = -ik_g * (p + q) / (p - q)

        dam_nums, dam_mods = wave_nums[:n_dam], comp_mods[:n_dam]
        r_base = depths_base[-1]
        r_top = _top_of_layer(depths_base, n_dam - 1, crest_offset)
        odds = np.arange(1, 2 * canyon.n_modes, 2)
        modes_a = np.zeros((len(odds), n_dam, n_freqs), dtype=complex)
        modes_b = np.zeros_like(modes_a)
        disp_base = np.ones(n_freqs, dtype=complex)
        for m, j in enumerate(odds):
            kappa = j * np.pi / canyon.length
            part = 4 / (j * np.pi)
            mode_nums = _mode_wave_nums(dam_nums, kappa)
            # Particular solution of each layer, which is constant
            vp = -part * dam_nums**2 / mode_nums**2
            hom_a, hom_b = _calc_wedge_waves(
                mode_nums, dam_mods, depths_base, crest_offset
            )
            par_a, par_b = _calc_wedge_waves(
                mode_nums,
                dam_mods,
                depths_base,
                crest_offset,
                disp_jumps=vp[:-1] - vp[1:],
                top_amp=0.0,
            )
            disp_h, strain_h = _wedge_state(
                mode_nums[-1], hom_a[-1], hom_b[-1], r_base, r_top
            )
            disp_p, strain_p = _wedge_state(
                mode_nums[-1], par_a[-1], par_b[-1], r_base, r_top
            )
            # Robin condition at the base, G dV/dr - Z1 V = P (Z1 + Z0), where
            # dV/dr = -k_j strain
            res_h = -dam_mods[-1] * mode_nums[-1] * strain_h - z_disp * disp_h
            res_p = -dam_mods[-1] * mode_nums[-1] * strain_p - z_disp * (
                disp_p + vp[-1]
            )
            coeff = (part * (z_disp + z_out) - res_p) / res_h
            mode_a = coeff * hom_a + par_a
            mode_b = coeff * hom_b + par_b
            # Modes that overflow are strongly evanescent: only the particular
            # solution is left
            bad = ~(np.isfinite(mode_a).all(0) & np.isfinite(mode_b).all(0))
            mode_a[:, bad], mode_b[:, bad] = 0.0, 0.0
            modes_a[m], modes_b[m] = mode_a, mode_b
            disp, _ = _wedge_state(mode_nums[-1], mode_a[-1], mode_b[-1], r_base, r_top)
            disp_base = disp_base + np.sin(kappa * x) * (disp + vp[-1])

        # Site column at the position, for the displacement at the base of the dam
        wave_a = (0.5 - q * disp_base) / (p - q)
        wave_b = disp_base - wave_a
        site_a, site_b = _calc_site_waves(
            wave_a, wave_b, site_nums, site_mods, thicknesses
        )
        # Rigid-body response at zero frequency, with an outcrop motion of one
        site_a[:, zero], site_b[:, zero] = 0.5, 0.5

        self._modes_a, self._modes_b = modes_a, modes_b
        self._mode_odds = odds
        self._waves_a = np.concatenate([np.zeros((n_dam, n_freqs), complex), site_a])
        self._waves_b = np.concatenate([np.zeros((n_dam, n_freqs), complex), site_b])
        self._wave_nums = np.where(zero, 0.0, wave_nums)

    def _canyon_field(self, loc, r):
        """Displacement, ``dw/dr`` and ``dw/dx`` in the dam at a distance ``r``."""
        canyon, x = self._canyon, self._canyon_x
        if loc.index > 0:
            r_top = loc.layer.depth + self._crest_offset
        else:
            r_top = self._crest_offset if self._crest_offset > 0 else None
        nums = self._wave_nums[loc.index]
        disp = np.ones(nums.shape, dtype=complex)
        d_dr = np.zeros_like(disp)
        d_dx = np.zeros_like(disp)
        for m, j in enumerate(self._mode_odds):
            kappa = j * np.pi / canyon.length
            mode_nums = _mode_wave_nums(nums, kappa)
            vp = -4 / (j * np.pi) * nums**2 / mode_nums**2
            mode_disp, mode_strain = _wedge_state(
                mode_nums,
                self._modes_a[m, loc.index],
                self._modes_b[m, loc.index],
                r,
                r_top,
            )
            mode_disp = mode_disp + vp
            disp = disp + np.sin(kappa * x) * mode_disp
            d_dr = d_dr - np.sin(kappa * x) * mode_nums * mode_strain
            d_dx = d_dx + kappa * np.cos(kappa * x) * mode_disp
        zero = self._zero_freq
        return (
            np.where(zero, 1.0, disp),
            np.where(zero, 0.0, d_dr),
            np.where(zero, 0.0, d_dx),
        )

    def _wedge_wave(self, loc):
        """Displacement and strain of the wedge solution at a location."""
        # The location is at a depth below the crest; the solution is in the
        # distance from the apex
        r = loc.layer.depth + loc.depth_within + self._crest_offset
        with np.errstate(over="ignore", invalid="ignore", divide="ignore"):
            return self._wedge_wave_at(loc, r)

    def _wedge_wave_at(self, loc, r):
        if self._canyon is not None:
            disp, d_dr, _ = self._canyon_field(loc, r)
            return disp, d_dr
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

    def calc_strain_xy_tf(self, lin, lout):
        r"""Longitudinal strain transfer function from the acceleration at ``lin``.

        It is the strain :math:`\partial u / \partial x` along the axis of the dam,
        which is zero without a canyon and in the site.
        """
        ang_freqs = self.motion.angular_freqs
        if self._canyon is None or not isinstance(lout.layer, WedgeLayer):
            return np.zeros(len(ang_freqs), dtype=complex)
        if lout.wave_field != WaveField.within:
            raise ValueError("Only 'within' locations are defined in the dam.")
        r = lout.layer.depth + lout.depth_within + self._crest_offset
        with np.errstate(over="ignore", invalid="ignore", divide="ignore"):
            _, _, d_dx = self._canyon_field(lout, r)
            denom = -(ang_freqs**2) * self.wave_at_location(lin)
            mask = ~np.isclose(ang_freqs, 0)
            tf = np.zeros_like(mask, dtype=complex)
            tf[mask] = GRAVITY * d_dx[mask] / denom[mask]
        return np.where(np.isfinite(tf), tf, 0)

    def _calc_strain_max(self, loc_input, loc_layer, motion, *args):
        """Peak shear strain, which includes the longitudinal one in a canyon."""
        if self._canyon is None or not isinstance(loc_layer.layer, WedgeLayer):
            return super()._calc_strain_max(loc_input, loc_layer, motion, *args)
        tf_rz = self.calc_strain_tf(loc_input, loc_layer)
        tf_xy = self.calc_strain_xy_tf(loc_input, loc_layer)
        if hasattr(motion, "calc_time_series"):
            ts_rz = motion.calc_time_series(tf_rz)
            ts_xy = motion.calc_time_series(tf_xy)
            return float(np.sqrt(ts_rz**2 + ts_xy**2).max())
        # Random vibration motions: the components are added as independent
        return motion.calc_peak(np.sqrt(np.abs(tf_rz) ** 2 + np.abs(tf_xy) ** 2))

    def calc_accel_tf(self, lin, lout):
        """Acceleration transfer function; non-finite values are returned as 0."""
        with np.errstate(over="ignore", invalid="ignore", divide="ignore"):
            return super().calc_accel_tf(lin, lout)

    def transfer_function(self, freqs, profile: Profile, loc_in, loc_out, x=None):
        """Acceleration transfer function without running a motion.

        Parameters
        ----------
        freqs : array_like
            Frequencies [Hz].
        profile : :class:`~.model.DamProfile`
        loc_in, loc_out : :class:`~pystrata.site.Location`
        x : float, optional
            Position [m] along the canyon, if the profile has one; the default
            is ``canyon_x``, which is the center of the canyon.
        """
        if x is not None:
            self.canyon_x = x
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
