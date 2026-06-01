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
"""Constitutive models for nonlinear time-domain site response analysis.

This module implements the MKZ (Matasovic-Vucetic) and HH (Hybrid Hyperbolic)
constitutive models for computing stress-strain relationships in soils.

References
----------
.. [1] Matasovic, N., & Vucetic, M. (1993). Cyclic characterization of
   liquefiable sands. Journal of Geotechnical Engineering, 119(11), 1805-1822.

.. [2] Shi, J., & Asimaki, D. (2017). From stiffness to strength: Formulation
   and validation of a hybrid hyperbolic nonlinear soil model for site-response
   analyses. Bulletin of the Seismological Society of America, 107(3), 1336-1355.
"""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import TYPE_CHECKING

import numpy as np
import numpy.typing as npt

if TYPE_CHECKING:
    pass

try:
    import numba

    HAS_NUMBA = True
except ImportError:
    HAS_NUMBA = False


@dataclass
class MKZParams:
    """Parameters for the Modified Kondner-Zelasko (MKZ) model.

    The MKZ model computes shear stress as::

        tau(gamma) = Gmax * gamma / (1 + beta * (|gamma| / gamma_ref)^s)

    Parameters
    ----------
    gamma_ref : float
        Reference shear strain (unit: 1, not %).
    beta : float
        Shape parameter controlling curve steepness.
    s : float
        Exponent controlling curve shape.
    shear_mod : float
        Initial (maximum) shear modulus [Pa].
    """

    gamma_ref: float
    beta: float
    s: float
    shear_mod: float

    def __post_init__(self):
        if self.gamma_ref <= 0:
            raise ValueError("gamma_ref must be positive")
        if self.shear_mod <= 0:
            raise ValueError("shear_mod must be positive")


@dataclass
class HHParams:
    """Parameters for the Hybrid Hyperbolic (HH) model.

    The HH model blends the MKZ model (low strain) with the FKZ model
    (high strain) using a transition function::

        tau_HH = w * tau_MKZ + (1 - w) * tau_FKZ

    where w is a transition function that varies from 1 (low strain)
    to 0 (high strain). The transition function is defined as::

        w = 1 - 1 / (1 + 10^(-a * (log10(|gamma|/gamma_t) - c1 * a^(-c2))))

    Parameters
    ----------
    gamma_t : float
        Transition strain where blending occurs (unit: 1).
    a : float
        Shape parameter controlling transition steepness.
    gamma_ref : float
        Reference strain for MKZ model (unit: 1).
    beta : float
        Shape parameter for MKZ model.
    s : float
        Exponent for MKZ model.
    shear_mod : float
        Initial (maximum) shear modulus [Pa].
    mu : float
        Shape parameter for FKZ model.
    shear_strength : float
        Shear strength of soil (Tmax) [Pa].
    d : float
        Exponent for FKZ model.
    trans_c1 : float, optional
        First transition function constant. Default 4.039 from Shi & Asimaki (2017).
    trans_c2 : float, optional
        Second transition function constant. Default 1.036 from Shi & Asimaki (2017).
    """

    gamma_t: float
    a: float
    gamma_ref: float
    beta: float
    s: float
    shear_mod: float
    mu: float
    shear_strength: float
    d: float
    trans_c1: float = 4.039
    trans_c2: float = 1.036

    def __post_init__(self):
        if self.gamma_t <= 0:
            raise ValueError("gamma_t must be positive")
        if self.gamma_ref <= 0:
            raise ValueError("gamma_ref must be positive")
        if self.shear_mod <= 0:
            raise ValueError("shear_mod must be positive")
        if self.shear_strength <= 0:
            raise ValueError("shear_strength must be positive")

    def to_mkz(self) -> MKZParams:
        """Extract MKZ parameters from HH parameters."""
        return MKZParams(
            gamma_ref=self.gamma_ref,
            beta=self.beta,
            s=self.s,
            shear_mod=self.shear_mod,
        )


@dataclass
class MultiLayerParams:
    """Container for constitutive model parameters for multiple layers.

    Parameters
    ----------
    params : list[MKZParams | HHParams]
        List of parameters for each layer.
    """

    params: list[MKZParams | HHParams] = field(default_factory=list)

    def __len__(self) -> int:
        return len(self.params)

    def __getitem__(self, index: int) -> MKZParams | HHParams:
        return self.params[index]

    def __iter__(self):
        return iter(self.params)

    def append(self, param: MKZParams | HHParams) -> None:
        self.params.append(param)

    @property
    def n_layers(self) -> int:
        return len(self.params)


# -----------------------------------------------------------------------------
# Pure Python implementations
# -----------------------------------------------------------------------------


def _tau_mkz_python(
    strain: npt.NDArray[np.floating],
    gamma_ref: float,
    beta: float,
    s: float,
    shear_mod: float,
) -> npt.NDArray[np.floating]:
    """Compute MKZ shear stress (pure Python).

    Parameters
    ----------
    strain : np.ndarray
        Shear strain array (unit: 1, not %).
    gamma_ref : float
        Reference shear strain.
    beta : float
        Shape parameter.
    s : float
        Exponent parameter.
    shear_mod : float
        Initial shear modulus [Pa].

    Returns
    -------
    stress : np.ndarray
        Shear stress [Pa].
    """
    return shear_mod * strain / (1 + beta * (np.abs(strain) / gamma_ref) ** s)


def _tau_fkz_python(
    strain: npt.NDArray[np.floating],
    shear_mod: float,
    mu: float,
    d: float,
    shear_strength: float,
) -> npt.NDArray[np.floating]:
    """Compute FKZ shear stress (pure Python).

    The FKZ model from Shi & Asimaki (2017) Eq. 6::

        tau = mu * Gmax * gamma^d / (1 + Gmax/Tmax * mu * |gamma|^d)

    Parameters
    ----------
    strain : np.ndarray
        Shear strain array (unit: 1).
    shear_mod : float
        Initial shear modulus [Pa].
    mu : float
        Shape parameter.
    d : float
        Exponent parameter.
    shear_strength : float
        Shear strength (Tmax) [Pa].

    Returns
    -------
    stress : np.ndarray
        Shear stress [Pa].
    """
    gamma_d = np.abs(strain) ** d
    return (
        mu
        * shear_mod
        * np.sign(strain)
        * gamma_d
        / (1 + shear_mod / shear_strength * mu * gamma_d)
    )


def _transition_function_python(
    strain: npt.NDArray[np.floating],
    a: float,
    gamma_t: float,
    trans_c1: float = 4.039,
    trans_c2: float = 1.036,
) -> npt.NDArray[np.floating]:
    """Compute HH transition function (pure Python).

    From Shi & Asimaki (2017) Eq. 7. The function transitions from 1
    (MKZ dominates at low strain) to 0 (FKZ dominates at high strain).

    Parameters
    ----------
    strain : np.ndarray
        Shear strain array (unit: 1).
    a : float
        Shape parameter controlling transition steepness.
    gamma_t : float
        Transition strain.
    trans_c1 : float, optional
        First transition function constant. Default 4.039.
    trans_c2 : float, optional
        Second transition function constant. Default 1.036.

    Returns
    -------
    w : np.ndarray
        Transition weights, ranging from 0 to 1.
    """
    w = np.zeros_like(strain)
    abs_strain = np.abs(strain)

    for i, g in enumerate(abs_strain):
        if g <= 0:
            w[i] = 1.0
            continue

        intermediate = np.log10(g / gamma_t) - trans_c1 * a ** (-trans_c2)
        exponent = -a * intermediate

        if exponent > 305:
            w[i] = 1.0
        elif exponent < -305:
            w[i] = 0.0
        else:
            w[i] = 1 - 1.0 / (1 + 10**exponent)

    return w


def _tau_hh_python(
    strain: npt.NDArray[np.floating],
    gamma_t: float,
    a: float,
    gamma_ref: float,
    beta: float,
    s: float,
    shear_mod: float,
    mu: float,
    shear_strength: float,
    d: float,
    trans_c1: float = 4.039,
    trans_c2: float = 1.036,
) -> npt.NDArray[np.floating]:
    """Compute HH shear stress (pure Python).

    Blends MKZ and FKZ models using a transition function.

    Parameters
    ----------
    strain : np.ndarray
        Shear strain array (unit: 1).
    gamma_t : float
        Transition strain.
    a : float
        Transition steepness parameter.
    gamma_ref : float
        Reference strain for MKZ.
    beta : float
        MKZ shape parameter.
    s : float
        MKZ exponent.
    shear_mod : float
        Initial shear modulus [Pa].
    mu : float
        FKZ shape parameter.
    shear_strength : float
        Shear strength [Pa].
    d : float
        FKZ exponent.
    trans_c1 : float, optional
        First transition function constant. Default 4.039.
    trans_c2 : float, optional
        Second transition function constant. Default 1.036.

    Returns
    -------
    stress : np.ndarray
        Shear stress [Pa].
    """
    w = _transition_function_python(strain, a, gamma_t, trans_c1, trans_c2)
    tau_mkz = _tau_mkz_python(strain, gamma_ref, beta, s, shear_mod)
    tau_fkz = _tau_fkz_python(strain, shear_mod, mu, d, shear_strength)

    return w * tau_mkz + (1 - w) * tau_fkz


# -----------------------------------------------------------------------------
# Numba-accelerated implementations
# -----------------------------------------------------------------------------


if HAS_NUMBA:

    @numba.njit(cache=True)
    def _tau_mkz_numba(
        strain: npt.NDArray[np.floating],
        gamma_ref: float,
        beta: float,
        s: float,
        shear_mod: float,
    ) -> npt.NDArray[np.floating]:
        """Compute MKZ shear stress (Numba-accelerated)."""
        n = len(strain)
        stress = np.empty(n, dtype=np.float64)
        for i in range(n):
            g = strain[i]
            stress[i] = shear_mod * g / (1 + beta * (abs(g) / gamma_ref) ** s)
        return stress

    @numba.njit(cache=True)
    def _tau_fkz_numba(
        strain: npt.NDArray[np.floating],
        shear_mod: float,
        mu: float,
        d: float,
        shear_strength: float,
    ) -> npt.NDArray[np.floating]:
        """Compute FKZ shear stress (Numba-accelerated)."""
        n = len(strain)
        stress = np.empty(n, dtype=np.float64)
        for i in range(n):
            g = strain[i]
            gamma_d = abs(g) ** d
            sign = 1.0 if g >= 0 else -1.0
            stress[i] = (
                mu
                * shear_mod
                * sign
                * gamma_d
                / (1 + shear_mod / shear_strength * mu * gamma_d)
            )
        return stress

    @numba.njit(cache=True)
    def _transition_function_numba(
        strain: npt.NDArray[np.floating],
        a: float,
        gamma_t: float,
        trans_c1: float = 4.039,
        trans_c2: float = 1.036,
    ) -> npt.NDArray[np.floating]:
        """Compute HH transition function (Numba-accelerated)."""
        n = len(strain)
        w = np.empty(n, dtype=np.float64)
        log10_gamma_t = np.log10(gamma_t)
        offset = trans_c1 * a ** (-trans_c2)

        for i in range(n):
            g = abs(strain[i])
            if g <= 0:
                w[i] = 1.0
                continue

            intermediate = np.log10(g) - log10_gamma_t - offset
            exponent = -a * intermediate

            if exponent > 305:
                w[i] = 1.0
            elif exponent < -305:
                w[i] = 0.0
            else:
                w[i] = 1 - 1.0 / (1 + 10**exponent)

        return w

    @numba.njit(cache=True)
    def _tau_hh_numba(
        strain: npt.NDArray[np.floating],
        gamma_t: float,
        a: float,
        gamma_ref: float,
        beta: float,
        s: float,
        shear_mod: float,
        mu: float,
        shear_strength: float,
        d: float,
        trans_c1: float = 4.039,
        trans_c2: float = 1.036,
    ) -> npt.NDArray[np.floating]:
        """Compute HH shear stress (Numba-accelerated)."""
        w = _transition_function_numba(strain, a, gamma_t, trans_c1, trans_c2)
        tau_mkz = _tau_mkz_numba(strain, gamma_ref, beta, s, shear_mod)
        tau_fkz = _tau_fkz_numba(strain, shear_mod, mu, d, shear_strength)

        n = len(strain)
        stress = np.empty(n, dtype=np.float64)
        for i in range(n):
            stress[i] = w[i] * tau_mkz[i] + (1 - w[i]) * tau_fkz[i]
        return stress

    @numba.njit(cache=True)
    def _calc_damping_from_stress_strain_numba(
        strain: npt.NDArray[np.floating],
        stress: npt.NDArray[np.floating],
        shear_mod: float,
    ) -> npt.NDArray[np.floating]:
        """Compute hysteretic damping from stress-strain backbone (Numba-
        accelerated)."""
        n = len(strain)
        area = np.zeros(n, dtype=np.float64)
        damping = np.zeros(n, dtype=np.float64)

        # Compute G/Gmax
        ggmax = np.ones(n, dtype=np.float64)
        for i in range(n):
            if strain[i] > 0:
                ggmax[i] = (stress[i] / strain[i]) / shear_mod

        # First point: triangle from origin
        area[0] = 0.5 * strain[0] * ggmax[0] * strain[0]
        if ggmax[0] > 0 and strain[0] > 0:
            damping[0] = (2.0 / np.pi) * (
                2.0 * area[0] / (ggmax[0] * strain[0] ** 2) - 1
            )

        for i in range(1, n):
            area[i] = area[i - 1] + 0.5 * (
                strain[i - 1] * ggmax[i - 1] + strain[i] * ggmax[i]
            ) * (strain[i] - strain[i - 1])

            if ggmax[i] > 0 and strain[i] > 0:
                damping[i] = (
                    2.0 / np.pi * (2.0 * area[i] / (ggmax[i] * strain[i] ** 2) - 1)
                )

        for i in range(n):
            if damping[i] < 0.0:
                damping[i] = 0.0

        return damping

    # Dispatch to numba versions
    _tau_mkz_dispatch = _tau_mkz_numba
    _tau_fkz_dispatch = _tau_fkz_numba
    _transition_function_dispatch = _transition_function_numba
    _tau_hh_dispatch = _tau_hh_numba
    _calc_damping_from_stress_strain_dispatch = _calc_damping_from_stress_strain_numba

else:
    _tau_mkz_dispatch = _tau_mkz_python
    _tau_fkz_dispatch = _tau_fkz_python
    _transition_function_dispatch = _transition_function_python
    _tau_hh_dispatch = _tau_hh_python
    _calc_damping_from_stress_strain_dispatch = None


# -----------------------------------------------------------------------------
# Public API
# -----------------------------------------------------------------------------


def calc_stress_mkz(
    strain: npt.ArrayLike,
    params: MKZParams,
) -> npt.NDArray[np.floating]:
    """Compute shear stress using the MKZ model.

    Parameters
    ----------
    strain : array_like
        Shear strain (unit: 1, not %).
    params : MKZParams
        MKZ model parameters.

    Returns
    -------
    stress : np.ndarray
        Shear stress [Pa].
    """
    strain = np.atleast_1d(np.asarray(strain, dtype=np.float64))
    return _tau_mkz_dispatch(
        strain, params.gamma_ref, params.beta, params.s, params.shear_mod
    )


def calc_stress_hh(
    strain: npt.ArrayLike,
    params: HHParams,
) -> npt.NDArray[np.floating]:
    """Compute shear stress using the HH model.

    Parameters
    ----------
    strain : array_like
        Shear strain (unit: 1, not %).
    params : HHParams
        HH model parameters.

    Returns
    -------
    stress : np.ndarray
        Shear stress [Pa].
    """
    strain = np.atleast_1d(np.asarray(strain, dtype=np.float64))
    return _tau_hh_dispatch(
        strain,
        params.gamma_t,
        params.a,
        params.gamma_ref,
        params.beta,
        params.s,
        params.shear_mod,
        params.mu,
        params.shear_strength,
        params.d,
        params.trans_c1,
        params.trans_c2,
    )


def calc_stress(
    strain: npt.ArrayLike,
    params: MKZParams | HHParams,
) -> npt.NDArray[np.floating]:
    """Compute shear stress using appropriate constitutive model.

    Parameters
    ----------
    strain : array_like
        Shear strain (unit: 1, not %).
    params : MKZParams | HHParams
        Model parameters.

    Returns
    -------
    stress : np.ndarray
        Shear stress [Pa].
    """
    if isinstance(params, HHParams):
        return calc_stress_hh(strain, params)
    elif isinstance(params, MKZParams):
        return calc_stress_mkz(strain, params)
    else:
        raise TypeError(f"Unknown parameter type: {type(params)}")


def calc_mod_reduc(
    strain: npt.ArrayLike,
    params: MKZParams | HHParams,
) -> npt.NDArray[np.floating]:
    """Compute modulus reduction (G/Gmax) from constitutive model.

    Parameters
    ----------
    strain : array_like
        Shear strain (unit: 1, not %).
    params : MKZParams | HHParams
        Model parameters.

    Returns
    -------
    mod_reduc : np.ndarray
        Modulus reduction ratio G/Gmax.
    """
    strain = np.atleast_1d(np.asarray(strain, dtype=np.float64))
    stress = calc_stress(strain, params)

    # G/Gmax = (tau/gamma) / Gmax = tau / (gamma * Gmax)
    # Handle zero strain
    with np.errstate(divide="ignore", invalid="ignore"):
        mod_reduc = stress / (strain * params.shear_mod)
        mod_reduc = np.where(np.abs(strain) < 1e-12, 1.0, mod_reduc)

    return mod_reduc


def _calc_damping_from_stress_strain_python(
    strain: npt.NDArray[np.floating],
    stress: npt.NDArray[np.floating],
    shear_mod: float,
) -> npt.NDArray[np.floating]:
    """Compute hysteretic damping from stress-strain backbone (pure Python)."""
    n = len(strain)
    ggmax = np.where(
        strain > 0,
        (stress / strain) / shear_mod,
        1.0,
    )

    area = np.zeros(n)
    damping = np.zeros(n)

    # First point: triangle from origin to (strain[0], stress[0])
    area[0] = 0.5 * strain[0] * ggmax[0] * strain[0]
    if ggmax[0] > 0 and strain[0] > 0:
        damping[0] = (2.0 / np.pi) * (2.0 * area[0] / (ggmax[0] * strain[0] ** 2) - 1)

    for i in range(1, n):
        area[i] = area[i - 1] + 0.5 * (
            strain[i - 1] * ggmax[i - 1] + strain[i] * ggmax[i]
        ) * (strain[i] - strain[i - 1])

        if ggmax[i] > 0 and strain[i] > 0:
            damping[i] = 2.0 / np.pi * (2.0 * area[i] / (ggmax[i] * strain[i] ** 2) - 1)

    damping = np.maximum(damping, 0.0)
    return damping


def calc_damping_from_stress_strain(
    strain: npt.NDArray[np.floating],
    stress: npt.NDArray[np.floating],
    shear_mod: float,
) -> npt.NDArray[np.floating]:
    """Compute hysteretic damping ratio from stress-strain backbone curve.

    Uses the Masing rule to compute the energy dissipated per cycle
    and relates it to damping ratio.

    Parameters
    ----------
    strain : np.ndarray
        Shear strain array (monotonically increasing, unit: 1).
    stress : np.ndarray
        Corresponding shear stress array [Pa].
    shear_mod : float
        Initial shear modulus [Pa].

    Returns
    -------
    damping : np.ndarray
        Damping ratio (unit: 1, not %).

    Notes
    -----
    The damping ratio is computed as::

        xi = (2/pi) * (2 * W / (G/Gmax * gamma^2) - 1)

    where W is the area under the backbone curve (integrated as the
    normalised product ``strain * G/Gmax``).
    """
    if _calc_damping_from_stress_strain_dispatch is not None:
        return _calc_damping_from_stress_strain_dispatch(strain, stress, shear_mod)
    return _calc_damping_from_stress_strain_python(strain, stress, shear_mod)


def calc_damping(
    strain: npt.ArrayLike,
    params: MKZParams | HHParams,
) -> npt.NDArray[np.floating]:
    """Compute damping ratio from constitutive model using Masing rules.

    Parameters
    ----------
    strain : array_like
        Shear strain (unit: 1, not %).
    params : MKZParams | HHParams
        Model parameters.

    Returns
    -------
    damping : np.ndarray
        Damping ratio (unit: 1, not %).
    """
    strain = np.atleast_1d(np.asarray(strain, dtype=np.float64))

    # Ensure positive strain for backbone curve
    strain_pos = np.abs(strain)

    # Sort to ensure monotonic
    idx_sort = np.argsort(strain_pos)
    strain_sorted = strain_pos[idx_sort]

    # Compute stress for sorted positive strains
    stress_sorted = np.abs(calc_stress(strain_sorted, params))

    # Compute damping
    damping_sorted = calc_damping_from_stress_strain(
        strain_sorted, stress_sorted, params.shear_mod
    )

    # Map back to original order
    damping = np.empty_like(strain)
    damping[idx_sort] = damping_sorted

    return damping


# -----------------------------------------------------------------------------
# Fused misfit functions for curve fitting
# -----------------------------------------------------------------------------


def _hh_misfit_python(
    strain: npt.NDArray[np.floating],
    mod_reduc_target: npt.NDArray[np.floating],
    damping_target: npt.NDArray[np.floating],
    gamma_t: float,
    a: float,
    gamma_ref: float,
    beta: float,
    s: float,
    shear_mod: float,
    mu: float,
    shear_strength: float,
    d: float,
    trans_c1: float,
    trans_c2: float,
) -> float:
    """Compute combined mod-reduc + damping MSE for HH model (pure Python)."""
    stress = _tau_hh_python(
        strain,
        gamma_t,
        a,
        gamma_ref,
        beta,
        s,
        shear_mod,
        mu,
        shear_strength,
        d,
        trans_c1,
        trans_c2,
    )
    damping = _calc_damping_from_stress_strain_python(strain, stress, shear_mod)

    n = len(strain)
    err_mod = 0.0
    err_damp = 0.0
    for i in range(n):
        if strain[i] > 0:
            mr = (stress[i] / strain[i]) / shear_mod
        else:
            mr = 1.0
        err_mod += (mr - mod_reduc_target[i]) ** 2
        if i > 0:
            err_damp += (damping[i] - damping_target[i]) ** 2

    err_mod /= n
    if n > 1:
        err_damp /= n - 1
    return err_mod + err_damp


def _mkz_damping_misfit_python(
    strain: npt.NDArray[np.floating],
    damping_target: npt.NDArray[np.floating],
    gamma_ref: float,
    beta: float,
    s: float,
    shear_mod: float,
) -> float:
    """Compute damping MSE for MKZ model (pure Python)."""
    stress = _tau_mkz_python(strain, gamma_ref, beta, s, shear_mod)
    damping = _calc_damping_from_stress_strain_python(strain, stress, shear_mod)

    n = len(strain)
    err = 0.0
    for i in range(1, n):
        err += (damping[i] - damping_target[i]) ** 2

    if n > 1:
        err /= n - 1
    return err


if HAS_NUMBA:

    @numba.njit(cache=True)
    def _hh_misfit_numba(
        strain: npt.NDArray[np.floating],
        mod_reduc_target: npt.NDArray[np.floating],
        damping_target: npt.NDArray[np.floating],
        gamma_t: float,
        a: float,
        gamma_ref: float,
        beta: float,
        s: float,
        shear_mod: float,
        mu: float,
        shear_strength: float,
        d: float,
        trans_c1: float,
        trans_c2: float,
    ) -> float:
        """Compute combined mod-reduc + damping MSE for HH model (Numba)."""
        n = len(strain)

        # --- compute stress and G/Gmax in one pass ---
        stress = np.empty(n)
        ggmax = np.ones(n)
        for i in range(n):
            g = abs(strain[i])
            # MKZ component
            tau_mkz = shear_mod * strain[i] / (1 + beta * (g / gamma_ref) ** s)
            # FKZ component
            gamma_d = g**d
            sign = 1.0 if strain[i] >= 0 else -1.0
            tau_fkz = (
                mu
                * shear_mod
                * sign
                * gamma_d
                / (1 + shear_mod / shear_strength * mu * gamma_d)
            )
            # Transition function
            if g <= 0:
                w = 1.0
            else:
                intermediate = np.log10(g / gamma_t) - trans_c1 * a ** (-trans_c2)
                exponent = -a * intermediate
                if exponent > 305:
                    w = 1.0
                elif exponent < -305:
                    w = 0.0
                else:
                    w = 1 - 1.0 / (1 + 10**exponent)
            stress[i] = w * tau_mkz + (1 - w) * tau_fkz
            if strain[i] > 0:
                ggmax[i] = (stress[i] / strain[i]) / shear_mod

        # --- Masing damping (cumulative area) ---
        area = np.zeros(n)
        damping = np.zeros(n)
        area[0] = 0.5 * strain[0] * ggmax[0] * strain[0]
        if ggmax[0] > 0 and strain[0] > 0:
            damping[0] = (2.0 / np.pi) * (
                2.0 * area[0] / (ggmax[0] * strain[0] ** 2) - 1
            )
        for i in range(1, n):
            area[i] = area[i - 1] + 0.5 * (
                strain[i - 1] * ggmax[i - 1] + strain[i] * ggmax[i]
            ) * (strain[i] - strain[i - 1])
            if ggmax[i] > 0 and strain[i] > 0:
                damping[i] = (
                    2.0 / np.pi * (2.0 * area[i] / (ggmax[i] * strain[i] ** 2) - 1)
                )
            if damping[i] < 0.0:
                damping[i] = 0.0

        # --- MSE ---
        err_mod = 0.0
        err_damp = 0.0
        for i in range(n):
            err_mod += (ggmax[i] - mod_reduc_target[i]) ** 2
            if i > 0:
                err_damp += (damping[i] - damping_target[i]) ** 2
        err_mod /= n
        if n > 1:
            err_damp /= n - 1
        return err_mod + err_damp

    @numba.njit(cache=True)
    def _mkz_damping_misfit_numba(
        strain: npt.NDArray[np.floating],
        damping_target: npt.NDArray[np.floating],
        gamma_ref: float,
        beta: float,
        s: float,
        shear_mod: float,
    ) -> float:
        """Compute damping MSE for MKZ model (Numba)."""
        n = len(strain)
        # Compute stress and G/Gmax
        ggmax = np.ones(n)
        for i in range(n):
            g = abs(strain[i])
            stress_i = shear_mod * strain[i] / (1 + beta * (g / gamma_ref) ** s)
            if strain[i] > 0:
                ggmax[i] = (stress_i / strain[i]) / shear_mod

        # Masing damping
        area = np.zeros(n)
        damping = np.zeros(n)
        area[0] = 0.5 * strain[0] * ggmax[0] * strain[0]
        if ggmax[0] > 0 and strain[0] > 0:
            damping[0] = (2.0 / np.pi) * (
                2.0 * area[0] / (ggmax[0] * strain[0] ** 2) - 1
            )
        for i in range(1, n):
            area[i] = area[i - 1] + 0.5 * (
                strain[i - 1] * ggmax[i - 1] + strain[i] * ggmax[i]
            ) * (strain[i] - strain[i - 1])
            if ggmax[i] > 0 and strain[i] > 0:
                damping[i] = (
                    2.0 / np.pi * (2.0 * area[i] / (ggmax[i] * strain[i] ** 2) - 1)
                )
            if damping[i] < 0.0:
                damping[i] = 0.0

        # MSE (skip first point)
        err = 0.0
        for i in range(1, n):
            err += (damping[i] - damping_target[i]) ** 2
        if n > 1:
            err /= n - 1
        return err

    hh_misfit = _hh_misfit_numba
    mkz_damping_misfit = _mkz_damping_misfit_numba
else:
    hh_misfit = _hh_misfit_python
    mkz_damping_misfit = _mkz_damping_misfit_python


# -----------------------------------------------------------------------------
# Shear Strength Estimation Functions
# -----------------------------------------------------------------------------

# Default values for shear strength estimation
DEFAULT_FRICTION_ANGLE = 30.0  # degrees
DEFAULT_DYNA_COEFF = 1.2  # dynamic coefficient for strain rate


def calc_lateral_pressure_coeff(
    ocr: npt.ArrayLike,
    friction_angle: float = DEFAULT_FRICTION_ANGLE,
) -> npt.NDArray[np.floating]:
    """Calculate lateral earth pressure coefficient at rest (K0) from OCR.

    Uses the empirical formula by Mayne & Kulhawy (1982).

    Parameters
    ----------
    ocr : array_like
        Over-consolidation ratio.
    friction_angle : float, optional
        Effective internal friction angle in degrees.
        Default is 30.0 degrees.

    Returns
    -------
    k0 : np.ndarray
        Lateral earth pressure coefficient at rest.
    """
    ocr = np.atleast_1d(np.asarray(ocr, dtype=np.float64))
    phi_rad = np.deg2rad(friction_angle)
    return (1 - np.sin(phi_rad)) * ocr ** np.sin(phi_rad)


def calc_shear_strength(
    vs: npt.ArrayLike,
    ocr: npt.ArrayLike,
    stress_vert: npt.ArrayLike,
    k0: npt.ArrayLike | None = None,
    friction_angle: float = DEFAULT_FRICTION_ANGLE,
    dyna_coeff: float = DEFAULT_DYNA_COEFF,
    vs_threshold: float = 760.0,
) -> npt.NDArray[np.floating]:
    """Estimate shear strength of soil layers.

    Uses undrained shear strength (Ladd 1991) for soft soils (Vs <= vs_threshold)
    and Mohr-Coulomb criterion for stiffer soils.

    Parameters
    ----------
    vs : array_like
        Shear wave velocity [m/s] for each layer.
    ocr : array_like
        Over-consolidation ratio for each layer.
    stress_vert : array_like
        Vertical effective stress [Pa] for each layer.
    k0 : array_like, optional
        Lateral earth pressure coefficient. If None, computed from OCR
        and friction_angle using Mayne & Kulhawy (1982).
    friction_angle : float, optional
        Effective internal friction angle in degrees.
        Default is 30.0 degrees.
    dyna_coeff : float, optional
        Dynamic coefficient to account for strain rate effects.
        Default is 1.2 (based on Vardanega & Bolton 2013, assuming
        strain rate of 0.01 s^-1).
    vs_threshold : float, optional
        Shear wave velocity threshold [m/s] below which undrained strength
        is used. Default is 760 m/s.

    Returns
    -------
    shear_strength : np.ndarray
        Estimated shear strength [Pa] for each layer.

    Notes
    -----
    For soft soils (Vs <= vs_threshold), uses Ladd (1991):
        Su = dyna_coeff * 0.28 * OCR^0.8 * sigma_v0

    For stiffer soils (Vs > vs_threshold), uses Mohr-Coulomb criterion
    based on the stress state and friction angle.

    References
    ----------
    .. [1] Ladd, C. C. (1991). Stability evaluation during staged construction.
       Journal of Geotechnical Engineering, 117(4), 540-615.
    .. [2] Vardanega, P. J., & Bolton, M. D. (2013). Strength mobilization
       in clays and silts. Canadian Geotechnical Journal, 50(7), 749-763.
    """
    vs = np.atleast_1d(np.asarray(vs, dtype=np.float64))
    ocr = np.atleast_1d(np.asarray(ocr, dtype=np.float64))
    stress_vert = np.atleast_1d(np.asarray(stress_vert, dtype=np.float64))

    n = len(vs)
    if k0 is None:
        k0 = calc_lateral_pressure_coeff(ocr, friction_angle)
    else:
        k0 = np.atleast_1d(np.asarray(k0, dtype=np.float64))

    phi_rad = np.deg2rad(friction_angle)
    shear_strength = np.zeros(n, dtype=np.float64)

    for j in range(n):
        if vs[j] <= vs_threshold:
            # Soft soils: undrained shear strength (Ladd 1991)
            shear_strength[j] = dyna_coeff * 0.28 * ocr[j] ** 0.8 * stress_vert[j]
        else:
            # Stiffer soils: Mohr-Coulomb criterion
            stress_h = k0[j] * stress_vert[j]
            sigma_1 = max(stress_vert[j], stress_h)  # largest principal stress
            sigma_3 = min(stress_vert[j], stress_h)  # smallest principal stress

            # Normal effective stress on slip plane
            sigma_n = (sigma_1 + sigma_3) / 2.0 - (sigma_1 - sigma_3) / 2.0 * np.sin(
                phi_rad
            )

            shear_strength[j] = dyna_coeff * sigma_n * np.tan(phi_rad)

    return shear_strength


def calc_ocr_from_vs(
    vs: npt.ArrayLike,
    density: npt.ArrayLike,
    stress_vert: npt.ArrayLike,
    ocr_max: float | None = None,
) -> npt.NDArray[np.floating]:
    """Estimate over-consolidation ratio (OCR) from shear wave velocity.

    Uses the empirical formula by Mayne, Robertson & Lunne (1998).

    Parameters
    ----------
    vs : array_like
        Shear wave velocity [m/s].
    density : array_like
        Mass density [kg/m³] (not used directly but included for API consistency).
    stress_vert : array_like
        Vertical effective stress [Pa].
    ocr_max : float, optional
        Maximum allowable OCR value. If None, no limit is applied.

    Returns
    -------
    ocr : np.ndarray
        Estimated over-consolidation ratio.

    References
    ----------
    .. [1] Mayne, P. W., Robertson, P. K., & Lunne, T. (1998). Clay stress
       history evaluated from seismic piezocone tests. Geotechnical Site
       Characterization, 2, 1113-1118.
    """
    vs = np.atleast_1d(np.asarray(vs, dtype=np.float64))
    stress_vert = np.atleast_1d(np.asarray(stress_vert, dtype=np.float64))

    # Pre-consolidation pressure from Mayne et al. (1998): sigma_p = 0.106 * Vs^1.47 [kPa]
    sigma_p = 0.106 * vs**1.47 * 1000  # Convert kPa to Pa

    ocr = sigma_p / stress_vert

    if ocr_max is not None:
        ocr = np.minimum(ocr, ocr_max)

    return ocr
