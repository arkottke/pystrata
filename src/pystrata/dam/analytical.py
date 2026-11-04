r"""Closed-form shear-wedge solutions of :cite:t:`gazetas1992seismic`, Table 2.

The dam is an infinitely long (plane-strain) triangular wedge that deforms as
a shear beam. With the radial coordinate ``r`` measured from the crest and a
shear-wave velocity :math:`V_s = a r^{m/2}` (shear modulus proportional to
:math:`r^m`) the free-vibration modes are

.. math::
    u_n(r) = x^{-q} J_q(x), \quad x = \frac{\omega r^{c}}{a c}, \quad
    q = \frac{m}{2 - m}, \quad c = 1 - \frac{m}{2}.

For the homogeneous wedge (:math:`m = 0`) this reduces to
:math:`J_0(\omega r / V_s)`.
"""

from __future__ import annotations

import numpy as np
import numpy.typing as npt
from scipy.optimize import brentq
from scipy.special import gamma, jv, yv

from .. import site


def wedge_order(m: float) -> float:
    r"""Order :math:`q = m / (2 - m)` for :math:`G \propto z^m`.

    This is the order of the Bessel function of the mode shapes.
    """
    if not 0 <= m < 2:
        raise ValueError("The inhomogeneity parameter m must satisfy 0 <= m < 2.")
    return m / (2 - m)


def comp_shear_vel(shear_vel: npt.ArrayLike, damping: float) -> np.ndarray:
    """Complex shear-wave velocity with the pyStrata complex modulus model.

    Uses ``site.COMP_MODULUS_MODEL``, so it matches the layers of a profile.
    """
    return np.asarray(shear_vel) * np.sqrt(site.calc_comp_factor(damping))


def wedge_base_vel(shear_vel_avg: float, m: float = 0.0) -> float:
    """Shear-wave velocity at the base from the area-weighted average."""
    return shear_vel_avg * (2 + m / 2) / 2


def _bessel_zeros(order: float, count: int) -> np.ndarray:
    """First ``count`` positive zeros of ``J_order``."""
    step = 0.1
    roots: list[float] = []
    x0 = step
    f0 = jv(order, x0)
    while len(roots) < count:
        x1 = x0 + step
        f1 = jv(order, x1)
        if f0 * f1 < 0:
            roots.append(brentq(lambda x: jv(order, x), x0, x1, xtol=1e-14))
        x0, f0 = x1, f1
    return np.array(roots)


def wedge_natural_freqs(
    height: float, shear_vel_avg: float, m: float = 0.0, n: int = 3
) -> np.ndarray:
    r"""Natural frequencies [Hz] of the first ``n`` modes of the shear wedge.

    Parameters
    ----------
    height : float
        Height of the dam [m].
    shear_vel_avg : float
        Area-weighted average shear-wave velocity of the dam [m/s].
    m : float, default=0
        Shear modulus varies as :math:`z^m`.
    n : int, default=3
        Number of modes.

    Notes
    -----
    For a homogeneous dam (:math:`m = 0`) the first frequency is
    :math:`f_1 = 2.405 V_{s,base} / (2 \pi H)`, i.e., about
    :math:`V_s / (2.61 H)`, with a rigid base. Hale and Abrahamson (2019,
    :cite:p:`Hale2019`), as implemented in ``pygmm.slope_displacement``, relate
    the natural frequency of the dam to the shear-wave velocity by
    :math:`f = V_s / (3 H)`, which is an empirical relation for a dam on a
    flexible foundation; the two are not expected to agree.
    """
    q = wedge_order(m)
    c = 1 - m / 2
    vel_base = wedge_base_vel(shear_vel_avg, m)
    return _bessel_zeros(q, n) * c * vel_base / height / (2 * np.pi)


def wedge_mode_shape(
    depth: npt.ArrayLike,
    freq: float,
    height: float,
    shear_vel_avg: float,
    m: float = 0.0,
) -> np.ndarray:
    """Normalized displacement :math:`u(r) / u(0)` at frequency ``freq`` [Hz]."""
    q = wedge_order(m)
    c = 1 - m / 2
    vel_base = wedge_base_vel(shear_vel_avg, m)
    # x = omega r^c / (a c), with a = Vb / H^(m/2)
    scale = 2 * np.pi * freq * height ** (m / 2) / (vel_base * c)
    x = scale * np.asarray(depth, dtype=float) ** c
    with np.errstate(divide="ignore", invalid="ignore"):
        u = x ** (-q) * jv(q, x)
    u0 = 2.0 ** (-q) / gamma(q + 1)
    return np.where(x == 0, 1.0, u / u0)


def wedge_rigid_base_tf(
    freqs: npt.ArrayLike,
    height: float,
    shear_vel_avg: float,
    m: float = 0.0,
    damping: float = 0.0,
) -> np.ndarray:
    """Crest-to-base transfer function of a dam on a rigid base.

    This is :math:`1 / J_0(a_0)` for the homogeneous wedge (Table 2, column I).
    """
    freqs = np.asarray(freqs, dtype=float)
    q = wedge_order(m)
    c = 1 - m / 2
    vel_base = comp_shear_vel(wedge_base_vel(shear_vel_avg, m), damping)
    x = 2 * np.pi * freqs * height / (vel_base * c)
    with np.errstate(divide="ignore", invalid="ignore"):
        u_base = x ** (-q) * jv(q, x)
    u_crest = 2.0 ** (-q) / gamma(q + 1)
    return np.where(freqs == 0, 1.0, u_crest / u_base)


def wedge_elastic_base_tf(
    freqs: npt.ArrayLike,
    height: float,
    shear_vel: float,
    damping: float,
    imped_ratio: float,
) -> np.ndarray:
    r"""Crest-to-outcrop transfer function of a homogeneous dam on elastic rock.

    Table 2, column II: :math:`1 / (J_0(a_0) + i \alpha J_1(a_0))`, where :math:`\alpha`
    is the impedance ratio of the dam to the rock, :math:`(\rho V)_{dam} / (\rho
    V)_{rock}`, and :math:`a_0 = \omega H / V^*` (complex velocity, which includes
    damping). The rock is assumed elastic.
    """
    freqs = np.asarray(freqs, dtype=float)
    vel = comp_shear_vel(shear_vel, damping)
    a0 = 2 * np.pi * freqs * height / vel
    # The dam impedance is complex when damped; the rock is taken as elastic
    alpha = imped_ratio * vel / shear_vel
    return 1.0 / (jv(0, a0) + 1j * alpha * jv(1, a0))


# Truncated wedge: a homogeneous dam with a crest width ---------------------------
#
# The wedge is cut at a distance ``r0 = b / (2 s)`` from its apex, where ``b`` is
# the crest width and ``s`` the horizontal run per unit height of each slope, and
# the crest is a free surface. The solution is
# ``u(r) = J0(k r) + K Y0(k r)`` with ``K = -J1(k r0) / Y1(k r0)``, so that the
# shear stress is zero at the crest.


def _truncated_constant(wave_num: npt.ArrayLike, crest_offset: float) -> np.ndarray:
    r"""Constant :math:`K = -J_1(k r_0) / Y_1(k r_0)` of the truncated wedge."""
    x0 = np.asarray(wave_num) * crest_offset
    return -jv(1, x0) / yv(1, x0)


def wedge_truncated_rigid_base_tf(
    freqs: npt.ArrayLike,
    height: float,
    shear_vel: float,
    crest_offset: float,
    damping: float = 0.0,
) -> np.ndarray:
    r"""Crest-to-base transfer function of a truncated homogeneous dam on rock.

    The base is rigid. With :math:`R = r_0 + H`, the distance from the apex to the
    base, and the Wronskian of the Bessel functions the transfer function is

    .. math::
        \frac{-2 / (\pi k r_0)}{J_0(k R) Y_1(k r_0) - Y_0(k R) J_1(k r_0)},
        \quad k = \omega / V^*,

    which is :math:`1 / J_0(k H)`, :func:`wedge_rigid_base_tf`, as :math:`r_0 \to 0`.

    Parameters
    ----------
    freqs : array_like
        Frequencies [Hz].
    height : float
        Height of the dam from the crest to the base [m].
    shear_vel : float
        Shear-wave velocity [m/s].
    crest_offset : float
        Distance from the apex of the wedge to the crest, :math:`r_0 = b / (2 s)`
        [m]. It must be positive.
    damping : float, default=0
        Damping ratio [decimal].
    """
    if not crest_offset > 0:
        raise ValueError("The crest offset must be positive; see wedge_rigid_base_tf.")
    freqs = np.asarray(freqs, dtype=float)
    k = 2 * np.pi * freqs / comp_shear_vel(shear_vel, damping)
    x0, x_base = k * crest_offset, k * (crest_offset + height)
    with np.errstate(divide="ignore", invalid="ignore"):
        denom = jv(0, x_base) * yv(1, x0) - yv(0, x_base) * jv(1, x0)
        tf = (-2 / (np.pi * x0)) / denom
    return np.where(freqs == 0, 1.0, tf)


def wedge_truncated_elastic_base_tf(
    freqs: npt.ArrayLike,
    height: float,
    shear_vel: float,
    damping: float,
    imped_ratio: float,
    crest_offset: float,
) -> np.ndarray:
    r"""Crest-to-outcrop transfer function of a truncated dam on elastic rock.

    With :math:`K = -J_1(k r_0) / Y_1(k r_0)` and :math:`R = r_0 + H` it is

    .. math::
        \frac{J_0(k r_0) + K Y_0(k r_0)}
        {J_0(k R) + K Y_0(k R) + i \alpha \left[J_1(k R) + K Y_1(k R)\right]},

    where :math:`\alpha` is the impedance ratio of the dam to the rock. It is
    :func:`wedge_elastic_base_tf` as :math:`r_0 \to 0`. The rock is taken as
    elastic and the dam is homogeneous.

    Parameters
    ----------
    freqs : array_like
        Frequencies [Hz].
    height : float
        Height of the dam from the crest to the base [m].
    shear_vel : float
        Shear-wave velocity of the dam [m/s].
    damping : float
        Damping ratio of the dam [decimal].
    imped_ratio : float
        Impedance ratio of the dam to the rock, :math:`(\rho V)_{dam} /
        (\rho V)_{rock}`.
    crest_offset : float
        Distance from the apex of the wedge to the crest [m]. It must be positive.
    """
    if not crest_offset > 0:
        raise ValueError(
            "The crest offset must be positive; see wedge_elastic_base_tf."
        )
    freqs = np.asarray(freqs, dtype=float)
    vel = comp_shear_vel(shear_vel, damping)
    k = 2 * np.pi * freqs / vel
    x0, x_base = k * crest_offset, k * (crest_offset + height)
    alpha = imped_ratio * vel / shear_vel
    with np.errstate(divide="ignore", invalid="ignore"):
        const = _truncated_constant(k, crest_offset)
        crest = jv(0, x0) + const * yv(0, x0)
        disp = jv(0, x_base) + const * yv(0, x_base)
        strain = jv(1, x_base) + const * yv(1, x_base)
        tf = crest / (disp + 1j * alpha * strain)
    return np.where(freqs == 0, 1.0, tf)


def wedge_truncated_natural_freqs(
    height: float, shear_vel: float, crest_offset: float, n: int = 3
) -> np.ndarray:
    r"""Natural frequencies [Hz] of a truncated homogeneous dam on a rigid base.

    They are the roots of :math:`J_0(k R) Y_1(k r_0) - Y_0(k R) J_1(k r_0)`, with
    :math:`R = r_0 + H`, which tend to those of :func:`wedge_natural_freqs`, the
    roots of :math:`J_0`, as :math:`r_0 \to 0`.

    Parameters
    ----------
    height : float
        Height of the dam from the crest to the base [m].
    shear_vel : float
        Shear-wave velocity [m/s].
    crest_offset : float
        Distance from the apex of the wedge to the crest [m]. It must be positive.
    n : int, default=3
        Number of modes.
    """
    if not crest_offset > 0:
        raise ValueError("The crest offset must be positive; see wedge_natural_freqs.")
    r_base = crest_offset + height
    ratio = crest_offset / r_base

    def char(x: float) -> float:
        # In the dimensionless distance of the base, x = k R
        return jv(0, x) * yv(1, ratio * x) - yv(0, x) * jv(1, ratio * x)

    step = 0.05
    roots: list[float] = []
    x0 = step
    f0 = char(x0)
    while len(roots) < n:
        x1 = x0 + step
        f1 = char(x1)
        if f0 * f1 < 0:
            roots.append(brentq(char, x0, x1, xtol=1e-14))
        x0, f0 = x1, f1
    return np.array(roots) * shear_vel / (2 * np.pi * r_base)
