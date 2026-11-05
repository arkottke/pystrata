r"""Canyon (three-dimensional) effects on a shear-wedge dam.

The dam of :mod:`pystrata.dam.analytical` is infinitely long. In a canyon of
finite length :math:`L` with rigid abutments that move with the input motion the
displacement is :math:`u = u_b w(x, r)`, with :math:`w = 1` on the abutments.
Each homogeneous layer satisfies

.. math::
    \frac{1}{r} \frac{\partial}{\partial r}\left(r \frac{\partial w}{\partial r}\right)
    + \frac{\partial^2 w}{\partial x^2} + k^2 w = 0.

Writing :math:`w = 1 + v` and expanding :math:`v` in :math:`\sin(\kappa_j x)`,
:math:`\kappa_j = j \pi / L` with odd :math:`j`, gives for each term the Bessel
equation of the 2D wedge with the wave number :math:`k_j^2 = k^2 - \kappa_j^2`
and a constant forcing :math:`-k^2 P_j`, where :math:`P_j = 4 / (j \pi)`.
"""

from __future__ import annotations

from dataclasses import dataclass

import numpy as np
import numpy.typing as npt
from scipy.special import hankel1e, hankel2e, jv

from .analytical import _bessel_zeros, comp_shear_vel, wedge_truncated_natural_freqs


@dataclass(frozen=True)
class RectangularCanyon:
    """Canyon of constant length with rigid abutments.

    Parameters
    ----------
    length : float
        Length of the canyon along the axis of the dam [m].
    n_modes : int, default=25
        Number of odd longitudinal modes in the series of the response. Use more
        than :math:`2 f L / V_s` to resolve the response near the abutments.
    """

    length: float
    n_modes: int = 25

    def __post_init__(self):
        if not self.length > 0:
            raise ValueError("The canyon length must be positive.")
        if self.n_modes < 1:
            raise ValueError("At least one mode is required.")


def canyon_rect_natural_freqs(
    height: float,
    shear_vel: float,
    length: float,
    n: int = 3,
    n_long: int = 1,
    crest_offset: float = 0.0,
) -> np.ndarray:
    r"""Natural frequencies [Hz] of a homogeneous dam in a rectangular canyon.

    The base is rigid. The frequency is the 2D frequency :math:`f_n` of the
    wedge combined with the longitudinal half wave number :math:`j`,

    .. math::
        f = \sqrt{f_n^2 + (j V_s / 2 L)^2}.

    Parameters
    ----------
    height : float
        Height of the dam [m].
    shear_vel : float
        Shear-wave velocity [m/s].
    length : float
        Length of the canyon [m].
    n : int, default=3
        Number of vertical modes.
    n_long : int, default=1
        Longitudinal mode number :math:`j`; 1 is the fundamental.
    crest_offset : float, default=0
        Distance from the apex to the crest for a truncated wedge [m].

    Returns
    -------
    np.ndarray
        Frequencies of the first ``n`` vertical modes.
    """
    if crest_offset > 0:
        freqs_2d = wedge_truncated_natural_freqs(height, shear_vel, crest_offset, n)
    else:
        freqs_2d = _bessel_zeros(0.0, n) * shear_vel / (2 * np.pi * height)
    return np.sqrt(freqs_2d**2 + (n_long * shear_vel / (2 * length)) ** 2)


def canyon_freq_ratio(
    height: float, shear_vel: float, length: float, crest_offset: float = 0.0
) -> float:
    """Ratio of the fundamental frequency in the canyon to the 2D value."""
    f_3d = canyon_rect_natural_freqs(
        height, shear_vel, length, n=1, crest_offset=crest_offset
    )[0]
    f_2d = f_3d**2 - (shear_vel / (2 * length)) ** 2
    return float(f_3d / np.sqrt(f_2d))


def canyon_rect_rigid_base_tf(
    freqs: npt.ArrayLike,
    x: float,
    height: float,
    shear_vel: float,
    length: float,
    damping: float = 0.0,
    depth: float = 0.0,
    crest_offset: float = 0.0,
    n_modes: int = 25,
) -> np.ndarray:
    r"""Transfer function from the base to a point of a dam in a rectangular canyon.

    The dam is homogeneous with a rigid base. It is :math:`w(x, r)` of the module
    description, with :math:`w = 1 + \sum_j P_j \sin(\kappa_j x) R_j(r)`.

    Parameters
    ----------
    freqs : array_like
        Frequencies [Hz].
    x : float
        Position along the canyon, 0 to ``length`` [m].
    height : float
        Height of the dam [m].
    shear_vel : float
        Shear-wave velocity [m/s].
    length : float
        Length of the canyon [m].
    damping : float, default=0
        Damping ratio [decimal].
    depth : float, default=0
        Depth below the crest [m]; 0 is the crest.
    crest_offset : float, default=0
        Distance from the apex of the wedge to the crest [m]. If positive the
        wedge is truncated and the crest is a free surface.
    n_modes : int, default=25
        Number of odd longitudinal modes in the series.
    """
    freqs = np.asarray(freqs, dtype=float)
    if not 0 <= x <= length:
        raise ValueError("The position must be within the canyon.")
    r = crest_offset + depth
    r_base = crest_offset + height
    k = 2 * np.pi * freqs / comp_shear_vel(shear_vel, damping)
    w = np.ones_like(k, dtype=complex)
    with np.errstate(divide="ignore", invalid="ignore", over="ignore"):
        for j in range(1, 2 * n_modes, 2):
            kappa = j * np.pi / length
            part = 4 / (j * np.pi)
            kj = np.sqrt(k**2 - kappa**2 + 0j)
            if crest_offset > 0:
                # Free crest, dR/dr = 0. Hankel functions scaled as in
                # ``propagation._wedge_state`` stay bounded in evanescent modes
                x0 = kj * crest_offset
                ratio = -hankel1e(1, x0) / hankel2e(1, x0)

                def wave(dist, kj=kj, ratio=ratio):
                    shift = 1j * kj * (dist - crest_offset)
                    return hankel1e(0, kj * dist) * np.exp(shift) + ratio * hankel2e(
                        0, kj * dist
                    ) * np.exp(-shift)

                shape = wave(r) / wave(r_base)
            else:
                shape = jv(0, kj * r) / jv(0, kj * r_base)
            w = w + part * np.sin(kappa * x) * (k**2 / kj**2) * (shape - 1)
    # Rigid-body motion at zero frequency
    w = np.where(freqs == 0, 1.0, w)
    # Transfer function from the rigid base (= abutments) to the point
    return w
