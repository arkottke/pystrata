"""Failure surfaces of a dam and the transfer function of the sliding mass.

A dam fails by sliding of the mass above a failure surface. The acceleration of
the mass that moves is its average acceleration, which is what the transfer
function of :cite:t:`Hale2019` describes. The shear-beam solution of the dam has an
acceleration at each depth, and the acceleration of the sliding mass is their
average, weighted by the width of the mass at each depth.

Coordinates are in the cross-section of the dam: ``x`` is horizontal, from the
centerline, and ``y`` is the depth below the crest, positive down. The
cross-section is symmetric, so a surface and its mirror image have the same
mass.
"""

from __future__ import annotations

from dataclasses import dataclass

import numpy as np
import numpy.typing as npt
from scipy import integrate

from .model import Dam, DamProfile


@dataclass(frozen=True)
class CircularSurface:
    """Circular failure surface, defined by its center and radius.

    The sliding mass is the part of the dam inside the circle.

    Parameters
    ----------
    x : float
        Horizontal position of the center from the centerline of the dam [m].
    y : float
        Depth of the center below the crest [m]. It is negative if the center is
        above the crest, which is usual for a slope failure.
    radius : float
        Radius of the circle [m].
    """

    x: float
    y: float
    radius: float

    def __post_init__(self):
        if not self.radius > 0:
            raise ValueError("The radius must be positive.")

    def mass_extent(
        self, dam: Dam, depths: npt.ArrayLike
    ) -> tuple[np.ndarray, np.ndarray]:
        """Horizontal extent of the sliding mass at depths below the crest.

        Parameters
        ----------
        dam : Dam
            Dam with a slope.
        depths : array_like
            Depths below the crest [m].

        Returns
        -------
        left, right : np.ndarray
            Horizontal limits of the mass [m]. Where there is no mass the limits
            are equal.
        """
        depths = np.asarray(depths, dtype=float)
        half_width = 0.5 * dam.width(depths)

        offset = depths - self.y
        with np.errstate(invalid="ignore"):
            half_chord = np.sqrt(np.maximum(self.radius**2 - offset**2, 0.0))
        left = np.maximum(self.x - half_chord, -half_width)
        right = np.minimum(self.x + half_chord, half_width)

        inside = (depths >= 0) & (depths <= dam.height) & (right > left)
        return np.where(inside, left, 0.0), np.where(inside, right, 0.0)

    def mass_width(self, dam: Dam, depths: npt.ArrayLike) -> np.ndarray:
        """Width [m] of the sliding mass at depths below the crest.

        Parameters
        ----------
        dam : Dam
            Dam with a slope.
        depths : array_like
            Depths below the crest [m].
        """
        left, right = self.mass_extent(dam, depths)
        return right - left

    def area(self, dam: Dam) -> float:
        """Area [m²] of the sliding mass in the cross-section of the dam."""
        area, _ = integrate.quad(
            lambda y: float(self.mass_width(dam, y)),
            0.0,
            dam.height,
            limit=500,
            epsabs=1e-10,
            epsrel=1e-10,
        )
        return float(area)


def toe_circle(dam: Dam, radius_ratio: float = 1.0) -> CircularSurface:
    """Create the circle through the toe and the far side of the crest.

    The failure surface passes through the toe of the slope that fails and the
    edge of the crest that is furthest from it, so the whole crest moves. Two
    points do not fix a circle; the third condition is the radius, as a multiple
    of the distance between them, and the center is above the dam.

    Parameters
    ----------
    dam : Dam
        Dam with a slope and, optionally, a crest width.
    radius_ratio : float, default=1
        Radius of the circle divided by the distance from the toe to the far
        side of the crest. It must be at least 0.5, which is the circle that has
        that distance as its diameter and is the deepest surface. A larger ratio
        is a flatter, shallower surface, which tends to the straight line from
        the toe to the far side of the crest.

    Returns
    -------
    CircularSurface
        The failure surface.
    """
    slope = dam._require_slope()
    if not radius_ratio >= 0.5:
        raise ValueError("The radius ratio must be at least 0.5.")

    crest_corner = np.array([-0.5 * dam.crest_width, 0.0])
    toe = np.array([0.5 * dam.crest_width + slope * dam.height, dam.height])
    chord = toe - crest_corner
    length = np.hypot(*chord)

    radius = radius_ratio * length
    middle = 0.5 * (crest_corner + toe)
    # Distance of the center from the middle of the chord, perpendicular to it and
    # on the side that is above the dam
    distance = np.sqrt(radius**2 - (0.5 * length) ** 2)
    normal = np.array([chord[1], -chord[0]]) / length
    center = middle + distance * normal

    return CircularSurface(float(center[0]), float(center[1]), float(radius))


def plot_failure_surface(
    dam: Dam,
    surface: CircularSurface,
    ax=None,
    show_mass: bool = True,
    **kwds,
):
    """Plot the cross-section of a dam and a failure surface.

    Parameters
    ----------
    dam : Dam
        Dam with a slope.
    surface : CircularSurface
        Failure surface.
    ax : matplotlib.axes.Axes, optional
        Axes to plot on. A new figure is created if not provided.
    show_mass : bool, default=True
        If the sliding mass is shaded.
    **kwds
        Keywords of :meth:`matplotlib.axes.Axes.plot` for the failure surface.

    Returns
    -------
    matplotlib.axes.Axes
        Axes with the plot. Depth increases downward and the aspect is equal.
    """
    import matplotlib.pyplot as plt

    if ax is None:
        _, ax = plt.subplots()

    height = dam.height
    half_crest, half_base = 0.5 * dam.crest_width, 0.5 * dam.base_width
    outline_x = [-half_crest, half_crest, half_base, -half_base, -half_crest]
    outline_y = [0.0, 0.0, height, height, 0.0]
    ax.plot(outline_x, outline_y, "k-", lw=1.5, label="Dam")

    if show_mass:
        depths = np.linspace(0.0, height, 400)
        left, right = surface.mass_extent(dam, depths)
        ax.fill_betweenx(
            depths,
            left,
            right,
            where=right > left,
            color="C1",
            alpha=0.3,
            lw=0,
            label="Sliding mass",
        )

    # The part of the circle that is inside the dam
    angles = np.linspace(0.0, 2 * np.pi, 2001)
    x = surface.x + surface.radius * np.cos(angles)
    y = surface.y + surface.radius * np.sin(angles)
    inside = (y >= 0) & (y <= height) & (np.abs(x) <= 0.5 * dam.width(np.maximum(y, 0)))
    kwds = {"color": "C3", "lw": 2, "label": "Failure surface"} | kwds
    ax.plot(np.where(inside, x, np.nan), np.where(inside, y, np.nan), **kwds)

    ax.set(
        xlabel="Distance from the centerline (m)", ylabel="Depth below the crest (m)"
    )
    ax.set_aspect("equal")
    ax.invert_yaxis()
    ax.grid(True, alpha=0.3)
    return ax


def sliding_mass_transfer_function(
    calc,
    loc_in,
    profile: DamProfile,
    surface: CircularSurface,
    n_depths: int = 101,
) -> np.ndarray:
    """Compute the transfer function to the average acceleration of the sliding mass.

    The acceleration transfer function from the input location to each depth of
    the dam is averaged, weighted by the width of the sliding mass at that depth.
    This assumes that the acceleration is the same across the width of the dam at
    a depth, as in the shear-beam solution.

    Parameters
    ----------
    calc : Dam*Calculator
        Dam calculator that has been run (or whose waves have been computed with
        ``transfer_function``). Equivalent-linear calculators give the
        strain-compatible transfer function.
    loc_in : Location
        Location of the input motion, e.g., the outcrop of the half space.
    profile : DamProfile
        Profile of the dam and site that `calc` was run for.
    surface : CircularSurface
        Failure surface.
    n_depths : int, default=101
        Number of slices of the dam, which are evaluated at their middle. The
        error is mainly where the surface leaves the dam.

    Returns
    -------
    np.ndarray
        Complex transfer function at the frequencies of `calc`.

    Raises
    ------
    ValueError
        If the failure surface does not cut the dam.
    """
    dam = profile.dam
    height = dam.height
    depths = (np.arange(n_depths) + 0.5) * height / n_depths

    weights = surface.mass_width(dam, depths)
    if not weights.sum() > 0:
        raise ValueError("The failure surface does not cut the dam.")

    total = 0.0
    for depth, weight in zip(depths, weights):
        if weight == 0:
            continue
        loc_out = profile.location("within", depth=float(depth))
        total = total + weight * calc.calc_accel_tf(loc_in, loc_out)
    return total / weights.sum()
