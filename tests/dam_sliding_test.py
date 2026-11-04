"""Failure surfaces of a dam and the transfer function of the sliding mass."""

import matplotlib
import numpy as np
import pytest
from numpy.testing import assert_allclose

import pystrata
from pystrata import dam, site

from . import FPATH_DATA

matplotlib.use("Agg")

SOIL = site.SoilType("dam", 20.0, None, 0.05)
ROCK = site.SoilType("rock", 25.0, None, 0.0)
FREQS = np.linspace(0.2, 20, 80)


def rock_site(vel=1500.0):
    return site.Profile([site.Layer(ROCK, 1.0, vel)])


def make_dam(height=100.0, slope=2.5, crest_width=10.0, vel=300.0, n_layers=4):
    layers = [site.Layer(SOIL, height / n_layers, vel) for _ in range(n_layers)]
    return dam.Dam(layers, slope=slope, crest_width=crest_width)


def toe_and_crest_corner(d):
    toe = np.array([0.5 * d.crest_width + d.slope * d.height, d.height])
    corner = np.array([-0.5 * d.crest_width, 0.0])
    return toe, corner


# toe_circle


@pytest.mark.parametrize("radius_ratio", [1.0, 1.5, 4.0])
@pytest.mark.parametrize("slope,crest_width", [(2.5, 10.0), (2.0, 0.0), (3.0, 30.0)])
def test_toe_circle_passes_through_the_toe_and_the_far_crest_corner(
    radius_ratio, slope, crest_width
):
    d = make_dam(slope=slope, crest_width=crest_width)
    surface = dam.toe_circle(d, radius_ratio)
    center = np.array([surface.x, surface.y])
    toe, corner = toe_and_crest_corner(d)

    assert np.linalg.norm(center - toe) == pytest.approx(surface.radius, rel=1e-12)
    assert np.linalg.norm(center - corner) == pytest.approx(surface.radius, rel=1e-12)
    # The radius is the ratio of that of the circle tangent to the base at the toe
    length = np.linalg.norm(toe - corner)
    assert surface.radius == pytest.approx(
        radius_ratio * length**2 / (2 * d.height), rel=1e-12
    )
    # The center is above the crest and the surface stays above the base
    assert surface.y < 0
    assert surface.y + surface.radius >= d.height - 1e-9


def test_toe_circle_of_ratio_one_is_tangent_to_the_base_at_the_toe():
    d = make_dam()
    surface = dam.toe_circle(d, 1.0)
    toe, _ = toe_and_crest_corner(d)
    assert_allclose([surface.x, surface.y + surface.radius], toe, atol=1e-9)


def test_toe_circle_moves_the_whole_crest():
    """The failure surface passes through the far side of the crest."""
    d = make_dam(crest_width=10.0)
    surface = dam.toe_circle(d)
    # At the crest the mass is as wide as the crest
    assert surface.mass_width(d, 1e-6) == pytest.approx(10.0, rel=1e-3)
    # Below it the surface is inside of the far slope, so the mass is narrower
    assert 0 < surface.mass_width(d, 1.0) < d.width(1.0)


def test_flatter_surfaces_hold_less_of_the_dam():
    """A larger radius ratio tends to the straight line from the toe to the crest."""
    d = make_dam()
    areas = [dam.toe_circle(d, ratio).area(d) for ratio in (1.0, 1.2, 2.0, 5.0)]
    assert np.all(np.diff(areas) < 0)
    # The straight line from the far side of the crest to the toe leaves a triangle
    # with the crest as its base and the height of the dam
    limit = dam.toe_circle(d, 1e6).area(d)
    assert limit == pytest.approx(0.5 * d.crest_width * d.height, rel=1e-3)
    assert areas[-1] > limit


def test_toe_circle_requires_a_slope_and_a_valid_ratio():
    pure = dam.Dam([site.Layer(SOIL, 100.0, 300.0)])
    with pytest.raises(ValueError, match="requires a slope"):
        dam.toe_circle(pure)
    for ratio in (0.0, 0.5, 0.9999, -1.0, np.nan):
        with pytest.raises(ValueError, match="at least 1"):
            dam.toe_circle(make_dam(), ratio)


# CircularSurface


def test_circle_must_have_a_radius():
    with pytest.raises(ValueError, match="radius must be positive"):
        dam.CircularSurface(0.0, 0.0, 0.0)


def test_circle_around_the_dam_holds_all_of_it():
    d = make_dam()
    surface = dam.CircularSurface(0.0, 50.0, 1000.0)
    depths = np.linspace(0, d.height, 11)
    assert_allclose(surface.mass_width(d, depths), d.width(depths))
    assert surface.area(d) == pytest.approx(d.area, rel=1e-9)


def test_circle_inside_a_wide_dam_has_the_area_of_a_circle():
    # A very slight slope, so the width is 100 m everywhere
    d = make_dam(height=50.0, slope=1e-9, crest_width=100.0)
    surface = dam.CircularSurface(0.0, 25.0, 10.0)
    assert surface.area(d) == pytest.approx(np.pi * 100.0, rel=1e-8)


def test_circle_away_from_the_dam_holds_nothing():
    d = make_dam()
    surface = dam.CircularSurface(1e4, 50.0, 10.0)
    assert_allclose(surface.mass_width(d, [10.0, 50.0]), 0.0)
    assert surface.area(d) == 0.0
    # Above the crest and below the base there is no dam
    above = dam.CircularSurface(0.0, -500.0, 100.0)
    assert above.area(d) == 0.0


def test_mass_area_agrees_with_a_grid():
    d = make_dam()
    surface = dam.toe_circle(d, 1.2)
    x = np.linspace(-0.5 * d.base_width, 0.5 * d.base_width, 4001)
    y = np.linspace(0.0, d.height, 2001)
    gx, gy = np.meshgrid(x, y)
    in_dam = np.abs(gx) <= 0.5 * d.width(gy)
    in_circle = (gx - surface.x) ** 2 + (gy - surface.y) ** 2 <= surface.radius**2
    cell = (x[1] - x[0]) * (y[1] - y[0])
    assert surface.area(d) == pytest.approx(np.sum(in_dam & in_circle) * cell, rel=2e-3)


def test_geometry_needs_a_slope():
    pure = dam.Dam([site.Layer(SOIL, 100.0, 300.0)])
    with pytest.raises(ValueError, match="requires a slope"):
        dam.CircularSurface(0.0, 0.0, 10.0).mass_width(pure, [1.0])


# plot_failure_surface


def test_plot_failure_surface():
    import matplotlib.pyplot as plt

    d = make_dam()
    surface = dam.toe_circle(d)
    ax = dam.plot_failure_surface(d, surface)

    labels = [line.get_label() for line in ax.get_lines()]
    assert "Dam" in labels and "Failure surface" in labels
    assert len(ax.collections) == 1  # the shaded sliding mass
    assert ax.yaxis_inverted()
    assert ax.get_aspect() == 1.0

    # The arc that is drawn is inside the dam and on the circle
    arc = next(line for line in ax.get_lines() if line.get_label() == "Failure surface")
    x, y = arc.get_data()
    keep = np.isfinite(x)
    assert keep.any()
    radius = np.hypot(x[keep] - surface.x, y[keep] - surface.y)
    assert_allclose(radius, surface.radius, rtol=1e-9)
    assert np.all((y[keep] >= 0) & (y[keep] <= d.height))
    plt.close(ax.figure)


def test_plot_options():
    import matplotlib.pyplot as plt

    d = make_dam()
    fig, ax = plt.subplots()
    returned = dam.plot_failure_surface(
        d, dam.toe_circle(d), ax=ax, show_mass=False, color="k"
    )
    assert returned is ax
    assert len(ax.collections) == 0
    plt.close(fig)


# sliding_mass_transfer_function


def weighted_average(d, profile, surface, n):
    """Independent calculation: transfer functions at depths, weighted by width."""
    depths = (np.arange(n) + 0.5) * d.height / n
    weights = surface.mass_width(d, depths)
    tfs = np.array([profile.transfer_function(FREQS, y) for y in depths])
    return (weights[:, None] * tfs).sum(axis=0) / weights.sum()


def test_sliding_mass_transfer_function_is_the_width_weighted_average():
    d = make_dam()
    profile = dam.DamProfile(d, rock_site())
    surface = dam.toe_circle(d)
    expected = weighted_average(d, profile, surface, 25)
    actual = profile.sliding_mass_transfer_function(FREQS, surface, n_depths=25)
    assert_allclose(actual, expected, rtol=1e-10)


def test_whole_dam_surface_weights_by_the_width_of_the_dam():
    d = make_dam()
    profile = dam.DamProfile(d, rock_site())
    surface = dam.CircularSurface(0.0, 50.0, 1e4)
    n = 25
    depths = (np.arange(n) + 0.5) * d.height / n
    tfs = np.array([profile.transfer_function(FREQS, y) for y in depths])
    expected = (d.width(depths)[:, None] * tfs).sum(axis=0) / d.width(depths).sum()
    assert_allclose(
        profile.sliding_mass_transfer_function(FREQS, surface, n), expected, rtol=1e-10
    )


def test_mass_near_the_crest_has_the_crest_transfer_function():
    d = make_dam()
    profile = dam.DamProfile(d, rock_site())
    # A circle that only holds the upper 5 cm of the dam; the response changes
    # with the square of the distance from the free surface at the crest
    surface = dam.CircularSurface(0.0, -1000.0, 1000.05)
    sliding = profile.sliding_mass_transfer_function(FREQS, surface, n_depths=2000)
    assert_allclose(sliding, profile.transfer_function(FREQS, 0.0), rtol=1e-3)


def test_sliding_mass_is_smoother_than_the_crest():
    """Averaging over the mass lowers the resonance peak below the crest value."""
    d = make_dam()
    profile = dam.DamProfile(d, rock_site())
    crest = np.abs(profile.transfer_function(np.linspace(0.2, 5, 400), 0.0))
    sliding = np.abs(
        profile.sliding_mass_transfer_function(
            np.linspace(0.2, 5, 400), dam.toe_circle(d)
        )
    )
    assert sliding.max() < crest.max()
    assert sliding.max() > 1.0


def test_surface_that_misses_the_dam_raises():
    d = make_dam()
    profile = dam.DamProfile(d, rock_site())
    with pytest.raises(ValueError, match="does not cut the dam"):
        profile.sliding_mass_transfer_function(
            FREQS, dam.CircularSurface(1e4, 50.0, 10.0)
        )


def test_works_with_an_equivalent_linear_calculator():
    """The function applies to any dam calculator that has been run."""
    d = make_dam(n_layers=3)
    profile = dam.DamProfile(d, rock_site())
    motion = pystrata.motion.TimeSeriesMotion.load_at2_file(
        FPATH_DATA / "NIS090.AT2", scale=0.3
    )
    loc_in = profile.location("outcrop", index=len(profile) - 1)
    surface = dam.toe_circle(d)

    calc = dam.DamEquivalentLinearCalculator()
    calc(motion, profile, loc_in)
    eql = dam.sliding_mass_transfer_function(calc, loc_in, profile, surface, 21)

    # The soil is linear, so this is the linear transfer function
    expected = profile.sliding_mass_transfer_function(motion.freqs, surface, 21)
    assert_allclose(eql, expected, rtol=1e-6)
