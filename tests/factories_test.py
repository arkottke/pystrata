#!/usr/bin/env python
"""Test factories module."""

# This program is free software; you can redistribute it and/or modify
# it under the terms of the GNU General Public License as published by
# the Free Software Foundation; either version 2 of the License, or
# (at your option) any later version.
#
# This program is distributed in the hope that it will be useful,
# but WITHOUT ANY WARRANTY; without even the implied warranty of
# MERCHANTABILITY or FITNESS FOR A PARTICULAR PURPOSE.  See the
# GNU General Public License for more details.
#
# You should have received a copy of the GNU General Public License
# along with this program; if not, write to the Free Software
# Foundation, Inc., 59 Temple Place - Suite 330, Boston, MA 02111-1307, USA.
import numpy as np
import pytest
from numpy.testing import assert_allclose
from scipy.integrate import quad

from pystrata.factories import LayerFactory, darendeli_soil_type, menq_soil_type
from pystrata.site import Profile
from pystrata.units import GRAVITY


def _stress_mean_at(depth, unit_wt, wt_depth=0.0, k0=0.5):
    """Reference implementation of mean effective stress at `depth` [m],
    matching LayerFactory's internal calculation, used to independently
    verify velocity/stress wiring without relying on soil_type internals."""
    stress_vert_total = depth * unit_wt
    pore_pressure = GRAVITY * max(depth - wt_depth, 0.0)
    stress_vert_eff = stress_vert_total - pore_pressure
    return (1 + 2 * k0) / 3 * stress_vert_eff


def _integral_travel_times(layers, coef, power, param, wt_depth=0.0, k0=0.5, depth0=0.0):
    """Reference travel times computed by directly integrating 1/velocity(z)
    over each layer's exact depth range, independent of the layer's assigned
    (midpoint) shear_vel. Used to verify the partition boundaries themselves
    -- rather than the constant-velocity-per-layer approximation -- give
    equal travel time.
    """

    def velocity(depth, unit_wt):
        x = depth if param == "depth" else _stress_mean_at(depth, unit_wt, wt_depth, k0)
        return coef * x**power

    depth = depth0
    times = []
    for layer in layers:
        unit_wt = layer.unit_wt
        t, _ = quad(lambda z: 1.0 / velocity(z, unit_wt), depth, depth + layer.thickness)
        times.append(t)
        depth += layer.thickness
    return times


def test_total_thickness_preserved():
    factory = LayerFactory()
    layers = factory.add(10.0, 18.0, 300, 0.2, "depth", 5, darendeli_soil_type)

    assert len(layers) == 5
    assert_allclose(sum(layer.thickness for layer in layers), 10.0)


def test_equal_travel_time_depth():
    coef, power = 300, 0.25
    factory = LayerFactory()
    layers = factory.add(15.0, 18.0, coef, power, "depth", 6, darendeli_soil_type)

    # The boundaries are chosen so that the *exact* (integral) travel time is
    # equal across sub-layers; Layer.travel_time (which uses a single
    # midpoint velocity) is only an approximation of this, so we verify the
    # boundaries directly rather than Layer.travel_time.
    travel_times = _integral_travel_times(layers, coef, power, "depth")
    assert_allclose(travel_times, travel_times[0], rtol=1e-6)


def test_equal_travel_time_mean_eff_stress():
    coef, power = 250, 0.3
    factory = LayerFactory(wt_depth=3.0)
    layers = factory.add(
        12.0, 19.0, coef, power, "mean_eff_stress", 5, menq_soil_type, {"coef_unif": 8}
    )

    travel_times = _integral_travel_times(
        layers, coef, power, "mean_eff_stress", wt_depth=3.0
    )
    assert_allclose(travel_times, travel_times[0], rtol=1e-6)


def test_velocity_follows_power_law_depth():
    coef, power = 300, 0.25
    factory = LayerFactory()
    layers = factory.add(20.0, 18.0, coef, power, "depth", 5, darendeli_soil_type)

    depth = 0.0
    for layer in layers:
        depth_mid = depth + layer.thickness / 2
        expected = coef * depth_mid**power
        assert_allclose(layer.shear_vel, expected, rtol=1e-6)
        depth += layer.thickness


def test_velocity_follows_power_law_mean_eff_stress():
    coef, power = 250, 0.3
    factory = LayerFactory()
    layers = factory.add(
        20.0, 18.0, coef, power, "mean_eff_stress", 5, darendeli_soil_type
    )

    depth = 0.0
    for layer in layers:
        depth_mid = depth + layer.thickness / 2
        expected_stress = _stress_mean_at(depth_mid, 18.0)
        expected_vel = coef * expected_stress**power
        assert_allclose(layer.shear_vel, expected_vel, rtol=1e-6)
        depth += layer.thickness


def test_multiple_spans_are_continuous():
    factory = LayerFactory(wt_depth=100.0)
    layers_1 = factory.add(10.0, 18.0, 300, 0.2, "depth", 3, darendeli_soil_type)
    depth_after_1 = factory.depth
    stress_after_1 = factory.stress_mean

    assert_allclose(depth_after_1, 10.0)
    # Dry condition (water table far below): mean_eff_stress = (1+2k0)/3 * sigma_v
    assert_allclose(stress_after_1, 10.0 * 18.0 * (1 + 2 * 0.5) / 3)

    layers_2 = factory.add(
        5.0, 20.0, 150, 0.3, "mean_eff_stress", 2, menq_soil_type, {"coef_unif": 6}
    )

    # Second span's first layer's stress reflects continuity with the end of
    # the first span, plus overburden accumulated within its own thickness
    # (evaluated at its midpoint, per the "midpoint" velocity-evaluation
    # convention).
    mid_offset = layers_2[0].thickness / 2
    expected_stress = (1 + 2 * 0.5) / 3 * (10.0 * 18.0 + mid_offset * 20.0)
    expected_vel = 150 * expected_stress**0.3
    assert_allclose(layers_2[0].shear_vel, expected_vel, rtol=1e-6)
    assert_allclose(factory.depth, 15.0)
    assert factory.layers == layers_1 + layers_2


def test_water_table_crossing_within_span():
    coef, power = 300, 0.25
    factory = LayerFactory(wt_depth=5.0)
    layers = factory.add(10.0, 18.0, coef, power, "depth", 8, darendeli_soil_type)

    assert_allclose(sum(layer.thickness for layer in layers), 10.0)

    travel_times = _integral_travel_times(layers, coef, power, "depth", wt_depth=5.0)
    assert_allclose(travel_times, travel_times[0], rtol=1e-6)


def test_invalid_param_raises():
    factory = LayerFactory()
    with pytest.raises(ValueError):
        factory.add(10.0, 18.0, 300, 0.2, "bogus", 3, darendeli_soil_type)


def test_invalid_count_raises():
    factory = LayerFactory()
    with pytest.raises(ValueError):
        factory.add(10.0, 18.0, 300, 0.2, "depth", 0, darendeli_soil_type)


def test_join_two_factories_into_profile():
    factory_1 = LayerFactory()
    factory_1.add(10.0, 18.0, 300, 0.25, "depth", 4, darendeli_soil_type)

    factory_2 = LayerFactory(initial_depth=10.0)
    factory_2.add(
        15.0, 20.0, 150, 0.3, "depth", 4, menq_soil_type, {"coef_unif": 6}
    )

    combined = factory_1.layers + factory_2.layers
    profile = Profile(combined)

    assert len(profile) == 8
    assert_allclose(
        sum(layer.thickness for layer in profile), 25.0
    )
