#!/usr/bin/env python
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
#
# Copyright (C) Albert Kottke, 2013-2016
import warnings

import numpy as np
import pygmm
import pytest
import xarray as xr
from numpy.testing import assert_allclose
from pygmm.fourier_spectrum import SourceTheoryModel
from scipy.stats import norm, uniform

import pystrata
from pystrata import motion, output, propagation, site, variation
from pystrata.output import stack_columns

# All concrete ProfileBasedOutput subclasses. Used to check that the
# ref/value bookkeeping stays consistent across the family.
PROFILE_OUTPUTS = [
    output.MaxStrainProfile,
    output.DampingProfile,
    output.ShearModReducProfile,
    output.InitialVelProfile,
    output.CompatVelProfile,
    output.CyclicStressRatioProfile,
    output.MaxAccelProfile,
]


def _darendeli(**kw) -> site.SoilType:
    """Create a pystrata SoilType from pygmm's DarendeliSoilType."""
    return site.SoilType.from_curves(pygmm.DarendeliSoilType(**kw).curves())


def _profile(vel_scale: float = 1.0) -> site.Profile:
    """Three nonlinear layers over rock.

    Base of the soil column is at 40 m.
    """
    return site.Profile(
        [
            site.Layer(
                _darendeli(unit_wt=18.0, plas_index=0, ocr=1, stress_mean=200),
                thickness,
                vel_scale * shear_vel,
            )
            for thickness, shear_vel in [(10, 300), (10, 400), (20, 600)]
        ]
        + [site.Layer(site.SoilType("Rock", 24.0, None, 0.01), 0, 1200)]
    )


@pytest.fixture(scope="module")
def profile():
    return _profile()


@pytest.fixture(scope="module")
def rvt_motion():
    return motion.RvtMotion.from_fas(SourceTheoryModel(6.0, 30, "wna"))


@pytest.fixture(scope="module")
def calc(profile, rvt_motion):
    """An equivalent-linear calculator that has been run on ``profile``."""
    calculator = propagation.EquivalentLinearCalculator()
    calculator(rvt_motion, profile, profile.location("outcrop", index=-1))
    return calculator


@pytest.mark.parametrize("cls", PROFILE_OUTPUTS, ids=lambda c: c.__name__)
def test_profile_output_ref_value_lengths(cls, calc):
    out = cls()
    out(calc)

    assert out.refs.ndim == 1
    assert len(out.refs) == len(out.values)
    assert out.refs[0] == 0
    # Depths increase monotonically from the surface
    assert np.all(np.diff(out.refs) > 0)
    assert np.all(np.isfinite(out.values))


@pytest.mark.parametrize("cls", PROFILE_OUTPUTS, ids=lambda c: c.__name__)
def test_profile_output_ragged_accumulation(cls, rvt_motion):
    """Profiles with differing layer counts accumulate into a 2-D, NaN-padded array."""
    out = cls()
    calculator = propagation.EquivalentLinearCalculator()
    for count in [3, 5, 4]:
        # Vary the layer count so each realization has a different ref length
        prof = _profile().auto_discretize(max_freq=count * 5.0)
        calculator(rvt_motion, prof, prof.location("outcrop", index=-1))
        out(calculator)

    assert out.refs.ndim == 2
    assert out.values.ndim == 2
    assert out.refs.shape == out.values.shape
    assert out.refs.shape[1] == 3
    assert out.names == ["r1", "r2", "r3"]


def test_initial_vel_profile_values(calc):
    """The surface value is repeated, and each layer reports its initial velocity."""
    out = output.InitialVelProfile()
    out(calc)

    assert_allclose(out.refs, [0, 10, 20, 40])
    assert_allclose(out.values, [300, 300, 400, 600])


def test_mid_depth_outputs_reach_the_base(calc):
    """MaxStrainProfile and CyclicStressRatioProfile report layer mid-depths.

    A final point is reported at the base of the soil column (40 m) so that the deepest
    layer is represented over its full thickness, rather than only down to its mid-depth
    at 30 m.
    """
    for cls in [output.MaxStrainProfile, output.CyclicStressRatioProfile]:
        out = cls()
        out(calc)
        assert_allclose(out.refs, [0, 5, 15, 30, 40])
        # The deepest layer's value carries to the base
        assert_allclose(out.values[-1], out.values[-2])

    for cls in [output.DampingProfile, output.InitialVelProfile]:
        out = cls()
        out(calc)
        assert_allclose(out.refs, [0, 10, 20, 40])


def test_max_strain_is_zero_at_the_surface(calc):
    out = output.MaxStrainProfile()
    out(calc)
    assert out.values[0] == 0


def test_profile_output_reset(calc):
    out = output.InitialVelProfile()
    out(calc)
    out(calc)
    assert out.values.shape[1] == 2

    out.reset()
    assert out.values is None
    assert out.names == []
    # ProfileBasedOutput refs are not constant, so they are cleared as well
    assert len(out.refs) == 0


def test_depth_grid_default_spacing(profile):
    grid = profile.depth_grid()

    # Slowest layer is 300 m/s: 300 / 50 Hz * 0.2 = 1.2 m
    assert_allclose(np.diff(grid), 1.2)
    assert grid[0] == 0
    assert grid[-1] >= profile[-2].depth_base


def test_depth_grid_explicit_spacing(profile):
    grid = profile.depth_grid(spacing=0.5)

    assert_allclose(np.diff(grid), 0.5)
    assert grid[-1] >= profile[-2].depth_base


@pytest.mark.parametrize(
    "dist", [norm(loc=60, scale=5), uniform(loc=30, scale=40)], ids=["norm", "uniform"]
)
def test_depth_grid_covers_depth_variation(dist, profile):
    var_depth = variation.HalfSpaceDepthVariation(dist)

    grid = profile.depth_grid(depth_var=var_depth)

    assert grid[-1] >= dist.ppf(0.999)
    # A bare frozen distribution is accepted as well
    assert_allclose(grid, profile.depth_grid(depth_var=dist))


def test_depth_grid_warns_when_too_shallow(profile):
    var_depth = variation.HalfSpaceDepthVariation(norm(loc=60, scale=5))

    with pytest.warns(UserWarning, match="shallower than"):
        grid = profile.depth_grid(max_depth=20, depth_var=var_depth)

    # Even so, the seed profile is always covered
    assert grid[-1] >= profile[-2].depth_base


def test_depth_grid_rejects_excessive_size(profile):
    with pytest.raises(ValueError, match="Increase spacing"):
        profile.depth_grid(spacing=1e-4, max_depth=1e4)


def test_depth_grid_rejects_bad_depth_var(profile):
    with pytest.raises(TypeError, match="depth_limit"):
        profile.depth_grid(depth_var="nonsense")


@pytest.mark.parametrize("cls", PROFILE_OUTPUTS, ids=lambda c: c.__name__)
def test_gridded_refs_are_constant(cls, calc, profile):
    grid = profile.depth_grid(spacing=1.0)
    out = cls(depths=grid)
    for _ in range(3):
        out(calc)

    assert out._const_ref
    assert out.refs.ndim == 1
    assert_allclose(out.refs, grid)
    assert_allclose(out.depths, grid)
    assert out.values.shape == (len(grid), 3)

    out.reset()
    # The grid survives a reset; the values do not
    assert_allclose(out.refs, grid)
    assert out.values is None


@pytest.mark.parametrize("cls", PROFILE_OUTPUTS, ids=lambda c: c.__name__)
def test_gridded_matches_ungridded(cls, calc, profile):
    """Resampling at collection time agrees with resampling at reporting time."""
    grid = profile.depth_grid(spacing=1.0)

    gridded = cls(depths=grid)
    gridded(calc)

    # Match the fill mode so the comparison isolates the resampling itself
    ungridded = cls(fill_below="nan")
    ungridded(calc)

    assert_allclose(
        gridded.values, ungridded.to_dataframe(ref=grid).iloc[:, 0].to_numpy()
    )


def _truncated(rvt_motion, depth):
    """Run a calculator on a profile truncated to *depth*.

    The depth is chosen away from a layer boundary, since truncating exactly at one
    leaves a degenerate zero-thickness layer, and away from a grid node, so that the
    sampling jitter cannot change which nodes the profile covers.
    """
    prof = variation.HalfSpaceDepthVariation(norm(loc=depth, scale=1e-9))(_profile())
    calculator = propagation.EquivalentLinearCalculator()
    calculator(rvt_motion, prof, prof.location("outcrop", index=-1))
    return calculator


def test_gridded_fill_below_nan(rvt_motion):
    """Below a realization's base, a gridded output reports NaN by default."""
    grid = np.arange(0, 51, 1.0)
    out = output.InitialVelProfile(depths=grid)

    # 34.5 m and 14.5 m deep realizations on a common 50 m grid
    for depth in [34.5, 14.5]:
        out(_truncated(rvt_motion, depth))

    assert np.all(np.isfinite(out.values[grid < 14.5, 1]))
    assert np.all(np.isnan(out.values[grid > 14.5, 1]))
    assert np.all(np.isfinite(out.values[grid < 34.5, 0]))
    assert np.all(np.isnan(out.values[grid > 34.5, 0]))

    stats = out.calc_stats()
    assert_allclose(stats["count"][grid < 14.5], 2)
    assert_allclose(stats["count"][(grid > 14.5) & (grid < 34.5)], 1)
    assert_allclose(stats["count"][grid > 34.5], 0)
    # With no contributing realizations the median is undefined
    assert np.all(np.isnan(stats["median"][grid > 34.5]))


def test_gridded_fill_below_hold(rvt_motion):
    grid = np.arange(0, 51, 1.0)
    out = output.InitialVelProfile(depths=grid, fill_below="hold")

    out(_truncated(rvt_motion, 14.5))

    assert np.all(np.isfinite(out.values))
    # The deepest value is repeated below the base
    deepest = out.values[grid < 14.5][-1]
    assert_allclose(out.values[grid > 14.5], deepest)
    assert_allclose(out.calc_stats()["count"], len(grid) * [1])


def test_gridded_warns_once_when_realization_is_deeper(calc, profile):
    """The grid stops at 20 m but the profile reaches 40 m."""
    out = output.InitialVelProfile(depths=np.arange(0, 21, 1.0))

    with warnings.catch_warnings(record=True) as caught:
        warnings.simplefilter("always")
        for _ in range(5):
            out(calc)

    messages = [str(w.message) for w in caught]
    assert len(messages) == 1
    assert "below the fixed depth grid" in messages[0]


def test_default_ref_is_shared_by_stats_and_dataframe(calc):
    out = output.InitialVelProfile()
    out(calc)
    out(calc)

    assert_allclose(out.calc_stats()["ref"], out.to_dataframe().index.to_numpy())


def test_gridded_stats_use_the_grid(calc, profile):
    grid = profile.depth_grid(spacing=1.0)
    out = output.InitialVelProfile(depths=grid)
    out(calc)
    out(calc)

    assert_allclose(out.calc_stats()["ref"], grid)
    assert_allclose(out.to_dataframe().index.to_numpy(), grid)
    # Both realizations are identical, so the median reproduces them exactly
    assert_allclose(out.calc_stats()["median"], out.values[:, 0])


@pytest.mark.parametrize(
    "depths", [[0, 5, 3], [[0, 1], [2, 3]], [0]], ids=["unsorted", "2d", "single"]
)
def test_invalid_depths_raise(depths):
    with pytest.raises(ValueError, match="depths must be"):
        output.InitialVelProfile(depths=depths)


def test_invalid_fill_below_raises():
    with pytest.raises(ValueError, match="fill_below"):
        output.InitialVelProfile(fill_below="extrapolate")


def test_max_accel_interpolates_linearly(calc, profile):
    """Acceleration is a continuous field, so it is not a step function."""
    assert output.MaxAccelProfile._interp_kind == "linear"
    assert output.InitialVelProfile._interp_kind == "next"

    grid = profile.depth_grid(spacing=1.0)
    out = output.MaxAccelProfile(depths=grid)
    out(calc)

    raw = output.MaxAccelProfile()
    raw(calc)

    # Between the 0 m and 10 m nodes the gridded value is strictly between the
    # two, rather than snapping to one of them
    mid = out.values[grid == 5.0][0]
    lo, hi = sorted(raw.values[:2])
    assert lo < mid < hi


def _spec_output(freqs=None):
    return output.ResponseSpectrumOutput(
        np.logspace(-1, 1, 21) if freqs is None else freqs,
        output.OutputLocation("outcrop", index=0),
        0.05,
    )


def test_stack_columns_pads_ragged():
    stacked = stack_columns([np.array([1.0, 2, 3]), np.array([4.0, 5])])

    assert stacked.shape == (3, 2)
    assert_allclose(stacked[:, 0], [1, 2, 3])
    assert_allclose(stacked[:, 1], [4, 5, np.nan])


def test_stack_columns_preserves_complex():
    stacked = stack_columns([np.array([1 + 2j, 3 + 0j]), np.array([0 + 1j, 1 + 1j])])

    assert np.iscomplexobj(stacked)
    assert_allclose(stacked[:, 0], [1 + 2j, 3 + 0j])


def test_values_shape_matches_realization_count(calc):
    out = _spec_output()

    out(calc)
    # A single realization is stored as a 1-D array
    assert out.values.ndim == 1
    assert out.names == ["r1"]

    out(calc)
    assert out.values.shape == (21, 2)
    assert out.names == ["r1", "r2"]


def test_reserve_allows_out_of_order_writes(calc):
    grid = np.logspace(-1, 1, 21)

    ordered = _spec_output(grid)
    for i in range(3):
        ordered(calc, index=i)

    shuffled = _spec_output(grid)
    shuffled.reserve(3)
    for i in [2, 0, 1]:
        shuffled(calc, index=i)

    assert shuffled.values.shape == (21, 3)
    assert_allclose(shuffled.values, ordered.values)
    # Each realization lands in its own column regardless of write order
    assert shuffled.names == ["r1", "r2", "r3"]


def test_reserve_leaves_skipped_realizations_as_nan(calc):
    out = _spec_output()
    out.reserve(3)

    out(calc, index=0)
    # Realization 1 failed and was skipped
    out(calc, index=2)

    assert out.values.shape == (21, 3)
    assert np.all(np.isfinite(out.values[:, 0]))
    assert np.all(np.isnan(out.values[:, 1]))
    assert np.all(np.isfinite(out.values[:, 2]))
    # The skipped realization stays identifiable rather than shifting left
    assert out.names[1] is None


def test_reserve_requires_constant_refs():
    with pytest.raises(RuntimeError, match="constant references"):
        output.InitialVelProfile().reserve(4)

    # A gridded profile output does have constant references
    output.InitialVelProfile(depths=np.arange(0, 21, 1.0)).reserve(4)


def test_complex_transfer_function_survives_preallocation(calc):
    """Assigning complex values into a real buffer would silently drop them."""
    freqs = np.logspace(-1, 1, 21)
    kwds = {
        "location_in": output.OutputLocation("outcrop", index=-1),
        "location_out": output.OutputLocation("outcrop", index=0),
        "absolute": False,
    }

    appended = output.AccelTransferFunctionOutput(freqs, **kwds)
    appended(calc)

    reserved = output.AccelTransferFunctionOutput(freqs, **kwds)
    reserved.reserve(1)
    reserved(calc, index=0)

    assert np.iscomplexobj(appended.values)
    assert np.iscomplexobj(reserved.values)
    assert np.any(appended.values.imag != 0)
    assert_allclose(reserved.values[:, 0], appended.values)


def test_absolute_transfer_function_stays_real(calc):
    out = output.AccelTransferFunctionOutput(
        np.logspace(-1, 1, 21),
        output.OutputLocation("outcrop", index=-1),
        output.OutputLocation("outcrop", index=0),
    )
    out(calc)

    assert not np.iscomplexobj(out.values)


def test_extend_combines_results(calc):
    grid = np.logspace(-1, 1, 21)

    whole = _spec_output(grid)
    for _ in range(4):
        whole(calc)

    first, second = _spec_output(grid), _spec_output(grid)
    for _ in range(2):
        first(calc)
        second(calc)
    first.extend(second)

    assert first.values.shape == whole.values.shape
    assert_allclose(first.values, whole.values)


def test_extend_rejects_mismatched_types(calc):
    with pytest.raises(TypeError, match="Cannot extend"):
        _spec_output().extend(output.InitialVelProfile())


def test_output_collection_extend(calc):
    def make():
        return output.OutputCollection([_spec_output(), output.InitialVelProfile()])

    first, second = make(), make()
    first(calc)
    second(calc)
    first.extend(second)

    assert first[0].values.shape[1] == 2
    assert first[1].values.shape[1] == 2


def test_to_xarray_without_tree_constant_refs(calc):
    out = _spec_output()
    out(calc)
    out(calc)

    da = out.to_xarray()

    assert da.dims == ("freq", "realization")
    assert da.shape == (21, 2)
    assert_allclose(da.coords["freq"], out.refs)
    assert list(da.coords["realization"].values) == ["r1", "r2"]


def test_to_xarray_without_tree_ragged_refs_stays_compact(rvt_motion):
    """Varying depths are a data variable, not a coordinate.

    Aligning on them would outer-join every distinct depth, giving an array that is
    almost entirely empty.
    """
    out = output.InitialVelProfile()
    calculator = propagation.EquivalentLinearCalculator()
    for count in [3, 5, 4]:
        prof = _profile().auto_discretize(max_freq=count * 5.0)
        calculator(rvt_motion, prof, prof.location("outcrop", index=-1))
        out(calculator)

    ds = out.to_xarray()

    assert isinstance(ds, xr.Dataset)
    assert "depth" not in ds.coords
    assert ds["depth"].dims == ("index", "realization")
    assert ds["value"].dims == ("index", "realization")
    # Densely packed rather than one row per distinct depth
    assert ds["value"].shape == out.values.shape
    assert ds["value"].shape[0] < np.unique(out.refs[np.isfinite(out.refs)]).size * 2


def test_accumulation_is_linear_in_realizations(calc):
    """Appending must not re-copy the whole array on every realization."""
    out = _spec_output()

    for _ in range(400):
        out(calc)

    # One stack, not 400 reallocations
    assert len(out._value_cols) == 400
    assert out._values_cache is None
    assert out.values.shape == (21, 400)
    assert out._values_cache is not None


def test_reset_clears_reserved_buffer(calc):
    out = _spec_output()
    out.reserve(2)
    out(calc, index=0)

    out.reset()

    assert np.all(np.isnan(out.values))
    assert out.names == [None, None]


def test_add_refs():
    output = pystrata.output.Output()
    refs = [1.1, 2, 3]
    output._add_refs(refs)
    assert_allclose(refs, output.refs)


# FIXME: Is this important?
@pytest.mark.xfail
def test_add_refs_same():
    output = pystrata.output.Output()
    # Force float arrays
    a = [1.1, 2, 3]
    b = [1.1, 2, 3]

    output._add_refs(a)
    output._add_refs(b)

    assert np.ndim(output.refs) == 1
    assert_allclose(output.refs, a)


def test_add_refs_diff():
    output = pystrata.output.Output()
    # Force float arrays
    a = [1.1, 2, 3]
    b = [1.1, 2, 3, 4, 5]

    output._add_refs(a)
    output._add_refs(b)

    assert np.ndim(output.refs) == 2
    assert len(output.refs) == len(b)
    assert_allclose(output.refs[:, 0], a + 2 * [np.nan])
    assert_allclose(output.refs[:, 1], b)
