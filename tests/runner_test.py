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
# Copyright (C) Albert Kottke, 2013-2025
import multiprocessing

import numpy as np
import pygmm
import pytest
from numpy.testing import assert_allclose
from pygmm.fourier_spectrum import SourceTheoryModel

from pystrata import motion, output, propagation, runner, site, variation

FREQS = np.logspace(-1, 2, 31)


def _darendeli(**kw) -> site.SoilType:
    return site.SoilType.from_curves(pygmm.DarendeliSoilType(**kw).curves())


@pytest.fixture(scope="module")
def profile():
    return site.Profile(
        [
            site.Layer(
                _darendeli(unit_wt=18.0, plas_index=0, ocr=1, stress_mean=200),
                thickness,
                shear_vel,
            )
            for thickness, shear_vel in [(10, 300), (10, 400), (20, 600)]
        ]
        + [site.Layer(site.SoilType("Rock", 24.0, None, 0.01), 0, 1200)]
    )


@pytest.fixture(scope="module")
def rvt_motion():
    return motion.RvtMotion.from_fas(SourceTheoryModel(6.5, 20, "wna"))


@pytest.fixture(scope="module")
def calc():
    return propagation.EquivalentLinearCalculator()


def _outputs():
    return output.OutputCollection(
        [
            output.ResponseSpectrumOutput(
                FREQS, output.OutputLocation("outcrop", index=0), 0.05
            )
        ]
    )


def _variations():
    return {
        "var_velocity": variation.ToroVelocityVariation.generic_model("USGS C"),
        "var_thickness": variation.ToroThicknessVariation(),
    }


def test_run_realization(profile, rvt_motion, calc):
    outputs = _outputs()

    returned = runner.run_realization(profile.copy(), rvt_motion, calc, outputs)

    assert returned is outputs
    assert outputs[0].values.shape == (len(FREQS),)
    assert np.all(np.isfinite(outputs[0].values))


def test_serial_matches_hand_written_loop(profile, rvt_motion, calc):
    """n_jobs=1 reproduces the loop the examples spell out by hand."""
    expected = _outputs()
    for prof in variation.iter_varied_profiles(profile, 4, seed=11, **_variations()):
        prof = prof.auto_discretize()
        calc(rvt_motion, prof, prof.location("outcrop", index=-1))
        expected(calc)

    result = runner.run_ensemble(
        profile,
        rvt_motion,
        calc,
        _outputs(),
        count=4,
        seed=11,
        discretize={},
        n_jobs=1,
        **_variations(),
    )

    assert_allclose(result[0].values, expected[0].values)
    assert result[0].names == expected[0].names


@pytest.mark.parametrize("chunksize", [None, 1, 3])
def test_parallel_matches_serial(profile, rvt_motion, calc, chunksize):
    kwds = dict(count=6, seed=99, discretize={}, **_variations())

    serial = runner.run_ensemble(
        profile, rvt_motion, calc, _outputs(), n_jobs=1, **kwds
    )
    parallel = runner.run_ensemble(
        profile, rvt_motion, calc, _outputs(), n_jobs=2, chunksize=chunksize, **kwds
    )

    assert_allclose(serial[0].values, parallel[0].values)
    assert serial[0].names == parallel[0].names


def test_parallel_matches_serial_multiple_motions(profile, calc):
    motions = [
        motion.RvtMotion.from_fas(SourceTheoryModel(mag, 20, "wna"))
        for mag in (6.0, 7.0)
    ]
    kwds = dict(count=3, seed=5, discretize={}, **_variations())

    serial = runner.run_ensemble(profile, motions, calc, _outputs(), n_jobs=1, **kwds)
    parallel = runner.run_ensemble(profile, motions, calc, _outputs(), n_jobs=2, **kwds)

    # Realizations x motions are flattened into a single task list
    assert serial[0].values.shape == (len(FREQS), 6)
    assert serial[0].names[:3] == [("p0", "m0"), ("p0", "m1"), ("p1", "m0")]
    assert_allclose(serial[0].values, parallel[0].values)
    assert serial[0].names == parallel[0].names


@pytest.mark.parametrize("mp_context", ["fork", "spawn"])
def test_start_methods_agree(profile, rvt_motion, calc, mp_context):
    """The task state is picklable, so every start method works.

    'fork' is fastest but unsafe from a threaded parent; 'spawn' and 'forkserver' re-
    import the calling module and pay a startup cost.
    """
    if mp_context not in multiprocessing.get_all_start_methods():
        pytest.skip(f"{mp_context} is not available on this platform")

    kwds = dict(count=3, seed=17, **_variations())

    serial = runner.run_ensemble(
        profile, rvt_motion, calc, _outputs(), n_jobs=1, **kwds
    )
    parallel = runner.run_ensemble(
        profile, rvt_motion, calc, _outputs(), n_jobs=2, mp_context=mp_context, **kwds
    )

    assert_allclose(serial[0].values, parallel[0].values)


def test_explicit_profiles(profile, rvt_motion, calc):
    """Pre-generated realizations bypass the seed-to-index mapping."""
    profiles = list(variation.iter_varied_profiles(profile, 3, seed=2, **_variations()))

    serial = runner.run_ensemble(
        profile, rvt_motion, calc, _outputs(), profiles=profiles, n_jobs=1
    )
    parallel = runner.run_ensemble(
        profile, rvt_motion, calc, _outputs(), profiles=profiles, n_jobs=2
    )

    assert serial[0].values.shape == (len(FREQS), 3)
    assert_allclose(serial[0].values, parallel[0].values)


def test_gridded_profile_output_runs_in_parallel(profile, rvt_motion, calc):
    grid = profile.depth_grid(spacing=2.0)

    def outputs():
        return output.OutputCollection([output.MaxStrainProfile(depths=grid)])

    kwds = dict(count=4, seed=3, discretize={}, **_variations())
    serial = runner.run_ensemble(profile, rvt_motion, calc, outputs(), n_jobs=1, **kwds)
    parallel = runner.run_ensemble(
        profile, rvt_motion, calc, outputs(), n_jobs=2, **kwds
    )

    assert serial[0].values.shape == (len(grid), 4)
    assert_allclose(serial[0].values, parallel[0].values, equal_nan=True)


def test_ragged_profile_output_runs_in_parallel(profile, rvt_motion, calc):
    """Outputs whose depths vary per realization are collected in index order."""

    def outputs():
        return output.OutputCollection([output.InitialVelProfile()])

    kwds = dict(count=4, seed=8, discretize={}, **_variations())
    serial = runner.run_ensemble(profile, rvt_motion, calc, outputs(), n_jobs=1, **kwds)
    parallel = runner.run_ensemble(
        profile, rvt_motion, calc, outputs(), n_jobs=2, **kwds
    )

    assert serial[0].refs.ndim == 2
    assert_allclose(serial[0].refs, parallel[0].refs, equal_nan=True)
    assert_allclose(serial[0].values, parallel[0].values, equal_nan=True)


def test_does_not_mutate_the_template_outputs(profile, rvt_motion, calc):
    template = _outputs()

    result = runner.run_ensemble(
        profile, rvt_motion, calc, template, count=2, seed=1, **_variations()
    )

    assert result is not template
    assert template[0].values is None


def test_seed_makes_the_ensemble_repeatable(profile, rvt_motion, calc):
    kwds = dict(count=3, seed=42, **_variations())

    first = runner.run_ensemble(profile, rvt_motion, calc, _outputs(), **kwds)
    again = runner.run_ensemble(profile, rvt_motion, calc, _outputs(), **kwds)

    assert_allclose(first[0].values, again[0].values)


def test_invalid_count(profile, rvt_motion, calc):
    with pytest.raises(ValueError, match="count must be at least 1"):
        runner.run_ensemble(profile, rvt_motion, calc, _outputs(), count=0)


def test_worker_error_is_propagated(profile, rvt_motion):
    class Boom(propagation.EquivalentLinearCalculator):
        def __call__(self, *args, **kwargs):
            raise RuntimeError("calculator exploded")

    with pytest.raises(RuntimeError, match="calculator exploded"):
        runner.run_ensemble(
            profile, rvt_motion, Boom(), _outputs(), count=2, seed=1, n_jobs=2
        )
