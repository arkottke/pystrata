# The MIT License (MIT)
#
# Copyright (c) 2016-2025 Albert Kottke
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
"""Run ensembles of site response analyses, optionally across processes.

An ensemble is the cross product of profile realizations, input motions, and
logic-tree branches. These are flattened into a single list of tasks, so that
distributing the work never nests one pool inside another.

Realizations are identified by index rather than by arrival order, and each is
generated from a seed rather than shipped to the worker. Results are therefore
independent of ``n_jobs``, of the chunk size, and of the order in which workers
finish.
"""

from __future__ import annotations

import copy
import logging
from collections.abc import Sequence

from . import variation
from .motion import Motion
from .output import OutputCollection, OutputLocation

logger = logging.getLogger(__name__)

__all__ = ["run_realization", "run_ensemble"]

#: Invariants shared by every task, populated once per worker process.
_STATE: dict = {}


def run_realization(
    profile,
    motion,
    calc,
    outputs,
    name=None,
    index: int | None = None,
    loc_input: OutputLocation | None = None,
):
    """Propagate one motion through one profile and collect the outputs.

    Parameters
    ----------
    profile : site.Profile
        Profile realization. Modified in place by the calculator.
    motion : motion.Motion
        Input motion.
    calc : propagation.AbstractCalculator
        Calculator. Holds state from the run, so it must not be shared between
        concurrent tasks -- pass a copy per task.
    outputs : output.OutputCollection or output.Output
        Outputs to populate.
    name : optional
        Name recorded for this realization.
    index : int, optional
        Column the result is stored in. Defaults to the next one.
    loc_input : output.OutputLocation, optional
        Location of the input motion. Defaults to an outcrop at the base of the
        profile.

    Returns
    -------
    output.OutputCollection or output.Output
        The *outputs* argument, populated.
    """
    if loc_input is None:
        loc_input = OutputLocation("outcrop", index=-1)

    calc(motion, profile, loc_input(profile))
    outputs(calc, name=name, index=index)

    return outputs


def _init_worker(state: dict) -> None:
    """Store the invariants that every task in this process shares."""
    _STATE.clear()
    _STATE.update(state)


def _build_profile(realization: int) -> object:
    """Regenerate a profile realization inside the worker."""
    state = _STATE

    profile = state["profiles"]
    if profile is not None:
        # Explicit profiles were supplied, so they were shipped with the state
        profile = profile[realization]
    else:
        profile = variation.varied_profile(
            state["profile"],
            realization,
            seed=state["seed"],
            var_depth=state["var_depth"],
            var_thickness=state["var_thickness"],
            var_velocity=state["var_velocity"],
            var_soiltypes=state["var_soiltypes"],
        )

    if state["discretize"] is not None:
        profile = profile.auto_discretize(**state["discretize"])

    return profile


def _run_task(task: tuple[int, int, int]) -> tuple[int, list]:
    """Run one task and return its results as plain arrays.

    Only the arrays travel back to the parent; the parent owns the real
    :class:`~pystrata.output.Output` objects and writes results into them.
    """
    index, realization, motion_index = task

    profile = _build_profile(realization)
    # The calculator carries state from the previous run, so use a fresh copy
    calc = copy.deepcopy(_STATE["calc"])
    outputs = copy.deepcopy(_STATE["outputs"])

    run_realization(
        profile,
        _STATE["motions"][motion_index],
        calc,
        outputs,
        loc_input=_STATE["loc_input"],
    )

    results = []
    for out in outputs:
        refs = out.refs if not out._const_ref else None
        results.append((out.values, refs))

    return index, results


def _run_chunk(tasks: list) -> list:
    return [_run_task(task) for task in tasks]


def _chunked(items: list, size: int):
    for i in range(0, len(items), size):
        yield items[i : i + size]


def run_ensemble(
    profile,
    motion,
    calc,
    outputs,
    *,
    count: int = 1,
    profiles: Sequence | None = None,
    var_depth=None,
    var_thickness=None,
    var_velocity=None,
    var_soiltypes=None,
    seed: int | None = None,
    discretize: dict | None = None,
    loc_input: OutputLocation | None = None,
    n_jobs: int = 1,
    chunksize: int | None = None,
    mp_context: str | None = None,
) -> OutputCollection:
    """Run an ensemble of realizations, optionally across processes.

    Parameters
    ----------
    profile : site.Profile
        Seed profile from which realizations are generated. Ignored when
        *profiles* is supplied.
    motion : motion.Motion or sequence of motion.Motion
        Input motion, or motions. Every motion is applied to every realization.
    calc : propagation.AbstractCalculator
        Calculator used as a template. A copy is made for each task, since a
        calculator holds state from its run.
    outputs : output.OutputCollection
        Outputs used as a template. The returned collection is a copy.
    count : int, optional
        Number of profile realizations. Ignored when *profiles* is supplied.
    profiles : sequence of site.Profile, optional
        Explicit realizations, instead of generating them from the variation
        models. Use this when realizations are rejection-sampled, since a
        rejected realization breaks the mapping from index to seed.
    var_depth, var_thickness, var_velocity, var_soiltypes
        Variation models, applied in that order.
    seed : int, optional
        Base seed. Required for reproducible results; without it each
        realization draws from the module-level generator.
    discretize : dict, optional
        Keyword arguments for :meth:`~pystrata.site.Profile.auto_discretize`,
        applied to each realization. Pass ``{}`` to use its defaults.
    loc_input : output.OutputLocation, optional
        Location of the input motion. Defaults to an outcrop at the base.
    n_jobs : int, optional
        Number of worker processes. ``1`` (default) runs serially in this
        process and takes no dependency on multiprocessing.
    chunksize : int, optional
        Tasks dispatched per unit of work. Defaults to spreading the tasks
        evenly over the workers. Larger chunks amortize the cost of sending a
        task at the expense of load balancing.
    mp_context : str, optional
        Start method for the worker processes: ``'fork'``, ``'forkserver'`` or
        ``'spawn'``. Defaults to the platform default. ``'fork'`` needs no
        cooperation from the calling module but is unsafe when the parent
        process already holds threads; the other two are safe but re-import
        the calling module, so a script must guard its entry point with
        ``if __name__ == "__main__":``.

    Returns
    -------
    output.OutputCollection
        A copy of *outputs*, populated with one column per task. With multiple
        motions the names are ``(profile, motion)`` tuples, matching the
        convention used for nested loops.

    Notes
    -----
    Results do not depend on *n_jobs* or *chunksize*: realization *i* is
    generated from *seed* and stored in column *i* regardless of which worker
    computes it or when it finishes.

    Processes are used rather than threads because the compiled kernels hold
    the GIL.
    """
    if profiles is not None:
        count = len(profiles)
    if count < 1:
        raise ValueError(f"count must be at least 1, not {count}.")

    motions = [motion] if isinstance(motion, Motion) else list(motion)
    multi_motion = len(motions) > 1

    results = copy.deepcopy(outputs)
    if not isinstance(results, OutputCollection):
        results = OutputCollection([results])

    # Flatten the axes into one task list, so that the work is spread evenly
    # and no pool is nested inside another
    tasks = [
        (i * len(motions) + j, i, j) for i in range(count) for j in range(len(motions))
    ]
    names = [
        (f"p{i}", f"m{j}") if multi_motion else "r%d" % (i + 1)
        for i in range(count)
        for j in range(len(motions))
    ]

    if seed is None and n_jobs != 1:
        logger.warning(
            "Running in parallel without a seed. Results will be correct but "
            "not reproducible; pass seed= to make the ensemble repeatable."
        )

    state = {
        "profile": profile,
        "profiles": list(profiles) if profiles is not None else None,
        "motions": motions,
        "calc": calc,
        "outputs": results,
        "var_depth": var_depth,
        "var_thickness": var_thickness,
        "var_velocity": var_velocity,
        "var_soiltypes": var_soiltypes,
        "seed": seed,
        "discretize": discretize,
        "loc_input": loc_input,
    }

    if n_jobs == 1:
        _init_worker(state)
        collected = [_run_task(task) for task in tasks]
    else:
        collected = _run_parallel(tasks, state, n_jobs, chunksize, mp_context)

    _collect(results, collected, names)

    return results


def _run_parallel(tasks, state, n_jobs, chunksize, mp_context):
    """Dispatch *tasks* across a process pool."""
    import multiprocessing
    from concurrent.futures import BrokenExecutor, ProcessPoolExecutor

    # Compile the numba kernels once here rather than in every worker at the
    # same time, which would contend on the on-disk cache.
    _init_worker(state)
    warmup = _run_task(tasks[0])

    remaining = tasks[1:]
    if not remaining:
        return [warmup]

    if chunksize is None:
        chunksize = max(1, len(remaining) // (4 * n_jobs))

    kwds = {}
    if mp_context is not None:
        kwds["mp_context"] = multiprocessing.get_context(mp_context)

    collected = [warmup]
    try:
        with ProcessPoolExecutor(
            max_workers=n_jobs,
            initializer=_init_worker,
            initargs=(state,),
            **kwds,
        ) as executor:
            for chunk in executor.map(_run_chunk, _chunked(remaining, chunksize)):
                collected.extend(chunk)
    except BrokenExecutor as err:
        method = mp_context or multiprocessing.get_start_method()
        raise RuntimeError(
            f"The worker pool failed while using the {method!r} start method. "
            "The 'spawn' and 'forkserver' methods re-import the calling module, "
            "so a script must guard its entry point with "
            '`if __name__ == "__main__":`, and objects defined interactively '
            "cannot be sent to a worker. Pass mp_context='fork' to run the "
            "workers without re-importing, or n_jobs=1 to run serially."
        ) from err

    return collected


def _collect(results: OutputCollection, collected: list, names: list) -> None:
    """Write the returned arrays into the real outputs, ordered by index."""
    for index, task_results in sorted(collected, key=lambda item: item[0]):
        for out, (values, refs) in zip(results, task_results):
            out._names.append(names[index])
            if refs is not None:
                out._add_refs(refs)
            out._value_cols.append(values)
            out._values_cache = None
