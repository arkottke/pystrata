Running Ensembles
=================

An ensemble is the cross product of profile realizations, input motions, and
logic-tree branches. :func:`pystrata.runner.run_ensemble` flattens those axes
into a single list of tasks and evaluates them, optionally across processes.

.. code-block:: python

    import pystrata

    results = pystrata.runner.run_ensemble(
        profile,
        motion,
        pystrata.propagation.EquivalentLinearCalculator(),
        outputs,
        count=500,
        seed=20250730,
        var_velocity=pystrata.variation.ToroVelocityVariation.generic_model("USGS C"),
        var_thickness=pystrata.variation.ToroThicknessVariation(),
        discretize={},
        n_jobs=8,
    )

``n_jobs=1`` (the default) runs serially and is equivalent to writing the loop
by hand. The returned :class:`~pystrata.output.OutputCollection` is a copy, so
the template passed in is left empty and can be reused.

Reproducibility
---------------

Each realization is generated from ``SeedSequence([seed, index])`` rather than
from a shared stream, so realization *i* is identical no matter how many
realizations were requested, which worker computed it, or when it finished.
Results therefore do not depend on ``n_jobs`` or ``chunksize``.

This also means a single realization can be reproduced on its own, which is
useful when one of them fails:

.. code-block:: python

    profile_7 = pystrata.variation.varied_profile(
        profile, 7, seed=20250730, var_velocity=var_velocity
    )

.. warning::

   Without ``seed``, every variation draws from a module-level generator. The
   results are still correct, but they are not repeatable.

   Do not rely on :func:`numpy.random.seed`; the variation models no longer
   read the global NumPy state. Pass ``seed=``, or ``rng=`` to an individual
   variation model.

Why processes
-------------

The compiled kernels hold the GIL, so a thread pool would not help. Work is
distributed to processes instead, and only a seed and a task index are sent to
each worker -- the profile is regenerated there, rather than being pickled and
shipped for every realization.

Start methods
-------------

``mp_context`` selects how the workers are started, defaulting to the platform
default:

``'fork'``
    Fastest to start and needs no cooperation from the calling module, so it
    works from a notebook or an unguarded script. Unsafe when the parent
    process already holds threads, which NumPy and matplotlib routinely do.

``'forkserver'``, ``'spawn'``
    Safe from a threaded parent, but they re-import the calling module. A
    script must then guard its entry point with
    ``if __name__ == "__main__":``, and objects defined interactively cannot be
    sent to a worker. Expect a few seconds of startup, which only pays for
    itself on larger ensembles.

Rejection sampling
------------------

``iter_varied_profiles`` accepts a ``check`` callable that rejects
realizations. A rejected realization breaks the mapping from index to seed, so
generate the profiles first and pass them explicitly:

.. code-block:: python

    profiles = list(
        pystrata.variation.iter_varied_profiles(
            profile, 200, seed=1, var_velocity=var_velocity, check=check
        )
    )
    results = pystrata.runner.run_ensemble(
        profile, motion, calc, outputs, profiles=profiles, n_jobs=8
    )

Collecting results yourself
---------------------------

If you drive the loop directly, results can still be collected out of order.
:meth:`~pystrata.output.Output.reserve` pre-allocates the storage so that each
realization is written to its own column:

.. code-block:: python

    outputs.reserve(count)
    outputs(calc, index=i)      # column i, whatever order they arrive in

:meth:`~pystrata.output.Output.extend` combines results accumulated separately.
Columns that are never written stay NaN, so a realization that failed remains
identifiable instead of shifting the ones after it.
