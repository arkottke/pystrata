.. _usage:

User Guide
==========

pyStrata computes the response of a layered soil column to an input motion.
Regardless of the analysis, the same four objects are used, and each of the
pages below describes one of them. This page describes what they are and how they
fit together.

.. toctree::
   :maxdepth: 2

   motions
   profiles
   calculations
   outputs
   variation
   logic_trees
   parallel
   dams

The pieces
----------

**Input motion** (:doc:`motions`)
    A :class:`~pystrata.motion.TimeSeriesMotion` is a recorded acceleration time
    series. A :class:`~pystrata.motion.RvtMotion` and a
    :class:`~pystrata.motion.CompatibleRvtMotion` are defined in the frequency
    domain (Fourier amplitudes, or a target response spectrum) and use random
    vibration theory to compute the peak response.

**Site profile** (:doc:`profiles`)
    A :class:`~pystrata.site.SoilType` holds the unit weight, damping and
    optional modulus reduction and damping curves. A :class:`~pystrata.site.Layer`
    is a soil type with a thickness and a shear-wave velocity, and a
    :class:`~pystrata.site.Profile` is a list of layers, the last of which is the
    half space. A :class:`~pystrata.site.Location` points to a depth and a wave
    field (outcrop or within) in a profile.

**Calculator** (:doc:`calculations`)
    A calculator is called with the motion, the profile and the location of the
    input: ``calc(motion, profile, loc_input)``. It propagates the motion through
    the profile and keeps the waves, so that any output can be computed afterward.

**Outputs** (:doc:`outputs`)
    An output is registered in an :class:`~pystrata.output.OutputCollection`
    with the location it is computed at. Calling the collection with the
    calculator that has run stores the results, as :class:`xarray.Dataset` objects.

**Variability** (:doc:`variation`, :doc:`logic_trees`, :doc:`parallel`)
    Variation models perturb a profile to give realizations of it. A logic tree
    holds alternatives for the profile, motion or calculator with weights. The
    runner evaluates the combinations, in one process or across several.

**Dams** (:doc:`dams`)
    The shear-wedge dam joins a dam to a profile and uses the same calculators
    and outputs.

How they fit together
---------------------

.. code-block:: text

   motion ─────────────┐
                        ├──► calculator(motion, profile, loc_input) ──► outputs(calculator)
   SoilType ► Layer ► Profile ─┘            ▲                                    │
                  ▲                         │                                    ▼
          variation models           logic-tree branch                    OutputCollection
          (realizations)             (alternatives)                       (xarray datasets)

A complete analysis with the objects above:

.. code-block:: python

   import numpy as np
   import pystrata

   motion = pystrata.motion.TimeSeriesMotion.load_at2_file("motion.at2")

   soil = pystrata.site.SoilType("soil", 18.0, None, 0.05)
   rock = pystrata.site.SoilType("rock", 24.0, None, 0.01)
   profile = pystrata.site.Profile(
       [
           pystrata.site.Layer(soil, 10.0, 200.0),
           pystrata.site.Layer(rock, 0.0, 1000.0),
       ]
   )

   calc = pystrata.propagation.LinearElasticCalculator()
   osc_freqs = np.geomspace(0.1, 100, 100)
   loc_in = pystrata.output.OutputLocation("outcrop", index=-1)
   loc_out = pystrata.output.OutputLocation("outcrop", index=0)
   outputs = pystrata.output.OutputCollection(
       [pystrata.output.ResponseSpectrumOutput(osc_freqs, loc_out, 0.05)]
   )

   calc(motion, profile, profile.location("outcrop", index=-1))
   outputs(calc)

Repeating the last two lines for the profiles that a variation model returns, or
for the branches of a logic tree, gives the distribution of the outputs;
:func:`pystrata.runner.run_ensemble` does so for all of the combinations of
profiles, motions and branches.

Other libraries
---------------

pyStrata uses the output of the sibling libraries. It has no runtime dependency on pyGMM, which it reads by duck typing; it depends on pyRVT.

.. list-table::
   :header-rows: 1
   :widths: 25 40 35

   * - Library
     - What pyStrata takes from it
     - Where
   * - pyGMM
     - Nonlinear curves (``NonlinearSoilCurves``), shear-wave velocity profiles
       (``VelocityProfile``) and Fourier amplitude spectrum models
     - :doc:`profiles`, :doc:`motions`
   * - pyRVT
     - The random vibration theory calculation of the peak response, and the peak
       factor calculators
     - :doc:`motions`

Examples
--------

.. list-table::
   :header-rows: 1
   :widths: 45 55

   * - Topic
     - Examples
   * - A time-series and an RVT analysis
     - :doc:`../examples/example-01`, :doc:`../examples/example-02`
   * - Several calculators on the same input
     - :doc:`../examples/example-03`, :doc:`../examples/example-09`
   * - Motions and peak factors
     - :doc:`../examples/example-07`, :doc:`../examples/example-10`,
       :doc:`../examples/example-14`
   * - Profiles and soil properties
     - :doc:`../examples/example-06`, :doc:`../examples/example-15`,
       :doc:`../examples/example-17`, :doc:`../examples/example-18`
   * - Variation of profiles and soil properties
     - :doc:`../examples/example-04`, :doc:`../examples/example-05`,
       :doc:`../examples/example-08`
   * - Logic trees
     - :doc:`../examples/example-16`
   * - Calculator settings
     - :doc:`../examples/example-11`, :doc:`../examples/example-12`,
       :doc:`../examples/example-13`, :doc:`../examples/example-19`,
       :doc:`../examples/example-24`
   * - Dams
     - :doc:`../examples/example-20`, :doc:`../examples/example-21`,
       :doc:`../examples/example-22`,
       :doc:`../examples/example-25`
