Collecting Outputs
==================

pyStrata separates *what to compute* from *how to run the analysis*. Outputs are registered
before the calculator is called, then populated automatically when the analysis runs.

.. currentmodule:: pystrata.output

OutputCollection
----------------

:class:`OutputCollection` holds a list of output objects and passes them to the calculator:

.. code-block:: python

    import pystrata

    outputs = pystrata.output.OutputCollection([
        pystrata.output.ResponseSpectrumOutput(),
        pystrata.output.AccelTransferFunctionOutput(),
        pystrata.output.MaxStrainProfile(),
    ])

    # Run calculator — outputs are populated here
    calc(motion, profile, profile.location("outcrop", index=-1))
    outputs(calc)

    # Iterate over results
    for output in outputs:
        print(output.refs, output.values)

Specifying Output Locations
---------------------------

Location-based outputs require an :class:`OutputLocation` specifying depth and wave-field:

.. code-block:: python

    # Surface within wave-field
    loc = pystrata.output.OutputLocation("within", index=0)

    # Specific depth
    loc = pystrata.output.OutputLocation("within", depth=10.0)

    # Bedrock outcrop
    loc = pystrata.output.OutputLocation("outcrop", index=-1)

Response Spectra
----------------

:class:`ResponseSpectrumOutput` computes the 5 %-damped pseudo-spectral acceleration at a
specified location:

.. code-block:: python

    rs = pystrata.output.ResponseSpectrumOutput(
        periods=None,        # default log-spaced 0.01–10 s
        osc_damping=0.05,
        location=pystrata.output.OutputLocation("within", index=0),
    )

:class:`ResponseSpectrumRatioOutput` returns the ratio between surface and input spectra:

.. code-block:: python

    rsr = pystrata.output.ResponseSpectrumRatioOutput(
        periods=None,
        osc_damping=0.05,
        location=pystrata.output.OutputLocation("within", index=0),
        ref_location=pystrata.output.OutputLocation("outcrop", index=-1),
    )

Transfer Functions
------------------

:class:`AccelTransferFunctionOutput` computes the complex acceleration transfer function
(ratio of output to input Fourier spectra):

.. code-block:: python

    tf = pystrata.output.AccelTransferFunctionOutput(
        location=pystrata.output.OutputLocation("within", index=0),
        ref_location=pystrata.output.OutputLocation("outcrop", index=-1),
    )

    # Amplification (absolute value)
    amp = abs(tf.values)

Fourier Amplitude Spectra
--------------------------

:class:`FourierAmplitudeSpectrumOutput` stores the Fourier amplitude spectrum at a location:

.. code-block:: python

    fas = pystrata.output.FourierAmplitudeSpectrumOutput(
        location=pystrata.output.OutputLocation("within", index=0),
    )

Time Series
-----------

:class:`AccelerationTSOutput` saves the acceleration time series at a location
(requires a :class:`~pystrata.motion.TimeSeriesMotion`):

.. code-block:: python

    ts = pystrata.output.AccelerationTSOutput(
        location=pystrata.output.OutputLocation("within", index=0),
    )

:class:`StrainTSOutput` and :class:`StressTSOutput` provide strain and stress histories
within a layer.

Profile-Based Outputs
---------------------

Profile outputs compute a quantity at every layer and return a depth profile:

.. list-table::
   :header-rows: 1

   * - Class
     - Quantity
   * - :class:`MaxStrainProfile`
     - Peak shear strain vs. depth
   * - :class:`MaxAccelProfile`
     - Peak acceleration vs. depth
   * - :class:`DampingProfile`
     - Strain-compatible damping vs. depth
   * - :class:`ShearModReducProfile`
     - G/G\ :sub:`max` vs. depth
   * - :class:`InitialVelProfile`
     - Initial shear-wave velocity vs. depth
   * - :class:`CompatVelProfile`
     - Strain-compatible shear-wave velocity vs. depth
   * - :class:`CyclicStressRatioProfile`
     - CSR for liquefaction screening vs. depth

.. code-block:: python

    strain = pystrata.output.MaxStrainProfile()
    # strain.refs → depths (m); strain.values → peak shear strain (decimal)

Fixed Depth Grid
~~~~~~~~~~~~~~~~

By default each realization stores its own layer depths. Those differ whenever
the layering is randomized or :meth:`~pystrata.site.Profile.auto_discretize` is
applied per realization, so ``refs`` becomes a 2-D, NaN-padded array and
results cannot be compared depth-by-depth without interpolating first.

Passing ``depths`` resamples every realization onto a fixed grid instead. Build
one with :meth:`~pystrata.site.Profile.depth_grid`, which by default resolves
the same wavelengths as ``auto_discretize`` and extends past the range a
half-space depth variation can sample:

.. code-block:: python

    var_depth = pystrata.variation.HalfSpaceDepthVariation(norm(loc=60, scale=5))

    # Spacing from wave_frac * min(Vs) / max_freq; extent from the variation
    grid = profile.depth_grid(depth_var=var_depth)

    # Or choose the spacing yourself
    grid = profile.depth_grid(spacing=0.5, depth_var=var_depth)

    strain = pystrata.output.MaxStrainProfile(depths=grid)

The results then have a constant shape, which also lets them be pre-allocated
and written by index -- see :doc:`parallel`.

Resampling at collection time is **lossy**: layer interfaces are snapped to the
next grid node and the original depths cannot be recovered. Leave ``depths``
unset if you need the exact interfaces, and pass ``ref=`` to
:meth:`calc_stats` when reporting.

If the grid is shorter than a realization's profile, the results below it are
discarded and a warning is issued once.

Below the base of a realization
"""""""""""""""""""""""""""""""

A grid that covers the depth variation is deeper than most realizations. What
is reported below a given realization's base is set by ``fill_below``:

``'nan'``
    Default in gridded mode. Those depths are excluded from the statistics, so
    only realizations that actually reach a depth contribute to it.

``'hold'``
    Default otherwise, and the historical behavior. The deepest soil value is
    repeated, which mixes real values with extrapolated ones at depth.

Because the number of contributing realizations then varies with depth,
:meth:`calc_stats` reports a ``count`` alongside the median.

Statistics
~~~~~~~~~~

:meth:`calc_stats` returns the lognormal median, ``ln_std``, and ``count`` over
the realizations, and :meth:`to_dataframe` returns the realizations themselves.
Both accept a ``ref`` grid; without one, gridded outputs report on their own
grid and others on a common grid spanning the deepest realization.

.. code-block:: python

    stats = strain.calc_stats(as_dataframe=True)
    stats["median"].plot()

    # Report on a grid of your choosing
    stats = strain.calc_stats(ref=np.arange(0, 50, 0.5))

Accessing Results
-----------------

All outputs expose:

- ``.refs`` — independent variable (periods, frequencies, or depths)
- ``.values`` — computed quantity (1-D for single run, 2-D for multiple runs)
- ``.names`` — label for each run
- ``.iter_results()`` — iterator yielding ``(name, refs, values)`` tuples

Exporting to xarray
~~~~~~~~~~~~~~~~~~~

:meth:`Output.to_xarray` collects all runs into a labeled array:

.. code-block:: python

    da = rs.to_xarray()          # dims: (freq, realization)
    da.mean("realization").plot()

When the references vary between realizations -- an ungridded profile output --
an ``xarray.Dataset`` is returned instead, with the depths as a 2-D data
variable rather than a coordinate. Making them a coordinate would align on
every distinct depth, producing an array that is almost entirely empty.

Given a logic tree, results are reshaped into one dimension per node instead:

.. code-block:: python

    da = rs.to_xarray(tree)      # dims: (freq, node1, node2, ...)
