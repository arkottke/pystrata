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
(only meaningful with :class:`~pystrata.propagation.TimeDomainCalculator` or
:class:`~pystrata.motion.TimeSeriesMotion`):

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

Accessing Results
-----------------

All outputs expose:

- ``.refs`` — independent variable (periods, frequencies, or depths)
- ``.values`` — computed quantity (1-D for single run, 2-D for multiple runs)
- ``.names`` — label for each run
- ``.iter_results()`` — iterator yielding ``(name, refs, values)`` tuples

Exporting to xarray
~~~~~~~~~~~~~~~~~~~

:meth:`Output.to_xarray` collects all runs into a labeled ``xarray.Dataset``:

.. code-block:: python

    ds = rs.to_xarray()
    ds["values"].mean("name").plot()  # mean over realizations
