Working with Ground Motions
===========================

Ground motions are the input to site response analysis, representing the seismic excitation at the base of the soil profile. pyStrata supports several motion types and formats to accommodate different analysis workflows.

Motion Types
------------

pyStrata provides several motion classes for different types of seismic input:

**Time Series Motions**
    Direct representation of ground motion as acceleration, velocity, or displacement time series

**Random Vibration Theory (RVT) Motions**
    Frequency-domain representation using Fourier amplitude spectra and duration

**Source-Based Motions**
    Theoretical motions generated from earthquake source parameters

Time Series Motions
--------------------

The most common input format is acceleration time series from recorded or simulated earthquakes.

**Loading from Files**

pyStrata supports several standard formats:

.. code-block:: python

    # AT2 format (common in earthquake engineering)
    motion = pystrata.motion.TimeSeriesMotion.load_at2_file("motion.at2")

    # SMC format (from CESMD/COSMOS)
    motion = pystrata.motion.TimeSeriesMotion.load_smc_file("motion.smc")

**Manual Creation**

You can also create time series motions directly:

.. code-block:: python

    motion = pystrata.motion.TimeSeriesMotion(
        filename="custom_motion",
        description="Custom acceleration time series",
        time_step=0.01,  # seconds
        accels=[0.1, 0.2, 0.15, 0.05, ...]  # acceleration values in g
    )

**Properties and Methods**

Time series motions provide several useful properties:

.. code-block:: python

    motion.duration        # Total duration (s)
    motion.time_step       # Time step (s)
    motion.times           # Time vector
    motion.accels          # Acceleration values (g)
    motion.freqs           # Frequency vector
    motion.fourier_amps    # Fourier amplitude spectrum

Random Vibration Theory Motions
-------------------------------

RVT motions are defined by their Fourier amplitude spectrum and strong motion duration. They are computationally efficient for parametric studies.

**Source-Based RVT Motions**

Generate theoretical motions from earthquake source parameters using
``pygmm.fourier_spectrum.SourceTheoryModel`` and the
:meth:`~pystrata.motion.RvtMotion.from_fas` factory:

.. code-block:: python

    import pygmm.fourier_spectrum as fs
    fas = fs.SourceTheoryModel(
        magnitude=6.5,    # Moment magnitude
        distance=20,      # Source-to-site distance (km)
        region="wna",     # Western North America attenuation
    )
    motion = pystrata.motion.RvtMotion.from_fas(fas)

**Custom RVT Motions**

Define motions from Fourier amplitude spectra:

.. code-block:: python

    motion = pystrata.motion.RvtMotion(
        freqs=[0.1, 0.2, 0.5, 1.0, 2.0, 5.0, 10.0],
        fourier_amps=[0.01, 0.02, 0.05, 0.08, 0.06, 0.03, 0.01],
        duration=20.0,          # Strong motion duration (s)
        peak_factor=2.5         # Peak factor for conversion
    )

Motion Processing
-----------------

**Frequency Domain Calculations**

All motions provide access to frequency domain representations:

.. code-block:: python

    # Compute Fourier amplitude spectrum
    motion.calc_fourier_amps()

    # Access frequency domain data
    freqs = motion.freqs
    fourier_amps = motion.fourier_amps

**Response Spectra**

Compute response spectra for any motion:

.. code-block:: python

    periods = [0.01, 0.02, 0.05, 0.1, 0.2, 0.5, 1.0, 2.0, 5.0]
    damping = 0.05  # 5% damping

    resp_spec = motion.calc_response_spectrum(periods, damping)

**Intensity Measures**

Common intensity measures are available:

.. code-block:: python

    motion.pga             # Peak ground acceleration
    motion.pgv             # Peak ground velocity (if available)
    motion.arias_intensity # Arias intensity
    motion.significant_duration  # Duration for 5-95% of Arias intensity

File Format Details
-------------------

**AT2 Format**

AT2 is a text format commonly used in earthquake engineering:

::

    FILENAME.AT2
    Description line
    NPTS=  1000, DT= 0.005 SEC
    acceleration_value_1  acceleration_value_2  acceleration_value_3
    acceleration_value_4  acceleration_value_5  acceleration_value_6
    ...

**Units and Conventions**

- **Time**: seconds
- **Acceleration**: g (acceleration of gravity)
- **Frequency**: Hz
- **Fourier Amplitudes**: g·s

Examples
--------

- :doc:`../examples/example-01`: a time series motion.
- :doc:`../examples/example-02`: an RVT motion.
- :doc:`../examples/example-07` and :doc:`../examples/example-10`: peak factor
  calculators.
- :doc:`../examples/example-14`: several input motions.
