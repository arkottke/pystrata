Quickstart Guide
================

This guide will get you up and running with PyStrata for basic site response analysis.

Basic Concepts
--------------

PyStrata models site response using four main components:

1. **Motion** - Input ground motion (time series or frequency domain)
2. **Profile** - Layered soil model with material properties
3. **Calculator** - Analysis method (time series, RVT, linear, nonlinear)
4. **Outputs** - Computed quantities (response spectra, transfer functions, etc.)

Simple Linear Analysis
----------------------

Let's start with the most basic analysis - linear elastic site response:

.. code-block:: python

    import pystrata

    # Load an input motion
    motion = pystrata.motion.TimeSeriesMotion.load_at2_file("path/to/motion.at2")

    # Create a simple two-layer profile
    profile = pystrata.site.Profile([
        # Soil layer: 30m thick, Vs=400 m/s, damping=5%
        pystrata.site.Layer(
            pystrata.site.SoilType("Soil", 18.0, None, 0.05),
            30, 400
        ),
        # Bedrock: half-space, Vs=1200 m/s, damping=1%
        pystrata.site.Layer(
            pystrata.site.SoilType("Rock", 24.0, None, 0.01),
            0, 1200
        ),
    ])

    # Define outputs of interest
    outputs = pystrata.output.OutputCollection([
        pystrata.output.ResponseSpectrumOutput(),
        pystrata.output.AccelTransferFunctionOutput(),
        pystrata.output.MaxAccelProfile(),
    ])

    # Create calculator and run analysis
    calc = pystrata.propagation.LinearElasticCalculator()
    calc(motion, profile, profile.location("outcrop", index=-1))

    # Access results
    for output in outputs:
        print(f"{output.name}: {output.values}")

Understanding the Profile
-------------------------

The soil profile is the foundation of any site response analysis. Each layer is defined by:

**Soil Properties**
    * Unit weight (kN/m³ or lb/ft³)
    * Shear modulus or shear wave velocity
    * Material damping ratio
    * Nonlinear stress-strain curves (optional)

**Layer Geometry**
    * Thickness (0 for half-space)
    * Depth to top of layer

**Example with More Layers**

.. code-block:: python

    profile = pystrata.site.Profile([
        # Surface clay
        pystrata.site.Layer(
            pystrata.site.SoilType("Clay", 17.0, None, 0.08),
            5, 200
        ),
        # Medium sand
        pystrata.site.Layer(
            pystrata.site.SoilType("Sand", 19.0, None, 0.04),
            15, 350
        ),
        # Dense sand
        pystrata.site.Layer(
            pystrata.site.SoilType("Dense Sand", 20.0, None, 0.03),
            20, 600
        ),
        # Bedrock
        pystrata.site.Layer(
            pystrata.site.SoilType("Rock", 24.0, None, 0.01),
            0, 1200
        ),
    ])

Working with Different Motion Types
-----------------------------------

**Time Series from File**

.. code-block:: python

    # AT2 format (common in earthquake engineering)
    motion = pystrata.motion.TimeSeriesMotion.load_at2_file("motion.at2")

    # Direct specification
    motion = pystrata.motion.TimeSeriesMotion(
        filename="custom_motion",
        description="Custom time series",
        time_step=0.01,  # seconds
        accels=[0.1, 0.2, 0.15, ...]  # acceleration values in g
    )

**Random Vibration Theory (RVT)**

.. code-block:: python

    # Source-based theoretical motion
    motion = pystrata.motion.SourceTheoryRvtMotion(
        magnitude=6.5,      # Moment magnitude
        distance=20,        # Source-to-site distance (km)
        region="wna"        # Western North America
    )

    # RVT from Fourier amplitude spectrum
    motion = pystrata.motion.RvtMotion(
        freqs=[0.1, 0.2, 0.5, 1.0, 2.0, 5.0, 10.0],
        fourier_amps=[0.01, 0.02, 0.05, 0.08, 0.06, 0.03, 0.01],
        duration=20.0,      # Strong motion duration
        peak_factor=2.5     # Peak factor for conversion to peak values
    )

Analysis Methods
----------------

**Linear Elastic**
    Fastest method, assumes constant soil properties

.. code-block:: python

    calc = pystrata.propagation.LinearElasticCalculator()

**Equivalent Linear**
    Iterative method accounting for strain-dependent soil properties

.. code-block:: python

    calc = pystrata.propagation.EquivalentLinearCalculator()

**Frequency Domain Method**
    Advanced nonlinear approach using frequency domain convolution

.. code-block:: python

    calc = pystrata.propagation.FrequencyDomainCalculator()

Output Options
--------------

PyStrata provides many output types for different analysis needs:

**Response Spectra**

.. code-block:: python

    # Surface response spectrum
    pystrata.output.ResponseSpectrumOutput()

    # Response spectrum ratio (surface/input)
    pystrata.output.ResponseSpectrumRatioOutput()

**Transfer Functions**

.. code-block:: python

    # Acceleration transfer function
    pystrata.output.AccelTransferFunctionOutput()

    # Amplification function (absolute value)
    pystrata.output.AccelTransferFunctionOutput(ko_bandwidth=30)

**Profile Outputs**

.. code-block:: python

    # Maximum acceleration vs. depth
    pystrata.output.MaxAccelProfile()

    # Maximum strain vs. depth
    pystrata.output.MaxStrainProfile()

**Time Series**

.. code-block:: python

    # Surface acceleration time series
    pystrata.output.AccelTimeSeriesOutput(pystrata.output.OutputLocation("outcrop", 0))

Visualization Example
---------------------

Here's a complete example that generates plots:

.. code-block:: python

    import matplotlib.pyplot as plt
    import pystrata

    # Setup
    motion = pystrata.motion.SourceTheoryRvtMotion(6.5, 20, "wna")
    profile = pystrata.site.Profile([
        pystrata.site.Layer(pystrata.site.SoilType("Soil", 18.0, None, 0.05), 30, 400),
        pystrata.site.Layer(pystrata.site.SoilType("Rock", 24.0, None, 0.01), 0, 1200),
    ])

    outputs = pystrata.output.OutputCollection([
        pystrata.output.ResponseSpectrumOutput(),
        pystrata.output.AccelTransferFunctionOutput(),
    ])

    # Run analysis
    calc = pystrata.propagation.LinearElasticCalculator()
    calc(motion, profile, profile.location("outcrop", index=-1))

    # Plot results
    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(12, 5))

    # Response spectrum
    rs_output = outputs[0]
    ax1.loglog(rs_output.periods, rs_output.values)
    ax1.set_xlabel("Period (s)")
    ax1.set_ylabel("Spectral Acceleration (g)")
    ax1.set_title("Surface Response Spectrum")
    ax1.grid(True)

    # Transfer function
    tf_output = outputs[1]
    ax2.semilogx(tf_output.freqs, abs(tf_output.values))
    ax2.set_xlabel("Frequency (Hz)")
    ax2.set_ylabel("Amplification Factor")
    ax2.set_title("Acceleration Transfer Function")
    ax2.grid(True)

    plt.tight_layout()
    plt.show()

Next Steps
----------

Now that you understand the basics, explore:

* :doc:`examples/index` - Real-world analysis examples
* :doc:`user_guide/index` - Detailed guidance on each component
* :doc:`theory/index` - Mathematical background and theory
* :doc:`api/index` - Complete API reference

For advanced features like uncertainty analysis with logic trees, see `example-16.ipynb <../examples/example-16.ipynb>`__.
