Introduction
============

What is Site Response Analysis?
-------------------------------

Site response analysis is a fundamental tool in earthquake engineering used to predict how seismic waves are modified as they propagate through soil layers from bedrock to the ground surface. This analysis is crucial for:

* **Seismic hazard assessment** - Understanding local amplification effects
* **Building design** - Determining design ground motions for structures
* **Risk assessment** - Evaluating earthquake impacts on infrastructure
* **Code development** - Supporting seismic design provisions

The method models the subsurface as a series of horizontal layers, each with distinct soil properties. As seismic waves travel upward through these layers, they undergo amplification, filtering, and nonlinear modification depending on the soil characteristics and input motion intensity.

Key Physical Processes
----------------------

**Wave Propagation**
    Seismic waves travel as shear waves through the soil column, with velocities determined by soil stiffness and density. The one-dimensional assumption treats wave propagation as vertically incident SH waves.

**Impedance Contrasts**
    Differences in soil properties between layers create impedance contrasts that reflect and transmit energy, leading to amplification patterns that depend on the layer geometry and property contrasts.

**Soil Nonlinearity**
    At higher strain levels, soil exhibits nonlinear stress-strain behavior and hysteretic damping. This reduces soil stiffness and increases damping, which can limit amplification during strong shaking.

**Resonance Effects**
    The soil column acts as a resonator, with fundamental and higher-mode frequencies determined by layer thicknesses and shear wave velocities. Ground motions are amplified near these resonant frequencies.

PyStrata Capabilities
---------------------

PyStrata implements state-of-the-art methods for site response analysis:

**Analysis Methods**
    * Time series analysis for nonstationary and nonlinear behavior
    * Random vibration theory (RVT) for efficient statistical analysis
    * Linear elastic and equivalent-linear approaches
    * Frequency-domain methods for soil nonlinearity

**Input Motion Handling**
    * Time series from accelerometer records
    * Fourier amplitude spectra
    * Source-based theoretical motions
    * Response spectrum compatible time series

**Soil Modeling**
    * Laboratory-based nonlinear soil models (Darendeli, Zhang et al.)
    * User-defined stress-strain curves
    * Frequency-dependent material damping
    * Layered profile generation and simulation

**Uncertainty Analysis**
    * Logic trees for systematic uncertainty propagation
    * Monte Carlo simulation capabilities
    * Sensitivity analysis tools
    * Statistical output processing

Scientific Foundation
---------------------

PyStrata is built on decades of research in computational seismology and geotechnical earthquake engineering. The theoretical foundation draws from:

* **Wave propagation theory** :cite:p:`Kramer1996` for fundamental wave mechanics
* **Soil dynamics** :cite:p:`Ishihara1996` for nonlinear soil behavior models
* **Computational methods** :cite:p:`Yoshida2002` for numerical implementation
* **Uncertainty quantification** :cite:p:`Bommer2005` for probabilistic analysis

The library implements peer-reviewed algorithms and has been validated against industry-standard software and recorded earthquake data.

Applications
------------

PyStrata is used for a wide range of applications in earthquake engineering:

**Research**
    * Development of new site response methods
    * Validation studies using earthquake recordings
    * Parametric studies of soil behavior effects
    * Ground motion prediction equation development

**Engineering Practice**
    * Site-specific ground motion studies
    * Seismic hazard analysis for critical facilities
    * Building code development and calibration
    * Performance-based earthquake engineering

**Education**
    * Teaching site response fundamentals
    * Demonstrating wave propagation concepts
    * Exploring parameter sensitivity
    * Hands-on analysis experience

Getting Started
---------------

The best way to get started with PyStrata is to:

1. :doc:`Install the package <install>` using pip or conda
2. Work through the :doc:`quickstart guide <quickstart>` for basic concepts
3. Explore the :doc:`examples gallery <examples/index>` for real-world applications
4. Consult the :doc:`API reference <api/index>` for detailed documentation

The :doc:`theory section <theory/index>` provides mathematical background, while the :doc:`user guide <user_guide/index>` offers practical guidance for analysis workflows.
