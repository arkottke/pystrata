.. _example-gallery:

Example Gallery
===============

The examples are notebooks in the ``examples`` directory of the repository, and are
executed by the test suite. They are grouped by the part of pyStrata that they
use; :doc:`../user-guide/index` describes how the parts fit together, and each page
of the guide lists its examples.

Analyses
--------

**Basic Time Series Analysis** (:doc:`example-01`)
    Time series analysis to compute surface response spectrum and site amplification functions.

**Random Vibration Theory** (:doc:`example-02`)
    RVT analysis to compute surface response spectrum and site amplification functions.

**Multiple Calculators** (:doc:`example-03`)
    RVT input motion with a variety of site response calculators.

**Quarter-Wave Analysis** (:doc:`example-09`)
    Quarter-wave length site amplification.

**Multiple Motions** (:doc:`example-14`)
    Example with multiple input motions and simulated soil profiles.

Calculator settings
-------------------

**FDM Properties** (:doc:`example-11`)
    Extract soil properties from FDM site response calculations.

**Frequency Dependent Damping** (:doc:`example-12`)
    Show the impact of frequency dependent D_min on FDM methods.

**Strain Limits** (:doc:`example-13`)
    Show the impact of the strain limit on the EQL calculation.

**Incidence Angle and Inhomogeneity** (:doc:`example-19`)
    Effect of an obliquely incident or inhomogeneous SH wave, compared with the default vertically incident, homogeneous wave.

Motions and peak factors
------------------------

**Peak Factor Methods** (:doc:`example-07`)
    RVT calculation using different peak factor calculators.

**Peak Factor Suites** (:doc:`example-10`)
    Compute ground motion using a suite of peak factors.

Profiles and soil properties
----------------------------

**Geopsy Profiles** (:doc:`example-06`)
    Time series analysis using velocity profiles from geopsy.

**Nonlinear Curves** (:doc:`example-15`)
    List of provided nonlinear curves.

**USGS National Crustal Model Profile** (:doc:`example-17`)
    Load a geophysical profile from the USGS NCM, apply Darendeli nonlinear curves, and adjust minimum damping to match a target site kappa (κ₀).

**Unit Conversion with Pint** (:doc:`example-18`)
    Define the same profile three ways — plain floats, metric pint Quantities, and Imperial units — demonstrating transparent unit handling.

Variability and uncertainty
---------------------------

**Profile Generation** (:doc:`example-04`)
    Generate simulated shear-wave velocity profiles.

**Nonlinear Properties** (:doc:`example-05`)
    Generate simulated nonlinear properties.

**Profile Suites** (:doc:`example-08`)
    RVT calculation using a suite of simulated profiles.

**Logic Trees** (:doc:`example-16`)
    Comprehensive example of using logic trees for uncertainty analysis.

Dams
----

**Dam Response** (:doc:`example-20`)
    Shear-wedge dam of Gazetas and Dakoulas (1992) joined to a layered site: power-law velocity profiles, tuning the dam to a target frequency, and linear and equivalent-linear analyses.

**Dam Response vs. Hale and Abrahamson (2019)** (:doc:`example-21`)
    Compares the Hale and Abrahamson sliding-mass transfer function in pygmm with the average acceleration of the sliding mass inside a toe circle of a trapezoidal shear-wedge dam, including nonlinearity for M 6.5 events at 5 and 30 km, the sensitivity to the failure surface, and a dam stacked on a site profile.

**Truncated Wedge vs. the Apex Solution** (:doc:`example-22`)
    Shows the effect of the crest width on the shear-wedge dam: the natural frequencies, the closed-form and numerical transfer functions of the truncated and apex wedges for a homogeneous dam, and an inhomogeneous dam.

.. toctree::
   :maxdepth: 1
   :hidden:

   example-01
   example-02
   example-03
   example-04
   example-05
   example-06
   example-07
   example-08
   example-09
   example-10
   example-11
   example-12
   example-13
   example-14
   example-15
   example-16
   example-17
   example-18
   example-19
   example-20
   example-21
   example-22

Running the examples
--------------------

Clone the repository and run a notebook from the ``examples`` directory. The data
files that the examples read are in ``examples/data``.
