.. _example-gallery:

Example Gallery
===============

.. toctree::
   :maxdepth: 1
   :caption: Example Notebooks
   :glob:

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

Complete Example Index
----------------------

All available examples with brief descriptions:

**Basic Time Series Analysis** (`example-01.ipynb <../../examples/example-01.ipynb>`__)
    Time series analysis to compute surface response spectrum and site amplification functions.

**Random Vibration Theory** (`example-02.ipynb <../../examples/example-02.ipynb>`__)
    RVT analysis to compute surface response spectrum and site amplification functions.

**Multiple Calculators** (`example-03.ipynb <../../examples/example-03.ipynb>`__)
    RVT input motion with a variety of site response calculators.

**Profile Generation** (`example-04.ipynb <../../examples/example-04.ipynb>`__)
    Generate simulated shear-wave velocity profiles.

**Nonlinear Properties** (`example-05.ipynb <../../examples/example-05.ipynb>`__)
    Generate simulated nonlinear properties.

**Geopsy Profiles** (`example-06.ipynb <../../examples/example-06.ipynb>`__)
    Time series analysis using velocity profiles from geopsy.

**Peak Factor Methods** (`example-07.ipynb <../../examples/example-07.ipynb>`__)
    RVT calculation using different peak factor calculators.

**Profile Suites** (`example-08.ipynb <../../examples/example-08.ipynb>`__)
    RVT calculation using a suite of simulated profiles.

**Quarter-Wave Analysis** (`example-09.ipynb <../../examples/example-09.ipynb>`__)
    Quarter-wave length site amplification.

**Peak Factor Suites** (`example-10.ipynb <../../examples/example-10.ipynb>`__)
    Compute ground motion using a suite of peak factors.

**FDM Properties** (`example-11.ipynb <../../examples/example-11.ipynb>`__)
    Extract soil properties from FDM site response calculations.

**Frequency Dependent Damping** (`example-12.ipynb <../../examples/example-12.ipynb>`__)
    Show the impact of frequency dependent D_min on FDM methods.

**Strain Limits** (`example-13.ipynb <../../examples/example-13.ipynb>`__)
    Show the impact of the strain limit on the EQL calculation.

**Multiple Motions** (`example-14.ipynb <../../examples/example-14.ipynb>`__)
    Example with multiple input motions and simulated soil profiles.

**Nonlinear Curves** (`example-15.ipynb <../../examples/example-15.ipynb>`__)
    List of provided nonlinear curves.

**Logic Trees** (`example-16.ipynb <../../examples/example-16.ipynb>`__)
    Comprehensive example of using logic trees for uncertainty analysis.

**USGS National Crustal Model Profile** (`example-17.ipynb <../../examples/example-17.ipynb>`__)
    Load a geophysical profile from the USGS NCM, apply Darendeli nonlinear curves, and adjust
    minimum damping to match a target site kappa (κ₀).

**Unit Conversion with Pint** (`example-18.ipynb <../../examples/example-18.ipynb>`__)
    Define the same profile three ways — plain floats, metric pint Quantities, and Imperial
    units — demonstrating transparent unit handling.

**Dam Response** (`example-20.ipynb <../../examples/example-20.ipynb>`__)
    Shear-wedge dam of Gazetas and Dakoulas (1992) joined to a layered site: power-law
    velocity profiles, tuning the dam to a target frequency, and linear and
    equivalent-linear analyses.

**Dam Response vs. Hale and Abrahamson (2019)** (`example-21.ipynb <../../examples/example-21.ipynb>`__)
    Compares the transfer function implied by the Hale and Abrahamson flexible
    sliding-mass model in pygmm with the shear-wedge dam, including nonlinearity for
    M 6.5 events at 5 and 30 km, and a dam stacked on a site profile.

Tips for Using Examples
------------------------

**Running Locally**
    Clone the repository and navigate to the examples directory. Each notebook can be run independently.

**Modifying Examples**
    The examples are designed to be educational and easily modified. Try changing parameters to see how they affect results.

**Data Files**
    Example data files are included in the `examples/data/` directory.
