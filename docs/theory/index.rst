Theory and Methods
==================

This section provides the mathematical and theoretical foundation for site response analysis methods implemented in PyStrata.

.. toctree::
   :maxdepth: 2

   site_response
   wave_propagation
   nonlinear_models
   uncertainty

Overview
--------

Site response analysis relies on the fundamental principles of wave propagation in layered media. The one-dimensional assumption treats seismic waves as vertically propagating shear waves (SH waves) through horizontally layered soil deposits.

The governing physics include:

**Wave Equation**
    The equation of motion for shear wave propagation in a continuous medium

**Boundary Conditions**
    Stress and displacement continuity at layer interfaces

**Material Constitutive Models**
    Relationships between stress, strain, and material properties

**Damping Mechanisms**
    Energy dissipation through material and radiation damping

Mathematical Framework
----------------------

The theoretical foundation is built on:

1. **Linear Wave Theory** - For small strain elastic wave propagation
2. **Transfer Functions** - Frequency domain representation of system response
3. **Equivalent Linear Method** - Iterative approach for strain-compatible properties
4. **Random Vibration Theory** - Statistical treatment of stochastic ground motion
5. **Uncertainty Propagation** - Monte Carlo and logic tree methods

Key Assumptions
---------------

Standard site response analysis makes several simplifying assumptions:

* **One-dimensional propagation** - Waves travel vertically through horizontal layers
* **Linear viscoelastic behavior** - For equivalent linear methods
* **Uniform layer properties** - Homogeneous properties within each layer
* **Perfect layer bonding** - No sliding at interfaces
* **Infinite lateral extent** - No boundary effects from finite dimensions

These assumptions are reasonable for most engineering applications but may require modification for complex site geometries or extreme ground motions.

Implementation Notes
--------------------

PyStrata implements these theoretical methods with careful attention to:

* **Numerical stability** - Robust algorithms for wave propagation calculations
* **Frequency resolution** - Adequate sampling for accurate results
* **Convergence criteria** - Appropriate tolerances for iterative methods
* **Validation** - Comparison with analytical solutions and benchmark problems
