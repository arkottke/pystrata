User Guide
==========

This guide provides practical information for using PyStrata effectively in your site response analysis workflows.

.. toctree::
   :maxdepth: 2

   motions
   profiles
   calculations
   outputs
   logic_trees

Overview
--------

PyStrata follows a modular design philosophy where site response analysis is broken down into distinct, interchangeable components:

**Input Motion** (:doc:`motions`)
    Specification of ground motion at the input location (typically bedrock)

**Site Profile** (:doc:`profiles`)
    Definition of the layered soil model with material properties

**Calculator** (:doc:`calculations`)
    Selection of the analysis method (linear, equivalent linear, nonlinear)

**Output Specification** (:doc:`outputs`)
    Definition of desired output quantities and locations

**Uncertainty Handling** (:doc:`logic_trees`)
    Tools for probabilistic analysis and uncertainty quantification

This modular approach allows users to mix and match components as needed for their specific analysis requirements.

Typical Workflow
----------------

A typical PyStrata analysis follows these steps:

1. **Load or define input motion**

   .. code-block:: python

      motion = pystrata.motion.TimeSeriesMotion.load_at2_file("motion.at2")

2. **Create site profile**

   .. code-block:: python

      profile = pystrata.site.Profile([
          pystrata.site.Layer(soil_type, thickness, vs),
          # ... additional layers
      ])

3. **Select analysis method**

   .. code-block:: python

      calc = pystrata.propagation.EquivalentLinearCalculator()

4. **Define outputs**

   .. code-block:: python

      outputs = pystrata.output.OutputCollection([
          pystrata.output.ResponseSpectrumOutput(),
          # ... additional outputs
      ])

5. **Run analysis**

   .. code-block:: python

      calc(motion, profile, profile.location("outcrop", index=-1))

6. **Extract and visualize results**

   .. code-block:: python

      for output in outputs:
          plt.plot(output.x_values, output.values, label=output.name)

Best Practices
--------------

**Profile Definition**
    * Use appropriate material damping values (typically 2-8% for soil)
    * Ensure adequate layer resolution for wave propagation
    * Include realistic property contrasts between layers
    * Specify bedrock as a half-space (thickness = 0)

**Motion Selection**
    * Use motions appropriate for the site hazard level
    * Consider frequency content compatibility with profile resonance
    * For RVT analysis, ensure adequate frequency range and resolution

**Method Selection**
    * Linear elastic: preliminary analyses, stiff sites, low ground motion levels
    * Equivalent linear: most common method, moderate nonlinearity
    * Frequency domain: strong nonlinearity, advanced applications

**Output Planning**
    * Select outputs relevant to your engineering application
    * Consider computational cost for large parametric studies
    * Use appropriate frequency/period ranges for your needs

Common Pitfalls
---------------

**Insufficient Frequency Resolution**
    Can lead to aliasing or missing important frequency content

**Inappropriate Boundary Conditions**
    Incorrect specification of input motion type or location

**Convergence Issues**
    Equivalent linear method may not converge for very soft soils or strong motions

**Units Consistency**
    Ensure consistent units throughout (SI or Imperial)

**Profile Truncation**
    Insufficient depth to bedrock can cause artificial reflections

Performance Considerations
--------------------------

**Computational Efficiency**
    * RVT methods are typically faster than time series
    * Linear methods are faster than nonlinear
    * Reduce frequency resolution for parametric studies

**Memory Usage**
    * Time series analyses require more memory
    * Profile suites and logic trees can consume significant memory
    * Consider batch processing for large studies

**Parallel Processing**
    * Logic tree analyses can be parallelized
    * Profile suite calculations benefit from parallel execution
    * Individual analyses are typically single-threaded

Validation and Quality Assurance
--------------------------------

**Check Results Reasonableness**
    * Verify amplification patterns match expected behavior
    * Compare with simplified hand calculations where possible
    * Check for conservation of energy in linear analyses

**Sensitivity Analysis**
    * Test sensitivity to key parameters
    * Vary profile properties within reasonable ranges
    * Assess impact of method selection

**Code Verification**
    * Compare with other site response codes
    * Use benchmark problems for validation
    * Test limiting cases with analytical solutions
