Running Calculations
====================

pyStrata provides five calculator classes, each implementing a different level of
approximation for 1D wave propagation.

.. currentmodule:: pystrata.propagation

Overview
--------

All calculators share the same calling interface:

.. code-block:: python

    calc(motion, profile, loc_input)

where ``loc_input`` is the :class:`~pystrata.site.Location` at which the input motion is defined
(typically the base of the profile with wave-field ``"outcrop"`` or ``"within"``).

After the call, outputs registered in an :class:`~pystrata.output.OutputCollection` are populated.

Available Calculators
---------------------

.. list-table::
   :header-rows: 1
   :widths: 30 70

   * - Class
     - Description
   * - :class:`LinearElasticCalculator`
     - Frequency-domain transfer-matrix method, constant properties
   * - :class:`EquivalentLinearCalculator`
     - Iterative EQL: updates G/Gmax and damping to strain-compatible values
   * - :class:`FrequencyDependentEqlCalculator`
     - Frequency-dependent EQL variant (Kausel & Assimaki 2002)
   * - :class:`TimeDomainCalculator`
     - True nonlinear FDM with MKZ or HH constitutive model
   * - :class:`QuarterWaveLenCalculator`
     - Quarter-wavelength approximation for site amplification

Linear Elastic
--------------

Use :class:`LinearElasticCalculator` for preliminary analysis or stiff/rock sites where strain
levels are low:

.. code-block:: python

    calc = pystrata.propagation.LinearElasticCalculator()
    calc(motion, profile, profile.location("outcrop", index=-1))

Equivalent Linear
-----------------

:class:`EquivalentLinearCalculator` is the most common choice for practical site response.
It iterates the frequency-domain solution until the effective shear strain in each layer is
consistent with the strain-dependent modulus and damping curves:

.. code-block:: python

    calc = pystrata.propagation.EquivalentLinearCalculator(
        strain_ratio=0.65,      # effective / peak strain
        tolerance=0.025,        # convergence tolerance (decimal)
        max_iterations=15,
        strain_limit=0.05,      # cap strains at 5 % (prevents runaway)
    )
    calc(motion, profile, profile.location("outcrop", index=-1))

Frequency-Dependent EQL
------------------------

:class:`FrequencyDependentEqlCalculator` extends the EQL approach with a frequency-dependent
strain transfer function, which better captures the behavior of soft soils under strong shaking:

.. code-block:: python

    calc = pystrata.propagation.FrequencyDependentEqlCalculator()
    calc(motion, profile, profile.location("outcrop", index=-1))

Time-Domain (Nonlinear)
-----------------------

:class:`TimeDomainCalculator` uses explicit central-difference integration and supports two
constitutive models:

- ``"mkz"`` — Modified Kondner–Zelasko (hyperbolic)
- ``"hh"`` — Hashash-Hardin (extended modified hyperbolic)

The profile **must be discretized** before use (``Profile.auto_discretize``).  If Numba is
installed, inner loops are JIT-compiled automatically.

.. code-block:: python

    # Discretize for adequate spatial resolution
    profile.auto_discretize(max_freq=50.0)

    calc = pystrata.propagation.TimeDomainCalculator(
        model="hh",           # constitutive model: 'mkz' or 'hh'
        boundary="elastic",   # transmitting base: 'elastic' or 'rigid'
    )

    # Optionally inspect fitted constitutive parameters before running
    params = calc.prepare(profile)

    calc(motion, profile, profile.location("outcrop", index=-1))

    # Access time-series results directly
    surface_accel = calc.accel_ts(profile.location("outcrop", index=0))

Quarter-Wavelength Approximation
---------------------------------

:class:`QuarterWaveLenCalculator` computes site amplification using the quarter-wavelength
method. It does not require a full wave-propagation solution and is useful for rapid screening:

.. code-block:: python

    calc = pystrata.propagation.QuarterWaveLenCalculator(site_atten=0.03)
    calc(motion, profile, profile.location("outcrop", index=-1))

Choosing a Calculator
---------------------

.. list-table::
   :header-rows: 1
   :widths: 25 25 25 25

   * - Scenario
     - Recommended
     - Why
     - Notes
   * - Stiff site / low shaking
     - LinearElastic
     - Fast; exact for linear behavior
     - No iteration
   * - General site response
     - EquivalentLinear
     - Industry standard
     - Needs nonlinear curves
   * - Soft soil / strong shaking
     - FrequencyDependentEql
     - Frequency-dependent damping
     - Slower than basic EQL
   * - True nonlinear / time histories
     - TimeDomain
     - Captures hysteresis, pore pressure
     - Requires discretized profile + constitutive fit
   * - Rapid amplification estimate
     - QuarterWaveLen
     - No profile wave propagation
     - Approximate only
