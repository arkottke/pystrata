Variation Module
================

The variation module provides tools for parameter variation and uncertainty analysis.

.. currentmodule:: pystrata.variation

Classes
-------

.. autosummary::
   :nosignatures:

   ToroVelocityVariation
   SpidVariation
   SoilTypeVariation
   LimitedNorm

Velocity Variations
-------------------

.. autoclass:: ToroVelocityVariation
   :members:
   :show-inheritance:

.. autoclass:: SpidVariation
   :members:
   :show-inheritance:

Soil Type Variations
--------------------

.. autoclass:: SoilTypeVariation
   :members:
   :show-inheritance:

Limiting the Variation
----------------------

The velocity and soil type variations limit the random variates to 2 standard
deviations. The ``limit_mode`` argument of the variations selects how, and
:class:`LimitedNorm` generates the variates.

.. list-table::
   :header-rows: 1
   :widths: 24 76

   * - ``limit_mode``
     - Distribution of the variates
   * - ``"normal"``
     - A normal distribution with no limit.
   * - ``"clipped"``
     - A normal distribution whose values beyond 2 standard deviations are set
       to the limit. Clipping removes variance, so the realized standard
       deviation is 0.96 of the model's.
   * - ``"clipped_inflated"`` (default)
     - The standard deviation is increased by a factor of 1.042 before the
       values are clipped at 2 of the increased standard deviations, so the
       realized standard deviation is the model's.

.. autoclass:: LimitedNorm
   :members:
   :show-inheritance:

.. autodata:: LIMIT_MODES
