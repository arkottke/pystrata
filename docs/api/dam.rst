Dam Module
==========

The dam module computes the seismic response of a dam modeled as a stack of
shear-wedge layers joined to a layered site (:cite:t:`gazetas1992seismic`). It reuses
the linear-elastic, equivalent-linear and frequency-dependent calculators.

.. currentmodule:: pystrata.dam

Classes
-------

.. autosummary::
   :nosignatures:

   Dam
   WedgeLayer
   DamProfile
   DamLinearElasticCalculator
   DamEquivalentLinearCalculator
   DamFrequencyDependentEqlCalculator

Dam and Profile
---------------

.. autoclass:: Dam
   :members:
   :show-inheritance:

.. autoclass:: WedgeLayer
   :show-inheritance:

.. autoclass:: DamProfile
   :members:
   :show-inheritance:

Calculators
-----------

.. autoclass:: DamLinearElasticCalculator
   :show-inheritance:

.. autoclass:: DamEquivalentLinearCalculator
   :show-inheritance:

.. autoclass:: DamFrequencyDependentEqlCalculator
   :show-inheritance:

Closed-Form Solutions
---------------------

Table 2 of Gazetas and Dakoulas (1992), for a homogeneous or power-law dam.

.. autofunction:: pystrata.dam.analytical.wedge_order

.. autofunction:: pystrata.dam.analytical.wedge_natural_freqs

.. autofunction:: pystrata.dam.analytical.wedge_mode_shape

.. autofunction:: pystrata.dam.analytical.wedge_rigid_base_tf

.. autofunction:: pystrata.dam.analytical.wedge_elastic_base_tf
