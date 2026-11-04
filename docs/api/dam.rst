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
   CircularSurface
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

Failure Surfaces and the Sliding Mass
-------------------------------------

The transfer function of :cite:t:`Hale2019` is to the average acceleration of the
mass that slides. A failure surface cuts the sliding mass out of the dam, and its
acceleration is the average of the accelerations of the shear-beam solution at
each depth, weighted by the width of the mass. A surface is a circle given by its
center and radius, or by :func:`~pystrata.dam.toe_circle`, which passes through
the toe and the far side of the crest.

.. autoclass:: CircularSurface
   :members:

.. autofunction:: toe_circle

.. autofunction:: plot_failure_surface

.. autofunction:: sliding_mass_transfer_function

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

Truncated Wedge
~~~~~~~~~~~~~~~

A dam with a crest width is a wedge that is cut off below its apex, and the crest
is a free surface. These are the solutions for a homogeneous dam, which tend to
the ones above as the crest width goes to zero.

.. autofunction:: pystrata.dam.analytical.wedge_truncated_natural_freqs

.. autofunction:: pystrata.dam.analytical.wedge_truncated_rigid_base_tf

.. autofunction:: pystrata.dam.analytical.wedge_truncated_elastic_base_tf
