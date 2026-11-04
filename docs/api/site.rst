Site Module
===========

The site module provides classes for defining layered soil profiles and material properties.

.. currentmodule:: pystrata.site

Classes
-------

.. autosummary::
   :nosignatures:

   Profile
   Layer
   SoilType
   Location
   NonlinearCurve
   ModulusReductionCurve
   DampingCurve
   layers_from_depth_power_law
   layers_from_stress_power_law

Site Profile
------------

.. autoclass:: Profile
   :members:
   :show-inheritance:

Layers and Materials
--------------------

.. autoclass:: Layer
   :members:
   :show-inheritance:

.. autoclass:: SoilType
   :members:
   :show-inheritance:

Output Locations
----------------

.. autoclass:: Location
   :members:
   :show-inheritance:

Nonlinear Curves
----------------

.. autoclass:: NonlinearCurve
   :members:
   :show-inheritance:

.. autoclass:: ModulusReductionCurve
   :members:
   :show-inheritance:

.. autoclass:: DampingCurve
   :members:
   :show-inheritance:

Power-Law Layers
----------------

.. autofunction:: layers_from_depth_power_law

.. autofunction:: layers_from_stress_power_law

Complex Modulus
---------------

.. autofunction:: calc_comp_factor
