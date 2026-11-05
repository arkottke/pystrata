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
   RectangularCanyon
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

Canyon Effects
--------------

A dam in a canyon of finite length is confined by the abutments, which raises its
natural frequencies and makes the response vary along its axis. Give the
:class:`Dam` a :class:`RectangularCanyon`, a canyon of constant length with rigid
abutments that move with the outcrop motion of the half space.

With :math:`w = u / u_{out}` and the longitudinal wave numbers
:math:`\kappa_j = j \pi / L` for odd :math:`j`, the displacement is
:math:`w = 1 + \sum_j \sin(\kappa_j x) V_j(r)`. In each layer :math:`V_j` solves the
Bessel equation of the 2D wedge with :math:`k_j^2 = k^2 - \kappa_j^2` and a constant
particular solution :math:`-4 k^2 / (j \pi k_j^2)`. The recursion of the 2D dam
is reused for every mode; the particular solutions only add a jump of the
displacement at the interfaces between layers.

The site is taken as independent 1D columns under the dam. The shear stress at
the base is then linear in the displacement, which is a Robin condition that
couples the base of each mode to the site and keeps the modes independent. The
solution tends to the 2D dam as the canyon gets long. The response is at a
position :math:`x` along the canyon, set with the ``x`` argument of
:meth:`DamProfile.transfer_function` or the ``canyon_x`` attribute of a
calculator, and the center of the canyon by default.

For equivalent-linear calculations the strain of a layer is the peak of
:math:`\sqrt{\gamma_{rz}^2 + \gamma_{xy}^2}` at that position (the
frequency-dependent calculator uses :math:`\gamma_{rz}` only). The cost grows
with ``n_modes``, which should exceed :math:`2 f L / V_s` to resolve the response
near the abutments; the center of a canyon is converged with fewer. The sliding
mass transfer function is averaged over the length of the canyon.

.. autoclass:: RectangularCanyon
   :members:

.. autofunction:: pystrata.dam.canyon.canyon_rect_natural_freqs

.. autofunction:: pystrata.dam.canyon.canyon_freq_ratio

.. autofunction:: pystrata.dam.canyon.canyon_rect_rigid_base_tf

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
