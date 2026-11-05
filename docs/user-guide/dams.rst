Dams
====

``pystrata.dam`` computes the response of an embankment dam, modeled as a shear
wedge (Gazetas and Dakoulas, 1992), joined to a layered site. The dam is described
by a stack of layers rather than by a profile, but it is analyzed with the same
calculators, locations and outputs.

.. code-block:: python

   from pystrata import dam, site

   fill = site.SoilType("rockfill", 21.0, None, 0.05)
   d = dam.Dam.from_inhomogeneity(
       fill, height=100.0, shear_vel_avg=300.0, m=2 / 3, n_layers=40
   )

   rock = site.SoilType("rock", 25.0, None, 0.0)
   foundation = site.Profile([site.Layer(rock, 1.0, 1500.0)])
   profile = dam.DamProfile(d, foundation)

   calc = dam.DamLinearElasticCalculator()
   loc_in = profile.location("outcrop", index=len(profile) - 1)
   calc(motion, profile, loc_in)

Geometry
--------

A :class:`~pystrata.dam.Dam` without a slope is the apex wedge of the paper: the
crest is the apex, and the width at a depth is proportional to the depth. With a
``slope`` (the horizontal run of each side per unit height) and a ``crest_width``
the wedge is truncated at the crest, which is then a free surface, and the width
is the crest width plus twice the slope times the depth. The closed-form
solutions of both are in ``pystrata.dam.analytical``.

Layers and tuning
-----------------

:meth:`~pystrata.dam.Dam.from_inhomogeneity` builds layers with a shear modulus
that is proportional to a power of the distance from the apex, and
:meth:`~pystrata.dam.Dam.tune` scales the velocities so that the fundamental
frequency matches a target, either of the dam on a rigid base or of the dam on
a site.

Calculators
-----------

:class:`~pystrata.dam.DamLinearElasticCalculator`,
:class:`~pystrata.dam.DamEquivalentLinearCalculator` and
:class:`~pystrata.dam.DamFrequencyDependentEqlCalculator` follow the calculators of
:doc:`calculations`, and ``calc.calc_accel_tf(loc_in, loc_out)`` gives the
transfer function between two locations of the dam profile.

Sliding mass
------------

A failure surface selects the part of the dam that slides.
:class:`~pystrata.dam.CircularSurface` is a circle given by its center and radius,
and :func:`~pystrata.dam.toe_circle` creates the circle through the toe of the
slope and the far side of the crest. :func:`~pystrata.dam.plot_failure_surface`
draws the section and the surface. The transfer function to the average
acceleration of the sliding mass, which is the quantity of Hale and Abrahamson
(2019), is the width-weighted average of the transfer functions over the mass:
:func:`~pystrata.dam.sliding_mass_transfer_function`. It works with any of the
calculators, so it can be used with strain-compatible properties.

Examples
--------

- :doc:`../examples/example-20`: a dam on a site, tuning, and the closed-form
  solutions.
- :doc:`../examples/example-21`: the sliding-mass transfer function compared with
  the model of Hale and Abrahamson in pyGMM.
- :doc:`../examples/example-22`: the truncated and apex wedges compared.
- :doc:`../examples/example-25`: the effect of the length of the canyon.
