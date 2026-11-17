Variation of profiles and soil properties
=========================================

The variation models in ``pystrata.variation`` take a profile (or a soil type)
and return a new one with randomly perturbed properties. They are the source of
the realizations of a Monte Carlo analysis. Each model is a callable object that
takes the profile and, optionally, a random generator (``rng``).

.. code-block:: python

   var_thickness = pystrata.variation.ToroThicknessVariation()
   var_velocity = pystrata.variation.ToroVelocityVariation.generic_model("USGS B")

   realization = var_velocity(var_thickness(profile))

Models
------

.. list-table::
   :header-rows: 1
   :widths: 35 65

   * - Class
     - Varies
   * - :class:`~pystrata.variation.ToroThicknessVariation`
     - Layer thicknesses (Toro, 1995)
   * - :class:`~pystrata.variation.ShiAsimakiThicknessVariation`
     - Layer thicknesses of basin sediments in California (Shi and Asimaki, 2018)
   * - :class:`~pystrata.variation.HalfSpaceDepthVariation`
     - The depth to the half space, from a ``scipy.stats`` distribution
   * - :class:`~pystrata.variation.ToroVelocityVariation`
     - Shear-wave velocity, with correlation between layers
   * - :class:`~pystrata.variation.DepthDependToroVelVariation`
     - Shear-wave velocity with a standard deviation that depends on depth
   * - :class:`~pystrata.variation.ShiAsimakiVelocityVariation`
     - Shear-wave velocity with a standard deviation that depends on depth and
       :math:`V_{S30}` (Shi and Asimaki, 2018)
   * - :class:`~pystrata.variation.DarendeliVariation`
     - Modulus reduction and damping curves of a soil type (Darendeli, 2001)
   * - :class:`~pystrata.variation.SpidVariation`
     - Modulus reduction and damping curves of a soil type (EPRI SPID, 2012)

``ToroVelocityVariation.generic_model`` takes the name of a site class, such as
``"USGS A"``, ``"USGS B"``, ``"USGS C"``, ``"USGS D"``, ``"Geomatrix AB"`` or
``"Geomatrix CD"``, and has the parameters of the velocity model of Toro (1995)
for it.

Shi and Asimaki (2018) sediment velocity model
-----------------------------------------------

The median profile of the sediment velocity model comes from pygmm. It is a
function of :math:`V_{S30}` and is valid for 173.1 to 1000 m/s, and the authors
recommend using it down to a velocity of about 1000 m/s, which is the depth that
:func:`pygmm.sa18_depth_to_vs` returns. The thickness and velocity models then
randomize a finely layered version of it:

.. code-block:: python

   import numpy as np
   import pygmm
   import pystrata

   vs30 = 300.0
   soil = pystrata.site.SoilType("soil", 18.0, None, 0.02)

   vp = pygmm.sa18_profile(np.arange(0, 100.5, 0.5), vs30)
   profile = pystrata.site.Profile.from_velocity_profile(vp, soil_types=soil)

   realizations = pystrata.variation.iter_varied_profiles(
       profile,
       100,
       var_thickness=pystrata.variation.ShiAsimakiThicknessVariation(),
       var_velocity=pystrata.variation.ShiAsimakiVelocityVariation(vs30),
       seed=1,
   )

``sa18_profile`` returns the lognormal median of the velocity, which is what the
velocity variation expects. The paper provides no correlation between the layers, so
:class:`~pystrata.variation.ShiAsimakiVelocityVariation` uses the model of Toro
(1995), with the parameters of the USGS site class of :math:`V_{S30}` unless
others are given.

Random numbers
--------------

The standard normal variates of the velocity and soil-type models can be limited
to a number of standard deviations, with :class:`~pystrata.variation.LimitedNorm`:
``"normal"`` (no limit), ``"clipped"`` (values beyond the limit are set to it) and
``"clipped_inflated"`` (the default: the standard deviation is increased so that the
clipped values have a unit standard deviation). The ``limit_mode`` argument of the
velocity models selects it.

Generating realizations
-----------------------

:func:`pystrata.variation.iter_varied_profiles` applies the depth, thickness,
velocity and soil-type models to a profile, those that are given, and yields
``count`` realizations.
A ``check`` that returns ``False`` rejects a realization, and
:class:`~pystrata.variation.DispersionCheck` rejects a profile whose surface-wave
dispersion curve is too far from a target.

.. code-block:: python

   for p in pystrata.variation.iter_varied_profiles(
       profile,
       20,
       var_thickness=var_thickness,
       var_velocity=var_velocity,
       var_soiltypes=pystrata.variation.SpidVariation(-0.5),
   ):
       p = p.auto_discretize()
       calc(motion, p, p.location("outcrop", index=-1))
       outputs(calc)

:func:`pystrata.runner.run_ensemble` takes the same models and runs the
realizations, possibly across processes (:doc:`parallel`).

Examples
--------

- :doc:`../examples/example-04`: realizations of the thicknesses, velocities and
  depth to bedrock.
- :doc:`../examples/example-05`: realizations of the nonlinear properties.
- :doc:`../examples/example-08`: a suite of profiles in an RVT analysis, with
  velocity models that vary with depth.
