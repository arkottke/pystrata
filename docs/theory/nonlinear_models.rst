Nonlinear Soil Models
=====================

Soil exhibits nonlinear stress-strain behavior under dynamic loading, with stiffness decreasing and damping increasing as shear strain increases. PyStrata implements several empirical models for capturing this behavior.

Equivalent Linear Approach
---------------------------

The equivalent linear method approximates nonlinear soil response by using strain-compatible linear properties. The effective shear modulus and damping ratio are determined iteratively:

.. math::

   G_{eff} = G_{max} \cdot G/G_{max}(\gamma_{eff})

.. math::

   \xi_{eff} = \xi(\gamma_{eff})

where :math:`\gamma_{eff}` is the effective shear strain, typically taken as 65% of the maximum strain.

Empirical Models
----------------

**Darendeli (2001) Model**

The Darendeli model provides normalized modulus reduction and damping curves:

.. math::

   \frac{G}{G_{max}} = \frac{1}{1 + (\gamma/\gamma_r)^a}

.. math::

   \xi = \xi_{min} + \frac{\xi_{max} - \xi_{min}}{1 + (\gamma_r/\gamma)^b}

Parameters depend on:
- Plasticity index (PI)
- Confining stress (σ')
- Over-consolidation ratio (OCR)
- Number of loading cycles

**Zhang et al. (2005) Model**

Alternative formulation with different parameter dependencies and curve shapes.

**User-Defined Curves**

PyStrata allows specification of custom modulus reduction and damping curves from laboratory data.

Implementation in PyStrata
---------------------------

Nonlinear soil properties are handled through the `SoilType` class with `NonlinearProperty` objects that define the strain-dependent curves.

.. code-block:: python

   # Darendeli model for clay
   nonlinear_property = pystrata.site.DarendeliNonlinearProperty(
       plasticity_index=30,
       stress_mean=100,  # kPa
   )

   soil_type = pystrata.site.SoilType(
       name="Clay",
       unit_wt=18.0,
       nonlinear_property=nonlinear_property,
       damping_min=0.02
   )
