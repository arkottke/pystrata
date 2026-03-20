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

**Published Models**

These models are published with fixed values:
   - Vucetic & Dobry (91), PI=0
   - Vucetic & Dobry (91), PI=15
   - Vucetic & Dobry (91), PI=30
   - Vucetic & Dobry (91), PI=50
   - Vucetic & Dobry (91), PI=100
   - Vucetic & Dobry (91), PI=200
   - EPRI (93), PI=10
   - EPRI (93), PI=30
   - EPRI (93), PI=50
   - EPRI (93), PI=70
   - EPRI (93), 0-20 ft
   - EPRI (93), 20-50 ft
   - EPRI (93), 50-120 ft
   - EPRI (93), 120-250 ft
   - EPRI (93), 250-500 ft
   - EPRI (93), 500-1000 ft
   - GEI (83), 0-50 ft
   - GEI (83), 50-100 ft
   - GEI (83), 100-250 ft
   - GEI (83), 250-500 ft
   - GEI (83), >500 ft
   - GeoMatrix (1990), 0-50 ft
   - GeoMatrix (1990), 50-150 ft
   - GeoMatrix (1990), >150 ft
   - GeoMatrix (1990), 150-300 ft
   - GeoMatrix (1990), >300 ft
   - Idriss (1990), Clay
   - Idriss (1990), Sand
   - Imperial Valley Soils, 0-300 ft
   - Imperial Valley Soils, >300 ft
   - Iwasaki, 0.25 atm
   - Iwasaki, 1.0 atm
   - Peninsular Range, Cohesionless 0-50 ft
   - Peninsular Range, Cohesionless 50-500 ft
   - Seed & Idriss, Sand Mean
   - Seed & Idriss, Sand Upper
   - Seed & Idriss, Sand Lower

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

**Menq (2003) Model**

The Menq model provides modulus reduction and damping curves for gravelly soils
using a modified hyperbolic formulation:

.. math::

   \gamma_r = \frac{0.12 \, C_u^{-0.6} \, (\sigma_m'/p_a)^{0.5 \, C_u^{-0.15}}}{100}

.. math::

   a = 0.86 + 0.1 \log_{10}(\sigma_m'/p_a)

Parameters depend on:
- Uniformity coefficient (:math:`C_u`)
- Mean grain diameter (:math:`D_{50}`)
- Mean effective stress (:math:`\sigma_m'`)
- Number of loading cycles

**Wang and Stokoe (2022) Model**

The Wang and Stokoe model uses a two-parameter modified hyperbolic form with
hierarchical levels of accuracy depending on available input data:

.. math::

   \frac{G}{G_{max}} = \frac{1}{\left(1 + \left(\gamma/\gamma_{mr}\right)^a\right)^b}

Three soil group classifications are supported:
- Clean sand and gravel (FC ≤ 12%)
- Nonplastic silty sand (FC > 12%, nonplastic)
- Clayey soil (FC > 12%, plastic)

Parameters depend on:
- Mean effective stress (:math:`\sigma_m'`)
- Void ratio
- Fines content
- Uniformity coefficient or plasticity index (group-dependent)

**Alemu et al. (2025) Model**

The Alemu et al. model provides curves for transitional silts, using the
Wang and Stokoe (2022) modified hyperbolic backbone with a Hardin–Drnevich
damping component:

.. math::

   \frac{G}{G_{max}} = \frac{1}{\left(1 + \left(\gamma/\gamma_{mr}\right)^A\right)^B}

.. math::

   \xi = \xi_{min} + \frac{E_1}{100} \frac{\gamma/\gamma_D}{1 + (\gamma/\gamma_D)^{E_6}}

Valid for: :math:`0 \le PI \le 32`, :math:`10 \le p' \le 125` kPa, :math:`1 \le OCR \le 9.1`.

Parameters depend on:
- Plasticity index (PI)
- Over-consolidation ratio (OCR)
- Mean effective stress (:math:`p'`)
- Fines content

**User-Defined Curves**

PyStrata allows specification of custom modulus reduction and damping curves from laboratory data.

Implementation in PyStrata
---------------------------

Nonlinear soil properties are handled through the `SoilType` class with `NonlinearCurve` objects that define the strain-dependent curves.

.. code-block:: python

   # Darendeli model for clay
   soil_type = pystrata.site.DarendeliSoilType(
       plas_index=30,
       stress_mean=100,  # kPa
       unit_wt=18.0,
   )

   # Menq model for gravel
   soil_type = pystrata.site.MenqSoilType(
       unit_wt=20.0,
       coef_unif=10,
       diam_mean=5,  # mm
       stress_mean=101.3,  # kPa
   )

   # Wang and Stokoe model for clean sand
   soil_type = pystrata.site.WangSoilType(
       soil_group="clean_sand_and_gravel",
       unit_wt=19.0,
       stress_mean=100,  # kPa
       void_ratio=0.6,
   )

   # Alemu et al. model for transitional silt
   soil_type = pystrata.site.AlemuEtAlSoilType(
       unit_wt=18.0,
       plas_index=15,
       ocr=1.5,
       stress_mean=50,  # kPa
   )
