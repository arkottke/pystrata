Defining Site Profiles
======================

The site profile is the central data structure of a pyStrata analysis. It represents the layered
soil column from the surface down to the bedrock half-space.

.. currentmodule:: pystrata.site

Soil Types
----------

A :class:`SoilType` describes the material properties of a soil layer:

- **Unit weight** (kN/m³) — used to compute effective stress and stiffness.
- **Shear-modulus reduction curve** — strain-dependent stiffness degradation.  Pass ``None`` for
  linear-elastic behavior (modulus reduction = 1 at all strains).
- **Damping curve** — strain-dependent material damping.  Pass a ``float`` for constant damping.

.. code-block:: python

    import pystrata

    # Linear soil layer with 5 % damping
    soil = pystrata.site.SoilType("Sand", unit_wt=18.0, mod_reduc=None, damping=0.05)

    # Nonlinear soil with explicit curves
    mr = pystrata.site.ModulusReductionCurve(
        strains=[1e-4, 1e-3, 1e-2, 1e-1],
        values=[1.0,  0.90, 0.60, 0.25],
    )
    d = pystrata.site.DampingCurve(
        strains=[1e-4, 1e-3, 1e-2, 1e-1],
        values=[0.02, 0.03, 0.07, 0.15],
    )
    soil_nl = pystrata.site.SoilType("Clay", unit_wt=17.0, mod_reduc=mr, damping=d)

Published Nonlinear Curves
~~~~~~~~~~~~~~~~~~~~~~~~~~

pyStrata ships a library of published nonlinear curves (Darendeli, Menq, Vucetic-Dobry, etc.).
Use :meth:`SoilType.from_published` to load them by name:

.. code-block:: python

    soil = pystrata.site.SoilType.from_published(
        name="Sand",
        unit_wt=18.0,
        model="Darendeli (2001)",
    )

    # List available curve names
    print(pystrata.site.known_published_curves())

Integration with pygmm
~~~~~~~~~~~~~~~~~~~~~~

:class:`SoilType` is duck-typed: pass any object with ``.strains``, ``.mod_reduc``,
``.damping``, ``.damping_min`` attributes (e.g. a ``pygmm`` soil-curve result) to
:meth:`SoilType.from_curves`:

.. code-block:: python

    import pygmm
    curves = pygmm.DarendeliSoilType(p_atm=1.0, sigma_v0=50.0, PI=0, OCR=1, freq=1, n_cycles=10)
    soil = pystrata.site.SoilType.from_curves(curves, name="Darendeli Sand", unit_wt=18.0)

Layers
------

A :class:`Layer` pairs a :class:`SoilType` with a thickness and shear-wave velocity:

.. code-block:: python

    layer = pystrata.site.Layer(soil, thickness=5.0, shear_vel=250.0)

The last layer must be the **half-space** (``thickness=0``).

Building a Profile
------------------

Pass an ordered list of :class:`Layer` objects to :class:`Profile`.  The final layer is the
infinite half-space:

.. code-block:: python

    profile = pystrata.site.Profile([
        pystrata.site.Layer(pystrata.site.SoilType("Clay",  17.0, None, 0.06),  5, 150),
        pystrata.site.Layer(pystrata.site.SoilType("Sand",  18.0, None, 0.04), 20, 350),
        pystrata.site.Layer(pystrata.site.SoilType("Rock",  24.0, None, 0.01),  0, 900),
    ])

From a DataFrame
~~~~~~~~~~~~~~~~

:meth:`Profile.from_dataframe` accepts a ``pandas.DataFrame`` with columns
``thickness`` (m), ``vel_shear`` (m/s), ``unit_wt`` (kN/m³), and ``damping`` (decimal):

.. code-block:: python

    import pandas as pd
    df = pd.read_csv("my_profile.csv")
    profile = pystrata.site.Profile.from_dataframe(df)

From a Velocity Profile Object
~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~

:meth:`Profile.from_velocity_profile` accepts any object with ``.depth`` and ``.vs_median``
arrays — including ``pygmm.contracts.VelocityProfile`` objects:

.. code-block:: python

    import pygmm
    vp = pygmm.kea16_profile(v_s30=400)
    soil = pystrata.site.SoilType("Generic", unit_wt=18.0, mod_reduc=None, damping=0.05)
    profile = pystrata.site.Profile.from_velocity_profile(vp, soil_types=soil)

Locations
---------

A :class:`Location` is a pointer into the profile at a specific depth and wave-field type.
Calculators and outputs use locations to specify *where* to evaluate results:

.. code-block:: python

    # Surface (index 0 = top of first layer)
    loc_surface = profile.location("within", index=0)

    # Base of profile (input motion location)
    loc_input = profile.location("outcrop", index=-1)

    # By depth
    loc_5m = profile.location("within", depth=5.0)

Wave-field options are ``"outcrop"``, ``"within"``, and ``"incoming_only"``
(see :class:`~pystrata.motion.WaveField`).

Profile Utilities
-----------------

.. code-block:: python

    # Time-averaged velocity (e.g. Vs30)
    vs30 = profile.time_average_vel(depth=30)

    # Auto-discretize for adequate frequency resolution
    profile.auto_discretize(max_freq=25.0, wave_frac=0.2)

    # Export to DataFrame
    df = profile.to_dataframe()
