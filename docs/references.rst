References
==========

This page contains the complete bibliography of scientific literature referenced in PyStrata documentation and implementation.

Core References
---------------

The theoretical foundation of PyStrata is based on these seminal works in geotechnical earthquake engineering and computational seismology.

**Fundamental Texts**

:cite:p:`Kramer1996` provides comprehensive coverage of geotechnical earthquake engineering principles, including site response analysis methods. :cite:p:`Ishihara1996` covers soil behavior in earthquake geotechnics with emphasis on dynamic soil properties and nonlinear response.

**Wave Propagation Theory**

The mathematical foundation for one-dimensional wave propagation in layered media comes from :cite:p:`Thomson1950` and :cite:p:`Haskell1953`, who developed the propagator matrix method for elastic wave transmission through stratified media.

Classical Site Response Methods
-------------------------------

**SHAKE Family of Codes**

The equivalent linear method was first implemented in :cite:p:`Schnabel1972` with the SHAKE program. :cite:p:`idriss1992shake91` extended this work with improved algorithms and broader applicability.

**Soil Dynamics and Nonlinearity**

Laboratory studies by :cite:p:`Seed1987` established fundamental relationships for modulus reduction and damping in soils. :cite:p:`Darendeli2001` developed normalized curves for a wide range of soil types, while :cite:p:`Zhang2005` provided alternative formulations for specific conditions.

Advanced Methods
----------------

**Frequency Domain Methods**

:cite:p:`kausel2002seismic` and :cite:p:`Yoshida2002` developed advanced frequency-domain approaches for incorporating soil nonlinearity more accurately than traditional equivalent linear methods.

**Uncertainty Quantification**

:cite:p:`Bommer2005` provides guidance on uncertainty treatment in earthquake loss modeling. :cite:p:`Toro1995` developed probabilistic models for site velocity profiles used in generic site response studies.

Validation and Benchmarking
---------------------------

**Laboratory and Field Studies**

Validation of site response methods relies on high-quality experimental data. :cite:p:`Matasovic1995` and :cite:p:`Zeghal1995` analyzed recorded earthquake data, while :cite:p:`Kutter2017` provides modern centrifuge validation data through the LEAP project.

**Computational Tools**

PyStrata results have been compared against established codes including :cite:p:`Hashash2016` (DEEPSOIL), :cite:p:`McKenna2000` (OpenSees), and :cite:p:`Kottke2013` (STRATA) to ensure consistency and accuracy.

Random Vibration Theory
-----------------------

**Ground Motion Simulation**

:cite:p:`Boore2003` provides the theoretical foundation for stochastic ground motion simulation used in RVT approaches. Peak factor calculations draw from :cite:p:`Cartwright1956` and :cite:p:`Vanmarcke1975`.

Complete Bibliography
---------------------

.. bibliography::
   :style: plain
   :all:

How to Cite PyStrata
--------------------

If you use PyStrata in your research or professional work, please cite it as:

.. code-block:: bibtex

   @misc{pystrata,
     title={PyStrata: A Python library for seismic site response analysis},
     author={Kottke, Albert R},
     year={2024},
     publisher={GitHub},
     url={https://github.com/arkottke/pystrata},
     note={Version X.X.X}
   }

Additionally, please cite the relevant methodological papers for the specific methods you use:

* **Equivalent Linear Method**: :cite:p:`Schnabel1972`
* **Random Vibration Theory**: :cite:p:`Boore2003`
* **Darendeli Soil Models**: :cite:p:`Darendeli2001`
* **Logic Tree Approach**: :cite:p:`Bommer2005`

Contributing References
-----------------------

If you notice missing or incorrect references, or would like to add citations for new methods implemented in PyStrata:

1. Add the reference to ``docs/refs.bib`` in BibTeX format
2. Update the relevant documentation with ``:cite:p:`key``` citations
3. Submit a pull request with your changes

For formatting guidelines, see our :doc:`developer/contributing` guide.
