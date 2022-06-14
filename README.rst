pyStrata
========

|PyPi Cheese Shop| |Build Status| |Documentation Status| |Code Quality| |Test Coverage|
|License| |Zenodo| |MyBinder|

Site response analyses implemented in Python. This Python packages aims
to implement many of the features found in
Strata_. These features include:

- Input motion characterization:
    - Time series
    - Random vibration theory
- Wave propagation or site amplification:
    - linear
    - equivalent-linear
    - equivalent-linear with frequency dependent properties
    - quarter wavelength
- Nonlinear curve models:
    - Darendeli (2001)
    - Menq (2004)
    - Kishida (2012)
- Site and soil property uncertainty:
    - Toro (1994) Vs correlation model
    - G/Gmax and D uncertainty:
        - Darendeli (2001)
        - EPRI SPID (2013)

Development of this software is on-going and any contributions are encouraged.
Previously named ``pysra``, but renamed after some sage and persistent advice to be
better associated with Strata_.

Installation
------------

``pystrata`` is available via ``pip`` and can be installed with::

   pip install pystrata

If you are using ``conda`` and a create a ``pystrata`` specific environmental make sure you install ``ipykernels`` and ``nb_conda_kernels`` so that the environment is discoverable by ``Jupyter`` with::

    conda install ipykernel nb_conda_kernels

Citation
--------

Please cite this software using the following DOI_.

Examples
--------

There are a variety of examples of using ``pystrata`` within the `examples
directory`_. An interactive Jupyter interface of these examples is available on
|MyBinder|.

.. _Strata: https://github.com/arkottke/strata
.. _DOI: https://zenodo.org/badge/latestdoi/8959678
.. _`examples directory`: https://github.com/arkottke/pystrata/tree/master/examples

.. |PyPi Cheese Shop| image:: https://img.shields.io/pypi/v/pystrata.svg
   :target: https://pypi.python.org/pypi/pystrata
.. |Build Status| image:: https://img.shields.io/travis/arkottke/pystrata.svg
   :target: https://travis-ci.org/arkottke/pystrata
.. |Documentation Status| image:: https://readthedocs.org/projects/pystrata/badge/?version=latest&style=flat
   :target: https://pystrata.readthedocs.org
.. |Code Quality| image:: https://api.codacy.com/project/badge/Grade/6dbbb3a4279744d697b9bfe08af19ded
   :target: https://www.codacy.com/app/arkottke/pystrata
.. |Test Coverage| image:: https://api.codacy.com/project/badge/Coverage/6dbbb3a4279744d697b9bfe08af19ded
   :target: https://www.codacy.com/app/arkottke/pystrata
.. |License| image:: https://img.shields.io/badge/license-MIT-blue.svg
.. |Zenodo| image:: https://zenodo.org/badge/8959678.svg
   :target: https://zenodo.org/badge/latestdoi/8959678
.. |MyBinder| image:: https://mybinder.org/badge_logo.svg
   :target: https://mybinder.org/v2/gh/arkottke/pystrata/master?filepath=examples
