pySRA
=====

|PyPi Cheese Shop| |Build Status| |Documentation Status| |Code Quality| |Test Coverage|
|License| |image|

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

Development of this software is on-going and any contributions are
encouraged.

Installation
------------

``pysra`` is available via ``pip`` and can be installed with::

   pip install pysra

Citation
--------

Please cite this software using the following DOI_.

.. _Strata: https://github.com/arkottke/strata
.. _DOI: https://zenodo.org/badge/latestdoi/8959678

.. |PyPi Cheese Shop| image:: https://img.shields.io/pypi/v/pysra.svg
   :target: https://pypi.python.org/pypi/pysra
.. |Build Status| image:: https://img.shields.io/travis/arkottke/pysra.svg
   :target: https://travis-ci.org/arkottke/pysra
.. |Documentation Status| image:: https://readthedocs.org/projects/pysra/badge/?version=latest&style=flat
   :target: https://pysra.readthedocs.org
.. |Code Quality| image:: https://api.codacy.com/project/badge/Grade/6dbbb3a4279744d697b9bfe08af19ded
   :target: https://www.codacy.com/app/arkottke/pysra
.. |Test Coverage| image:: https://api.codacy.com/project/badge/Coverage/6dbbb3a4279744d697b9bfe08af19ded
   :target: https://www.codacy.com/app/arkottke/pysra
.. |License| image:: https://img.shields.io/badge/license-MIT-blue.svg
.. |image| image:: https://zenodo.org/badge/8959678.svg
   :target: https://zenodo.org/badge/latestdoi/8959678
