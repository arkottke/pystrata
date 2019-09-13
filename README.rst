pySRA
=====

|PyPi Cheese Shop| |Build Status| |Code Quality| |Test Coverage| |License| |DOI|

Site response analyses implemented in Python. This Python packages aims
to implement many of the features found in
`Strata <https://github.com/arkottke/strata>`__. These features include:
- Input motion characterization: - Time series - Random vibration theory
- Wave propagation or site amplification: - linear - equivalent-linear -
equivalent-linear with frequency dependent properties - quarter
wavelength - Nonlinear curve models: - Darendeli (2001) - Menq (2004) -
Kishida (2012) - Site and soil property uncertainty: - Toro (1994) Vs
correlation model - G/Gmax and D uncertainty: - Darendeli (2001) - EPRI
SPID (2013)

Development of this software is on-going and any contributions are
encouraged.

Installation
------------

``pysra`` is available via ``pip`` and can be installed with:

::

   pip install pysra

Citation
--------

Please cite this software using the following
`DOI <https://zenodo.org/badge/latestdoi/8959678>`__.

.. |PyPi Cheese Shop| image:: https://img.shields.io/pypi/v/pysra.svg
   :target: https://pypi.python.org/pypi/pysra
.. |Build Status| image:: https://img.shields.io/travis/arkottke/pysra.svg
   :target: https://travis-ci.org/arkottke/pysra
.. |Documentation Status| image:: https://readthedocs.org/projects/pysra/badge/?version=latest&style=flat
   :target: https://pysra.readthedocs.org
.. |Test Coverage| image:: https://coveralls.io/repos/github/arkottke/pysra/badge.svg?branch=master
   :target: https://coveralls.io/github/arkottke/pysra?branch=master
.. |License| image:: https://img.shields.io/badge/license-MIT-blue.svg
   :target: https://github.com/arkottke/pysra/blob/master/LICENSE
.. |image| image:: https://zenodo.org/badge/8959678.svg
   :target: https://zenodo.org/badge/latestdoi/8959678
