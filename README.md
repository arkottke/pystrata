# pySRA


[![PyPi Cheese Shop](https://img.shields.io/pypi/v/pysra.svg)](https://pypi.python.org/pypi/pysra)
[![Build Status](https://img.shields.io/travis/arkottke/pysra.svg)](https://travis-ci.org/arkottke/pysra)
[![Documentation Status](https://readthedocs.org/projects/pysra/badge/?version=latest&style=flat)](https://pysra.readthedocs.org)
[![Test Coverage](https://coveralls.io/repos/github/arkottke/pysra/badge.svg?branch=master)](https://coveralls.io/github/arkottke/pysra?branch=master)
[![License](https://img.shields.io/badge/license-MIT-blue.svg)](https://github.com/arkottke/pysra/blob/master/LICENSE)
[![image](https://zenodo.org/badge/8959678.svg)](https://zenodo.org/badge/latestdoi/8959678)

Site response analyses implemented in Python. This Python packages aims to 
implement many of the features found in 
[Strata](https://github.com/arkottke/strata). These features include:
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

## Installation

`pysra` is available via `pip` and can be installed with:
```
pip install pysra
```

## Citation

Please cite this software using the following [DOI](https://zenodo.org/badge/latestdoi/8959678).
