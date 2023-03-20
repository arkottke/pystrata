# pyStrata

Python library for site response analysis.

[![PyPI](https://img.shields.io/pypi/v/pystrata)](https://pypi.org/project/pystrata/)
[![GitHub Workflow Status](https://img.shields.io/github/actions/workflow/status/arkottke/pystrata/python-app.yml)](https://github.com/arkottke/pystrata/actions/workflows/python-app.yml)
[![Read the
Docs](https://img.shields.io/readthedocs/pystrata)](https://pystrata.readthedocs.io/en/latest/)
[![Codacy coverage](https://img.shields.io/codacy/coverage/6dbbb3a4279744d697b9bfe08af19ded)](https://app.codacy.com/gh/arkottke/pystrata/dashboard)
![License](https://img.shields.io/badge/license-MIT-blue.svg)
[![Zenodo](https://zenodo.org/badge/8959678.svg)](https://zenodo.org/badge/latestdoi/8959678)
[![MyBinder](https://mybinder.org/badge_logo.svg)](https://mybinder.org/v2/gh/arkottke/pystrata/main?filepath=examples)

## Introduction

Site response analyses implemented in Python. This Python packages aims
to implement many of the features found in
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
    - Predictive models:
        - Darendeli (2001)
        - Menq (2004)
        - Kishida (2012)
    - Curves:
        - Vucetic & Dobry (1991)
        - EPRI (1993)
        - GEI (1983)
        - GeoMatrix (1990)
        - Idriss (1990)
        - Imperial Valley Soils
        - Iwasaki
        - Peninsular Range
        - Seed & Idriss
- Site and soil property uncertainty:
    - Toro (1994) Vs correlation model
    - G/Gmax and D uncertainty:
    - Darendeli (2001)
    - EPRI SPID (2013)

Development of this software is on-going and any contributions are
encouraged. Previously named `pysra`, but renamed after some sage and
persistent advice to be better associated with
[Strata](https://github.com/arkottke/strata).

## Installation

`pystrata` is available via `pip` and can be installed with:

    pip install pystrata

If you are using `conda` and a create a `pystrata` specific
environmental make sure you install `ipykernels` and `nb_conda_kernels`
so that the environment is discoverable by `Jupyter` with:

    conda install ipykernel nb_conda_kernels

## Citation

Please cite this software using the following
[DOI](https://zenodo.org/badge/latestdoi/8959678):

    Albert Kottke & Maxim Millen. (2023). arkottke/pystrata: v0.5.2 (v0.5.2). Zenodo. https://doi.org/10.5281/zenodo.7551992

or with BibTeX:

    @software{albert_kottke_2023_7551992,
      author       = {Albert Kottke and
                      Maxim Millen},
      title        = {arkottke/pystrata: v0.5.2},
      month        = jan,
      year         = 2023,
      publisher    = {Zenodo},
      version      = {v0.5.2},
      doi          = {10.5281/zenodo.7551992},
      url          = {https://doi.org/10.5281/zenodo.7551992}
    }

## Examples

There are a variety of examples of using `pystrata` within the [examples
directory](https://github.com/arkottke/pystrata/tree/main/examples). An
interactive Jupyter interface of these examples is available on
[![MyBinder](https://mybinder.org/badge_logo.svg)](https://mybinder.org/v2/gh/arkottke/pystrata/main?filepath=examples).
