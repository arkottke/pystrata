# pyStrata

[![PyPI](https://img.shields.io/pypi/v/pystrata.svg)](https://pypi.org/project/pystrata/)
[![Python](https://img.shields.io/python/required-version-toml?tomlFilePath=https://raw.githubusercontent.com/arkottke/pystrata/main/pyproject.toml)](https://pypi.org/project/pystrata/)
[![Documentation](https://readthedocs.org/projects/pystrata/badge/?version=latest)](https://pystrata.readthedocs.io/?badge=latest)
[![Build Status](https://github.com/arkottke/pystrata/actions/workflows/python-app.yml/badge.svg)](https://github.com/arkottke/pystrata/actions/workflows/python-app.yml)
[![Code Quality](https://api.codacy.com/project/badge/Grade/6dbbb3a4279744d697b9bfe08af19ded)](https://app.codacy.com/gh/arkottke/pystrata/dashboard)
[![Test Coverage](https://api.codacy.com/project/badge/Coverage/6dbbb3a4279744d697b9bfe08af19ded)](https://app.codacy.com/gh/arkottke/pystrata/dashboard)
[![License](https://img.shields.io/badge/license-MIT-blue.svg)](https://github.com/arkottke/pystrata/blob/main/LICENSE)
[![DOI](https://zenodo.org/badge/8959678.svg)](https://zenodo.org/badge/latestdoi/8959678)

Site response analysis implemented in Python.

pyStrata provides tools for modeling wave propagation through layered soil
profiles and computing surface ground motion from bedrock input motions. It
supports time series and random vibration theory (RVT) approaches,
equivalent-linear and frequency-domain methods, and logic trees for uncertainty
quantification.

## Installation

pyStrata requires Python 3.13 or later and can be installed with
[pip](https://pip.pypa.io):

```bash
python -m pip install pystrata
```

For development or the latest features, see the
[installation instructions](https://pystrata.readthedocs.io/en/latest/getting-started/installation.html).

## Documentation

Information on the installation and usage can be found in the
[documentation](https://pystrata.readthedocs.io/). To get started, see the
[Quickstart](https://pystrata.readthedocs.io/en/latest/getting-started/quickstart.html)
and the [examples](https://github.com/arkottke/pystrata/tree/main/examples). The
[API reference](https://pystrata.readthedocs.io/en/latest/api/index.html) has
the full details, and the
[changelog](https://pystrata.readthedocs.io/en/latest/changelog.html) lists the
changes in each release.

## Related packages

pyStrata is developed together with:

- [pyRVT](https://github.com/arkottke/pyrvt): random vibration theory
- [pyGMM](https://github.com/arkottke/pygmm): ground motion, Fourier amplitude
  spectrum and soil curve models
- [pyRotd](https://github.com/arkottke/pyrotd): response spectra and rotated
  response spectra from acceleration time series

## Citation

If you use pyStrata in your research, please cite the software using the
[DOI](https://zenodo.org/badge/latestdoi/8959678). Key references are listed in
the [documentation](https://pystrata.readthedocs.io/en/latest/references.html).

## License

pyStrata is made available under the MIT License. See
[LICENSE](https://github.com/arkottke/pystrata/blob/main/LICENSE).
