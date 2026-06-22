# CLAUDE.md

This file provides guidance to Claude Code (claude.ai/code) when working with code in this repository.

## Commands

```bash
# Run all tests
/home/albert/envs/py312/bin/python -m pytest tests/

# Run a single test file
/home/albert/envs/py312/bin/python -m pytest tests/propagation_test.py

# Run a specific test
/home/albert/envs/py312/bin/python -m pytest tests/propagation_test.py::test_name

# Run with coverage
/home/albert/envs/py312/bin/python -m pytest --cov=pystrata tests/

# Lint and format
uv run ruff check src/
uv run ruff format src/

# Build docs
uv run --group docs make -C docs html
```

## Architecture

pyStrata computes seismic site response — specifically, wave propagation through layered soil profiles from a bedrock input motion to the surface.

### Core data flow

1. **Motion** (`motion.py`) — defines the input ground motion. `TimeSeriesMotion` wraps a recorded acceleration time series; `RvtMotion` and `CompatibleRvtMotion` use Random Vibration Theory (via `pyrvt`) for a probabilistic approach. `WaveField` (enum: `outcrop`, `within`, `incoming_only`) specifies where in the column the motion is defined.

2. **Site** (`site.py`) — defines the soil column. `SoilType` holds material properties (unit weight, initial shear modulus, damping) and optional `ModulusReductionCurve`/`DampingCurve` for nonlinear behavior. `Layer` wraps a `SoilType` with a thickness. `Profile` is an ordered list of `Layer` objects (last layer is the half-space). `Location` is a pointer into the profile at a specific depth and wave field. Published nonlinear curves are loaded lazily from `src/pystrata/data/published_curves.toml`.

3. **Calculator / Propagation** (`propagation.py`) — runs the analysis. Calling `calculator(motion, profile, loc_input)` propagates the motion through the profile. Key classes:
   - `LinearElasticCalculator` — frequency-domain transfer matrix method, elastic.
   - `EquivalentLinearCalculator` — iterates to compatible strain-dependent modulus and damping.
   - `FrequencyDependentEqlCalculator` — frequency-dependent equivalent-linear variant.
   - `TimeDomainCalculator` — time-domain FDM solver (uses Numba if available).
   - `QuarterWaveLenCalculator` — simple quarter-wavelength approximation.

4. **Output** (`output.py`) — collects results after a calculator run. Outputs are registered before the run, then populated. Types: `AccelerationTSOutput`, `ResponseSpectrumOutput`, `FourierAmplitudeSpectrumOutput`, `AccelTransferFunctionOutput`, `ResponseSpectrumRatioOutput`, and depth-profile outputs (`MaxStrainProfile`, `DampingProfile`, etc.). All store results as `xarray.Dataset`.

5. **Variation** (`variation.py`) — stochastic perturbation of profiles for Monte Carlo analysis. `ToroVelocityVariation` and `DepthDependToroVelVariation` randomize shear-wave velocities; `ToroThicknessVariation` randomizes layer thicknesses; `DarendeliVariation` and `SpidVariation` randomize nonlinear curves.

6. **Logic Tree** (`logic_tree.py`) — weighted branching for epistemic uncertainty. `LogicTree` holds `Branch` objects with weights; each branch modifies the profile/motion/calculator. Used for systematic scenario analysis.

### Supporting modules

- `constitutive.py` — MKZ and Hashash-Hardin (HH) nonlinear constitutive models for use with `TimeDomainCalculator`.
- `curve_fitting.py` — fit MKZ/HH parameters to target nonlinear curves.
- `generic.py` — built-in generic velocity profiles: `aaa21_profile()` loads Al Atik & Abrahamson (2021) Vs30-indexed profiles; `get_profile_from_wus()` fetches WUS profiles.
- `tools.py` — utilities including `load_shake_inp()` (SHAKE format), `read_nrattle_ctl()`, `calc_atten_scatter()`, `adjust_damping_values()`.
- `units.py` — `pint`-based unit registry (`ureg`), `convert_units()`, and `convert_kwds_units()`.
- `time_integration.py` — numerical time integration helpers for `TimeDomainCalculator`.
- `_contracts.py` — private dataclasses (`NonlinearSoilCurves`, `VelocityProfile`) mirroring `pygmm.contracts` so pygmm is not a runtime dependency; `SoilType` and `Profile` accept duck-typed inputs matching these contracts.

### Optional dependencies

- **numba** — JIT-compiles inner loops in `propagation.py` and `time_integration.py`; automatically detected at import time via `HAS_NUMBA`.
- **disba** (`dispersion` extra) — surface-wave dispersion checks via `variation.DispersionCheck`.
- **pygmm** — no runtime dependency; interop uses `_contracts.py` duck-typing.
