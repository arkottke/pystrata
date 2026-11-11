# Changelog

## Unreleased

- Fixed: `NonlinearCurve` swapped the values returned outside the tabulated range for
  curves whose strains are in decreasing order, and evaluated every column of a
  frequency-dependent curve at the first strain of a strain vector. Columns now use
  their own strain, and a strain vector of the wrong length raises `ValueError`.
- Fixed: `FrequencyDependentEqlCalculator` ran the `zr15` method for any unknown
  `method`, including the documented but unimplemented `tan_avg`. Unknown methods now
  raise `ValueError`. The docstring no longer lists `tan_avg` and gives the real
  `strain_ratio` default (0.65).
- Fixed: `Profile.vs30()` ignored the travel time through the half-space when it
  starts above 30 m, overestimating Vs30 (600 instead of 428.6 m/s for 10 m at
  200 m/s over a 1000 m/s half-space). Profiles with soil to 30 m or deeper are
  unchanged.
- Fixed: `run_ensemble` without a `seed` drew every realization from the shared
  module-level generator, so workers started with `fork` produced repeated
  realizations. A base seed is now drawn once from OS entropy (logged at INFO) and
  each realization uses its own stream, for serial and parallel runs alike.
- Fixed: `VelocityVariation` (`ToroVelocityVariation`, `DepthDependToroVelVariation`)
  built a tridiagonal covariance matrix, which is not positive semi-definite for high
  correlations and gave only about half of the intended correlation between layers
  and a standard deviation inflated by 8-18 %. The covariance is now the
  first-order autoregressive one of Toro (1995), where the correlation of two layers
  is the product of the adjacent-layer correlations between them. Seeded velocity
  realizations of profiles with more than one soil layer change.

## 0.6.0 (2026-11-05)

- Fixed: restore `.readthedocs.yaml`, which was removed in June 2026, so that Read the
  Docs can build the documentation. It uses Python 3.13, the floor of
  `requires-python`, and installs pandoc for the example notebooks.
- Added: example 25 shows the effect of the canyon length on the response of a dam.
- Added: canyon (3D) effects for dams. `Dam(..., canyon=RectangularCanyon(length))`
  confines the dam between rigid abutments that move with the outcrop motion. The
  calculators give the response at a position `x` along the canyon (`canyon_x`, or the
  `x` argument of `DamProfile.transfer_function`), the site is independent 1D columns
  coupled by a Robin condition at the base, equivalent-linear strains include the
  longitudinal strain, and the sliding mass transfer function is averaged over the
  canyon. Closed forms for a dam on a rigid base: `canyon_rect_natural_freqs`,
  `canyon_freq_ratio` and `canyon_rect_rigid_base_tf`.
- Added: `KappaCorrectedCalculator` wraps a calculator and corrects the site
  attenuation (kappa) of the computed motions to a target: it reads the kappa of the
  input motion, runs the calculation and corrects the transfer function of each
  location, so every output is corrected. `TimeSeriesMotion.calc_attenuation` and
  `motion.fit_attenuation` compute kappa from a Fourier spectrum, with the arguments
  and results of the random vibration theory motions. Example 24. From the work of
  James Dea in pull request #68.
- Added: `Profile.auto_discretize` accepts a maximum frequency and a wave fraction for
  each layer. From James Dea, pull request #68.
- Changed: `TimeSeriesMotion.load_at2_file` finds the number of points and the time
  step in the header by their labels, which allows reversed order, missing commas and
  Fortran exponents, and warns when the number of points does not match the data.
  From James Dea, pull request #68.
- Changed: the documentation is reorganized: the user guide starts with how the
  motion, profile, calculator and outputs fit together, has pages for variation and
  dams, and lists the examples of each topic; the example gallery is grouped by
  topic; and best practices and calculator recommendations are removed.
- Added: example 22 compares the truncated wedge with the apex solution.
- Fixed: `Profile.simplified_rayliegh_vel` is renamed `simplified_rayleigh_vel`; the
  misspelled name remains as a deprecated alias.
- Added: failure surfaces and the transfer function of the sliding mass of a dam.
  `dam.CircularSurface(x, y, radius)` is a circle given by its center and radius,
  and `dam.toe_circle(dam, radius_ratio)` is the one through the toe (at least as shallow as the circle tangent to the base) and the far
  side of the crest. `dam.plot_failure_surface` draws the section, the surface and
  the sliding mass. `dam.sliding_mass_transfer_function` (and
  `DamProfile.sliding_mass_transfer_function` for the linear case) averages the
  acceleration transfer functions over the mass, weighted by its width, so it
  applies to the equivalent-linear calculators too. This is the quantity that
  Hale and Abrahamson (2019) model.
- Added: `Dam` takes a `slope` (horizontal run per unit height of each slope) and
  a `crest_width`. A crest width truncates the shear wedge, with a free surface
  at the crest, so the dynamics are for the trapezoidal section; without a slope
  the crest is the apex of the wedge, as before. `Dam.width`, `area`, `base_width`
  and `crest_offset` describe the section, and `Dam.from_inhomogeneity` takes the
  geometry (its `shear_vel_avg` is the average over the area of the section).
  For a 10 m crest on a 100 m dam the first-mode frequency is 1.7% lower than
  for the pure wedge, and 6.5% lower for a 40 m crest.
  `dam.analytical.wedge_truncated_*` give the closed forms of a truncated,
  homogeneous dam, which the numerical solution matches to 1e-8 up to 80 Hz.
- Changed: the velocity and soil type variations take a `limit_mode`, which
  selects how the random variates are limited to 2 standard deviations:
  `"normal"` (no limit), `"clipped"` (clipped, with the standard deviation of
  the model), or `"clipped_inflated"` (the default: the standard deviation is
  increased by 1.042 before clipping, so the clipped variates have the
  standard deviation of the model). `TruncatedNorm` is now `LimitedNorm`.
  **Results change.** The realized standard deviation of the ln velocity
  was 1.08 of the model's, with 4% of the draws on the limit, and the soil type
  curves varied by 0.88 (random mode) and 0.93 (fixed percentiles) of the
  model's, so the two modes sampled different distributions and the requested
  correlation of 0.7 came out as 0.63. All now have the model's standard
  deviation (0.96 in the `"clipped"` mode) and the requested correlation.
- Added: `pystrata.dam` for the seismic response of dams modeled as shear wedges
  (Gazetas and Dakoulas, 1992). A `Dam` (layered wedge, optionally with
  `G ~ z**m`) is joined to a site `Profile` as a `DamProfile` and analyzed with
  `DamLinearElasticCalculator`, `DamEquivalentLinearCalculator`, or
  `DamFrequencyDependentEqlCalculator`. `Dam.tune()` scales the velocities to a
  target fixed-base or coupled frequency. Closed-form solutions from Table 2 of
  the paper are in `pystrata.dam.analytical`. See example 20.
- Added: `site.layers_from_depth_power_law` (`Vs = a z**b`) and
  `site.layers_from_stress_power_law` (`Vs = a (sigma_m' / pa)**b`, from a
  constant unit weight, water-table depth and an initial vertical stress).
- Changed: `Layer.copy()` and `Profile.auto_discretize()` preserve `Layer` subclasses.
- Changed: adopted [SPEC 0](https://scientific-python.org/specs/spec-0000/) for
  version support -- Python minors from the last 36 months, core dependencies
  from the last 24 months. **Minimum Python is now 3.13** (was 3.10), and
  dependency lower bounds are declared explicitly: `matplotlib>=3.10`,
  `numba>=0.61`, `numpy>=2.2`, `pint>=0.25`, `pykooh>=0.5`, `pyrvt>=0.8`,
  `scipy>=1.15`, `xarray>=2024.11`. CI now runs 3.13 and 3.14. The dated
  drop schedule is in `pyproject.toml`.

    The old `>=3.10` floor was already unmet in practice: numpy, scipy, pandas,
    matplotlib and pint all require >=3.11, so a 3.10 install silently resolved
    to a much older stack (pandas 2.3 rather than 3.0, numpy 2.2 rather than
    2.5). That forced the lockfile to fork on `python_full_version < '3.11'` into 13 duplicated packages.

- Changed: dropped the `tomli` dependency; `tomllib` has been in the standard
  library since Python 3.11.

**Breaking changes**:

- Removed the `variation.random_state` global. It was never read by any model,
  and the `set_seed` usage its comment suggested is not a `RandomState` method.
  The variation models no longer draw from NumPy's global generator, so
  `np.random.seed()` does not control them.
  **Migration**: pass `rng=` to an individual variation model, or `seed=` to
  `iter_varied_profiles` / `run_ensemble`.
- `ProfileBasedOutput` subclasses now implement `_calc_profile(calc)` returning
  `(depths, values)`, instead of overriding `__call__`. Only affects code that
  subclasses them.
- `numba` is now a required dependency rather than an optional extra, and the
  `[numba]` extra has been removed. It was already effectively required, since
  `pykooh` and `pyrvt` both import it unconditionally.
  **Migration**: install `pystrata` instead of `pystrata[numba]`.

- Removed empirical soil-curve classes — now live in `pygmm.soil_curves`:
  `DarendeliSoilType`, `MenqSoilType`, `WangSoilType`, `AlemuEtAlSoilType`,
  `RollinsEtAlSoilType`, `KishidaSoilType`. The user-parameterized
  `ModifiedHyperbolicSoilType` and `TwoParamModifiedHyperbolicSoilType` were
  removed and have no replacement; pygmm has only the private base class
  `ModifiedHyperbolicBase`.
  **Migration**: `SoilType.from_curves(pygmm.DarendeliSoilType(...).curves())`
- Removed `kea16_profile()` — now `pygmm.velocity_profile.kea16_profile()`.
  **Migration**: `Profile.from_velocity_profile(pygmm.kea16_profile(...), soil_types=...)`

Added:

- `SoilType.from_curves(curves)` — create from any object with `.strains`,
  `.mod_reduc`, `.damping`, `.damping_min` (duck-typed; accepts `pygmm.contracts.NonlinearSoilCurves`).
- `Profile.from_velocity_profile(vp, soil_types, layer_thickness=1.0)` — create from
  any object with `.depth`, `.vs_median`, `.std_vs_ln` (accepts `pygmm.contracts.VelocityProfile`).
- `pystrata._contracts` — private `NonlinearSoilCurves` and `VelocityProfile` dataclasses
  (mirrors `pygmm.contracts`; no runtime pygmm dependency).
- General (inhomogeneous) SII/SH wave propagation for `LinearElasticCalculator`,
  `EquivalentLinearCalculator`, and `FrequencyDependentEqlCalculator` via new
  `incidence_angle` (θ) and `inhomogeneity` (γ) keyword arguments (degrees).
  Nonzero values engage the general viscoelastic transfer-matrix solution
  (Borcherdt, _Viscoelastic Waves in Layered Media_, Ch. 9), supporting oblique
  incidence and inhomogeneous body waves. Defaults (`0.0`) reproduce the previous
  normal-incidence, homogeneous-wave results unchanged.
- `pystrata.runner` — `run_ensemble(..., n_jobs=)` evaluates realizations,
  motions, and logic-tree branches as one flat task list, serially or across
  processes, and `run_realization()` runs a single one. Results do not depend
  on `n_jobs` or `chunksize`. `n_jobs=1` (the default) reproduces the loop the
  examples write by hand.
- `rng=` on every variation model's `__call__`, and `seed=` on
  `iter_varied_profiles`. Realization _i_ is derived from
  `SeedSequence([seed, i])`, so it is identical regardless of how many
  realizations are requested or the order they are consumed in.
- `variation.varied_profile(profile, index, seed=...)` — generate one
  realization without producing the ones before it.
- `var_depth=` slot in `iter_varied_profiles`. A `HalfSpaceDepthVariation` had
  to be passed through `var_thickness`, so the two could not be combined.
- `HalfSpaceDepthVariation.dist` and `.depth_limit(quantile)`.
- `Profile.depth_grid()` — uniform depth grid using the same
  `wave_frac * Vs / max_freq` criterion as `auto_discretize`, extending past
  the range a depth variation samples. Warns when a supplied `max_depth` is
  shallower than that range.
- `ProfileBasedOutput(depths=, fill_below=)` — opt-in fixed depth grid, so
  profile results have a constant shape and are directly comparable across
  realizations. `fill_below='nan'` (the default when gridded) excludes depths
  below a realization's base from the statistics.
- `calc_stats()` now reports `count`, the number of realizations contributing
  at each depth.
- `Output.reserve(n)` and `Output.extend(other)` — pre-allocate storage and
  write by index, or combine separately accumulated results.
- `Output.to_xarray()` without a logic tree, returning dimensions
  `(ref_name, realization)`. Varying references become a 2-D data variable
  rather than a coordinate, which would otherwise align on every distinct
  value.
- `LogicTree.__len__`.

Changed:

- Result accumulation buffers columns and stacks them once, instead of
  reallocating the whole array per realization. Collecting 800 realizations of
  4000 samples went from ~2.0 s to ~9 ms.
- `ProfileBasedOutput.to_dataframe()` now defaults to the same depth grid as
  `calc_stats()` (512 points with a margin, was 50 without).
- `MaxAccelProfile` interpolates linearly rather than as a step function, since
  acceleration is a continuous field rather than a layer property. This changes
  reported values between layer tops.
- `MaxStrainProfile` and `CyclicStressRatioProfile` report a final point at the
  base of the soil column. They previously stopped at the mid-depth of the
  deepest layer, leaving its lower half unrepresented.
- `CyclicStressRatioProfile` gained the `xlabel` it was missing, which made
  `plot()` raise `AttributeError`.

Fixed:

- `HalfSpaceDepthVariation` sub-divided the extended layer using
  `ceil(total // orig)`, where the floor division made the `ceil` a no-op. An
  overshoot of 19 m on a 20 m layer produced a single 39 m layer instead of
  two of 19.5 m. A non-positive sampled depth now raises instead of silently
  producing a malformed profile.
- `propagation.py` applied `@numba.jit` at module scope outside the `HAS_NUMBA`
  guard, so the pure-Python fallback beside it was unreachable and the module
  would raise `NameError` without numba. The duplicated function body it
  required is gone; numba is now a required dependency.

## 0.5.5 (2024-10-16)

- Change: Use `method` to define FrequencyDependentEqlCalculator options
- Add: FrequencyDependentEqlCalculator method based on smoothing of the strain spectrum
- Add: Tools for creating logic trees

## 0.5.4 (2024-03-29)

- Fix: error in example-08 that didn't reference the modified profiles.
- Change: method to create SoilTypes from published curves
- Add: Extended example-15 to show how to use published curves in site response models.

## 0.5.3 (2024-03-29)

- Added published curves

## 0.5.2 (2023-01-18)

- Fixed: Providing unsmoothed transfer function output
- Fixed #18: MenqSoilType

## 0.5.1 (2022-09-22)

- Fixed: Correlation model from Toro. Previously used rho_0 instead of
  d_0, and the wrong depth
- Renamed: BedrockDepthVariation to HalfSpaceDepthVariation
- Fixed: HalfSpaceDepthVariation was removing the last layer

## 0.5.0 (2022-06-14)

- Renamed to pyStrata

## 0.4.11 (2020-03-31)

- Added: Depth dependent velocity variation model
- Added: Output plotting functionality
- Added: Ability to exclude soil type variation from bedrock

## 0.4.10 (2020-03-27)

- Fixed: Error in SPID variation of G/Gmax
- Added: Scaling during read of SMC and AT2 input motions

## 0.4.9 (2020-03-09)

- Add InitialVelProfile and CompatVelProfile outputs

## 0.4.8 (2019-12-11)

- Remove Cython and cyko as dependencies
- Added a numba based Konno-Ohmachi smoothing

## 0.4.6 (2019-11-12)

- FIXED #11: Dependencies missing on install.

## 0.4.5 (2019-10-24)

- FIXED #9: Wrong stress for some Menq components.

## 0.4.4 (2019-05-22)

- Incremented version because of issue with automated builds.

## 0.4.3 (2019-05-22)

- FIXED: Bug in MANIFEST.in

## 0.4.2 (2019-05-22)

- Incremented version because of issue with automated builds.

## 0.4.1 (2019-05-22)

- Fixed strain profile to use `max_strain`.
- Changed README and HISTORY to markdown.

## 0.4.0 (2019-03-11)

- Added smoothed FourierAmplitudeSpectrum output.

## 0.3.2 (2018-12-02)

- Fixed building of docs.
- Removed stickler.
- Version double increment due to pypi naming conflict.

## 0.3.0 (2018-11-30)

- Converted all damping to decimal.
- Added tests for KishidaSoilType.
- Added tests against Deepsoil.

## 0.0.1 (2016-04-30)

- First release on PyPI.
