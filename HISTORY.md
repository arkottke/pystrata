# History

## Unreleased — 0.6.0

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
  `RollinsEtAlSoilType`, `KishidaSoilType`, `ModifiedHyperbolicSoilType`,
  `TwoParamModifiedHyperbolicSoilType`.
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

## v0.5.5 (2024-10-16)

- Change: Use `method` to define FrequencyDependentEqlCalculator options
- Add: FrequencyDependentEqlCalculator method based on smoothing of the strain spectrum
- Add: Tools for creating logic trees

## v0.5.4 (2024-03-29)

- Fix: error in example-08 that didn't reference the modified profiles.
- Change: method to create SoilTypes from published curves
- Add: Extended example-15 to show how to use published curves in site response models.

## v0.5.3 (2024-03-29)

- Added published curves

## v0.5.2 (2023-01-18)

- Fixed: Providing unsmoothed transfer function output
- Fixed #18: MenqSoilType

## v0.5.1 (2022-09-22)

- Fixed: Correlation model from Toro. Previously used rho_0 instead of
  d_0, and the wrong depth
- Renamed: BedrockDepthVariation to HalfSpaceDepthVariation
- Fixed: HalfSpaceDepthVariation was removing the last layer

## v0.5.0 (2022-06-14)

- Renamed to pyStrata

## v0.4.11 (2020-03-31)

- Added: Depth dependent velocity variation model
- Added: Output plotting functionality
- Added: Ability to exclude soil type variation from bedrock

## v0.4.10 (2020-03-27)

- Fixed: Error in SPID variation of G/Gmax
- Added: Scaling during read of SMC and AT2 input motions

## v0.4.9 (2020-03-09)

- Add InitialVelProfile and CompatVelProfile outputs

## v0.4.8 (2019-12-11)

- Remove Cython and cyko as dependencies
- Added a numba based Konno-Ohmachi smoothing

## v0.4.6 (2019-11-12)

- FIXED #11: Dependencies missing on install.

## v0.4.5 (2019-10-24)

- FIXED #9: Wrong stress for some Menq components.

## v0.4.4 (2019-05-22)

- Incremented version because of issue with automated builds.

## v0.4.3 (2019-05-22)

- FIXED: Bug in MANIFEST.in

## v0.4.2 (2019-05-22)

- Incremented version because of issue with automated builds.

## v0.4.1 (2019-05-22)

- Fixed strain profile to use `max_strain`.
- Changed README and HISTORY to markdown.

## v0.4.0 (2019-03-11)

- Added smoothed FourierAmplitudeSpectrum output.

## v0.3.2 (2018-12-02)

- Fixed building of docs.
- Removed stickler.
- Version double increment due to pypi naming conflict.

## v0.3.0 (2018-11-30)

- Converted all damping to decimal.
- Added tests for KishidaSoilType.
- Added tests against Deepsoil.

## v0.0.1 (2016-04-30)

- First release on PyPI.
