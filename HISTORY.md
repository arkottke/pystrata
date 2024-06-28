# History

## v0.5.4 (2024-03-29)

-   Fix: error in example-08 that didn't reference the modified profiles.
-   Change: method to create SoilTypes from published curves
-   Add: Extended example-15 to show how to use published curves in site response models.

## v0.5.3 (2024-03-29)

-   Added published curves

## v0.5.2 (2023-01-18)

-   Fixed: Providing unsmoothed transfer function output
-   Fixed #18: MenqSoilType

## v0.5.1 (2022-09-22)

-   Fixed: Correlation model from Toro. Previously used rho_0 instead of
    d_0, and the wrong depth
-   Renamed: BedrockDepthVariation to HalfSpaceDepthVariation
-   Fixed: HalfSpaceDepthVariation was removing the last layer

## v0.5.0 (2022-06-14)

-   Renamed to pyStrata

## v0.4.11 (2020-03-31)

-   Added: Depth dependent velocity variation model
-   Added: Output plotting functionality
-   Added: Ability to exclude soil type variation from bedrock

## v0.4.10 (2020-03-27)

-   Fixed: Error in SPID variation of G/Gmax
-   Added: Scaling during read of SMC and AT2 input motions

## v0.4.9 (2020-03-09)

-   Add InitialVelProfile and CompatVelProfile outputs

## v0.4.8 (2019-12-11)

-   Remove Cython and cyko as dependencies
-   Added a numba based Konno-Ohmachi smoothing

## v0.4.6 (2019-11-12)

-   FIXED #11: Dependencies missing on install.

## v0.4.5 (2019-10-24)

-   FIXED #9: Wrong stress for some Menq components.

## v0.4.4 (2019-05-22)

-   Incremented version because of issue with automated builds.

## v0.4.3 (2019-05-22)

-   FIXED: Bug in MANIFEST.in

## v0.4.2 (2019-05-22)

-   Incremented version because of issue with automated builds.

## v0.4.1 (2019-05-22)

-   Fixed strain profile to use `max_strain`.
-   Changed README and HISTORY to markdown.

## v0.4.0 (2019-03-11)

-   Added smoothed FourierAmplitudeSpectrum output.

## v0.3.2 (2018-12-02)

-   Fixed building of docs.
-   Removed stickler.
-   Version double increment due to pypi naming conflict.

## v0.3.0 (2018-11-30)

-   Converted all damping to decimal.
-   Added tests for KishidaSoilType.
-   Added tests against Deepsoil.

## v0.0.1 (2016-04-30)

-   First release on PyPI.
