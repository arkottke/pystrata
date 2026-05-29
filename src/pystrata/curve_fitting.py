# The MIT License (MIT)
#
# Copyright (c) 2016-2018 Albert Kottke
#
# Permission is hereby granted, free of charge, to any person obtaining a copy
# of this software and associated documentation files (the "Software"), to deal
# in the Software without restriction, including without limitation the rights
# to use, copy, modify, merge, publish, distribute, sublicense, and/or sell
# copies of the Software, and to permit persons to whom the Software is
# furnished to do so, subject to the following conditions:
#
# The above copyright notice and this permission notice shall be included in
# all copies or substantial portions of the Software.
#
# THE SOFTWARE IS PROVIDED "AS IS", WITHOUT WARRANTY OF ANY KIND, EXPRESS OR
# IMPLIED, INCLUDING BUT NOT LIMITED TO THE WARRANTIES OF MERCHANTABILITY,
# FITNESS FOR A PARTICULAR PURPOSE AND NONINFRINGEMENT. IN NO EVENT SHALL THE
# AUTHORS OR COPYRIGHT HOLDERS BE LIABLE FOR ANY CLAIM, DAMAGES OR OTHER
# LIABILITY, WHETHER IN AN ACTION OF CONTRACT, TORT OR OTHERWISE, ARISING FROM,
# OUT OF OR IN CONNECTION WITH THE SOFTWARE OR THE USE OR OTHER DEALINGS IN THE
# SOFTWARE.
"""Curve fitting utilities for constitutive model parameter calibration.

This module provides functions to fit MKZ and HH model parameters to modulus reduction
(G/Gmax) and damping curves.
"""

from __future__ import annotations

import logging
import time
from typing import TYPE_CHECKING

import numpy as np
from scipy.optimize import differential_evolution

from .constitutive import (
    HHParams,
    MKZParams,
    MultiLayerParams,
    calc_damping,
    calc_mod_reduc,
    hh_damping_only_misfit,
    hh_misfit,
    hh_modreduc_only_misfit,
    mkz_damping_misfit,
)

logger = logging.getLogger(__name__)

if TYPE_CHECKING:
    from .site import Layer, Profile


def fit_mkz_to_mod_reduc(
    strains: np.ndarray,
    mod_reduc: np.ndarray,
    shear_mod: float = 1.0,
    seed: int | None = None,
    **kwargs,
) -> MKZParams:
    """Fit MKZ model parameters to a modulus reduction curve.

    Parameters
    ----------
    strains : np.ndarray
        Strain values (unit: 1, not %).
    mod_reduc : np.ndarray
        G/Gmax values corresponding to strains.
    shear_mod : float, optional
        Initial shear modulus [Pa]. Default is 1.0 (normalized).
    seed : int, optional
        Random seed for reproducibility.
    **kwargs
        Additional arguments passed to scipy.optimize.differential_evolution.

    Returns
    -------
    params : MKZParams
        Fitted MKZ model parameters.
    """
    strains = np.asarray(strains, dtype=np.float64)
    mod_reduc = np.asarray(mod_reduc, dtype=np.float64)

    # Bounds for [log10(gamma_ref), log10(beta), log10(s)]
    # gamma_ref: 1e-6 to 1e-1 (typical reference strains)
    # beta: 0.1 to 10 (typical shape parameter)
    # s: 0.5 to 2.0 (typical exponent)
    bounds = [(-6, -1), (-1, 1), (-0.3, 0.3)]

    def misfit(x):
        gamma_ref = 10 ** x[0]
        beta = 10 ** x[1]
        s = 10 ** x[2]

        params = MKZParams(gamma_ref=gamma_ref, beta=beta, s=s, shear_mod=shear_mod)
        pred = calc_mod_reduc(strains, params)

        # Mean squared error
        return np.mean((pred - mod_reduc) ** 2)

    default_kwargs = {
        "maxiter": 500,
        "tol": 1e-8,
        "polish": True,
        "workers": 1,
    }
    default_kwargs.update(kwargs)

    result = differential_evolution(misfit, bounds, **default_kwargs)

    fitted = MKZParams(
        gamma_ref=10 ** result.x[0],
        beta=10 ** result.x[1],
        s=10 ** result.x[2],
        shear_mod=shear_mod,
    )

    logger.debug(
        "fit_mkz_to_mod_reduc: gamma_ref=%.4e, beta=%.3f, s=%.3f (mse=%.2e)",
        fitted.gamma_ref,
        fitted.beta,
        fitted.s,
        result.fun,
    )

    return fitted


def fit_mkz_to_damping(
    strains: np.ndarray,
    damping: np.ndarray,
    shear_mod: float = 1.0,
    seed: int | None = None,
    **kwargs,
) -> MKZParams:
    """Fit MKZ model parameters to a damping curve.

    The damping is computed from the Masing hysteresis loop area.

    Parameters
    ----------
    strains : np.ndarray
        Strain values (unit: 1, not %).
    damping : np.ndarray
        Damping ratio values (unit: 1, not %).
    shear_mod : float, optional
        Initial shear modulus [Pa]. Default is 1.0 (normalized).
    seed : int, optional
        Random seed for reproducibility.
    **kwargs
        Additional arguments passed to scipy.optimize.differential_evolution.

    Returns
    -------
    params : MKZParams
        Fitted MKZ model parameters.
    """
    strains = np.asarray(strains, dtype=np.float64)
    damping = np.asarray(damping, dtype=np.float64)

    # Remove small-strain damping offset
    damping_offset = damping[0] if len(damping) > 0 else 0
    damping_adjusted = damping - damping_offset

    # Bounds for [log10(gamma_ref), log10(beta), log10(s)]
    bounds = [(-6, -1), (-1, 1), (-0.3, 0.3)]

    def misfit(x):
        gamma_ref = 10 ** x[0]
        beta = 10 ** x[1]
        s = 10 ** x[2]

        return mkz_damping_misfit(
            strains,
            damping_adjusted,
            gamma_ref,
            beta,
            s,
            shear_mod,
        )

    default_kwargs = {
        "maxiter": 500,
        "tol": 1e-8,
        "polish": True,
        "workers": 1,
    }
    default_kwargs.update(kwargs)

    result = differential_evolution(misfit, bounds, **default_kwargs)

    return MKZParams(
        gamma_ref=10 ** result.x[0],
        beta=10 ** result.x[1],
        s=10 ** result.x[2],
        shear_mod=shear_mod,
    )


def fit_hh(
    strains: np.ndarray,
    mod_reduc: np.ndarray,
    damping: np.ndarray,
    shear_mod: float,
    shear_strength: float | None = None,
    seed: int | None = None,
    **kwargs,
) -> HHParams:
    """Fit HH model parameters to modulus reduction and damping curves.

    The fitting is done in two stages:
    1. Fit MKZ parameters to the modulus reduction curve
    2. Optimize FKZ and transition parameters to match damping

    Parameters
    ----------
    strains : np.ndarray
        Strain values (unit: 1, not %).
    mod_reduc : np.ndarray
        G/Gmax values corresponding to strains.
    damping : np.ndarray
        Damping ratio values (unit: 1, not %).
    shear_mod : float
        Initial shear modulus [Pa].
    shear_strength : float, optional
        Shear strength (Tmax) [Pa]. If None, estimated as 0.001 * shear_mod.
    seed : int, optional
        Random seed for reproducibility.
    **kwargs
        Additional arguments passed to scipy.optimize.differential_evolution.

    Returns
    -------
    params : HHParams
        Fitted HH model parameters.
    """
    strains = np.asarray(strains, dtype=np.float64)
    mod_reduc = np.asarray(mod_reduc, dtype=np.float64)
    damping = np.asarray(damping, dtype=np.float64)

    # Default shear strength estimate
    if shear_strength is None:
        shear_strength = 0.001 * shear_mod

    # Remove small-strain damping offset for fitting
    damping_offset = damping[0] if len(damping) > 0 else 0
    damping_adjusted = damping - damping_offset

    # Stage 1: Fit MKZ to modulus reduction
    mkz_params = fit_mkz_to_mod_reduc(
        strains, mod_reduc, shear_mod=shear_mod, seed=seed
    )

    # Stage 2: Optimize all 9 HH parameters jointly
    # Bounds for [log10(gamma_t), log10(a), log10(gamma_ref), log10(beta), log10(s),
    #             log10(mu), log10(Tmax), log10(d)]
    # Note: shear_mod is fixed, not optimized
    bounds = [
        (-6, -1),  # gamma_t
        (-1, 1),  # a
        (
            np.log10(mkz_params.gamma_ref) - 1,
            np.log10(mkz_params.gamma_ref) + 1,
        ),  # gamma_ref
        (np.log10(mkz_params.beta) - 1, np.log10(mkz_params.beta) + 1),  # beta
        (np.log10(mkz_params.s) - 0.5, np.log10(mkz_params.s) + 0.5),  # s
        (-1, 2),  # mu
        (np.log10(shear_strength) - 2, np.log10(shear_strength) + 2),  # Tmax
        (-0.5, 0.5),  # d
    ]

    def misfit(x):
        return hh_misfit(
            strains,
            mod_reduc,
            damping_adjusted,
            10 ** x[0],  # gamma_t
            10 ** x[1],  # a
            10 ** x[2],  # gamma_ref
            10 ** x[3],  # beta
            10 ** x[4],  # s
            shear_mod,
            10 ** x[5],  # mu
            10 ** x[6],  # shear_strength
            10 ** x[7],  # d
            4.039,  # trans_c1
            1.036,  # trans_c2
        )

    default_kwargs = {
        "maxiter": 1000,
        "tol": 1e-8,
        "polish": True,
        "workers": 1,
    }
    default_kwargs.update(kwargs)

    result = differential_evolution(misfit, bounds, **default_kwargs)

    return HHParams(
        gamma_t=10 ** result.x[0],
        a=10 ** result.x[1],
        gamma_ref=10 ** result.x[2],
        beta=10 ** result.x[3],
        s=10 ** result.x[4],
        shear_mod=shear_mod,
        mu=10 ** result.x[5],
        shear_strength=10 ** result.x[6],
        d=10 ** result.x[7],
    )


def fit_hh_modreduc_only(
    strains: np.ndarray,
    mod_reduc: np.ndarray,
    shear_mod: float,
    shear_strength: float | None = None,
    seed: int | None = None,
    **kwargs,
) -> HHParams:
    """Fit HH model parameters to modulus reduction curve only.

    The resulting parameters define the *loading* backbone whose G/Gmax
    matches the target modulus reduction. This is one half of the
    two-set fitting procedure used with the Li & Assimaki (2010) non-Masing
    hysteresis rule.

    Parameters
    ----------
    strains : np.ndarray
        Strain values (unit: 1, not %).
    mod_reduc : np.ndarray
        G/Gmax values corresponding to strains.
    shear_mod : float
        Initial shear modulus [Pa].
    shear_strength : float, optional
        Shear strength (Tmax) [Pa]. If None, estimated as 0.001 * shear_mod.
    seed : int, optional
        Random seed for reproducibility.
    **kwargs
        Additional arguments passed to scipy.optimize.differential_evolution.

    Returns
    -------
    params : HHParams
        Fitted HH model parameters (loading backbone).
    """
    strains = np.asarray(strains, dtype=np.float64)
    mod_reduc = np.asarray(mod_reduc, dtype=np.float64)

    if shear_strength is None:
        shear_strength = 0.001 * shear_mod

    # Initial MKZ fit for starting bounds
    mkz_params = fit_mkz_to_mod_reduc(
        strains, mod_reduc, shear_mod=shear_mod, seed=seed
    )

    bounds = [
        (-6, -1),  # gamma_t
        (-1, 1),  # a
        (np.log10(mkz_params.gamma_ref) - 1, np.log10(mkz_params.gamma_ref) + 1),
        (np.log10(mkz_params.beta) - 1, np.log10(mkz_params.beta) + 1),
        (np.log10(mkz_params.s) - 0.5, np.log10(mkz_params.s) + 0.5),
        (-1, 2),  # mu
        (np.log10(shear_strength) - 2, np.log10(shear_strength) + 2),
        (-0.5, 0.5),  # d
    ]

    def misfit(x):
        return hh_modreduc_only_misfit(
            strains,
            mod_reduc,
            10 ** x[0],
            10 ** x[1],
            10 ** x[2],
            10 ** x[3],
            10 ** x[4],
            shear_mod,
            10 ** x[5],
            10 ** x[6],
            10 ** x[7],
            4.039,
            1.036,
        )

    default_kwargs = {"maxiter": 1000, "tol": 1e-8, "polish": True, "workers": 1}
    default_kwargs.update(kwargs)

    result = differential_evolution(misfit, bounds, **default_kwargs)

    return HHParams(
        gamma_t=10 ** result.x[0],
        a=10 ** result.x[1],
        gamma_ref=10 ** result.x[2],
        beta=10 ** result.x[3],
        s=10 ** result.x[4],
        shear_mod=shear_mod,
        mu=10 ** result.x[5],
        shear_strength=10 ** result.x[6],
        d=10 ** result.x[7],
    )


def fit_hh_damping_only(
    strains: np.ndarray,
    damping: np.ndarray,
    shear_mod: float,
    shear_strength: float | None = None,
    seed: int | None = None,
    **kwargs,
) -> HHParams:
    """Fit HH model parameters to damping curve only.

    The resulting parameters define the *unloading* backbone whose
    Masing-derived damping matches the target damping curve. This is
    one half of the two-set fitting procedure used with the Li & Assimaki
    (2010) non-Masing hysteresis rule.

    Parameters
    ----------
    strains : np.ndarray
        Strain values (unit: 1, not %).
    damping : np.ndarray
        Damping ratio values (unit: 1, not %).
    shear_mod : float
        Initial shear modulus [Pa].
    shear_strength : float, optional
        Shear strength (Tmax) [Pa]. If None, estimated as 0.001 * shear_mod.
    seed : int, optional
        Random seed for reproducibility.
    **kwargs
        Additional arguments passed to scipy.optimize.differential_evolution.

    Returns
    -------
    params : HHParams
        Fitted HH model parameters (unloading backbone).
    """
    strains = np.asarray(strains, dtype=np.float64)
    damping = np.asarray(damping, dtype=np.float64)

    if shear_strength is None:
        shear_strength = 0.001 * shear_mod

    # Remove small-strain damping offset
    damping_offset = damping[0] if len(damping) > 0 else 0
    damping_adjusted = damping - damping_offset

    bounds = [
        (-6, -1),  # gamma_t
        (-1, 1),  # a
        (-6, -1),  # gamma_ref
        (-1, 1),  # beta
        (-0.3, 0.3),  # s
        (-1, 2),  # mu
        (np.log10(shear_strength) - 2, np.log10(shear_strength) + 2),
        (-0.5, 0.5),  # d
    ]

    def misfit(x):
        return hh_damping_only_misfit(
            strains,
            damping_adjusted,
            10 ** x[0],
            10 ** x[1],
            10 ** x[2],
            10 ** x[3],
            10 ** x[4],
            shear_mod,
            10 ** x[5],
            10 ** x[6],
            10 ** x[7],
            4.039,
            1.036,
        )

    default_kwargs = {"maxiter": 1000, "tol": 1e-8, "polish": True, "workers": 1}
    default_kwargs.update(kwargs)

    result = differential_evolution(misfit, bounds, **default_kwargs)

    return HHParams(
        gamma_t=10 ** result.x[0],
        a=10 ** result.x[1],
        gamma_ref=10 ** result.x[2],
        beta=10 ** result.x[3],
        s=10 ** result.x[4],
        shear_mod=shear_mod,
        mu=10 ** result.x[5],
        shear_strength=10 ** result.x[6],
        d=10 ** result.x[7],
    )


def fit_layer_mkz(layer: Layer, seed: int | None = None, **kwargs) -> MKZParams:
    """Fit MKZ parameters for a single layer.

    Parameters
    ----------
    layer : Layer
        pystrata Layer with soil_type that has nonlinear curves.
    seed : int, optional
        Random seed for reproducibility.
    **kwargs
        Additional arguments passed to fitting functions.

    Returns
    -------
    params : MKZParams
        Fitted MKZ model parameters.
    """
    soil_type = layer.soil_type

    if not soil_type.is_nonlinear:
        # Linear layer - use default values
        return MKZParams(
            gamma_ref=0.01,  # 1%
            beta=1.0,
            s=1.0,
            shear_mod=layer.initial_shear_mod,
        )

    # Get modulus reduction curve
    mod_reduc_curve = soil_type.mod_reduc
    strains = mod_reduc_curve.strains
    mod_reduc = mod_reduc_curve.values

    return fit_mkz_to_mod_reduc(
        strains, mod_reduc, shear_mod=layer.initial_shear_mod, seed=seed, **kwargs
    )


def fit_layer_hh(
    layer: Layer,
    depth: float | None = None,
    seed: int | None = None,
    **kwargs,
) -> HHParams:
    """Fit HH parameters for a single layer.

    Parameters
    ----------
    layer : Layer
        pystrata Layer with soil_type that has nonlinear curves.
    depth : float, optional
        Depth to layer midpoint [m], used for shear strength estimation.
    seed : int, optional
        Random seed for reproducibility.
    **kwargs
        Additional arguments passed to fitting functions.

    Returns
    -------
    params : HHParams
        Fitted HH model parameters.
    """
    soil_type = layer.soil_type
    shear_mod = layer.initial_shear_mod

    # Estimate shear strength from vertical stress (Ladd, 1991)
    # Su/sigma_v' ≈ 0.22 for normally consolidated clays
    if depth is not None:
        # Assume density ~1800 kg/m³ for stress calculation
        stress_vert = 1800 * 9.81 * depth
        shear_strength = 0.22 * stress_vert
    else:
        # Default: assume shear strength is 0.1% of shear modulus
        shear_strength = 0.001 * shear_mod

    if not soil_type.is_nonlinear:
        # Linear layer - use default values
        return HHParams(
            gamma_t=0.001,
            a=1.0,
            gamma_ref=0.01,
            beta=1.0,
            s=1.0,
            shear_mod=shear_mod,
            mu=1.0,
            shear_strength=shear_strength,
            d=1.0,
        )

    # Get modulus reduction and damping curves
    mod_reduc_curve = soil_type.mod_reduc
    strains = mod_reduc_curve.strains
    mod_reduc = mod_reduc_curve.values

    # Get damping values at same strains
    damping_curve = soil_type.damping
    if callable(damping_curve):
        # DampingCurve object
        damping = damping_curve(strains)
    else:
        # Constant damping
        damping = np.full_like(strains, damping_curve)

    return fit_hh(
        strains,
        mod_reduc,
        damping,
        shear_mod=shear_mod,
        shear_strength=shear_strength,
        seed=seed,
        **kwargs,
    )


def fit_profile(
    profile: Profile,
    model: str = "hh",
    seed: int | None = None,
    verbose: bool = False,
    **kwargs,
) -> MultiLayerParams:
    """Fit constitutive model parameters for all layers in a profile.

    Parameters
    ----------
    profile : Profile
        pystrata Profile with layers.
    model : str
        Model type: 'mkz' or 'hh'.
    seed : int, optional
        Random seed for reproducibility.
    verbose : bool
        Print progress information.
    **kwargs
        Additional arguments passed to fitting functions.

    Returns
    -------
    params : MultiLayerParams
        Container with fitted parameters for each layer.
    """
    start_time = time.perf_counter()
    multi_params = MultiLayerParams()
    n_layers = len(profile) - 1

    logger.debug(
        "fit_profile: fitting %s parameters for %d layers", model.upper(), n_layers
    )

    # Calculate cumulative depth for shear strength estimation
    depth = 0.0

    for i, layer in enumerate(profile[:-1]):  # Exclude halfspace
        layer_depth = depth + layer.thickness / 2

        if verbose:
            print(f"Fitting layer {i + 1}/{len(profile) - 1}: depth={layer_depth:.2f}m")

        if model.lower() == "mkz":
            params = fit_layer_mkz(layer, seed=seed, **kwargs)
        elif model.lower() == "hh":
            params = fit_layer_hh(layer, depth=layer_depth, seed=seed, **kwargs)
        else:
            raise ValueError(f"Unknown model: {model}. Use 'mkz' or 'hh'.")

        multi_params.append(params)
        depth += layer.thickness

    elapsed = time.perf_counter() - start_time
    logger.info(
        "fit_profile: fitted %d layers with %s model in %.3fs",
        n_layers,
        model.upper(),
        elapsed,
    )

    return multi_params


def _get_layer_curves(layer: Layer) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    """Extract strains, modulus reduction and damping arrays from a layer."""
    soil_type = layer.soil_type
    mod_reduc_curve = soil_type.mod_reduc
    strains = mod_reduc_curve.strains
    mod_reduc = mod_reduc_curve.values

    damping_curve = soil_type.damping
    if callable(damping_curve):
        damping = damping_curve(strains)
    else:
        damping = np.full_like(strains, damping_curve)

    return strains, mod_reduc, damping


def _estimate_shear_strength(layer: Layer, depth: float | None) -> float:
    """Estimate shear strength for a layer at a given depth."""
    shear_mod = layer.initial_shear_mod
    if depth is not None:
        stress_vert = 1800 * 9.81 * depth
        return 0.22 * stress_vert
    return 0.001 * shear_mod


def _default_hh_params(shear_mod: float, shear_strength: float) -> HHParams:
    """Return default HH parameters for a linear layer."""
    return HHParams(
        gamma_t=0.001,
        a=1.0,
        gamma_ref=0.01,
        beta=1.0,
        s=1.0,
        shear_mod=shear_mod,
        mu=1.0,
        shear_strength=shear_strength,
        d=1.0,
    )


def fit_layer_hh_two_set(
    layer: Layer,
    depth: float | None = None,
    seed: int | None = None,
    **kwargs,
) -> tuple[HHParams, HHParams]:
    """Fit separate modulus-reduction and damping HH parameters for one layer.

    Returns two parameter sets: ``mod_params`` controls the loading backbone
    (fitted to G/Gmax) and ``damp_params`` controls the unloading backbone
    (fitted to damping). Together they are used by the Li & Assimaki (2010)
    non-Masing hysteresis rule.

    Parameters
    ----------
    layer : Layer
        pystrata Layer with nonlinear soil type.
    depth : float, optional
        Depth to layer midpoint [m].
    seed : int, optional
        Random seed for reproducibility.
    **kwargs
        Additional arguments passed to scipy.optimize.differential_evolution.

    Returns
    -------
    mod_params : HHParams
        Parameters fitted to modulus reduction (loading backbone).
    damp_params : HHParams
        Parameters fitted to damping (unloading backbone).
    """
    soil_type = layer.soil_type
    shear_mod = layer.initial_shear_mod
    shear_strength = _estimate_shear_strength(layer, depth)

    if not soil_type.is_nonlinear:
        default = _default_hh_params(shear_mod, shear_strength)
        return default, default

    strains, mod_reduc, damping = _get_layer_curves(layer)

    mod_params = fit_hh_modreduc_only(
        strains,
        mod_reduc,
        shear_mod=shear_mod,
        shear_strength=shear_strength,
        seed=seed,
        **kwargs,
    )
    damp_params = fit_hh_damping_only(
        strains,
        damping,
        shear_mod=shear_mod,
        shear_strength=shear_strength,
        seed=seed,
        **kwargs,
    )
    return mod_params, damp_params


def fit_profile_two_set(
    profile: Profile,
    seed: int | None = None,
    verbose: bool = False,
    **kwargs,
) -> tuple[MultiLayerParams, MultiLayerParams]:
    """Fit separate modulus-reduction and damping parameters for a profile.

    For each nonlinear layer, fits two independent HH parameter sets:

    - ``mod_params`` — loading backbone matched to G/Gmax curves
    - ``damp_params`` — unloading backbone whose Masing-derived damping
      matches the target damping curves

    These two sets are used together by the Li & Assimaki (2010)
    non-Masing hysteresis rule to independently control stiffness and
    energy dissipation.

    Parameters
    ----------
    profile : Profile
        pystrata Profile with layers.
    seed : int, optional
        Random seed for reproducibility.
    verbose : bool
        Print progress information.
    **kwargs
        Additional arguments passed to fitting functions.

    Returns
    -------
    mod_params : MultiLayerParams
        Container with modulus-reduction parameters for each layer.
    damp_params : MultiLayerParams
        Container with damping parameters for each layer.
    """
    start_time = time.perf_counter()
    mod_multi = MultiLayerParams()
    damp_multi = MultiLayerParams()
    n_layers = len(profile) - 1

    logger.debug(
        "fit_profile_two_set: fitting HH two-set parameters for %d layers", n_layers
    )

    depth = 0.0
    for i, layer in enumerate(profile[:-1]):
        layer_depth = depth + layer.thickness / 2

        if verbose:
            print(
                f"Fitting layer {i + 1}/{n_layers}: depth={layer_depth:.2f}m (two-set)"
            )

        mod_p, damp_p = fit_layer_hh_two_set(
            layer,
            depth=layer_depth,
            seed=seed,
            **kwargs,
        )
        mod_multi.append(mod_p)
        damp_multi.append(damp_p)
        depth += layer.thickness

    elapsed = time.perf_counter() - start_time
    logger.info("fit_profile_two_set: fitted %d layers in %.3fs", n_layers, elapsed)

    return mod_multi, damp_multi


def plot_layer_fit(
    layer: Layer,
    params: MKZParams | HHParams,
    strains: np.ndarray | None = None,
    axes: np.ndarray | None = None,
    damp_params: HHParams | None = None,
) -> np.ndarray:
    """Plot target nonlinear curves vs constitutive model fit for one layer.

    Parameters
    ----------
    layer : Layer
        Layer with a nonlinear soil type providing target curves.
    params : MKZParams or HHParams
        Fitted constitutive model parameters for this layer.  In two-set
        mode this is the *modulus-reduction* parameter set.
    strains : np.ndarray, optional
        Strain array [decimal] for evaluation. Defaults to logspace(-6, -1.5).
    axes : np.ndarray of Axes, optional
        Two matplotlib Axes (G/Gmax, Damping). Created if not provided.
    damp_params : HHParams, optional
        Separate damping parameter set (two-set mode).  When provided, the
        damping panel shows the Masing-derived damping from this set instead
        of the primary *params*.

    Returns
    -------
    axes : np.ndarray
        Array of two Axes [ax_mod_reduc, ax_damping].
    """
    import matplotlib.pyplot as plt

    if strains is None:
        strains = np.logspace(-6, -1.5, 200)
    strains_pct = strains * 100

    if axes is None:
        _, axes = plt.subplots(1, 2, figsize=(10, 4))
    ax_mr, ax_d = axes

    st = layer.soil_type

    # Target curves from soil type
    if st.mod_reduc is not None and callable(st.mod_reduc):
        ax_mr.semilogx(strains_pct, st.mod_reduc(strains), "k-", lw=2, label="Target")
    if callable(getattr(st, "damping", None)):
        ax_d.semilogx(
            strains_pct, st.damping(strains) * 100, "k-", lw=2, label="Target"
        )

    # Fitted model predictions
    pred_mr = calc_mod_reduc(strains, params)
    label = "MKZ" if isinstance(params, MKZParams) else "HH"

    ax_mr.semilogx(strains_pct, pred_mr, "--", lw=1.5, label=f"{label} fit")

    # Damping: use damp_params if provided (two-set mode), otherwise params
    damp_p = damp_params if damp_params is not None else params
    pred_d = calc_damping(strains, damp_p)
    # Add back small-strain damping offset (D_min) that was subtracted during fitting
    if callable(getattr(st, "damping", None)):
        d_min = st.damping(strains[:1])[0]
        pred_d = pred_d + d_min
    damp_label = f"{label} damp fit" if damp_params is not None else f"{label} fit"
    ax_d.semilogx(strains_pct, pred_d * 100, "--", lw=1.5, label=damp_label)

    ax_mr.set(xlabel="Shear strain (%)", ylabel="G/Gmax", ylim=(0, 1.05))
    ax_d.set(xlabel="Shear strain (%)", ylabel="Damping ratio (%)")
    for ax in axes:
        ax.legend(fontsize=8)
        ax.grid(True, alpha=0.3)

    return axes


def plot_fit(
    profile: Profile,
    params: MultiLayerParams,
    indices: list[int] | None = None,
    step: int | None = None,
    strains: np.ndarray | None = None,
    axes: np.ndarray | None = None,
    damp_params: MultiLayerParams | None = None,
) -> np.ndarray:
    """Plot target nonlinear curves vs constitutive model fit for a profile.

    Parameters
    ----------
    profile : Profile
        Site profile with nonlinear soil types.
    params : MultiLayerParams
        Fitted constitutive model parameters (from :func:`fit_profile`).
        In two-set mode this is the modulus-reduction parameter set.
    indices : list of int, optional
        Layer indices to plot. If None, selects nonlinear layers filtered
        by *step*.
    step : int, optional
        Plot every *step*-th nonlinear layer. Useful for finely discretized
        profiles with many sub-layers sharing the same soil type. Ignored
        if *indices* is provided.
    strains : np.ndarray, optional
        Strain array [decimal] for evaluation. Defaults to logspace(-6, -1.5).
    axes : np.ndarray, optional
        Array of shape (n_layers, 2) of Axes. Created if not provided.
    damp_params : MultiLayerParams, optional
        Separate damping parameters (two-set mode).  When provided,
        damping panels show predictions from this set.

    Returns
    -------
    axes : np.ndarray
        Array of shape (n_rows, 2) of Axes.
    """
    import matplotlib.pyplot as plt

    # Determine which layers to plot
    if indices is not None:
        sel = indices
    else:
        # All nonlinear layer indices
        sel = [
            i for i, layer in enumerate(profile[:-1]) if layer.soil_type.is_nonlinear
        ]
        if step is not None and step > 1:
            sel = sel[::step]

    if not sel:
        raise ValueError("No nonlinear layers selected for plotting.")

    n_rows = len(sel)
    if axes is None:
        fig, axes = plt.subplots(n_rows, 2, figsize=(10, 4 * n_rows), squeeze=False)
    else:
        axes = np.atleast_2d(axes)

    depth = 0.0
    depths = []
    for layer in profile[:-1]:
        depths.append(depth + layer.thickness / 2)
        depth += layer.thickness

    for row, idx in enumerate(sel):
        layer = profile[idx]
        ax_pair = axes[row]
        dp = damp_params[idx] if damp_params is not None else None
        plot_layer_fit(
            layer, params[idx], strains=strains, axes=ax_pair, damp_params=dp
        )

        name = layer.soil_type.name or ""
        label = (
            f"Layer {idx}: {name}\n"
            f"Vs={layer.initial_shear_vel:.0f} m/s, z={depths[idx]:.1f} m"
        )
        ax_pair[0].set_title(f"{label} – G/Gmax", fontsize=8)
        ax_pair[1].set_title(f"{label} – Damping", fontsize=8)

    plt.tight_layout()
    return axes
