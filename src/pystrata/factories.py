# The MIT License (MIT)
#
# Copyright (c) 2016-2026 Albert Kottke
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
"""Factories for generating :class:`pystrata.site.Layer` stacks from simple
velocity-power-law models."""

from __future__ import annotations

from typing import Any, Callable

import numpy as np
from scipy.integrate import quad
from scipy.optimize import brentq

from .site import DampingCurve, Layer, ModulusReductionCurve, SoilType
from .units import GRAVITY, KPA_TO_ATM, convert_units

__all__ = ["LayerFactory", "darendeli_soil_type", "menq_soil_type"]


def _modified_hyperbolic_curves(
    name: str,
    strain_ref: float,
    curvature: float,
    masing_scaling: float,
    damping_min: float,
    strains: np.ndarray | None = None,
) -> tuple[ModulusReductionCurve, DampingCurve]:
    """Modified hyperbolic modulus-reduction/damping curves (Darendeli, 2001
    formulation), ported from the modulus-reduction/damping model previously
    provided by ``pystrata.site.ModifiedHyperbolicSoilType``.
    """
    if strains is None:
        strains = np.logspace(-6, -1.5, num=20)  # in decimal
    else:
        strains = np.asarray(strains)

    # Modified hyperbolic shear modulus reduction
    mod_reduc = 1 / (1 + (strains / strain_ref) ** curvature)

    # Masing damping based on shear-modulus reduction [%]
    strains_percent = strains * 100
    strain_ref_percent = strain_ref * 100
    damping_masing_a1 = (100.0 / np.pi) * (
        4
        * (
            strains_percent
            - strain_ref_percent
            * np.log((strains_percent + strain_ref_percent) / strain_ref_percent)
        )
        / (strains_percent**2 / (strains_percent + strain_ref_percent))
        - 2.0
    )
    # Correction between perfect hyperbolic strain model and modified model [%].
    c1 = -1.1143 * curvature**2 + 1.8618 * curvature + 0.2523
    c2 = 0.0805 * curvature**2 - 0.0710 * curvature - 0.0095
    c3 = -0.0005 * curvature**2 + 0.0002 * curvature + 0.0003
    damping_masing = (
        c1 * damping_masing_a1 + c2 * damping_masing_a1**2 + c3 * damping_masing_a1**3
    )

    # Compute the damping correction in percent
    d_correction = masing_scaling * damping_masing * mod_reduc**0.1

    # Prevent the damping from reducing as it can at large strains
    damping = np.maximum.accumulate(d_correction / 100.0)
    # Add the minimum damping component and convert to decimal values
    damping = damping + damping_min

    return (
        ModulusReductionCurve(name, strains, mod_reduc),
        DampingCurve(name, strains, damping),
    )


def darendeli_soil_type(
    unit_wt: float,
    stress_mean: float,
    name: str = "",
    plas_index: float = 0,
    ocr: float = 1,
    freq: float = 1,
    num_cycles: float = 10,
    strains: np.ndarray | None = None,
) -> SoilType:
    """Darendeli (2001) modified hyperbolic model for fine-grained soils.

    A stress-dependent replacement for the (now removed) built-in
    ``pystrata.site.DarendeliSoilType``, suitable for use as the `soil_type`
    passed to :meth:`LayerFactory.add`.

    Parameters
    ----------
    unit_wt: float
        unit weight of the material [kN/m³]
    stress_mean: float
        mean effective stress [kN/m²]
    name: str, optional
        used for identification. If empty, created from properties.
    plas_index: float, default=0
        plasticity index [percent]
    ocr: float, default=1
        over-consolidation ratio
    freq: float, default=1
        excitation frequency [Hz]
    num_cycles: float, default=10
        number of cycles of loading
    strains: `array_like`, optional
        shear strain levels [decimal]
    """
    damping_min = (
        (0.8005 + 0.0129 * plas_index * ocr**-0.1069)
        * (stress_mean * KPA_TO_ATM) ** -0.2889
        * (1 + 0.2919 * np.log(freq))
    ) / 100
    masing_scaling = 0.6329 - 0.00566 * np.log(num_cycles)
    strain_ref = (
        (0.0352 + 0.0010 * plas_index * ocr**0.3246) * (stress_mean * KPA_TO_ATM) ** 0.3483
    ) / 100
    curvature = 0.9190

    if not name:
        name = f"Darendeli (PI={plas_index:.0f}, OCR={ocr:.1f}, σₘ'={stress_mean:.1f} kN/m²)"

    mod_reduc, damping = _modified_hyperbolic_curves(
        name, strain_ref, curvature, masing_scaling, damping_min, strains
    )
    return SoilType(name=name, unit_wt=unit_wt, mod_reduc=mod_reduc, damping=damping)


def menq_soil_type(
    unit_wt: float,
    stress_mean: float,
    name: str = "",
    coef_unif: float = 10,
    diam_mean: float = 5,
    num_cycles: float = 10,
    strains: np.ndarray | None = None,
) -> SoilType:
    """Menq (2003) modified hyperbolic model for gravelly soils.

    A stress-dependent replacement for the (now removed) built-in
    ``pystrata.site.MenqSoilType``, suitable for use as the `soil_type`
    passed to :meth:`LayerFactory.add`.

    Parameters
    ----------
    unit_wt: float
        unit weight of the material [kN/m³]
    stress_mean: float
        mean effective stress [kN/m²]
    name: str, optional
        used for identification. If empty, created from properties.
    coef_unif: float, default=10
        uniformity coefficient (Cᵤ)
    diam_mean: float, default=5
        mean diameter (D₅₀) [mm]
    num_cycles: float, default=10
        number of cycles of loading
    strains: `array_like`, optional
        shear strain levels [decimal]
    """
    damping_min = (
        0.55 * coef_unif**0.1 * diam_mean**-0.3 * (stress_mean * KPA_TO_ATM) ** -0.08
    ) / 100
    masing_scaling = 0.6329 - 0.00566 * np.log(num_cycles)
    strain_ref = (
        0.12
        * coef_unif**-0.6
        * (stress_mean * KPA_TO_ATM) ** (0.5 * coef_unif**-0.15)
    ) / 100
    curvature = 0.86 + 0.1 * np.log10(stress_mean * KPA_TO_ATM)

    if not name:
        name = (
            f"Menq (Cᵤ={coef_unif:.1f}, D₅₀={diam_mean:.1f} mm, "
            f"σₘ'={stress_mean:.1f} kN/m²)"
        )

    mod_reduc, damping = _modified_hyperbolic_curves(
        name, strain_ref, curvature, masing_scaling, damping_min, strains
    )
    return SoilType(name=name, unit_wt=unit_wt, mod_reduc=mod_reduc, damping=damping)


class LayerFactory:
    """Builds a series of :class:`~pystrata.site.Layer` from a power-law
    velocity model.

    For each *span* added with :meth:`add`, the shear-wave velocity is
    modeled as ``coef * x ** power`` where ``x`` is either the depth or the
    mean effective stress. The span's total thickness is partitioned into
    ``count`` sub-layers such that each sub-layer has the same travel time
    (``thickness / shear_vel``). The mean effective stress at the midpoint
    of each sub-layer is computed and passed to `soil_type` (as the
    ``stress_mean`` keyword) so that its nonlinear curve can be initialized
    appropriately -- this requires `soil_type` to accept a ``stress_mean``
    keyword argument (e.g. :func:`darendeli_soil_type` or
    :func:`menq_soil_type`).

    The factory tracks depth and mean effective stress continuously across
    calls to :meth:`add`, so multiple spans -- each potentially with a
    different `unit_wt`, `coef`, `power`, or `soil_type` -- can be
    concatenated into one continuous stack. Because the output
    (:attr:`layers`) is just a plain ``list`` of `Layer`, the output of two
    independently created factories can be joined with simple list
    concatenation (``factory_1.layers + factory_2.layers``).

    Parameters
    ----------
    wt_depth: float, default=0
        depth to the water table [m], in the factory's own running-depth
        coordinate (i.e., 0 corresponds to the depth of the first layer
        added by this factory instance, not necessarily the ground
        surface).
    k0: float, default=0.5
        coefficient of lateral earth pressure at rest, used to compute the
        mean effective stress from the vertical effective stress. Matches
        the default used by :meth:`pystrata.site.Layer.stress_mean`.
    initial_depth: float, default=0
        depth [m] at the top of the first layer this factory will create.
        Non-zero when the factory continues an existing stack of layers.
    initial_stress_mean: float, default=0
        mean effective stress [kN/m²] already acting at `initial_depth`,
        e.g. from an unmodeled overburden. Non-zero when the factory
        continues an existing stack of layers.
    """

    @convert_units(wt_depth="meter", initial_depth="meter")
    def __init__(
        self,
        wt_depth: float = 0.0,
        k0: float = 0.5,
        initial_depth: float = 0.0,
        initial_stress_mean: float = 0.0,
    ):
        self.wt_depth = wt_depth
        self.k0 = k0

        self._depth = initial_depth
        self._stress_vert_eff = self._stress_mean_to_vert_eff(initial_stress_mean)

        self._layers: list[Layer] = []

    @property
    def layers(self) -> list[Layer]:
        """List of :class:`~pystrata.site.Layer` created so far."""
        return self._layers

    @property
    def depth(self) -> float:
        """Current (running) depth [m] -- the depth at which the *next*
        call to :meth:`add` will start."""
        return self._depth

    @property
    def stress_mean(self) -> float:
        """Current (running) mean effective stress [kN/m²] -- the value at
        the depth at which the *next* call to :meth:`add` will start."""
        return self._vert_eff_to_stress_mean(self._stress_vert_eff)

    def _stress_mean_to_vert_eff(self, stress_mean: float) -> float:
        return 3 * stress_mean / (1 + 2 * self.k0)

    def _vert_eff_to_stress_mean(self, stress_vert_eff: float) -> float:
        return (1 + 2 * self.k0) / 3 * stress_vert_eff

    def _pore_pressure(self, depth: float) -> float:
        """Hydrostatic pore pressure [kN/m²] at the given (global) depth."""
        return GRAVITY * max(depth - self.wt_depth, 0.0)

    def _stress_vert_eff_at(self, unit_wt: float, offset: float) -> float:
        """Vertical effective stress [kN/m²] at `offset` [m] below the
        start of the span currently being built."""
        depth = self._depth + offset
        stress_vert_total = (
            self._stress_vert_eff
            + self._pore_pressure(self._depth)
            + offset * unit_wt
        )
        return stress_vert_total - self._pore_pressure(depth)

    def _mean_eff_stress_at(self, unit_wt: float, offset: float) -> float:
        """Mean effective stress [kN/m²] at `offset` [m] below the start of
        the span currently being built."""
        return self._vert_eff_to_stress_mean(
            self._stress_vert_eff_at(unit_wt, offset)
        )

    def _param_at(self, param: str, unit_wt: float, offset: float) -> float:
        if param == "depth":
            return self._depth + offset
        elif param == "mean_eff_stress":
            return self._mean_eff_stress_at(unit_wt, offset)
        else:
            raise ValueError(
                f"param must be 'depth' or 'mean_eff_stress', not {param!r}"
            )

    @convert_units(thickness="meter", unit_wt="kilonewton / meter ** 3")
    def add(
        self,
        thickness: float,
        unit_wt: float,
        coef: float,
        power: float,
        param: str,
        count: int,
        soil_type: Callable[..., SoilType],
        soil_type_kwds: dict[str, Any] | None = None,
    ) -> list[Layer]:
        """Add a span of `count` layers spanning `thickness`, with shear-wave
        velocity following ``coef * x ** power``.

        Parameters
        ----------
        thickness: float
            total thickness of the span [m], to be partitioned into `count`
            sub-layers of equal travel time.
        unit_wt: float
            unit weight of the soil in this span [kN/m³], constant across
            the span.
        coef: float
            coefficient of the power-law velocity model.
        power: float
            exponent of the power-law velocity model.
        param: str
            the variable ``x`` driving the velocity model: either
            ``"depth"`` or ``"mean_eff_stress"``.
        count: int
            number of sub-layers to partition `thickness` into.
        soil_type: callable
            callable used to construct each sub-layer's soil type. Must
            accept `unit_wt` and `stress_mean` keyword arguments and return
            a :class:`~pystrata.site.SoilType` (e.g. :func:`darendeli_soil_type`,
            :func:`menq_soil_type`).
        soil_type_kwds: dict, optional
            additional keyword arguments passed to `soil_type` (e.g.
            `plas_index`, `ocr`).

        Returns
        -------
        list of :class:`~pystrata.site.Layer`
            the newly created sub-layers (also appended to :attr:`layers`).
        """
        if count < 1:
            raise ValueError("count must be at least 1")

        soil_type_kwds = dict(soil_type_kwds or {})

        def velocity(offset: float) -> float:
            x = self._param_at(param, unit_wt, offset)
            return coef * x**power

        def inv_velocity(offset: float) -> float:
            return 1.0 / velocity(offset)

        total_time, _ = quad(inv_velocity, 0.0, thickness)
        target_time = total_time / count

        def cum_time(offset: float) -> float:
            value, _ = quad(inv_velocity, 0.0, offset)
            return value

        # Find the `count - 1` interior boundaries such that each of the
        # `count` sub-layers has the same travel time.
        boundaries = [0.0]
        for i in range(1, count):
            target = i * target_time

            def residual(offset: float, target=target) -> float:
                return cum_time(offset) - target

            boundaries.append(brentq(residual, boundaries[-1], thickness))
        boundaries.append(thickness)

        new_layers = []
        for start, end in zip(boundaries[:-1], boundaries[1:]):
            mid = 0.5 * (start + end)
            sub_thickness = end - start

            shear_vel = velocity(mid)
            stress_mean_mid = self._mean_eff_stress_at(unit_wt, mid)

            st = soil_type(unit_wt=unit_wt, stress_mean=stress_mean_mid, **soil_type_kwds)
            layer = Layer(st, sub_thickness, shear_vel)
            new_layers.append(layer)

        # Advance the running depth/stress state to the base of the span.
        self._stress_vert_eff = self._stress_vert_eff_at(unit_wt, thickness)
        self._depth += thickness

        self._layers.extend(new_layers)

        return new_layers
