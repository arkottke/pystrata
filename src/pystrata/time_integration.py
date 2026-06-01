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
"""Time-domain wave propagation using central difference integration.

This module implements true nonlinear time-domain site response analysis
using explicit central difference time integration of the 1D wave equation.

The wave equation being solved is:
    ρ ∂²u/∂t² = ∂τ/∂z

where:
    u = horizontal displacement
    τ = shear stress (from constitutive model)
    ρ = mass density
    z = depth
"""

from __future__ import annotations

import logging
import time as time_module
from dataclasses import dataclass
from typing import TYPE_CHECKING, Literal

import numpy as np
import numpy.typing as npt

from .constitutive import HHParams, MKZParams, MultiLayerParams

logger = logging.getLogger(__name__)

if TYPE_CHECKING:
    pass

try:
    import numba

    HAS_NUMBA = True
except ImportError:
    HAS_NUMBA = False


def _extract_params_to_arrays(
    params_list: list[MKZParams | HHParams],
) -> tuple[
    npt.NDArray[np.int32],  # model_type: 0=MKZ, 1=HH
    npt.NDArray[np.floating],  # gamma_ref
    npt.NDArray[np.floating],  # beta
    npt.NDArray[np.floating],  # s
    npt.NDArray[np.floating],  # shear_mod
    npt.NDArray[np.floating],  # gamma_t (HH only)
    npt.NDArray[np.floating],  # a (HH only)
    npt.NDArray[np.floating],  # mu (HH only)
    npt.NDArray[np.floating],  # shear_strength (HH only)
    npt.NDArray[np.floating],  # d (HH only)
    npt.NDArray[np.floating],  # trans_c1 (HH only)
    npt.NDArray[np.floating],  # trans_c2 (HH only)
]:
    """Extract constitutive parameters to arrays for Numba."""
    n = len(params_list)
    model_type = np.zeros(n, dtype=np.int32)
    gamma_ref = np.zeros(n, dtype=np.float64)
    beta = np.zeros(n, dtype=np.float64)
    s = np.zeros(n, dtype=np.float64)
    shear_mod = np.zeros(n, dtype=np.float64)
    # HH-specific parameters
    gamma_t = np.zeros(n, dtype=np.float64)
    a = np.zeros(n, dtype=np.float64)
    mu = np.zeros(n, dtype=np.float64)
    shear_strength = np.zeros(n, dtype=np.float64)
    d = np.zeros(n, dtype=np.float64)
    trans_c1 = np.zeros(n, dtype=np.float64)
    trans_c2 = np.zeros(n, dtype=np.float64)

    for i, p in enumerate(params_list):
        gamma_ref[i] = p.gamma_ref
        beta[i] = p.beta
        s[i] = p.s
        shear_mod[i] = p.shear_mod
        if isinstance(p, HHParams):
            model_type[i] = 1
            gamma_t[i] = p.gamma_t
            a[i] = p.a
            mu[i] = p.mu
            shear_strength[i] = p.shear_strength
            d[i] = p.d
            trans_c1[i] = p.trans_c1
            trans_c2[i] = p.trans_c2

    return (
        model_type,
        gamma_ref,
        beta,
        s,
        shear_mod,
        gamma_t,
        a,
        mu,
        shear_strength,
        d,
        trans_c1,
        trans_c2,
    )


@dataclass
class TimeDomainResults:
    """Results from time-domain wave propagation.

    Attributes
    ----------
    times : np.ndarray
        Time array [s].
    depths : np.ndarray
        Depth to each node [m].
    accel : np.ndarray
        Acceleration time history at each depth [m/s²].
        Shape: (n_times, n_depths).
    veloc : np.ndarray
        Velocity time history at each depth [m/s].
        Shape: (n_times, n_depths).
    displ : np.ndarray
        Displacement time history at each depth [m].
        Shape: (n_times, n_depths).
    strain : np.ndarray
        Strain time history at each layer midpoint.
        Shape: (n_times, n_layers).
    stress : np.ndarray
        Stress time history at each layer midpoint [Pa].
        Shape: (n_times, n_layers).
    """

    times: npt.NDArray[np.floating]
    depths: npt.NDArray[np.floating]
    accel: npt.NDArray[np.floating]
    veloc: npt.NDArray[np.floating]
    displ: npt.NDArray[np.floating]
    strain: npt.NDArray[np.floating]
    stress: npt.NDArray[np.floating]

    @property
    def n_times(self) -> int:
        return len(self.times)

    @property
    def n_depths(self) -> int:
        return len(self.depths)

    @property
    def n_layers(self) -> int:
        return self.strain.shape[1]

    def max_accel(self) -> npt.NDArray[np.floating]:
        """Maximum absolute acceleration at each depth."""
        return np.max(np.abs(self.accel), axis=0)

    def max_veloc(self) -> npt.NDArray[np.floating]:
        """Maximum absolute velocity at each depth."""
        return np.max(np.abs(self.veloc), axis=0)

    def max_strain(self) -> npt.NDArray[np.floating]:
        """Maximum absolute strain in each layer."""
        return np.max(np.abs(self.strain), axis=0)

    def max_stress(self) -> npt.NDArray[np.floating]:
        """Maximum absolute stress in each layer."""
        return np.max(np.abs(self.stress), axis=0)


def calc_cfl_subcycles(
    time_step: float,
    thicknesses: npt.NDArray[np.floating],
    shear_vels: npt.NDArray[np.floating],
    safety_factor: float = 0.9,
) -> int:
    """Calculate number of subcycles needed for CFL stability.

    The CFL condition requires: dt < dz / Vs_max

    Parameters
    ----------
    time_step : float
        Input motion time step [s].
    thicknesses : np.ndarray
        Layer thicknesses [m].
    shear_vels : np.ndarray
        Layer shear velocities [m/s].
    safety_factor : float
        Safety factor for CFL (< 1.0 for stability margin).

    Returns
    -------
    subcycles : int
        Number of subcycles per input time step.
    """
    # Find minimum allowed dt from CFL condition
    dt_cfl = safety_factor * np.min(thicknesses / shear_vels)

    # Number of subcycles needed
    subcycles = max(1, int(np.ceil(time_step / dt_cfl)))

    return subcycles


# -----------------------------------------------------------------------------
# Pure Python implementation
# -----------------------------------------------------------------------------


def _integrate_python(
    n_times: int,
    n_nodes: int,
    n_layers: int,
    dt: float,
    dz: npt.NDArray[np.floating],
    rho: npt.NDArray[np.floating],
    shear_mod: npt.NDArray[np.floating],
    damping: npt.NDArray[np.floating],
    input_accel: npt.NDArray[np.floating],
    boundary: str,
    rho_base: float,
    vs_base: float,
) -> tuple[
    npt.NDArray[np.floating],
    npt.NDArray[np.floating],
    npt.NDArray[np.floating],
    npt.NDArray[np.floating],
]:
    """Perform time integration using central difference method (pure Python).

    Uses linear elastic stress-strain relationship. For nonlinear analysis,
    use the wrapper that updates stress externally.

    Parameters
    ----------
    n_times : int
        Number of output time steps.
    n_nodes : int
        Number of depth nodes.
    n_layers : int
        Number of layers.
    dt : float
        Integration time step [s].
    dz : np.ndarray
        Layer thicknesses [m].
    rho : np.ndarray
        Layer densities [kg/m³].
    shear_mod : np.ndarray
        Layer shear moduli [Pa].
    damping : np.ndarray
        Layer damping ratios.
    input_accel : np.ndarray
        Input acceleration at base [m/s²].
    boundary : str
        'elastic' or 'rigid'.
    rho_base : float
        Base layer density [kg/m³].
    vs_base : float
        Base layer shear velocity [m/s].

    Returns
    -------
    displ : np.ndarray
        Displacement at each node, shape (n_times, n_nodes).
    veloc : np.ndarray
        Velocity at each node, shape (n_times, n_nodes).
    accel : np.ndarray
        Acceleration at each node, shape (n_times, n_nodes).
    stress : np.ndarray
        Stress at each layer midpoint, shape (n_times, n_layers).
    """
    # Initialize arrays
    displ = np.zeros((n_times, n_nodes))
    veloc = np.zeros((n_times, n_nodes))
    accel = np.zeros((n_times, n_nodes))
    stress = np.zeros((n_times, n_layers))

    # Node 0 is surface, node n_nodes-1 is base (bottom of last layer)
    # Layer i has top at node i and bottom at node i+1

    # Calculate mass per node (for lumped mass approach)
    mass = np.zeros(n_nodes)
    for i in range(n_layers):
        # Distribute layer mass to top and bottom nodes
        layer_mass = rho[i] * dz[i]
        mass[i] += layer_mass / 2
        mass[i + 1] += layer_mass / 2

    # Stiffness per layer: k = G * A / h = G / h (per unit area)
    stiff = shear_mod / dz

    # Damping coefficient per layer (viscous): c = 2 * xi * sqrt(k * m)
    # Using Rayleigh damping proportional to stiffness
    damp_coeff = 2 * damping * np.sqrt(rho * shear_mod) / dz

    # Base motion: integrate acceleration to get displacement
    base_veloc = np.cumsum(input_accel) * dt
    base_displ = np.cumsum(base_veloc) * dt

    # Previous and current displacement
    u_prev = np.zeros(n_nodes)
    u_curr = np.zeros(n_nodes)

    for n in range(n_times):
        u_next = np.zeros(n_nodes)

        # Apply base displacement for rigid boundary
        if boundary == "rigid":
            u_next[-1] = base_displ[n]

        # Update interior nodes (1 to n_nodes-2)
        for i in range(1, n_nodes - 1):
            # Force from layer above (i-1) and below (i)
            # Layer i-1: connects node i-1 (top) to node i (bottom)
            # Layer i: connects node i (top) to node i+1 (bottom)

            # Velocity for damping
            v_curr = (u_curr[i] - u_prev[i]) / dt

            # Force from spring (stress * area, area = 1)
            # Force pointing upward is positive (toward surface)
            f_above = -stiff[i - 1] * (u_curr[i] - u_curr[i - 1])  # From layer above
            f_below = stiff[i] * (u_curr[i + 1] - u_curr[i])  # From layer below

            # Damping force
            f_damp = -(damp_coeff[i - 1] + damp_coeff[i]) * v_curr * 0.5

            # Total force
            f_total = f_above + f_below + f_damp

            # Central difference: u_next = 2*u_curr - u_prev + dt^2 * f / m
            u_next[i] = 2 * u_curr[i] - u_prev[i] + dt**2 * f_total / mass[i]

        # Surface boundary (free surface: zero stress)
        # Only connected to layer 0 below
        v_surf = (u_curr[0] - u_prev[0]) / dt
        f_below = stiff[0] * (u_curr[1] - u_curr[0])
        f_damp = -damp_coeff[0] * v_surf * 0.5
        f_total = f_below + f_damp
        u_next[0] = 2 * u_curr[0] - u_prev[0] + dt**2 * f_total / mass[0]

        # Base boundary
        if boundary == "elastic":
            # Semi-implicit absorbing + incoming-wave boundary.
            # Solving the central-difference equation with the radiation-damping
            # term treated implicitly removes the stability constraint
            # Z*dt / (2*m) <= 1 that plagues fully-explicit formulations.
            #
            #   m*(u_next - 2u + u_prev)/dt² = f_spring
            #                                 + 2*Z*v_in        (incoming wave)
            #                                 - Z*(u_next-u_prev)/(2*dt)  (radiation, implicit)
            #
            # Solving for u_next:
            #   u_next = (2u - (1-α)*u_prev + dt²/m*(f_spring + 2Z*v_in)) / (1+α)
            #   where α = Z*dt / (2*m)  [always >= 0, so always stable]
            impedance = rho_base * vs_base
            v_in = base_veloc[n]
            f_above = -stiff[-1] * (u_curr[-1] - u_curr[-2])
            f_incoming = 2.0 * impedance * v_in
            alpha = impedance * dt / (2.0 * mass[-1])
            u_next[-1] = (
                2 * u_curr[-1]
                - (1 - alpha) * u_prev[-1]
                + dt**2 / mass[-1] * (f_above + f_incoming)
            ) / (1 + alpha)

        # Store results
        displ[n, :] = u_next
        veloc[n, :] = (u_next - u_prev) / (2 * dt) if n > 0 else (u_next - u_curr) / dt
        accel[n, :] = (u_next - 2 * u_curr + u_prev) / dt**2

        # Compute stress at each layer (average of top/bottom strain)
        for i in range(n_layers):
            strain_i = (u_next[i + 1] - u_next[i]) / dz[i]
            stress[n, i] = shear_mod[i] * strain_i

        # Advance time step
        u_prev = u_curr.copy()
        u_curr = u_next.copy()

    return displ, veloc, accel, stress


def _integrate_nonlinear_python(
    n_times: int,
    n_nodes: int,
    n_layers: int,
    dt: float,
    dz: npt.NDArray[np.floating],
    rho: npt.NDArray[np.floating],
    params_list: list[MKZParams | HHParams],
    damping_min: npt.NDArray[np.floating],
    input_accel: npt.NDArray[np.floating],
    boundary: str,
    rho_base: float,
    vs_base: float,
) -> tuple[
    npt.NDArray[np.floating],
    npt.NDArray[np.floating],
    npt.NDArray[np.floating],
    npt.NDArray[np.floating],
    npt.NDArray[np.floating],
]:
    """Perform nonlinear time integration (pure Python).

    Parameters
    ----------
    n_times : int
        Number of output time steps.
    n_nodes : int
        Number of depth nodes.
    n_layers : int
        Number of layers.
    dt : float
        Integration time step [s].
    dz : np.ndarray
        Layer thicknesses [m].
    rho : np.ndarray
        Layer densities [kg/m³].
    params_list : list
        Constitutive model parameters for each layer.
    damping_min : np.ndarray
        Minimum (small-strain) damping ratio for each layer.
    input_accel : np.ndarray
        Input acceleration at base [m/s²].
    boundary : str
        'elastic' or 'rigid'.
    rho_base : float
        Base layer density [kg/m³].
    vs_base : float
        Base layer shear velocity [m/s].

    Returns
    -------
    displ, veloc, accel : np.ndarray
        Displacement, velocity, acceleration at each node.
    strain, stress : np.ndarray
        Strain and stress at each layer midpoint.
    """
    from .constitutive import calc_stress

    # Initialize arrays
    displ = np.zeros((n_times, n_nodes))
    veloc = np.zeros((n_times, n_nodes))
    accel = np.zeros((n_times, n_nodes))
    strain = np.zeros((n_times, n_layers))
    stress = np.zeros((n_times, n_layers))

    # Previous time step displacement
    u_prev = np.zeros(n_nodes)
    u_curr = np.zeros(n_nodes)
    u_next = np.zeros(n_nodes)

    # Extract shear modulus for damping calculation
    shear_mod = np.array([p.shear_mod for p in params_list])

    # Rayleigh damping coefficient
    visc_coeff = 2 * damping_min * np.sqrt(rho * shear_mod)

    # Pre-integrate input acceleration for boundary conditions
    impedance_base = rho_base * vs_base
    base_veloc = np.cumsum(input_accel) * dt  # velocity  (rigid + elastic)
    base_displ = np.cumsum(base_veloc) * dt  # displacement (rigid boundary)
    mass_base = rho[-1] * dz[-1] / 2  # lumped mass at base node (per unit area)

    for n in range(n_times):
        # Compute strain at each layer midpoint
        strain_mid = np.zeros(n_layers)
        for i in range(n_layers):
            strain_mid[i] = (u_curr[i + 1] - u_curr[i]) / dz[i]

        # Compute stress from constitutive model
        stress_mid = np.zeros(n_layers)
        for i in range(n_layers):
            # Get stress from single strain value
            strain_arr = np.array([strain_mid[i]])
            stress_mid[i] = calc_stress(strain_arr, params_list[i])[0]

            # Add viscous damping contribution
            strain_rate = (strain_mid[i] - strain[n - 1, i] if n > 0 else 0) / dt
            stress_mid[i] += visc_coeff[i] * strain_rate * dz[i]

        # Update interior nodes
        for i in range(1, n_nodes - 1):
            # Determine which layers contribute at this node
            if i < n_layers:
                rho_node = 0.5 * (rho[i - 1] + rho[i])
                stress_above = stress_mid[i - 1]
                stress_below = stress_mid[i]
                dz_avg = 0.5 * (dz[i - 1] + dz[i])
            else:
                rho_node = rho[n_layers - 1]
                stress_above = stress_mid[n_layers - 1]
                stress_below = stress_mid[n_layers - 1]
                dz_avg = dz[n_layers - 1]

            # Force balance
            force = (stress_below - stress_above) / dz_avg

            # Central difference update
            u_next[i] = 2 * u_curr[i] - u_prev[i] + dt**2 / rho_node * force

        # Surface boundary (free surface)
        u_next[0] = (
            2 * u_curr[0] - u_prev[0] + dt**2 / rho[0] * stress_mid[0] / (dz[0] / 2)
        )

        # Base boundary
        if boundary == "rigid":
            u_next[-1] = base_displ[n]  # prescribed input displacement
        else:
            # Semi-implicit elastic (transmitting) boundary — see linear integrator
            # for derivation.  α = Z*dt/(2m); unconditionally stable for any α.
            v_in = base_veloc[n]
            f_above = -stress_mid[-1]  # stress from last soil layer (per unit area)
            f_incoming = 2.0 * impedance_base * v_in
            alpha = impedance_base * dt / (2.0 * mass_base)
            u_next[-1] = (
                2 * u_curr[-1]
                - (1 - alpha) * u_prev[-1]
                + dt**2 / mass_base * (f_above + f_incoming)
            ) / (1 + alpha)

        # Store results
        displ[n, :] = u_next
        veloc[n, :] = (u_next - u_prev) / (2 * dt)
        accel[n, :] = (u_next - 2 * u_curr + u_prev) / dt**2
        strain[n, :] = strain_mid
        stress[n, :] = stress_mid

        # Advance time step
        u_prev[:] = u_curr
        u_curr[:] = u_next

    return displ, veloc, accel, strain, stress


# -----------------------------------------------------------------------------
# Numba-accelerated implementation
# -----------------------------------------------------------------------------


if HAS_NUMBA:

    @numba.njit(cache=True)
    def _integrate_linear_numba(
        n_times: int,
        n_nodes: int,
        n_layers: int,
        dt: float,
        dz: npt.NDArray[np.floating],
        rho: npt.NDArray[np.floating],
        shear_mod: npt.NDArray[np.floating],
        damping: npt.NDArray[np.floating],
        input_accel: npt.NDArray[np.floating],
        boundary_code: int,  # 0 = rigid, 1 = elastic
        rho_base: float,
        vs_base: float,
    ) -> tuple[
        npt.NDArray[np.floating],
        npt.NDArray[np.floating],
        npt.NDArray[np.floating],
        npt.NDArray[np.floating],
    ]:
        """Numba-accelerated linear elastic integration.

        Mirrors _integrate_python exactly: lumped mass, spring-dashpot per
        layer, semi-implicit elastic base boundary.
        """
        # Initialize output arrays
        displ = np.zeros((n_times, n_nodes))
        veloc = np.zeros((n_times, n_nodes))
        accel_out = np.zeros((n_times, n_nodes))
        stress_out = np.zeros((n_times, n_layers))

        u_prev = np.zeros(n_nodes)
        u_curr = np.zeros(n_nodes)
        u_next = np.zeros(n_nodes)

        # Lumped mass per node
        mass = np.zeros(n_nodes)
        for i in range(n_layers):
            layer_mass = rho[i] * dz[i]
            mass[i] += layer_mass / 2
            mass[i + 1] += layer_mass / 2

        # Stiffness and damping coefficient per layer
        stiff = np.empty(n_layers)
        damp_coeff = np.empty(n_layers)
        for i in range(n_layers):
            stiff[i] = shear_mod[i] / dz[i]
            damp_coeff[i] = 2 * damping[i] * np.sqrt(rho[i] * shear_mod[i]) / dz[i]

        # Pre-integrate input acceleration for boundary conditions
        base_veloc = np.empty(n_times)
        base_displ = np.empty(n_times)
        base_veloc[0] = input_accel[0] * dt
        base_displ[0] = base_veloc[0] * dt
        for n in range(1, n_times):
            base_veloc[n] = base_veloc[n - 1] + input_accel[n] * dt
            base_displ[n] = base_displ[n - 1] + base_veloc[n] * dt

        for n in range(n_times):
            # --- Interior nodes (1 to n_nodes - 2) ---
            for i in range(1, n_nodes - 1):
                v_curr = (u_curr[i] - u_prev[i]) / dt

                f_above = -stiff[i - 1] * (u_curr[i] - u_curr[i - 1])
                f_below = stiff[i] * (u_curr[i + 1] - u_curr[i])
                f_damp = -(damp_coeff[i - 1] + damp_coeff[i]) * v_curr * 0.5
                f_total = f_above + f_below + f_damp

                u_next[i] = 2 * u_curr[i] - u_prev[i] + dt * dt * f_total / mass[i]

            # --- Surface (free surface: zero stress above) ---
            v_surf = (u_curr[0] - u_prev[0]) / dt
            f_below_surf = stiff[0] * (u_curr[1] - u_curr[0])
            f_damp_surf = -damp_coeff[0] * v_surf * 0.5
            f_total_surf = f_below_surf + f_damp_surf
            u_next[0] = 2 * u_curr[0] - u_prev[0] + dt * dt * f_total_surf / mass[0]

            # --- Base boundary ---
            if boundary_code == 0:  # rigid
                u_next[n_nodes - 1] = base_displ[n]
            else:  # elastic — semi-implicit absorbing boundary
                impedance = rho_base * vs_base
                v_in = base_veloc[n]
                f_above_base = -stiff[n_layers - 1] * (
                    u_curr[n_nodes - 1] - u_curr[n_nodes - 2]
                )
                f_incoming = 2.0 * impedance * v_in
                alpha = impedance * dt / (2.0 * mass[n_nodes - 1])
                u_next[n_nodes - 1] = (
                    2 * u_curr[n_nodes - 1]
                    - (1 - alpha) * u_prev[n_nodes - 1]
                    + dt * dt / mass[n_nodes - 1] * (f_above_base + f_incoming)
                ) / (1 + alpha)

            # --- Store results ---
            for j in range(n_nodes):
                displ[n, j] = u_next[j]
                if n > 0:
                    veloc[n, j] = (u_next[j] - u_prev[j]) / (2 * dt)
                else:
                    veloc[n, j] = (u_next[j] - u_curr[j]) / dt
                accel_out[n, j] = (u_next[j] - 2 * u_curr[j] + u_prev[j]) / (dt * dt)

            for j in range(n_layers):
                strain_j = (u_next[j + 1] - u_next[j]) / dz[j]
                stress_out[n, j] = shear_mod[j] * strain_j

            # --- Advance ---
            for j in range(n_nodes):
                u_prev[j] = u_curr[j]
                u_curr[j] = u_next[j]

        return displ, veloc, accel_out, stress_out

    @numba.njit(cache=True)
    def _calc_stress_mkz_scalar(
        strain: float,
        gamma_ref: float,
        beta: float,
        s: float,
        shear_mod: float,
    ) -> float:
        """Compute MKZ stress for a single strain value."""
        return shear_mod * strain / (1 + beta * (abs(strain) / gamma_ref) ** s)

    @numba.njit(cache=True)
    def _calc_stress_hh_scalar(
        strain: float,
        gamma_ref: float,
        beta: float,
        s: float,
        shear_mod: float,
        gamma_t: float,
        a: float,
        mu: float,
        shear_strength: float,
        d: float,
        trans_c1: float,
        trans_c2: float,
    ) -> float:
        """Compute HH stress for a single strain value."""
        g = abs(strain)

        # MKZ component
        tau_mkz = shear_mod * strain / (1 + beta * (g / gamma_ref) ** s)

        # FKZ component
        gamma_d = g**d
        sign = 1.0 if strain >= 0 else -1.0
        tau_fkz = (
            mu
            * shear_mod
            * sign
            * gamma_d
            / (1 + shear_mod / shear_strength * mu * gamma_d)
        )

        # Transition function
        if g <= 0:
            w = 1.0
        else:
            intermediate = np.log10(g / gamma_t) - trans_c1 * a ** (-trans_c2)
            exponent = -a * intermediate
            if exponent > 305:
                w = 1.0
            elif exponent < -305:
                w = 0.0
            else:
                w = 1 - 1.0 / (1 + 10**exponent)

        return w * tau_mkz + (1 - w) * tau_fkz

    @numba.njit(cache=True)
    def _integrate_nonlinear_numba(
        n_times: int,
        n_nodes: int,
        n_layers: int,
        dt: float,
        dz: npt.NDArray[np.floating],
        rho: npt.NDArray[np.floating],
        model_type: npt.NDArray[np.int32],
        gamma_ref: npt.NDArray[np.floating],
        beta: npt.NDArray[np.floating],
        s: npt.NDArray[np.floating],
        shear_mod: npt.NDArray[np.floating],
        gamma_t: npt.NDArray[np.floating],
        a: npt.NDArray[np.floating],
        mu: npt.NDArray[np.floating],
        shear_strength: npt.NDArray[np.floating],
        d: npt.NDArray[np.floating],
        trans_c1: npt.NDArray[np.floating],
        trans_c2: npt.NDArray[np.floating],
        damping_min: npt.NDArray[np.floating],
        input_accel: npt.NDArray[np.floating],
        boundary_code: int,
        rho_base: float,
        vs_base: float,
    ) -> tuple[
        npt.NDArray[np.floating],
        npt.NDArray[np.floating],
        npt.NDArray[np.floating],
        npt.NDArray[np.floating],
        npt.NDArray[np.floating],
    ]:
        """Numba-accelerated nonlinear time integration."""
        # Initialize arrays
        displ = np.zeros((n_times, n_nodes))
        veloc = np.zeros((n_times, n_nodes))
        accel = np.zeros((n_times, n_nodes))
        strain = np.zeros((n_times, n_layers))
        stress = np.zeros((n_times, n_layers))

        u_prev = np.zeros(n_nodes)
        u_curr = np.zeros(n_nodes)
        u_next = np.zeros(n_nodes)

        # Rayleigh damping coefficient
        visc_coeff = np.empty(n_layers)
        for i in range(n_layers):
            visc_coeff[i] = 2 * damping_min[i] * np.sqrt(rho[i] * shear_mod[i])

        # Pre-integrate input acceleration for boundary conditions
        impedance_base = rho_base * vs_base
        base_veloc = np.empty(n_times)
        base_displ = np.empty(n_times)
        base_veloc[0] = input_accel[0] * dt
        base_displ[0] = base_veloc[0] * dt
        for n in range(1, n_times):
            base_veloc[n] = base_veloc[n - 1] + input_accel[n] * dt
            base_displ[n] = base_displ[n - 1] + base_veloc[n] * dt

        mass_base = rho[n_layers - 1] * dz[n_layers - 1] / 2

        prev_strain_mid = np.zeros(n_layers)

        for n in range(n_times):
            # Compute strain at each layer midpoint
            strain_mid = np.empty(n_layers)
            for i in range(n_layers):
                strain_mid[i] = (u_curr[i + 1] - u_curr[i]) / dz[i]

            # Compute stress from constitutive model
            stress_mid = np.empty(n_layers)
            for i in range(n_layers):
                if model_type[i] == 0:  # MKZ
                    stress_mid[i] = _calc_stress_mkz_scalar(
                        strain_mid[i], gamma_ref[i], beta[i], s[i], shear_mod[i]
                    )
                else:  # HH
                    stress_mid[i] = _calc_stress_hh_scalar(
                        strain_mid[i],
                        gamma_ref[i],
                        beta[i],
                        s[i],
                        shear_mod[i],
                        gamma_t[i],
                        a[i],
                        mu[i],
                        shear_strength[i],
                        d[i],
                        trans_c1[i],
                        trans_c2[i],
                    )

                # Add viscous damping contribution
                if n > 0:
                    strain_rate = (strain_mid[i] - prev_strain_mid[i]) / dt
                else:
                    strain_rate = 0.0
                stress_mid[i] += visc_coeff[i] * strain_rate * dz[i]

            # Update interior nodes
            for i in range(1, n_nodes - 1):
                if i < n_layers:
                    rho_node = 0.5 * (rho[i - 1] + rho[i])
                    stress_above = stress_mid[i - 1]
                    stress_below = stress_mid[i]
                    dz_avg = 0.5 * (dz[i - 1] + dz[i])
                else:
                    rho_node = rho[n_layers - 1]
                    stress_above = stress_mid[n_layers - 1]
                    stress_below = stress_mid[n_layers - 1]
                    dz_avg = dz[n_layers - 1]

                force = (stress_below - stress_above) / dz_avg
                u_next[i] = 2 * u_curr[i] - u_prev[i] + dt * dt / rho_node * force

            # Surface boundary (free surface)
            u_next[0] = (
                2 * u_curr[0]
                - u_prev[0]
                + dt * dt / rho[0] * stress_mid[0] / (dz[0] / 2)
            )

            # Base boundary
            if boundary_code == 0:  # rigid
                u_next[n_nodes - 1] = base_displ[n]
            else:  # elastic
                v_in = base_veloc[n]
                f_above = -stress_mid[n_layers - 1]
                f_incoming = 2.0 * impedance_base * v_in
                alpha = impedance_base * dt / (2.0 * mass_base)
                u_next[n_nodes - 1] = (
                    2 * u_curr[n_nodes - 1]
                    - (1 - alpha) * u_prev[n_nodes - 1]
                    + dt * dt / mass_base * (f_above + f_incoming)
                ) / (1 + alpha)

            # Store results
            for j in range(n_nodes):
                displ[n, j] = u_next[j]
                veloc[n, j] = (u_next[j] - u_prev[j]) / (2 * dt)
                accel[n, j] = (u_next[j] - 2 * u_curr[j] + u_prev[j]) / (dt * dt)
            for j in range(n_layers):
                strain[n, j] = strain_mid[j]
                stress[n, j] = stress_mid[j]

            # Advance time step
            for j in range(n_nodes):
                u_prev[j] = u_curr[j]
                u_curr[j] = u_next[j]
            for j in range(n_layers):
                prev_strain_mid[j] = strain_mid[j]

        return displ, veloc, accel, strain, stress

    # For now, always use Python version until Numba version is updated
    # _integrate_linear_dispatch = _integrate_linear_numba


def _integrate_linear_dispatch(
    n_times,
    n_nodes,
    n_layers,
    dt,
    dz,
    rho,
    shear_mod,
    damping,
    input_accel,
    boundary_code,
    rho_base,
    vs_base,
):
    if HAS_NUMBA:
        return _integrate_linear_numba(
            n_times,
            n_nodes,
            n_layers,
            dt,
            dz,
            rho,
            shear_mod,
            damping,
            input_accel,
            boundary_code,
            rho_base,
            vs_base,
        )
    boundary = "rigid" if boundary_code == 0 else "elastic"
    return _integrate_python(
        n_times,
        n_nodes,
        n_layers,
        dt,
        dz,
        rho,
        shear_mod,
        damping,
        input_accel,
        boundary,
        rho_base,
        vs_base,
    )


def _integrate_nonlinear_dispatch(
    n_times: int,
    n_nodes: int,
    n_layers: int,
    dt: float,
    dz: npt.NDArray[np.floating],
    rho: npt.NDArray[np.floating],
    params_list: list[MKZParams | HHParams],
    damping_min: npt.NDArray[np.floating],
    input_accel: npt.NDArray[np.floating],
    boundary: str,
    rho_base: float,
    vs_base: float,
) -> tuple[
    npt.NDArray[np.floating],
    npt.NDArray[np.floating],
    npt.NDArray[np.floating],
    npt.NDArray[np.floating],
    npt.NDArray[np.floating],
]:
    """Dispatch to Numba or Python nonlinear integration."""
    if HAS_NUMBA:
        # Extract parameters to arrays for Numba
        (
            model_type,
            gamma_ref,
            beta,
            s,
            shear_mod,
            gamma_t,
            a,
            mu,
            shear_strength,
            d,
            trans_c1,
            trans_c2,
        ) = _extract_params_to_arrays(params_list)

        boundary_code = 0 if boundary == "rigid" else 1
        return _integrate_nonlinear_numba(
            n_times,
            n_nodes,
            n_layers,
            dt,
            dz,
            rho,
            model_type,
            gamma_ref,
            beta,
            s,
            shear_mod,
            gamma_t,
            a,
            mu,
            shear_strength,
            d,
            trans_c1,
            trans_c2,
            damping_min,
            input_accel,
            boundary_code,
            rho_base,
            vs_base,
        )
    else:
        return _integrate_nonlinear_python(
            n_times,
            n_nodes,
            n_layers,
            dt,
            dz,
            rho,
            params_list,
            damping_min,
            input_accel,
            boundary,
            rho_base,
            vs_base,
        )


# -----------------------------------------------------------------------------
# High-level API
# -----------------------------------------------------------------------------


def propagate_time_domain(
    times: npt.NDArray[np.floating],
    input_accel: npt.NDArray[np.floating],
    thicknesses: npt.NDArray[np.floating],
    densities: npt.NDArray[np.floating],
    shear_mods: npt.NDArray[np.floating],
    damping_ratios: npt.NDArray[np.floating],
    boundary: Literal["elastic", "rigid"] = "elastic",
    rho_base: float | None = None,
    vs_base: float | None = None,
    subcycles: int | None = None,
) -> TimeDomainResults:
    """Perform linear elastic time-domain wave propagation.

    Parameters
    ----------
    times : np.ndarray
        Time array [s].
    input_accel : np.ndarray
        Input acceleration time series [m/s²].
    thicknesses : np.ndarray
        Layer thicknesses [m].
    densities : np.ndarray
        Layer densities [kg/m³].
    shear_mods : np.ndarray
        Layer shear moduli [Pa].
    damping_ratios : np.ndarray
        Layer damping ratios.
    boundary : str
        'elastic' or 'rigid'.
    rho_base : float, optional
        Base layer density [kg/m³]. Required for elastic boundary.
    vs_base : float, optional
        Base layer shear velocity [m/s]. Required for elastic boundary.
    subcycles : int, optional
        Number of subcycles per time step. Auto-calculated if None.

    Returns
    -------
    results : TimeDomainResults
        Results containing time histories at all depths.
    """
    start_time = time_module.perf_counter()

    thicknesses = np.asarray(thicknesses, dtype=np.float64)
    densities = np.asarray(densities, dtype=np.float64)
    shear_mods = np.asarray(shear_mods, dtype=np.float64)
    damping_ratios = np.asarray(damping_ratios, dtype=np.float64)
    input_accel = np.asarray(input_accel, dtype=np.float64)

    n_layers = len(thicknesses)
    n_nodes = n_layers + 1
    n_times = len(times)

    # Calculate time step
    dt_input = times[1] - times[0] if len(times) > 1 else 0.01

    # Auto-calculate subcycles for CFL stability
    shear_vels = np.sqrt(shear_mods / densities)
    if subcycles is None:
        subcycles = calc_cfl_subcycles(dt_input, thicknesses, shear_vels)

    dt = dt_input / subcycles

    logger.debug(
        "propagate_time_domain: %d layers, %d time steps, dt=%.4fs, subcycles=%d, boundary=%s",
        n_layers,
        n_times,
        dt_input,
        subcycles,
        boundary,
    )

    # Default base properties
    if rho_base is None:
        rho_base = densities[-1]
    if vs_base is None:
        vs_base = shear_vels[-1]

    # Interpolate input motion to subcycled time steps
    n_times_sub = (n_times - 1) * subcycles + 1
    times_sub = np.linspace(times[0], times[-1], n_times_sub)
    input_accel_sub = np.interp(times_sub, times, input_accel)

    # Run integration
    boundary_code = 0 if boundary == "rigid" else 1
    displ, veloc, accel, stress = _integrate_linear_dispatch(
        n_times_sub,
        n_nodes,
        n_layers,
        dt,
        thicknesses,
        densities,
        shear_mods,
        damping_ratios,
        input_accel_sub,
        boundary_code,
        rho_base,
        vs_base,
    )

    # Downsample results back to original time step
    indices = np.arange(0, n_times_sub, subcycles)
    if len(indices) > n_times:
        indices = indices[:n_times]

    displ_out = displ[indices, :]
    veloc_out = veloc[indices, :]
    accel_out = accel[indices, :]
    stress_out = stress[indices, :]

    # Compute strain from displacement
    strain_out = np.zeros((len(indices), n_layers))
    for i in range(n_layers):
        strain_out[:, i] = (displ_out[:, i + 1] - displ_out[:, i]) / thicknesses[i]

    # Calculate depths
    depths = np.zeros(n_nodes)
    depths[1:] = np.cumsum(thicknesses)

    elapsed = time_module.perf_counter() - start_time
    logger.info("propagate_time_domain: completed in %.3fs", elapsed)

    return TimeDomainResults(
        times=times[: len(indices)],
        depths=depths,
        accel=accel_out,
        veloc=veloc_out,
        displ=displ_out,
        strain=strain_out,
        stress=stress_out,
    )


def propagate_nonlinear(
    times: npt.NDArray[np.floating],
    input_accel: npt.NDArray[np.floating],
    thicknesses: npt.NDArray[np.floating],
    densities: npt.NDArray[np.floating],
    params: MultiLayerParams,
    damping_min: npt.NDArray[np.floating],
    boundary: Literal["elastic", "rigid"] = "elastic",
    rho_base: float | None = None,
    vs_base: float | None = None,
    subcycles: int | None = None,
) -> TimeDomainResults:
    """Perform nonlinear time-domain wave propagation.

    Parameters
    ----------
    times : np.ndarray
        Time array [s].
    input_accel : np.ndarray
        Input acceleration time series [m/s²].
    thicknesses : np.ndarray
        Layer thicknesses [m].
    densities : np.ndarray
        Layer densities [kg/m³].
    params : MultiLayerParams
        Constitutive model parameters for each layer.
    damping_min : np.ndarray
        Minimum (small-strain) damping ratios.
    boundary : str
        'elastic' or 'rigid'.
    rho_base : float, optional
        Base layer density. Required for elastic boundary.
    vs_base : float, optional
        Base layer shear velocity. Required for elastic boundary.
    subcycles : int, optional
        Number of subcycles per time step.

    Returns
    -------
    results : TimeDomainResults
        Results containing time histories at all depths.
    """
    start_time = time_module.perf_counter()

    thicknesses = np.asarray(thicknesses, dtype=np.float64)
    densities = np.asarray(densities, dtype=np.float64)
    damping_min = np.asarray(damping_min, dtype=np.float64)
    input_accel = np.asarray(input_accel, dtype=np.float64)

    n_layers = len(thicknesses)
    n_nodes = n_layers + 1
    n_times = len(times)

    # Extract shear moduli from parameters
    shear_mods = np.array([p.shear_mod for p in params])
    shear_vels = np.sqrt(shear_mods / densities)

    dt_input = times[1] - times[0] if len(times) > 1 else 0.01

    if subcycles is None:
        subcycles = calc_cfl_subcycles(dt_input, thicknesses, shear_vels)

    dt = dt_input / subcycles

    # Determine model type from first parameter
    model_type = type(params[0]).__name__ if len(params) > 0 else "unknown"
    logger.debug(
        "propagate_nonlinear: %d layers, %d time steps, dt=%.4fs, subcycles=%d, model=%s",
        n_layers,
        n_times,
        dt_input,
        subcycles,
        model_type,
    )

    # Default base properties
    rho_base_val: float = densities[-1] if rho_base is None else rho_base
    vs_base_val: float = shear_vels[-1] if vs_base is None else vs_base

    # Interpolate input motion
    n_times_sub = (n_times - 1) * subcycles + 1
    times_sub = np.linspace(times[0], times[-1], n_times_sub)
    input_accel_sub = np.interp(times_sub, times, input_accel)

    # Run nonlinear integration (uses Numba if available)
    displ, veloc, accel, strain, stress = _integrate_nonlinear_dispatch(
        n_times_sub,
        n_nodes,
        n_layers,
        dt,
        thicknesses,
        densities,
        list(params),
        damping_min,
        input_accel_sub,
        boundary,
        rho_base_val,
        vs_base_val,
    )

    # Downsample
    indices = np.arange(0, n_times_sub, subcycles)
    if len(indices) > n_times:
        indices = indices[:n_times]

    depths = np.zeros(n_nodes)
    depths[1:] = np.cumsum(thicknesses)

    elapsed = time_module.perf_counter() - start_time
    logger.info("propagate_nonlinear: completed in %.3fs", elapsed)

    return TimeDomainResults(
        times=times[: len(indices)],
        depths=depths,
        accel=accel[indices, :],
        veloc=veloc[indices, :],
        displ=displ[indices, :],
        strain=strain[indices, :],
        stress=stress[indices, :],
    )
