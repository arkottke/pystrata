"""Verify time-domain linear integration against frequency-domain solution.

Uses a discretized profile to ensure spatial resolution is adequate. Compares the linear
and nonlinear integrators for LINEAR soil.
"""

import numpy as np

import pystrata
from pystrata.constitutive import MKZParams, MultiLayerParams
from pystrata.time_integration import (
    propagate_nonlinear,
    propagate_time_domain,
)

GRAVITY = 9.81


def make_profile():
    return pystrata.site.Profile(
        [
            pystrata.site.Layer(
                pystrata.site.SoilType("Soil", 18.0, mod_reduc=None, damping=0.02),
                thickness=20,
                shear_vel=200,
            ),
            pystrata.site.Layer(
                pystrata.site.SoilType("Rock", 24.0, mod_reduc=None, damping=0.01),
                thickness=0,
                shear_vel=800,
            ),
        ]
    )


def make_ricker(f0=5.0, t0=0.5, dt=0.005, dur=5.0, pga=0.01):
    t = np.arange(0, dur, dt)
    u = (1 - 2 * (np.pi * f0 * (t - t0)) ** 2) * np.exp(-((np.pi * f0 * (t - t0)) ** 2))
    u = u / np.max(np.abs(u)) * pga
    return pystrata.motion.TimeSeriesMotion(
        filename="ricker",
        description="Ricker wavelet",
        time_step=dt,
        accels=u,
    )


def main():
    motion = make_ricker(f0=3.0, pga=0.001)  # low freq, tiny amplitude

    # Discretize manually
    n_sub = 10
    density = 18.0 / GRAVITY * 1000
    vs_soil = 200.0
    G = density * vs_soil**2
    dz = 20.0 / n_sub

    thicknesses = np.full(n_sub, dz)
    densities = np.full(n_sub, density)
    shear_mods = np.full(n_sub, G)
    damping_ratios = np.full(n_sub, 0.02)

    rho_base = 24.0 / GRAVITY * 1000
    vs_base = 800.0
    input_accel = motion.accels * GRAVITY / 2  # outcrop -> incident

    # --- Linear integrator ---
    res_lin = propagate_time_domain(
        times=motion.times,
        input_accel=input_accel,
        thicknesses=thicknesses,
        densities=densities,
        shear_mods=shear_mods,
        damping_ratios=damping_ratios,
        boundary="elastic",
        rho_base=rho_base,
        vs_base=vs_base,
    )

    # --- Nonlinear integrator with ~linear MKZ params ---
    # Use very large gamma_ref so MKZ is effectively linear
    params = MultiLayerParams()
    for i in range(n_sub):
        params.append(MKZParams(gamma_ref=1e10, beta=1.0, s=0.9, shear_mod=G))

    res_nl = propagate_nonlinear(
        times=motion.times,
        input_accel=input_accel,
        thicknesses=thicknesses,
        densities=densities,
        params=params,
        damping_min=damping_ratios,
        boundary="elastic",
        rho_base=rho_base,
        vs_base=vs_base,
    )

    # --- Frequency-domain linear ---
    profile_fd = make_profile()
    calc_fd = pystrata.propagation.LinearElasticCalculator()
    loc_in = profile_fd.location("outcrop", index=-1)
    calc_fd(motion, profile_fd, loc_in)

    # Get FD surface acceleration
    loc_surf = profile_fd.location("outcrop", index=0)
    tf = calc_fd.calc_accel_tf(loc_in, loc_surf)
    accel_surf_fd = calc_fd.motion.calc_time_series(tf)
    pga_fd = np.max(np.abs(accel_surf_fd))

    # Compare
    pga_lin = np.max(np.abs(res_lin.accel[:, 0])) / GRAVITY
    pga_nl = np.max(np.abs(res_nl.accel[:, 0])) / GRAVITY

    print(f"Input PGA: {motion.pga:.6f} g")
    print(f"FD surface PGA: {pga_fd:.6f} g")
    print(f"TD linear PGA:  {pga_lin:.6f} g  (ratio to FD: {pga_lin / pga_fd:.3f})")
    print(f"TD nonlin PGA:  {pga_nl:.6f} g  (ratio to FD: {pga_nl / pga_fd:.3f})")
    print(f"Lin vs NL ratio: {pga_lin / pga_nl:.4f}")
    print(f"NaN in linear: {np.any(np.isnan(res_lin.accel))}")
    print(f"NaN in nonlin: {np.any(np.isnan(res_nl.accel))}")

    # Time series comparison between linear and nonlinear
    n = min(res_lin.accel.shape[0], res_nl.accel.shape[0])
    a_lin = res_lin.accel[:n, 0]
    a_nl = res_nl.accel[:n, 0]
    rms_diff = np.sqrt(np.mean((a_lin - a_nl) ** 2))
    rms_lin = np.sqrt(np.mean(a_lin**2))
    print(
        f"RMS diff (lin vs nl): {rms_diff:.6f} m/s2 ({rms_diff / rms_lin * 100:.2f}% of lin)"
    )

    # --- Test the hysteretic integrator too ---
    from pystrata.time_integration import _integrate_hysteretic_dispatch

    # Same linear params for both mod and damp
    mod_params = list(params)
    damp_params = list(params)  # same as loading for linear check

    displ, veloc, accel, strain, stress = _integrate_hysteretic_dispatch(
        n_times_sub=len(motion.times),
        n_nodes=n_sub + 1,
        n_layers=n_sub,
        dt=motion.time_step,
        dz=thicknesses,
        rho=densities,
        mod_params_list=mod_params,
        damp_params_list=damp_params,
        damping_min=damping_ratios,
        input_accel=input_accel * GRAVITY,  # wait - is input already in m/s2?
        boundary="elastic",
        rho_base=rho_base,
        vs_base=vs_base,
        damp_form="rayleigh",
    )


if __name__ == "__main__":
    main()
