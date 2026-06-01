#!/usr/bin/env python
"""Benchmark pystrata RVT SRA with simulated profiles: numba vs pure Python.

Based on example-08.ipynb. Benchmarks both EquivalentLinearCalculator and
FrequencyDependentEqlCalculator with 60 realizations.
"""

import time

import numpy as np

import pystrata
from pystrata import propagation


def create_motion():
    m = pystrata.motion.SourceTheoryRvtMotion(6.0, 30, "wna")
    m.calc_fourier_amps()
    return m


def create_profile():
    return pystrata.site.Profile(
        [
            pystrata.site.Layer(
                pystrata.site.DarendeliSoilType(
                    18.0, plas_index=0, ocr=1, stress_mean=100
                ),
                10,
                400,
            ),
            pystrata.site.Layer(
                pystrata.site.DarendeliSoilType(
                    18.0, plas_index=0, ocr=1, stress_mean=200
                ),
                10,
                450,
            ),
            pystrata.site.Layer(
                pystrata.site.DarendeliSoilType(
                    18.0, plas_index=0, ocr=1, stress_mean=400
                ),
                30,
                600,
            ),
            pystrata.site.Layer(
                pystrata.site.SoilType("Rock", 24.0, None, 0.01), 0, 1200
            ),
        ]
    )


def create_outputs():
    freqs = np.logspace(-1, 2, num=500)
    return pystrata.output.OutputCollection(
        [
            pystrata.output.ResponseSpectrumOutput(
                freqs,
                pystrata.output.OutputLocation("outcrop", index=0),
                0.05,
            ),
            pystrata.output.ResponseSpectrumRatioOutput(
                freqs,
                pystrata.output.OutputLocation("outcrop", index=-1),
                pystrata.output.OutputLocation("outcrop", index=0),
                0.05,
            ),
            pystrata.output.InitialVelProfile(),
        ]
    )


def run_analysis(calc, motion, profile, outputs, count):
    var_thickness = pystrata.variation.ToroThicknessVariation()
    var_velocity = pystrata.variation.ToroVelocityVariation.generic_model("USGS C")
    var_soiltypes = pystrata.variation.SpidVariation(
        -0.5, std_mod_reduc=0.15, std_damping=0.30
    )

    outputs.reset()
    for p in pystrata.variation.iter_varied_profiles(
        profile,
        count,
        var_thickness=var_thickness,
        var_velocity=var_velocity,
        var_soiltypes=var_soiltypes,
    ):
        p = p.auto_discretize()
        calc(motion, p, p.location("outcrop", index=-1))
        outputs(calc)


def set_numba_dispatchers():
    """Restore the numba-accelerated dispatch functions."""
    propagation._calc_waves_dispatch = propagation._calc_waves_numba
    propagation._wave_at_location_dispatch = propagation._wave_at_location_numba
    propagation._calc_strain_tf_dispatch = propagation._calc_strain_tf_numba
    propagation.my_trapz = propagation._my_trapz_impl


def set_python_dispatchers():
    """Switch to the pure-Python dispatch functions."""
    propagation._calc_waves_dispatch = propagation._calc_waves_python
    propagation._wave_at_location_dispatch = propagation._wave_at_location_python
    propagation._calc_strain_tf_dispatch = propagation._calc_strain_tf_python
    propagation.my_trapz = propagation._my_trapz_python


def benchmark(calc_class, calc_name, motion, profile, count):
    """Benchmark a single calculator class with and without numba."""
    print(f"--- {calc_name} ---")

    # --- Run with numba ---
    if propagation.HAS_NUMBA:
        set_numba_dispatchers()

        # Warm-up run (trigger JIT compilation)
        calc = calc_class()
        outputs = create_outputs()
        run_analysis(calc, motion, profile, outputs, count=1)

        # Timed run
        calc = calc_class()
        outputs = create_outputs()
        t0 = time.perf_counter()
        run_analysis(calc, motion, profile, outputs, count)
        t_numba = time.perf_counter() - t0
        print(f"  With numba:    {t_numba:.2f} s")
    else:
        t_numba = None
        print("  Numba not installed — skipping numba run")

    # --- Run without numba ---
    set_python_dispatchers()

    calc = calc_class()
    outputs = create_outputs()
    t0 = time.perf_counter()
    run_analysis(calc, motion, profile, outputs, count)
    t_python = time.perf_counter() - t0
    print(f"  Without numba: {t_python:.2f} s")

    if t_numba is not None:
        speedup = t_python / t_numba
        print(f"  Speedup:       {speedup:.2f}x")

    print()
    return t_numba, t_python


def main():
    count = 60
    motion = create_motion()
    profile = create_profile()

    print(f"Numba available: {propagation.HAS_NUMBA}")
    print(f"Realizations: {count}")
    print()

    results = {}

    results["EQL"] = benchmark(
        pystrata.propagation.EquivalentLinearCalculator,
        "EquivalentLinearCalculator",
        motion,
        profile,
        count,
    )

    results["FDM"] = benchmark(
        pystrata.propagation.FrequencyDependentEqlCalculator,
        "FrequencyDependentEqlCalculator",
        motion,
        profile,
        count,
    )

    # --- Summary table ---
    print("=" * 50)
    print(f"{'Calculator':<12} {'Numba (s)':>10} {'Python (s)':>11} {'Speedup':>8}")
    print("-" * 50)
    for name, (t_numba, t_python) in results.items():
        numba_str = f"{t_numba:.2f}" if t_numba is not None else "N/A"
        speedup_str = f"{t_python / t_numba:.2f}x" if t_numba else "N/A"
        print(f"{name:<12} {numba_str:>10} {t_python:>11.2f} {speedup_str:>8}")
    print("=" * 50)


if __name__ == "__main__":
    main()
