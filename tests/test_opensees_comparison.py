"""Verification of pystrata time-domain against OpenSees 1D site response.

This module runs equivalent 1D wave-propagation analyses in both pystrata
(central-difference / nonlinear integrator) and OpenSees (implicit Newmark
with quad elements) and compares the surface acceleration time histories.

Two cases are tested:

1. **Linear-elastic** — both codes use identical elastic properties.
2. **Nonlinear (hyperbolic)** — pystrata uses MKZ (β=1, s=1) and OpenSees
   uses ``PressureIndependMultiYield``, which share the same hyperbolic
   backbone: τ = Gmax·γ / (1 + γ/γ_ref).

The profile is a simple 20 m soil layer (Vs=200 m/s) over rock (Vs=800 m/s),
discretized into 10 sub-layers (dz=2 m).  Input is a low-frequency Ricker
wavelet so that spatial discretization is adequate.

Rayleigh damping is used in both codes (with the same target frequencies) so
that viscous dissipation is comparable.

Tests are automatically skipped when ``openseespy`` is not installed.
"""

import numpy as np
import pytest

import pystrata
from pystrata.constitutive import MKZParams, MultiLayerParams
from pystrata.time_integration import propagate_nonlinear, propagate_time_domain

try:
    from pystrata.opensees_helpers import run_opensees_linear, run_opensees_nonlinear

    HAS_OPENSEES = True
except Exception:
    HAS_OPENSEES = False

pytestmark = pytest.mark.skipif(not HAS_OPENSEES, reason="openseespy not installed")

GRAVITY = 9.81


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------


def _make_ricker(
    f0: float = 3.0, pga: float = 0.001, dt: float = 0.005, dur: float = 5.0
):
    """Create a Ricker-wavelet TimeSeriesMotion."""
    t = np.arange(0, dur, dt)
    t0 = 1.0 / f0 + 0.2
    u = (1 - 2 * (np.pi * f0 * (t - t0)) ** 2) * np.exp(-((np.pi * f0 * (t - t0)) ** 2))
    u = u / np.max(np.abs(u)) * pga
    return pystrata.motion.TimeSeriesMotion(
        filename="ricker",
        description="Ricker wavelet",
        time_step=dt,
        accels=u,
    )


def _build_profile_arrays(
    n_sub: int = 10,
    vs_soil: float = 200.0,
    unit_wt_soil: float = 18.0,
    damping_soil: float = 0.02,
):
    """Build flat arrays for a single uniform soil layer over rock.

    Returns thicknesses, densities, shear_mods, damping_ratios, rho_base, vs_base.
    """
    rho_soil = unit_wt_soil / GRAVITY * 1000  # kN/m³ → kg/m³
    G_soil = rho_soil * vs_soil**2
    dz = 20.0 / n_sub

    thicknesses = np.full(n_sub, dz)
    densities = np.full(n_sub, rho_soil)
    shear_mods = np.full(n_sub, G_soil)
    damping_ratios = np.full(n_sub, damping_soil)

    rho_base = 24.0 / GRAVITY * 1000
    vs_base = 800.0

    return thicknesses, densities, shear_mods, damping_ratios, rho_base, vs_base


# ===========================================================================
# Linear-Elastic Comparison
# ===========================================================================


class TestLinearElastic:
    """Compare linear-elastic surface response from pystrata and OpenSees."""

    @pytest.fixture(scope="class")
    def results(self):
        """Run both pystrata (linear TD) and OpenSees linear."""
        motion = _make_ricker(f0=3.0, pga=0.001)
        thicknesses, densities, shear_mods, damping_ratios, rho_base, vs_base = (
            _build_profile_arrays()
        )
        dt = motion.time_step
        # Outcrop → incident: divide by 2
        input_accel_ms2 = motion.accels * GRAVITY / 2.0

        # --- pystrata ---
        res_ps = propagate_time_domain(
            times=motion.times,
            input_accel=input_accel_ms2,
            thicknesses=thicknesses,
            densities=densities,
            shear_mods=shear_mods,
            damping_ratios=damping_ratios,
            boundary="elastic",
            rho_base=rho_base,
            vs_base=vs_base,
        )
        accel_ps = res_ps.accel[:, 0]  # surface node, m/s²

        # --- OpenSees ---
        input_accel_outcrop = motion.accels * GRAVITY  # full outcrop in m/s²
        accel_os = run_opensees_linear(
            thicknesses=thicknesses,
            densities=densities,
            shear_mods=shear_mods,
            damping_ratios=damping_ratios,
            input_accel=input_accel_outcrop,
            dt=dt,
            rho_base=rho_base,
            vs_base=vs_base,
        )

        return {
            "pystrata": accel_ps,
            "opensees": accel_os,
            "dt": dt,
            "input_pga_g": motion.pga,
        }

    def test_no_nan(self, results):
        assert not np.any(np.isnan(results["pystrata"])), "NaN in pystrata"
        assert not np.any(np.isnan(results["opensees"])), "NaN in OpenSees"

    def test_pga_ratio(self, results):
        pga_ps = np.max(np.abs(results["pystrata"]))
        pga_os = np.max(np.abs(results["opensees"]))
        ratio = pga_ps / pga_os
        assert 0.85 <= ratio <= 1.15, (
            f"PGA ratio {ratio:.3f} out of [0.85, 1.15] "
            f"(pystrata={pga_ps:.6f}, OpenSees={pga_os:.6f})"
        )

    def test_waveform_correlation(self, results):
        a_ps = results["pystrata"]
        a_os = results["opensees"]
        n = min(len(a_ps), len(a_os))
        corr = np.corrcoef(a_ps[:n], a_os[:n])[0, 1]
        assert corr > 0.90, f"Waveform correlation {corr:.3f} < 0.90"


# ===========================================================================
# Nonlinear (Hyperbolic) Comparison
# ===========================================================================


class TestNonlinear:
    """Compare nonlinear (MKZ β=1, s=1 ↔ PIMY) surface response."""

    @pytest.fixture(scope="class")
    def results(self):
        motion = _make_ricker(f0=3.0, pga=0.1)
        thicknesses, densities, shear_mods, damping_ratios, rho_base, vs_base = (
            _build_profile_arrays()
        )
        dt = motion.time_step
        n_layers = len(thicknesses)

        gamma_ref = 0.01  # reference strain = 1 %
        shear_strengths = shear_mods * gamma_ref  # τ_max = G * γ_ref

        # --- pystrata nonlinear (MKZ β=1, s=1) ---
        params = MultiLayerParams()
        for i in range(n_layers):
            params.append(
                MKZParams(gamma_ref=gamma_ref, beta=1.0, s=1.0, shear_mod=shear_mods[i])
            )

        input_accel_ms2 = motion.accels * GRAVITY / 2.0  # outcrop → incident
        res_ps = propagate_nonlinear(
            times=motion.times,
            input_accel=input_accel_ms2,
            thicknesses=thicknesses,
            densities=densities,
            params=params,
            damping_min=damping_ratios,
            boundary="elastic",
            rho_base=rho_base,
            vs_base=vs_base,
            damp_form="rayleigh",
        )
        accel_ps = res_ps.accel[:, 0]  # surface node, m/s²

        # --- OpenSees nonlinear (PressureIndependMultiYield) ---
        input_accel_outcrop = motion.accels * GRAVITY
        accel_os = run_opensees_nonlinear(
            thicknesses=thicknesses,
            densities=densities,
            shear_mods=shear_mods,
            damping_ratios=damping_ratios,
            shear_strengths=shear_strengths,
            ref_strains=np.full(n_layers, gamma_ref),
            input_accel=input_accel_outcrop,
            dt=dt,
            rho_base=rho_base,
            vs_base=vs_base,
        )

        return {
            "pystrata": accel_ps,
            "opensees": accel_os,
            "dt": dt,
            "input_pga_g": motion.pga,
        }

    def test_no_nan(self, results):
        assert not np.any(np.isnan(results["pystrata"])), "NaN in pystrata"
        assert not np.any(np.isnan(results["opensees"])), "NaN in OpenSees"

    def test_pga_ratio(self, results):
        pga_ps = np.max(np.abs(results["pystrata"]))
        pga_os = np.max(np.abs(results["opensees"]))
        ratio = pga_ps / pga_os
        assert 0.70 <= ratio <= 1.30, (
            f"PGA ratio {ratio:.3f} out of [0.70, 1.30] "
            f"(pystrata={pga_ps:.6f}, OpenSees={pga_os:.6f})"
        )

    def test_max_accel_order(self, results):
        """Max accelerations should be in the same order of magnitude."""
        max_ps = np.max(np.abs(results["pystrata"]))
        max_os = np.max(np.abs(results["opensees"]))
        ratio = max_ps / max_os if max_os > 0 else float("inf")
        assert 0.3 < ratio < 3.0, (
            f"Max accel ratio {ratio:.3f} outside [0.3, 3.0] "
            f"(pystrata={max_ps:.4f}, OpenSees={max_os:.4f})"
        )
