"""Cross-validation tests against PySeismoSoil.

These tests compare pyStrata's constitutive model implementations against PySeismoSoil's
independent implementations of the same models (MKZ, HH, FKZ). This gives confidence
that both libraries produce physically consistent results.

Skipped automatically if PySeismoSoil is not installed.
"""

from __future__ import annotations

import numpy as np
import pytest

pss_mkz = pytest.importorskip("PySeismoSoil.helper_mkz_model")
pss_hh = pytest.importorskip("PySeismoSoil.helper_hh_model")
pss_sr = pytest.importorskip("PySeismoSoil.helper_site_response")
pss_calib = pytest.importorskip("PySeismoSoil.helper_hh_calibration")

import pystrata.site as site  # noqa: E402
from pystrata.constitutive import (  # noqa: E402
    HHParams,
    MKZParams,
    calc_damping,
    calc_damping_from_stress_strain,
    calc_lateral_pressure_coeff,
    calc_mod_reduc,
    calc_ocr_from_vs,
    calc_shear_strength,
    calc_stress_hh,
    calc_stress_mkz,
)
from pystrata.curve_fitting import fit_layer_mkz  # noqa: E402

# ---------------------------------------------------------------------------
# Parameter sets
# ---------------------------------------------------------------------------

MKZ_CASES = [
    dict(gamma_ref=0.005, beta=1.2, s=0.85, Gmax=50e6),
    dict(gamma_ref=0.01, beta=1.0, s=0.9, Gmax=100e6),
    dict(gamma_ref=0.02, beta=0.5, s=0.95, Gmax=30e6),
]

HH_CASES = [
    dict(
        gamma_t=0.001,
        a=1.0,
        gamma_ref=0.01,
        beta=1.0,
        s=0.9,
        Gmax=50e6,
        mu=1.0,
        Tmax=100e3,
        d=1.0,
    ),
    dict(
        gamma_t=0.005,
        a=0.8,
        gamma_ref=0.005,
        beta=1.2,
        s=0.85,
        Gmax=80e6,
        mu=0.9,
        Tmax=200e3,
        d=0.8,
    ),
    dict(
        gamma_t=0.0001,
        a=1.5,
        gamma_ref=0.02,
        beta=0.6,
        s=0.95,
        Gmax=30e6,
        mu=1.2,
        Tmax=50e3,
        d=1.2,
    ),
]

# Strains for comparison (unit: 1, not %)
STRAINS = np.geomspace(1e-6, 0.05, 50)

# At very small strains damping is dominated by numerical integration error;
# use a moderate range for damping comparison.
STRAINS_DAMPING = np.geomspace(1e-4, 0.05, 30)


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------


def _ps_mkz(pss_dict: dict) -> MKZParams:
    """Convert a PySeismoSoil-style dict to a pyStrata MKZParams."""
    return MKZParams(
        gamma_ref=pss_dict["gamma_ref"],
        beta=pss_dict["beta"],
        s=pss_dict["s"],
        shear_mod=pss_dict["Gmax"],
    )


def _ps_hh(pss_dict: dict) -> HHParams:
    """Convert a PySeismoSoil-style dict to a pyStrata HHParams."""
    return HHParams(
        gamma_t=pss_dict["gamma_t"],
        a=pss_dict["a"],
        gamma_ref=pss_dict["gamma_ref"],
        beta=pss_dict["beta"],
        s=pss_dict["s"],
        shear_mod=pss_dict["Gmax"],
        mu=pss_dict["mu"],
        shear_strength=pss_dict["Tmax"],
        d=pss_dict["d"],
    )


# ===========================================================================
# MKZ stress tests
# ===========================================================================


class TestMKZStress:
    """Verify MKZ backbone stress is identical between libraries."""

    @pytest.mark.parametrize("case", MKZ_CASES, ids=lambda c: f"gref={c['gamma_ref']}")
    def test_stress(self, case):
        ps_params = _ps_mkz(case)
        stress_ps = calc_stress_mkz(STRAINS, ps_params)
        stress_pss = pss_mkz.tau_MKZ(STRAINS, **case)
        np.testing.assert_allclose(stress_ps, stress_pss, rtol=1e-12)


# ===========================================================================
# HH stress tests
# ===========================================================================


class TestHHStress:
    """Verify HH backbone stress is identical between libraries."""

    @pytest.mark.parametrize("case", HH_CASES, ids=lambda c: f"gref={c['gamma_ref']}")
    def test_stress(self, case):
        ps_params = _ps_hh(case)
        stress_ps = calc_stress_hh(STRAINS, ps_params)
        stress_pss = pss_hh.tau_HH(STRAINS, **case)
        np.testing.assert_allclose(stress_ps, stress_pss, rtol=1e-12)


# ===========================================================================
# Modulus reduction tests
# ===========================================================================


class TestModulusReduction:
    """Verify G/Gmax from MKZ backbone matches PySeismoSoil."""

    @pytest.mark.parametrize("case", MKZ_CASES, ids=lambda c: f"gref={c['gamma_ref']}")
    def test_ggmax_mkz(self, case):
        ps_params = _ps_mkz(case)
        ggmax_ps = calc_mod_reduc(STRAINS, ps_params)

        stress_pss = pss_mkz.tau_MKZ(STRAINS, **case)
        ggmax_pss = pss_sr.calc_GGmax_from_stress_strain(
            STRAINS, stress_pss, Gmax=case["Gmax"]
        )
        np.testing.assert_allclose(ggmax_ps, ggmax_pss, rtol=1e-10)

    @pytest.mark.parametrize("case", HH_CASES, ids=lambda c: f"gref={c['gamma_ref']}")
    def test_ggmax_hh(self, case):
        """G/Gmax from HH backbone should match PySeismoSoil."""
        ps_params = _ps_hh(case)
        stress_ps = calc_stress_hh(STRAINS, ps_params)
        ggmax_ps = (stress_ps / STRAINS) / ps_params.shear_mod

        stress_pss = pss_hh.tau_HH(STRAINS, **case)
        ggmax_pss = pss_sr.calc_GGmax_from_stress_strain(
            STRAINS, stress_pss, Gmax=case["Gmax"]
        )
        np.testing.assert_allclose(ggmax_ps, ggmax_pss, rtol=1e-10)


# ===========================================================================
# Damping tests
# ===========================================================================


class TestDamping:
    """Compare Masing-based damping between libraries.

    Both use trapezoidal integration of the backbone stress-strain area. Small
    differences at tiny strains are expected; we compare at moderate strains where the
    integration converges.
    """

    @pytest.mark.parametrize("case", MKZ_CASES, ids=lambda c: f"gref={c['gamma_ref']}")
    def test_damping_mkz(self, case):
        ps_params = _ps_mkz(case)
        d_ps = calc_damping(STRAINS_DAMPING, ps_params)

        pss_dict = dict(
            gamma_ref=case["gamma_ref"],
            beta=case["beta"],
            s=case["s"],
            Gmax=case["Gmax"],
        )
        d_pss = pss_sr.calc_damping_from_param(
            pss_dict, STRAINS_DAMPING, pss_mkz.tau_MKZ
        )
        np.testing.assert_allclose(d_ps, d_pss, atol=5e-4)

    @pytest.mark.parametrize("case", HH_CASES, ids=lambda c: f"gref={c['gamma_ref']}")
    def test_damping_hh(self, case):
        ps_params = _ps_hh(case)
        d_ps = calc_damping(STRAINS_DAMPING, ps_params)

        d_pss = pss_sr.calc_damping_from_param(case, STRAINS_DAMPING, pss_hh.tau_HH)
        np.testing.assert_allclose(d_ps, d_pss, atol=5e-4)


# ===========================================================================
# Damping from stress-strain backbone
# ===========================================================================


class TestDampingFromStressStrain:
    """Compare calc_damping_from_stress_strain against PySeismoSoil."""

    def test_mkz_backbone(self):
        case = MKZ_CASES[0]
        ps_params = _ps_mkz(case)
        stress = calc_stress_mkz(STRAINS_DAMPING, ps_params)

        d_ps = calc_damping_from_stress_strain(STRAINS_DAMPING, stress, case["Gmax"])
        d_pss = pss_sr.calc_damping_from_stress_strain(
            STRAINS_DAMPING, stress, Gmax=case["Gmax"]
        )
        np.testing.assert_allclose(d_ps, d_pss, atol=5e-4)

    def test_hh_backbone(self):
        case = HH_CASES[0]
        ps_params = _ps_hh(case)
        stress = calc_stress_hh(STRAINS_DAMPING, ps_params)

        d_ps = calc_damping_from_stress_strain(STRAINS_DAMPING, stress, case["Gmax"])
        d_pss = pss_sr.calc_damping_from_stress_strain(
            STRAINS_DAMPING, stress, Gmax=case["Gmax"]
        )
        np.testing.assert_allclose(d_ps, d_pss, atol=5e-4)


# ===========================================================================
# Geotechnical parameter estimation
# ===========================================================================


class TestK0:
    """Compare K0 (lateral pressure coefficient) estimation."""

    @pytest.mark.parametrize("ocr", [1.0, 2.0, 4.0, 8.0])
    def test_k0_matches(self, ocr):
        k0_ps = calc_lateral_pressure_coeff(np.array([ocr]))[0]
        k0_pss = pss_calib._calc_K0(np.array([ocr]))[0]
        np.testing.assert_allclose(k0_ps, k0_pss, rtol=1e-10)


class TestOCR:
    """Compare OCR estimation from Vs."""

    def test_ocr_matches(self):
        vs = np.array([150.0, 250.0, 400.0, 600.0])
        # Need density for sigma_v0 — use same density formula as PySeismoSoil
        h = np.array([5.0, 5.0, 10.0, 10.0])
        rho_pss = pss_calib._calc_rho(h, vs)
        sigma_v0 = pss_calib._calc_vertical_stress(h, rho_pss)

        ocr_ps = calc_ocr_from_vs(vs, rho_pss, sigma_v0)
        ocr_pss = pss_calib._calc_OCR(vs, rho_pss, sigma_v0)
        np.testing.assert_allclose(ocr_ps, ocr_pss, rtol=1e-10)


class TestShearStrength:
    """Compare shear strength estimation."""

    def test_soft_soil(self):
        """Soft soil (Vs <= 760): Ladd (1991) undrained strength."""
        vs = np.array([150.0, 300.0, 500.0])
        ocr = np.array([1.5, 2.0, 3.0])
        sigma_v0 = np.array([25e3, 75e3, 150e3])
        k0 = calc_lateral_pressure_coeff(ocr)

        tmax_ps = calc_shear_strength(vs, ocr, sigma_v0, k0=k0)
        tmax_pss = pss_calib._calc_shear_strength(vs, ocr, sigma_v0, K0=k0)
        np.testing.assert_allclose(tmax_ps, tmax_pss, rtol=1e-10)

    def test_stiff_soil(self):
        """Stiff soil (Vs > 760): Mohr-Coulomb criterion."""
        vs = np.array([800.0, 1000.0])
        ocr = np.array([5.0, 8.0])
        sigma_v0 = np.array([200e3, 400e3])
        k0 = calc_lateral_pressure_coeff(ocr)

        tmax_ps = calc_shear_strength(vs, ocr, sigma_v0, k0=k0)
        tmax_pss = pss_calib._calc_shear_strength(vs, ocr, sigma_v0, K0=k0)
        np.testing.assert_allclose(tmax_ps, tmax_pss, rtol=1e-10)


# ===========================================================================
# Curve fitting cross-check
# ===========================================================================


class TestFitCrossCheck:
    """Verify pyStrata's fitted MKZ parameters reproduce PySeismoSoil curves.

    We use PySeismoSoil to generate target G/Gmax and damping curves from known HH
    parameters, then fit pyStrata's MKZ model to those curves and verify agreement.
    """

    def _make_pss_curves(self, case):
        """Generate G/Gmax and damping curves from PySeismoSoil HH params."""
        strains_pct = np.geomspace(0.0001, 10, 100)  # strain in %
        strains = strains_pct / 100  # strain in unit 1
        stress = pss_hh.tau_HH(strains, **case)
        ggmax = pss_sr.calc_GGmax_from_stress_strain(strains, stress, Gmax=case["Gmax"])
        damping = pss_sr.calc_damping_from_stress_strain(
            strains, stress, Gmax=case["Gmax"]
        )
        return strains_pct, ggmax, damping

    def test_mkz_fit_reproduces_pss_ggmax(self):
        """Fit MKZ to PySeismoSoil-generated G/Gmax and check accuracy."""
        case = HH_CASES[0]
        strains_pct, ggmax_target, _ = self._make_pss_curves(case)

        # Create a layer with a custom soil type whose mod_reduc
        # is the PySeismoSoil-generated curve
        strains_dec = strains_pct / 100
        from scipy.interpolate import interp1d

        mod_reduc_fn = interp1d(
            strains_dec,
            ggmax_target,
            kind="linear",
            fill_value=(1.0, ggmax_target[-1]),
            bounds_error=False,
        )

        st = site.SoilType("test", unit_wt=18.0, mod_reduc=mod_reduc_fn, damping=0.02)
        layer = site.Layer(st, 10, 200)
        params = fit_layer_mkz(layer, seed=42)

        # Evaluate pyStrata fit
        eval_strains = np.geomspace(1e-6, 0.05, 50)
        ggmax_fit = calc_mod_reduc(eval_strains, params)
        ggmax_ref = mod_reduc_fn(eval_strains)

        rms = np.sqrt(np.mean((ggmax_fit - ggmax_ref) ** 2))
        assert rms < 0.05, f"MKZ fit RMS {rms:.4f} exceeds 0.05"
