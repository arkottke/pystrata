"""Tests for numba vs python agreement in constitutive.py."""

from __future__ import annotations

import numpy as np
import pytest

from pystrata.constitutive import (
    _calc_damping_from_stress_strain_python,
    _hh_misfit_python,
    _mkz_damping_misfit_python,
    _tau_mkz_python,
)
from pystrata.time_integration import _integrate_python

try:
    from pystrata.constitutive import (
        _calc_damping_from_stress_strain_numba,
        _hh_misfit_numba,
        _mkz_damping_misfit_numba,
    )
    from pystrata.time_integration import _integrate_linear_numba

    HAS_NUMBA = True
except ImportError:
    HAS_NUMBA = False


def _make_strain_stress(shear_mod: float, gamma_ref: float, beta: float, s: float):
    """Build monotonic strain/stress arrays from an MKZ model."""
    strain = np.linspace(1e-6, 0.05, 200)
    stress = _tau_mkz_python(strain, gamma_ref, beta, s, shear_mod)
    return strain, stress


@pytest.mark.skipif(not HAS_NUMBA, reason="numba not installed")
class TestDampingFromStressStrainNumba:
    """Verify numba and python implementations of calc_damping_from_stress_strain
    agree."""

    SHEAR_MOD = 50e6  # 50 MPa
    GAMMA_REF = 0.001
    BETA = 1.0
    S = 0.9

    def test_basic_agreement(self):
        strain, stress = _make_strain_stress(
            self.SHEAR_MOD, self.GAMMA_REF, self.BETA, self.S
        )
        result_py = _calc_damping_from_stress_strain_python(
            strain, stress, self.SHEAR_MOD
        )
        result_nb = _calc_damping_from_stress_strain_numba(
            strain, stress, self.SHEAR_MOD
        )

        np.testing.assert_allclose(result_nb, result_py, rtol=1e-12, atol=1e-15)

    def test_single_point(self):
        strain = np.array([0.001])
        stress = _tau_mkz_python(
            strain, self.GAMMA_REF, self.BETA, self.S, self.SHEAR_MOD
        )

        result_py = _calc_damping_from_stress_strain_python(
            strain, stress, self.SHEAR_MOD
        )
        result_nb = _calc_damping_from_stress_strain_numba(
            strain, stress, self.SHEAR_MOD
        )

        np.testing.assert_allclose(result_nb, result_py, rtol=1e-12, atol=1e-15)

    def test_zero_strain_first(self):
        """First strain value is zero — should not produce NaN or error."""
        strain = np.array([0.0, 1e-5, 1e-4, 1e-3])
        stress = np.zeros_like(strain)
        stress[1:] = _tau_mkz_python(
            strain[1:], self.GAMMA_REF, self.BETA, self.S, self.SHEAR_MOD
        )

        result_py = _calc_damping_from_stress_strain_python(
            strain, stress, self.SHEAR_MOD
        )
        result_nb = _calc_damping_from_stress_strain_numba(
            strain, stress, self.SHEAR_MOD
        )

        assert not np.any(np.isnan(result_nb))
        np.testing.assert_allclose(result_nb, result_py, rtol=1e-12, atol=1e-15)

    def test_large_strain_range(self):
        """Logarithmically spaced strains spanning several decades."""
        strain = np.logspace(-7, -1, 500)
        stress = _tau_mkz_python(
            strain, self.GAMMA_REF, self.BETA, self.S, self.SHEAR_MOD
        )

        result_py = _calc_damping_from_stress_strain_python(
            strain, stress, self.SHEAR_MOD
        )
        result_nb = _calc_damping_from_stress_strain_numba(
            strain, stress, self.SHEAR_MOD
        )

        np.testing.assert_allclose(result_nb, result_py, rtol=1e-12, atol=1e-15)

    def test_damping_non_negative(self):
        strain, stress = _make_strain_stress(
            self.SHEAR_MOD, self.GAMMA_REF, self.BETA, self.S
        )
        result_nb = _calc_damping_from_stress_strain_numba(
            strain, stress, self.SHEAR_MOD
        )

        assert np.all(result_nb >= 0.0)


@pytest.mark.skipif(not HAS_NUMBA, reason="numba not installed")
class TestLinearIntegratorNumba:
    """Verify numba and python linear integrators agree."""

    def _make_inputs(self, boundary="rigid"):
        n_layers = 3
        n_nodes = n_layers + 1
        n_times = 500
        dt = 0.001

        dz = np.array([5.0, 8.0, 12.0])
        rho = np.array([1800.0, 1900.0, 2100.0])
        shear_mod = np.array([50e6, 80e6, 150e6])
        damping = np.array([0.02, 0.015, 0.01])

        # Simple pulse
        input_accel = np.zeros(n_times)
        input_accel[50:60] = 0.5

        rho_base = 2200.0
        vs_base = 800.0

        return (
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

    def test_rigid_boundary(self):
        args = self._make_inputs("rigid")
        (
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
        ) = args

        d_py, v_py, a_py, s_py = _integrate_python(
            n_times,
            n_nodes,
            n_layers,
            dt,
            dz,
            rho,
            shear_mod,
            damping,
            input_accel,
            "rigid",
            rho_base,
            vs_base,
        )
        d_nb, v_nb, a_nb, s_nb = _integrate_linear_numba(
            n_times,
            n_nodes,
            n_layers,
            dt,
            dz,
            rho,
            shear_mod,
            damping,
            input_accel,
            0,
            rho_base,
            vs_base,
        )

        np.testing.assert_allclose(d_nb, d_py, rtol=1e-10, atol=1e-12)
        np.testing.assert_allclose(a_nb, a_py, rtol=1e-10, atol=1e-10)
        np.testing.assert_allclose(s_nb, s_py, rtol=1e-10, atol=1e-10)

    def test_elastic_boundary(self):
        args = self._make_inputs("elastic")
        (
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
        ) = args

        d_py, v_py, a_py, s_py = _integrate_python(
            n_times,
            n_nodes,
            n_layers,
            dt,
            dz,
            rho,
            shear_mod,
            damping,
            input_accel,
            "elastic",
            rho_base,
            vs_base,
        )
        d_nb, v_nb, a_nb, s_nb = _integrate_linear_numba(
            n_times,
            n_nodes,
            n_layers,
            dt,
            dz,
            rho,
            shear_mod,
            damping,
            input_accel,
            1,
            rho_base,
            vs_base,
        )

        np.testing.assert_allclose(d_nb, d_py, rtol=1e-10, atol=1e-12)
        np.testing.assert_allclose(a_nb, a_py, rtol=1e-10, atol=1e-10)
        np.testing.assert_allclose(s_nb, s_py, rtol=1e-10, atol=1e-10)

    def test_velocity_agreement(self):
        args = self._make_inputs("elastic")
        (
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
        ) = args

        _, v_py, _, _ = _integrate_python(
            n_times,
            n_nodes,
            n_layers,
            dt,
            dz,
            rho,
            shear_mod,
            damping,
            input_accel,
            "elastic",
            rho_base,
            vs_base,
        )
        _, v_nb, _, _ = _integrate_linear_numba(
            n_times,
            n_nodes,
            n_layers,
            dt,
            dz,
            rho,
            shear_mod,
            damping,
            input_accel,
            1,
            rho_base,
            vs_base,
        )

        np.testing.assert_allclose(v_nb, v_py, rtol=1e-12, atol=1e-15)


@pytest.mark.skipif(not HAS_NUMBA, reason="numba not installed")
class TestFusedMisfitNumba:
    """Verify fused numba misfit functions match python versions."""

    SHEAR_MOD = 50e6
    GAMMA_REF = 0.005
    BETA = 1.2
    S = 0.85

    def test_hh_misfit_agreement(self):
        strains = np.logspace(-6, -1, 50)
        mod_reduc = np.ones_like(strains) * 0.5  # dummy target
        damping = np.linspace(0, 0.15, len(strains))

        kwargs = dict(
            gamma_t=0.001,
            a=1.0,
            gamma_ref=self.GAMMA_REF,
            beta=self.BETA,
            s=self.S,
            shear_mod=self.SHEAR_MOD,
            mu=1.0,
            shear_strength=100e3,
            d=1.0,
            trans_c1=4.039,
            trans_c2=1.036,
        )

        val_py = _hh_misfit_python(strains, mod_reduc, damping, **kwargs)
        val_nb = _hh_misfit_numba(strains, mod_reduc, damping, **kwargs)

        np.testing.assert_allclose(val_nb, val_py, rtol=1e-10)

    def test_mkz_damping_misfit_agreement(self):
        strains = np.logspace(-6, -1, 50)
        damping = np.linspace(0, 0.10, len(strains))

        val_py = _mkz_damping_misfit_python(
            strains,
            damping,
            self.GAMMA_REF,
            self.BETA,
            self.S,
            self.SHEAR_MOD,
        )
        val_nb = _mkz_damping_misfit_numba(
            strains,
            damping,
            self.GAMMA_REF,
            self.BETA,
            self.S,
            self.SHEAR_MOD,
        )

        np.testing.assert_allclose(val_nb, val_py, rtol=1e-10)
