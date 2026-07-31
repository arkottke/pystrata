"""Interop tests: pystrata ↔ pygmm (soil curves + velocity profile + stub objects)."""

from types import SimpleNamespace

import numpy as np
import pytest
from numpy.testing import assert_allclose

pygmm = pytest.importorskip("pygmm")
# Imported after the skip check so the module is not required to collect this file
import pystrata  # noqa: E402


@pytest.fixture
def simple_profile_for_eql():
    """A minimal elastic profile for a quick EQL smoke-test."""
    bedrock = pystrata.site.SoilType(name="Rock", unit_wt=22.0, damping=0.02)
    return pystrata.site.Profile(
        [
            pystrata.site.Layer(bedrock, 30.0, 400.0),
            pystrata.site.Layer(bedrock, 0.0, 800.0),
        ]
    )


# ---------------------------------------------------------------------------
# SoilType.from_curves
# ---------------------------------------------------------------------------


def test_from_curves_darendeli():
    """Pygmm DarendeliSoilType curves → pystrata SoilType.from_curves round-trip."""
    pgm_st = pygmm.DarendeliSoilType(unit_wt=18.0, stress_mean=50.0)
    curves = pgm_st.curves()

    pst = pystrata.site.SoilType.from_curves(curves)
    assert pst.name == curves.name
    # The interpolated mod_reduc at a mid-range strain should match
    mid_strain = curves.strains[len(curves.strains) // 2]
    assert_allclose(
        pst.mod_reduc(mid_strain),
        curves.mod_reduc[len(curves.strains) // 2],
        rtol=0.02,
    )


def test_from_curves_stub():
    """SoilType.from_curves accepts a SimpleNamespace stub — no isinstance check."""
    strains = np.logspace(-6, -1.5, 20)
    stub = SimpleNamespace(
        strains=strains,
        mod_reduc=1 / (1 + (strains / 1e-4) ** 0.9),
        damping=np.full(20, 0.03),
        damping_min=0.03,
        unit_wt=18.0,
        name="stub",
    )
    pst = pystrata.site.SoilType.from_curves(stub)
    assert pst.unit_wt == 18.0
    assert pst.damping_min == pytest.approx(0.03, rel=0.01)


# ---------------------------------------------------------------------------
# Profile.from_velocity_profile
# ---------------------------------------------------------------------------


def test_from_velocity_profile_kea16():
    """Pygmm kea16_profile → pystrata Profile.from_velocity_profile round-trip."""
    depth = np.arange(0, 30, 5, dtype=float)
    vp = pygmm.kea16_profile(depth, vs30=400.0, region="california")

    bedrock = pystrata.site.SoilType(name="rock", unit_wt=22.0, damping=0.02)
    profile = pystrata.site.Profile.from_velocity_profile(vp, soil_types=bedrock)

    assert len(profile) == len(depth)
    for i, layer in enumerate(profile):
        assert layer.initial_shear_vel == pytest.approx(vp.vs_median[i], rel=1e-6)


def test_from_velocity_profile_stub():
    """Profile.from_velocity_profile accepts a SimpleNamespace stub."""
    depth = np.array([0.0, 5.0, 10.0, 20.0])
    stub = SimpleNamespace(
        depth=depth,
        vs_median=np.array([200.0, 300.0, 400.0, 600.0]),
        std_vs_ln=np.full(4, 0.2),
    )
    bedrock = pystrata.site.SoilType(name="rock", unit_wt=22.0, damping=0.02)
    profile = pystrata.site.Profile.from_velocity_profile(stub, soil_types=bedrock)
    assert len(profile) == 4
    assert profile[0].initial_shear_vel == pytest.approx(200.0)
    assert profile[-1].initial_shear_vel == pytest.approx(600.0)
