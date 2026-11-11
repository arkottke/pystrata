from collections import Counter

import numpy as np
import pandas as pd
import pytest
from scipy.constants import g as GRAVITY

import pystrata

from . import FPATH_DATA


def create_pystrata_profile(
    fpath, depth_wt, shear_vel_hs=None, randomize_nl_curves=False
):
    """Create a PyStrata profile from CSV data.

    Parameters
    ----------
    fpath : str or Path
        Path to CSV file containing profile data
    depth_wt : float
        Water table depth in meters
    shear_vel_hs : float, optional
        Shear wave velocity of half-space layer in m/s
    randomize_nl_curves : bool, optional
        If True, randomize nonlinear curves using SPID variation

    Returns
    -------
    pystrata.site.Profile
        Site profile object with specified layers and properties
    """
    df = pd.read_csv(fpath)

    layers = []
    counter = Counter()
    for _, row in df.iterrows():
        if row["material"] in ["rock", "crust"]:
            soil_type = pystrata.site.SoilType(
                row["material"], GRAVITY * row["density"], damping=0.0
            )
        else:
            soil_type = pystrata.site.SoilType(
                row["material"] + "-" + str(counter[row["material"]]),
                GRAVITY * row["density"],
                damping=0.05,
            )
            counter[row["material"]] += 1

        layer = pystrata.site.Layer(
            soil_type,
            row["thick"],
            row["vs"],
        )
        layers.append(layer)

    if shear_vel_hs is not None:
        layers.append(
            # Half-space
            pystrata.site.Layer(
                pystrata.site.SoilType(
                    "half-space", GRAVITY * df.iloc[-1]["density"], damping=0.0
                ),
                0,
                shear_vel_hs,
            )
        )

    profile = pystrata.site.Profile(layers, wt_depth=depth_wt)

    if randomize_nl_curves:
        st_variation = pystrata.variation.SpidVariation(-0.5)
        profile = next(
            pystrata.variation.iter_varied_profiles(
                profile, 1, var_soiltypes=st_variation
            )
        )

    return profile


def iter_profiles():
    """Generate profile and site attenuation test cases.

    Yields
    ------
    tuple
        Contains:
        - pystrata.site.Profile : Site profile with randomized properties
        - float : Site attenuation value
    """
    samples = 10
    for fpath in (FPATH_DATA / "profiles").glob("*.csv.gz"):
        for _ in range(samples):
            profile = create_pystrata_profile(fpath, 0, 3500, randomize_nl_curves=True)
            site_atten = np.exp(np.random.normal(np.log(0.04), 0.5))
            yield profile, site_atten


@pytest.mark.parametrize("profile, site_atten", iter_profiles())
def test_adjust_profile_damping(profile, site_atten):
    """Test profile damping adjustment matches target site attenuation.

    Parameters
    ----------
    profile : pystrata.site.Profile
        Input site profile to adjust
    site_atten : float
        Target site attenuation value to match

    Raises
    ------
    AssertionError
        If adjusted profile damping does not match target within 1% tolerance
    """
    adjusted, site_atten_scatter = pystrata.tools.adjust_damping_values(
        profile, site_atten
    )
    assert np.isclose(
        adjusted.site_attenuation() + site_atten_scatter, site_atten, rtol=0.01
    )


@pytest.mark.parametrize("exclude", ["Mid", "Rock"])
def test_adjust_damping_with_excluded_layers(exclude):
    """The target is met when layers are excluded from the adjustment.

    Differencing the depths of the selected layers spans an excluded middle layer, and
    gives the last selected layer no thickness if the half-space is excluded; either way
    the adjusted profile misses the target.
    """

    def layer(name, thick, vel):
        soil = pystrata.site.SoilType(name, 18.0, None, 0.02)
        return pystrata.site.Layer(soil, thick, vel)

    profile = pystrata.site.Profile(
        [
            layer("Top", 10.0, 200.0),
            layer("Mid", 20.0, 350.0),
            layer("Bot", 30.0, 500.0),
            layer("Rock", 0.0, 1500.0),
        ]
    )
    target = 0.06
    adjusted, scatter = pystrata.tools.adjust_damping_values(
        profile, target, exclude=exclude
    )
    assert np.isclose(adjusted.site_attenuation() + scatter, target, rtol=1e-3)


# SHAKE91 input files
FPATH_SHAKE = FPATH_DATA / "shake_minimal.inp"


def test_load_shake_inp():
    inp = pystrata.tools.load_shake_inp(FPATH_SHAKE)
    profile = inp["profile"]

    np.testing.assert_allclose([layer.thickness for layer in profile], [2, 3, 0])
    np.testing.assert_allclose(profile.initial_shear_vel, [150, 250, 800])
    np.testing.assert_allclose([layer.unit_wt for layer in profile], [17, 18, 22])

    # The strains and damping are tabulated in percent
    mod_reduc = inp["curves"][(1, "mod_reduc")]
    damping = inp["curves"][(1, "damping")]
    np.testing.assert_allclose(mod_reduc.strains, [1e-6, 1e-4, 1e-2, 1e-1])
    np.testing.assert_allclose(mod_reduc.values, [1.0, 0.95, 0.5, 0.1])
    np.testing.assert_allclose(damping.values, [0.01, 0.02, 0.12, 0.25])
    assert damping.values.max() < 0.5

    calc = inp["run_control"]
    assert isinstance(calc, pystrata.propagation.EquivalentLinearCalculator)
    assert calc.strain_ratio == pytest.approx(0.65)
    assert calc.max_iterations == 10

    loc = inp["input_loc"]
    assert loc.index == 2
    assert loc.wave_field == pystrata.motion.WaveField.outcrop

    # The motion is scaled, flattened, and uses the file's FFT length
    motion = inp["motion"]
    assert motion.accels.shape == (16,)
    assert motion.accels.max() == pytest.approx(0.2)
    assert len(motion.freqs) == 32 // 2 + 1
    assert "output" not in inp


def _shake_text(units="METRIC", length=1.0, unit_wt=1.0, outputs=False, end=True):
    """Write the minimal input with the numbers scaled to another unit system."""
    lines = ["Minimal SHAKE91 input for testing", units, "    1", "    1"]
    for name, values in [
        ("G/Gmax", [1.0, 0.95, 0.5, 0.1]),
        ("Damping", [1.0, 2.0, 12.0, 25.0]),
    ]:
        lines.append(f"{4:5d}{name:<65s}")
        lines.append("".join(f"{v:10.4f}" for v in [0.0001, 0.01, 1.0, 10.0]))
        lines.append("".join(f"{v:10.4f}" for v in values))
    lines += [f"{1:5d}{1:5d}", "    2", f"{1:5d}{3:5d}{0:5d}{'Profile':<55s}"]
    for i, (thick, wt, vs) in enumerate([(2, 17, 150), (3, 18, 250), (0, 22, 800)]):
        lines.append(
            f"{i + 1:5d}{1:5d}{thick / length:15.6f}{4000.0:10.1f}{0.05:10.3f}"
            f"{wt / unit_wt:10.7f}{vs / length:10.3f}"
        )
    lines += ["    3", f"{1:5d}{32:5d}{0.01:10.4f}{'shake_minimal.acc':<30s}"]
    lines[-1] += f"{'(8F10.6)':<30s}"
    lines += [f"{2.0:10.4f}{'':20s}{0:5d}{0:5d}", "    4", f"{3:5d}{0:5d}"]
    lines += ["    5", f"{1:5d}{10:5d}{0.65:10.4f}{0:5d}{0:5d}"]
    if outputs:
        # Output blocks are skipped; their contents are not interpreted
        for option in (6, 7, 9):
            lines += [f"{option:5d}", f"{1:5d}{1:5d}{1:5d}"]
    if end:
        lines.append("    0")
    return "\n".join(lines) + "\n"


def _write_shake(tmp_path, **kwds):
    fpath = tmp_path / "test.inp"
    fpath.write_text(_shake_text(**kwds))
    (tmp_path / "shake_minimal.acc").write_text(
        (FPATH_DATA / "shake_minimal.acc").read_text()
    )
    return fpath


def test_load_shake_inp_english_units(tmp_path):
    """English unit weights are in kcf and lengths in ft."""
    ft, kcf = 0.3048, 157.08746
    fpath = _write_shake(tmp_path, units="ENGLISH", length=ft, unit_wt=kcf)
    profile = pystrata.tools.load_shake_inp(fpath)["profile"]

    np.testing.assert_allclose(
        [layer.thickness for layer in profile], [2, 3, 0], rtol=1e-5
    )
    np.testing.assert_allclose(profile.initial_shear_vel, [150, 250, 800], rtol=1e-5)
    np.testing.assert_allclose(
        [layer.unit_wt for layer in profile], [17, 18, 22], rtol=1e-5
    )


def test_load_shake_inp_skips_outputs_and_keeps_the_last_block(tmp_path):
    with_outputs = pystrata.tools.load_shake_inp(_write_shake(tmp_path, outputs=True))
    assert with_outputs["output"] is None
    assert with_outputs["run_control"].max_iterations == 10

    # The last block is kept without a terminating option 0
    no_end = pystrata.tools.load_shake_inp(_write_shake(tmp_path, end=False))
    assert no_end["run_control"].max_iterations == 10
