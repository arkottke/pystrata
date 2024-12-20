from collections import Counter

import numpy as np
import pandas as pd
import pystrata
import pytest
from scipy.constants import g as GRAVITY

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
            soil_type = pystrata.site.WangSoilType(
                row["material"],
                row["material"] + "-" + str(counter[row["material"]]),
                GRAVITY * row["density"],
                damping_min=0,
                stress_mean=row["stress_mean_eff"],
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
