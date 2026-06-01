"""Generic velocity profiles."""

from __future__ import annotations

import gzip as _gzip
import json
import os
import urllib.request
import warnings

import numpy as np
import numpy.typing as npt
import pandas as pd

from .tools import calc_mean_eff_stress


def kea16_profile(depth: npt.ArrayLike, vs30: float, region: str) -> pd.DataFrame:
    """Calculate the median shear-wave velocity profile and its standard deviation at a
    given depth, based on the VS30 value and region, using the model from Kamai et al.
    (2016).

    Parameters
    ----------
    depth : float or np.ndarray
        The depth(s) in meters at which to calculate the profile.
    vs30 : float
        The VS30 value in m/sec.
    region : str
        The region for the model. Must be either 'California' or 'Japan'.

    Returns
    -------
    pd.DataFrame
        DataFrame with index as depth and columns:
        - 'vs_median': float or np.ndarray
            The median shear-wave velocity at the given depth(s) in m/sec.
        - 'std_vs_ln_units': float
            The standard deviation of the shear-wave velocity in ln units.

    Raises
    ------
    ValueError
        If the region is not 'California' or 'Japan'.

    Warns
    -----
    UserWarning
        If vs30 is outside the recommended range [250, 850] m/sec.

    References
    ----------
    Kamai, R., et al. (2016). "Nonlinear site response from the KiK-net
    database." Bulletin of the Seismological Society of America, 106(4),
    1710-1723.
    """
    # Coefficients from Table 2 [cite: 184, 185]
    if region == "california":
        b1 = 0.25
        b2 = 104
        b3 = 0.22
        b4 = 166
        b5 = 53
        b6 = -0.00388
        b7 = 0.0002
        b8 = 0.195
    elif region == "japan":
        b1 = 0.2
        b2 = 40
        b3 = 0.25
        b4 = 260
        b5 = 28
        b6 = -0.00296
        b7 = 0
        b8 = 0.4
    else:
        raise ValueError("Region must be 'California' or 'Japan'")

    if not (250 <= vs30 <= 850):
        # The model is recommended for VS30 between 250 and 850 m/sec.
        warnings.warn(
            "The model is recommended for VS30 between 250 and 850 m/sec. "
            "Values outside this range may lead to inaccurate results.",
            UserWarning,
        )

    a0 = b1 * vs30 + b2  # Equation 3a
    a1 = b3 * vs30 + b4  # Equation 3b
    a2 = b5 * np.exp(b6 * vs30)  # Equation 3c

    # Ensure the argument to log is positive and handle depth=0 or negative depths
    log_arg = (np.array(depth) + a2) / a2
    vs_median = a0 + a1 * np.log(
        np.maximum(log_arg, 1e-9)
    )  # Equation 3, using maximum to handle non-positive log arguments

    a3 = b7 * vs30 + b8  # Equation 4a
    std_vs_ln_units = a3  # Equation 4

    df = pd.DataFrame(
        {"vs_median": vs_median, "std_vs_ln_units": std_vs_ln_units},
        index=pd.Index(depth, name="depth"),
    )

    return df


NCM_BASE_URL = "https://earthquake.usgs.gov/ws/nshmp/ncm/geophysical"


def fetch_ncm_profile(
    latitude: float,
    longitude: float,
    fpath: str | os.PathLike,
    depth_start: float = 0,
    depth_step: float = 5,
    depth_end: float = 1e4,
    gzip_output: bool = True,
) -> None:
    """Fetch a geophysical profile from the USGS National Crustal Model.

    Queries the NCM geophysical endpoint for a single location and saves the
    JSON response to *fpath*.

    Parameters
    ----------
    latitude : float
        Latitude of the site [degrees].
    longitude : float
        Longitude of the site [degrees].
    fpath : str or path-like
        Output file path.  If *gzip_output* is True the file is gzip-compressed.
    depth_start : float, optional
        Minimum depth [m] (default 0).
    depth_step : float, optional
        Depth increment [m] (default 5).
    depth_end : float, optional
        Maximum depth [m] (default 10 000).
    gzip_output : bool, optional
        If True (default), write a gzip-compressed JSON file.

    Raises
    ------
    urllib.error.HTTPError
        If the request to the NCM service fails.
    RuntimeError
        If the response status is not ``"sucess"`` (sic).
    """
    url = (
        f"{NCM_BASE_URL}"
        f"?location={latitude},{longitude}"
        f"&depths={depth_start},{depth_step},{depth_end}"
    )
    with urllib.request.urlopen(url) as resp:
        data = json.loads(resp.read().decode())

    if data.get("status") not in ("sucess", "success"):
        raise RuntimeError(f"NCM request failed with status: {data.get('status')}")

    fpath = os.fspath(fpath)
    if gzip_output:
        with _gzip.open(fpath, "wt") as f:
            json.dump(data, f)
    else:
        with open(fpath, "w") as f:
            json.dump(data, f)


def load_ncm_profile(
    fpath: str | os.PathLike,
    simplify: bool = False,
    simplify_tol: float = 0.02,
) -> tuple[pd.DataFrame, float]:
    """Load an NCM geophysical profile and return a DataFrame.

    Parameters
    ----------
    fpath : str or path-like
        Path to the NCM JSON file (plain or gzip-compressed).
    simplify : bool, optional
        If True, merge adjacent layers whose shear-wave velocities differ by
        less than *simplify_tol* (default False).
    simplify_tol : float, optional
        Relative tolerance on ``vel_shear`` for merging adjacent layers
        (default 0.02, i.e. 2 %).

    Returns
    -------
    df : pandas.DataFrame
        DataFrame with columns ``depth`` [m], ``vel_shear`` [m/s], and
        ``density`` [kg/m³].
    water_table_depth : float
        Depth to the water table [m].

    Raises
    ------
    ValueError
        If the file cannot be parsed or contains no results.
    """
    fpath = os.fspath(fpath)

    # Try gzip first, fall back to plain JSON
    try:
        with _gzip.open(fpath, "rt") as f:
            data = json.load(f)
    except _gzip.BadGzipFile:
        with open(fpath) as f:
            data = json.load(f)

    results = data["response"]["results"]
    if not results:
        raise ValueError("NCM response contains no results")

    result = results[0]
    water_table_depth = result["location"]["water_table_depth"]

    depths = np.array(data["request"]["depths"]["depth_vector"], dtype=float)
    vs = np.array(result["profile"]["vs"], dtype=float)
    density = np.array(result["profile"]["density"], dtype=float)
    unit_wt = density * 9.81 / 1000

    df = pd.DataFrame({"depth": depths, "vel_shear": vs, "unit_wt": unit_wt})

    if simplify:
        df = _simplify_profile(df, simplify_tol)

    # Compute mean effective stress
    df["mean_eff_stress"] = calc_mean_eff_stress(
        df["depth"].values, df["unit_wt"].values, water_table_depth
    )
    df["thickness"] = np.diff(df["depth"], append=df["depth"].iloc[-1])

    return df, water_table_depth


def _simplify_profile(
    df: pd.DataFrame,
    tol: float,
) -> pd.DataFrame:
    """Merge adjacent layers with similar shear-wave velocity.

    Within each group of consecutive layers whose ``vel_shear`` values are
    within *tol* of the first layer in the group, the time-averaged shear-wave
    velocity and the thickness-weighted average unit weight are computed.

    Parameters
    ----------
    df : pandas.DataFrame
        Must contain ``depth``, ``vel_shear``, and ``unit_wt`` columns.
    tol : float
        Relative tolerance for grouping (e.g. 0.02 = 2 %).

    Returns
    -------
    pandas.DataFrame
        Simplified profile with the same columns.
    """
    depths = df["depth"].values
    vs = df["vel_shear"].values
    unit_wt = df["unit_wt"].values

    # Compute layer thicknesses (last layer gets 0)
    thicknesses = np.diff(depths, append=depths[-1])

    groups: list[list[int]] = []
    current_group: list[int] = [0]

    for i in range(1, len(vs)):
        ref_vs = vs[current_group[0]]
        if abs(vs[i] - ref_vs) / ref_vs <= tol:
            current_group.append(i)
        else:
            groups.append(current_group)
            current_group = [i]
    groups.append(current_group)

    rows = []
    for group in groups:
        depth_top = depths[group[0]]
        thick = thicknesses[group]
        total_thick = thick.sum()

        if total_thick > 0:
            # Time-average velocity: Vs_avg = total_thickness / sum(thickness_i / Vs_i)
            vs_avg = total_thick / np.sum(thick / vs[group])
        else:
            # Halfspace (last layer)
            vs_avg = vs[group[0]]

        unit_wt_avg = np.average(unit_wt[group], weights=np.maximum(thick, 1e-10))
        rows.append({"depth": depth_top, "vel_shear": vs_avg, "unit_wt": unit_wt_avg})

    return pd.DataFrame(rows)
