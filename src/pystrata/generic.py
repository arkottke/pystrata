"""Generic velocity profiles."""

from __future__ import annotations

import gzip
import json
import os
import urllib.request
from pathlib import Path

import numpy as np
import pandas as pd

from .tools import calc_mean_eff_stress


def aaa21_profile(
    model: str,
    v_s30: int,
    simplify: bool = True,
    simplify_tol: float = 0.02,
) -> pd.DataFrame:
    """Load a generic shear-wave velocity profile from Ahdi, Ancheta & Abrahamson
    (2021).

    Parameters
    ----------
    model : str
        Ground-motion model identifier.  One of ``'ASK14'``, ``'BSSA14'``,
        ``'CB14'``, or ``'CY14'``.
    v_s30 : int
        Reference VS30 value in m/s.  One of ``620``, ``760``, or ``1100``.
    simplify : bool, optional
        If True (default), merge adjacent layers whose shear-wave velocities
        differ by less than *simplify_tol*.
    simplify_tol : float, optional
        Relative tolerance on velocity for merging adjacent layers
        (default 0.02, i.e. 2 %).

    Returns
    -------
    pd.DataFrame
        DataFrame with columns ``depth`` [m] and ``vel_shear`` [m/s].

    Raises
    ------
    KeyError
        If *model* or *v_s30* is not found in the data file.

    References
    ----------
    Al Atik, L., & Abrahamson, N. (2021). "A methodology for the development
    of 1D reference VS profiles compatible with ground-motion prediction
    equations: Application to NGA-West2 GMPEs." *Bulletin of the Seismological
    Society of America*, 111(4), 1765–1783.
    https://doi.org/10.1785/0120200312
    """
    fpath = Path(__file__).parent / "data" / "aa21-profiles.json.gz"
    with gzip.open(fpath, "rt") as f:
        data = json.load(f)

    model = model.upper()
    if model not in data:
        raise KeyError(f"Unknown model '{model}'. Choose from: {list(data.keys())}")

    vs30_key = f"vs30_{v_s30}_mps"
    if vs30_key not in data[model]:
        raise KeyError(
            f"Unknown v_s30={v_s30}. Choose from: "
            f"{[k.split('_')[1] for k in data[model]]}"
        )

    profile = data[model][vs30_key]
    depth_m = np.array(profile["depth_km"]) * 1000
    vs_mps = np.array(profile["vs_km_per_sec"]) * 1000

    df = pd.DataFrame({"depth": depth_m, "vel_shear": vs_mps})

    if simplify:
        df = _simplify_profile(df, simplify_tol)

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
        with gzip.open(fpath, "wt") as f:
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
        with gzip.open(fpath, "rt") as f:
            data = json.load(f)
    except gzip.BadGzipFile:
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
        Must contain ``depth`` and ``vel_shear`` columns.  If ``unit_wt`` is
        present it will also be averaged.
    tol : float
        Relative tolerance for grouping (e.g. 0.02 = 2 %).

    Returns
    -------
    pandas.DataFrame
        Simplified profile with the same columns.
    """
    depths = df["depth"].values
    vs = df["vel_shear"].values
    has_unit_wt = "unit_wt" in df.columns
    if has_unit_wt:
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

        row = {"depth": depth_top, "vel_shear": vs_avg}
        if has_unit_wt:
            row["unit_wt"] = np.average(
                unit_wt[group], weights=np.maximum(thick, 1e-10)
            )
        rows.append(row)

    return pd.DataFrame(rows)
