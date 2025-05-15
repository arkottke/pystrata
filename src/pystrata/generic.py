"""Generic velocity profiles."""

import warnings

import numpy as np
import pandas as pd


def kea16_profile(depth: float, vs30: float, region: str) -> pd.DataFrame:
    """
    Calculate the median shear-wave velocity profile and its standard deviation
    at a given depth, based on the VS30 value and region, using the model from
    Kamai et al. (2016).

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
