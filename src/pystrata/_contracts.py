"""Consumer-side data contracts for pystrata ↔ pygmm interop.

Canonical definitions live in ``pygmm.contracts``; this file is a private
~40-LOC duplicate so pystrata has no runtime dependency on pygmm.
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import Optional

import numpy as np
import numpy.typing as npt


@dataclass(frozen=True)
class NonlinearSoilCurves:
    """Strain-dependent modulus-reduction and damping curves.

    Field names match ``pygmm.contracts.NonlinearSoilCurves``.
    """

    strains: npt.NDArray[np.floating]
    mod_reduc: npt.NDArray[np.floating]
    damping: npt.NDArray[np.floating]
    damping_min: float
    unit_wt: Optional[float] = None
    name: Optional[str] = None


@dataclass(frozen=True)
class VelocityProfile:
    """Shear-wave velocity profile.

    Field names match ``pygmm.contracts.VelocityProfile``.
    """

    depth: npt.NDArray[np.floating]
    vs_median: npt.NDArray[np.floating]
    std_vs_ln: npt.NDArray[np.floating]
    region: Optional[str] = None
    site_class: Optional[str] = None
