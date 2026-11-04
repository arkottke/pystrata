"""Seismic response of dams modeled as shear wedges.

Based on :cite:t:`gazetas1992seismic`.
"""

from .analytical import (
    wedge_elastic_base_tf,
    wedge_mode_shape,
    wedge_natural_freqs,
    wedge_order,
    wedge_rigid_base_tf,
    wedge_truncated_elastic_base_tf,
    wedge_truncated_natural_freqs,
    wedge_truncated_rigid_base_tf,
)
from .model import Dam, DamProfile, WedgeLayer
from .propagation import (
    DamEquivalentLinearCalculator,
    DamFrequencyDependentEqlCalculator,
    DamLinearElasticCalculator,
)
from .sliding import (
    CircularSurface,
    plot_failure_surface,
    sliding_mass_transfer_function,
    toe_circle,
)

__all__ = [
    "CircularSurface",
    "Dam",
    "DamEquivalentLinearCalculator",
    "DamFrequencyDependentEqlCalculator",
    "DamLinearElasticCalculator",
    "DamProfile",
    "WedgeLayer",
    "plot_failure_surface",
    "sliding_mass_transfer_function",
    "toe_circle",
    "wedge_elastic_base_tf",
    "wedge_mode_shape",
    "wedge_natural_freqs",
    "wedge_order",
    "wedge_rigid_base_tf",
    "wedge_truncated_elastic_base_tf",
    "wedge_truncated_natural_freqs",
    "wedge_truncated_rigid_base_tf",
]
