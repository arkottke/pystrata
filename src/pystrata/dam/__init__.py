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
from .canyon import (
    RectangularCanyon,
    canyon_freq_ratio,
    canyon_rect_natural_freqs,
    canyon_rect_rigid_base_tf,
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
    "RectangularCanyon",
    "WedgeLayer",
    "canyon_freq_ratio",
    "canyon_rect_natural_freqs",
    "canyon_rect_rigid_base_tf",
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
