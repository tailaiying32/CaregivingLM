"""
Clinical Rules Module for SIRS Sampling.

This module provides clinical filtering rules to constrain the 4D joint space
for shoulder + elbow movement sampling.

Main components:
- joint_config: Joint limits and discretization (D0, S1)
- filters: Clinical filtering pipeline (B3, R4, C5, E6)
- coupling_rules: Allowed/disallowed coupling pairs (G7)
- connectivity: Seed-based connectivity enforcement (K8)
"""

from .joint_config import (
    JOINT_LIMITS_DEG,
    JOINT_NAME_MAP,
    GRID_STEP_DEG,
    MIN_SPAN_DEG,
    quantize_to_grid,
    quantize_joint_limits,
    check_box_sanity,
)

from .filters import (
    ClinicalFilterPipeline,
    ScapularPlaneBandFilter,
    ElevationRotationFilter,
    CrossBodyPruneFilter,
    ElbowElevationFilter,
)

from .coupling_rules import (
    ALLOWED_COUPLING_PAIRS,
    DISALLOWED_COUPLING_PAIRS,
    MAX_COUPLING_STRUCTURES,
    get_implementation_pairs,
    filter_to_allowed_pairs,
    is_pair_allowed,
    get_recommended_joint_pairs,
)

from .connectivity import (
    SEED_REGION,
    points_in_region,
    enforce_connectivity,
    find_closest_feasible_to_neutral,
    enforce_connectivity_2d,
)

from .archetypes import (
    ImpairmentArchetype,
    ARCHETYPES,
    get_archetype,
    list_archetypes,
    apply_archetype_to_limits,
)

__all__ = [
    # joint_config
    'JOINT_LIMITS_DEG',
    'JOINT_NAME_MAP',
    'GRID_STEP_DEG',
    'MIN_SPAN_DEG',
    'quantize_to_grid',
    'quantize_joint_limits',
    'check_box_sanity',
    # filters
    'ClinicalFilterPipeline',
    'ScapularPlaneBandFilter',
    'ElevationRotationFilter',
    'CrossBodyPruneFilter',
    'ElbowElevationFilter',
    # coupling_rules
    'ALLOWED_COUPLING_PAIRS',
    'DISALLOWED_COUPLING_PAIRS',
    'MAX_COUPLING_STRUCTURES',
    'get_implementation_pairs',
    'filter_to_allowed_pairs',
    'is_pair_allowed',
    'get_recommended_joint_pairs',
    # connectivity
    'SEED_REGION',
    'points_in_region',
    'enforce_connectivity',
    'find_closest_feasible_to_neutral',
    'enforce_connectivity_2d',
    # archetypes
    'ImpairmentArchetype',
    'ARCHETYPES',
    'get_archetype',
    'list_archetypes',
    'apply_archetype_to_limits',
]
