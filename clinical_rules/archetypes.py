"""
Impairment Archetypes (A9) - Optional.

Provides predefined impairment configurations that can be used to
generate samples representing common clinical presentations.

This module is optional and can be used to create samples that
represent specific patient populations or impairment types.
"""

import numpy as np
from typing import Dict, List, Tuple, Optional
from dataclasses import dataclass


@dataclass
class ImpairmentArchetype:
    """
    Defines an impairment archetype with modified joint limits and filter parameters.
    """
    name: str
    description: str

    # Modified joint limits (as factors of normal range)
    # e.g., 0.5 means limit to 50% of normal range
    fe_range_factor: float = 1.0
    aa_range_factor: float = 1.0
    rot_range_factor: float = 1.0
    el_range_factor: float = 1.0

    # Shifted neutral positions (in degrees)
    fe_offset: float = 0.0
    aa_offset: float = 0.0
    rot_offset: float = 0.0
    el_offset: float = 0.0

    # Filter modifications
    scapular_band_width_factor: float = 1.0  # Multiply half-widths
    rotation_threshold_factor: float = 1.0   # Multiply rotation thresholds

    # Whether specific filters are enabled
    enable_B3: bool = True
    enable_R4: bool = True
    enable_C5: bool = True
    enable_E6: bool = True


# =============================================================================
# Predefined Archetypes
# =============================================================================

ARCHETYPES = {
    'healthy': ImpairmentArchetype(
        name='healthy',
        description='No impairment - full range of motion',
    ),

    'frozen_shoulder_mild': ImpairmentArchetype(
        name='frozen_shoulder_mild',
        description='Mild frozen shoulder - reduced ROM especially in rotation',
        fe_range_factor=0.8,
        aa_range_factor=0.7,
        rot_range_factor=0.5,
        el_range_factor=1.0,
        scapular_band_width_factor=0.7,
    ),

    'frozen_shoulder_severe': ImpairmentArchetype(
        name='frozen_shoulder_severe',
        description='Severe frozen shoulder - significantly reduced ROM',
        fe_range_factor=0.5,
        aa_range_factor=0.4,
        rot_range_factor=0.3,
        el_range_factor=0.9,
        scapular_band_width_factor=0.5,
        rotation_threshold_factor=0.5,
    ),

    'rotator_cuff_tear': ImpairmentArchetype(
        name='rotator_cuff_tear',
        description='Rotator cuff tear - limited elevation and rotation',
        fe_range_factor=0.7,
        aa_range_factor=0.6,
        rot_range_factor=0.6,
        el_range_factor=1.0,
        rotation_threshold_factor=0.7,
    ),

    'post_stroke_flexion': ImpairmentArchetype(
        name='post_stroke_flexion',
        description='Post-stroke with flexor synergy pattern',
        fe_range_factor=0.6,
        aa_range_factor=0.5,
        rot_range_factor=0.4,
        el_range_factor=0.7,
        fe_offset=20.0,  # Tendency toward flexion
        el_offset=30.0,  # Elbow tends to flex
        scapular_band_width_factor=0.6,
    ),

    'elbow_contracture': ImpairmentArchetype(
        name='elbow_contracture',
        description='Elbow flexion contracture - limited extension',
        fe_range_factor=1.0,
        aa_range_factor=1.0,
        rot_range_factor=1.0,
        el_range_factor=0.5,
        el_offset=40.0,  # Can't fully extend
    ),

    'elderly_normal': ImpairmentArchetype(
        name='elderly_normal',
        description='Age-related ROM reduction without pathology',
        fe_range_factor=0.85,
        aa_range_factor=0.85,
        rot_range_factor=0.8,
        el_range_factor=0.95,
        scapular_band_width_factor=0.9,
    ),
}


def get_archetype(name: str) -> ImpairmentArchetype:
    """
    Get a predefined impairment archetype by name.

    Args:
        name: Name of the archetype

    Returns:
        ImpairmentArchetype instance

    Raises:
        KeyError: If archetype name not found
    """
    if name not in ARCHETYPES:
        raise KeyError(f"Unknown archetype: {name}. Available: {list(ARCHETYPES.keys())}")
    return ARCHETYPES[name]


def list_archetypes() -> List[str]:
    """Return list of available archetype names."""
    return list(ARCHETYPES.keys())


def apply_archetype_to_limits(
    joint_limits: Dict[str, Tuple[float, float]],
    archetype: ImpairmentArchetype,
    degrees: bool = True
) -> Dict[str, Tuple[float, float]]:
    """
    Apply an impairment archetype to modify joint limits.

    Args:
        joint_limits: Original joint limits dictionary
        archetype: Impairment archetype to apply
        degrees: If True, limits are in degrees; if False, radians

    Returns:
        Modified joint limits dictionary
    """
    # Mapping from joint names to archetype factors and offsets
    mapping = {
        'shoulder_flexion_r': ('fe', archetype.fe_range_factor, archetype.fe_offset),
        'shoulder_abduction_r': ('aa', archetype.aa_range_factor, archetype.aa_offset),
        'shoulder_rotation_r': ('rot', archetype.rot_range_factor, archetype.rot_offset),
        'elbow_flexion_r': ('el', archetype.el_range_factor, archetype.el_offset),
    }

    modified = {}

    for joint_name, (lower, upper) in joint_limits.items():
        if joint_name in mapping:
            _, range_factor, offset = mapping[joint_name]

            # Convert offset to radians if needed
            if not degrees:
                offset = np.radians(offset)

            # Compute range and center
            range_val = upper - lower
            center = (upper + lower) / 2

            # Apply range reduction around shifted center
            new_center = center + offset
            new_range = range_val * range_factor
            new_lower = new_center - new_range / 2
            new_upper = new_center + new_range / 2

            modified[joint_name] = (new_lower, new_upper)
        else:
            # Keep unchanged
            modified[joint_name] = (lower, upper)

    return modified


def get_modified_filter_params(archetype: ImpairmentArchetype) -> Dict:
    """
    Get filter parameters modified by archetype.

    Args:
        archetype: Impairment archetype

    Returns:
        Dictionary of filter parameters to pass to ClinicalFilterPipeline
    """
    return {
        'enable_B3': archetype.enable_B3,
        'enable_R4': archetype.enable_R4,
        'enable_C5': archetype.enable_C5,
        'enable_E6': archetype.enable_E6,
        # Note: band width and threshold factors would need filter modifications
        # to support. For now, this returns the enable flags.
    }
