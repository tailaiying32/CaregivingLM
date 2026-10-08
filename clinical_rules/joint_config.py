"""
Joint Configuration and Discretization.

Defines joint limits from the clinical rulebook and provides
discretization (D0) and box sanity checks (S1).
"""

import numpy as np
from typing import Dict, Tuple, Optional, Union


# =============================================================================
# Joint Limits (degrees) - from clinical rulebook
# =============================================================================

JOINT_LIMITS_DEG = {
    'shoulder_FE': (0, 180),        # shoulder_flexion_r: forward elevation
    'shoulder_AA': (-170, 150),     # shoulder_abduction_r: adduction-abduction
    'shoulder_ROT': (-120, 180),    # shoulder_rotation_r: internal-external rotation
    'elbow_FLEX': (-20, 170),       # elbow_flexion_r: elbow flexion
}

# Mapping from clinical abbreviations to actual joint names in the model
JOINT_NAME_MAP = {
    'shoulder_FE': 'shoulder_flexion_r',
    'shoulder_AA': 'shoulder_abduction_r',
    'shoulder_ROT': 'shoulder_rotation_r',
    'elbow_FLEX': 'elbow_flexion_r',
    # Reverse mapping
    'shoulder_flexion_r': 'shoulder_FE',
    'shoulder_abduction_r': 'shoulder_AA',
    'shoulder_rotation_r': 'shoulder_ROT',
    'elbow_flexion_r': 'elbow_FLEX',
}

# Short name mappings for convenience in filters
SHORT_NAMES = {
    'FE': 'shoulder_flexion_r',
    'AA': 'shoulder_abduction_r',
    'ROT': 'shoulder_rotation_r',
    'EL': 'elbow_flexion_r',
}

LONG_TO_SHORT = {v: k for k, v in SHORT_NAMES.items()}


# =============================================================================
# Discretization Parameters (D0)
# =============================================================================

GRID_STEP_DEG = 10.0  # 10-degree grid for quantization


def quantize_to_grid(
    x_deg: Union[float, np.ndarray],
    limits: Optional[Tuple[float, float]] = None
) -> Union[float, np.ndarray]:
    """
    Quantize values to 10-degree grid: x_q = round(x / 10) * 10

    Args:
        x_deg: Value(s) in degrees to quantize
        limits: Optional (min, max) limits to clip to after quantization

    Returns:
        Quantized value(s) in degrees
    """
    x_q = np.round(x_deg / GRID_STEP_DEG) * GRID_STEP_DEG
    if limits is not None:
        x_q = np.clip(x_q, limits[0], limits[1])
    return x_q


def quantize_joint_limits(
    joint_limits: Dict[str, Tuple[float, float]],
    degrees: bool = True
) -> Dict[str, Tuple[float, float]]:
    """
    Quantize all joint limits to 10-degree grid.

    Args:
        joint_limits: Dictionary mapping joint name to (lower, upper) tuple
        degrees: If False, input is in radians and will be converted

    Returns:
        Dictionary with quantized joint limits (in same units as input)
    """
    quantized = {}

    for joint_name, (lower, upper) in joint_limits.items():
        # Convert to degrees if needed
        if not degrees:
            lower_deg = np.degrees(lower)
            upper_deg = np.degrees(upper)
        else:
            lower_deg = lower
            upper_deg = upper

        # Quantize
        lower_q = quantize_to_grid(lower_deg)
        upper_q = quantize_to_grid(upper_deg)

        # Ensure lower < upper
        if lower_q >= upper_q:
            upper_q = lower_q + GRID_STEP_DEG

        # Convert back to radians if needed
        if not degrees:
            lower_q = np.radians(lower_q)
            upper_q = np.radians(upper_q)

        quantized[joint_name] = (lower_q, upper_q)

    return quantized


# =============================================================================
# Box Sanity Checks (S1)
# =============================================================================

MIN_SPAN_DEG = 20.0  # Minimum span per DOF in degrees
MIN_GRID_POINTS = 81  # Optional: at least 3^4 points for 4D grid


def check_box_sanity(
    joint_limits: Dict[str, Tuple[float, float]],
    degrees: bool = True,
    min_span_deg: float = MIN_SPAN_DEG,
    verbose: bool = False
) -> Tuple[bool, str]:
    """
    Reject boxes with span < MIN_SPAN_DEG per DOF.

    Args:
        joint_limits: Dictionary mapping joint name to (lower, upper) tuple
        degrees: If False, input is in radians
        min_span_deg: Minimum allowed span in degrees
        verbose: Print rejection reasons

    Returns:
        Tuple of (is_valid, message)
    """
    for joint_name, (lower, upper) in joint_limits.items():
        # Convert to degrees if needed
        if not degrees:
            lower_deg = np.degrees(lower)
            upper_deg = np.degrees(upper)
        else:
            lower_deg = lower
            upper_deg = upper

        span = upper_deg - lower_deg

        if span < min_span_deg:
            msg = f"{joint_name} span {span:.1f}° < {min_span_deg}°"
            if verbose:
                print(f"Box sanity check failed: {msg}")
            return False, msg

    return True, "OK"


def degrees_to_radians_limits(
    joint_limits: Dict[str, Tuple[float, float]]
) -> Dict[str, Tuple[float, float]]:
    """Convert joint limits from degrees to radians."""
    return {
        name: (np.radians(lo), np.radians(hi))
        for name, (lo, hi) in joint_limits.items()
    }


def radians_to_degrees_limits(
    joint_limits: Dict[str, Tuple[float, float]]
) -> Dict[str, Tuple[float, float]]:
    """Convert joint limits from radians to degrees."""
    return {
        name: (np.degrees(lo), np.degrees(hi))
        for name, (lo, hi) in joint_limits.items()
    }


def get_clinical_limits_rad() -> Dict[str, Tuple[float, float]]:
    """Get clinical joint limits in radians using model joint names."""
    return {
        SHORT_NAMES[k.split('_')[1] if '_' in k else k]:
        (np.radians(lo), np.radians(hi))
        for k, (lo, hi) in JOINT_LIMITS_DEG.items()
    }


# =============================================================================
# Helper Functions
# =============================================================================

def to_short_name(joint_name: str) -> str:
    """Convert long joint name to short form (FE, AA, ROT, EL)."""
    return LONG_TO_SHORT.get(joint_name, joint_name)


def to_long_name(short_name: str) -> str:
    """Convert short joint name to long form (shoulder_flexion_r, etc.)."""
    return SHORT_NAMES.get(short_name, short_name)
