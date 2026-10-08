"""
Seed-Based Connectivity Enforcement (K8).

Ensures that the feasible region is connected to a neutral seed region,
preventing isolated feasible pockets that may be unreachable.
"""

import numpy as np
from typing import Dict, Tuple, Optional, Union
from scipy import ndimage


# =============================================================================
# Seed Region Definition
# =============================================================================

# Primary seed region (grid-aligned, in degrees)
# This is a neutral position that should always be reachable
SEED_REGION = {
    'FE': (0, 20),      # Near neutral forward elevation
    'AA': (-10, 20),    # Near neutral abduction
    'ROT': (-20, 20),   # Near neutral rotation
    'EL': (10, 40),     # Slightly flexed elbow (natural resting)
}

# Neutral point (for fallback if no seed in region)
NEUTRAL_POINT = {
    'FE': 10.0,
    'AA': 0.0,
    'ROT': 0.0,
    'EL': 20.0,
}


# =============================================================================
# Connectivity Functions
# =============================================================================

def points_in_region(
    grid_points: Dict[str, np.ndarray],
    region: Dict[str, Tuple[float, float]]
) -> np.ndarray:
    """
    Find which grid points are within a specified region.

    Args:
        grid_points: Dictionary with joint names as keys and 1D arrays of values
        region: Dictionary with joint names as keys and (min, max) tuples

    Returns:
        Boolean mask of points within the region
    """
    # Get first key to determine number of points
    first_key = list(grid_points.keys())[0]
    n_points = len(grid_points[first_key])
    mask = np.ones(n_points, dtype=bool)

    for joint, (lo, hi) in region.items():
        if joint in grid_points:
            vals = grid_points[joint]
            mask &= (vals >= lo) & (vals <= hi)

    return mask


def find_closest_feasible_to_neutral(
    feasible_mask: np.ndarray,
    grid_points: Dict[str, np.ndarray],
    neutral: Dict[str, float] = None
) -> int:
    """
    Find the feasible point closest to neutral position.

    This is a fallback when no feasible point exists in the seed region.

    Args:
        feasible_mask: Boolean mask of feasible points
        grid_points: Dictionary with joint angle arrays
        neutral: Neutral position dictionary (default: NEUTRAL_POINT)

    Returns:
        Index of closest feasible point to neutral
    """
    if neutral is None:
        neutral = NEUTRAL_POINT

    feasible_indices = np.where(feasible_mask)[0]
    if len(feasible_indices) == 0:
        raise ValueError("No feasible points found")

    # Compute distances to neutral
    min_dist = np.inf
    closest_idx = feasible_indices[0]

    for idx in feasible_indices:
        dist_sq = 0
        for joint, neutral_val in neutral.items():
            if joint in grid_points:
                dist_sq += (grid_points[joint][idx] - neutral_val) ** 2
        if dist_sq < min_dist:
            min_dist = dist_sq
            closest_idx = idx

    return closest_idx


def enforce_connectivity(
    feasible_mask: np.ndarray,
    grid_shape: Tuple[int, ...],
    grid_points: Dict[str, np.ndarray],
    seed_region: Dict[str, Tuple[float, float]] = None,
    use_fallback: bool = True
) -> np.ndarray:
    """
    Keep only the connected component containing the seed region.

    This implements rule K8 - ensuring connectivity to a neutral seed.

    Args:
        feasible_mask: Boolean mask of feasible points (flat or shaped)
        grid_shape: Shape of the grid (e.g., (n_fe, n_aa, n_rot, n_el))
        grid_points: Dictionary with joint names as keys and arrays of grid values
        seed_region: Seed region definition (default: SEED_REGION)
        use_fallback: If True, use closest-to-neutral fallback (K8.2)

    Returns:
        Boolean mask with only seed-connected feasible points
    """
    if seed_region is None:
        seed_region = SEED_REGION

    # Reshape if necessary
    was_flat = feasible_mask.ndim == 1
    if was_flat:
        feasible_shaped = feasible_mask.reshape(grid_shape)
    else:
        feasible_shaped = feasible_mask

    # Find seed points
    seed_mask = points_in_region(grid_points, seed_region)
    seed_mask_shaped = seed_mask.reshape(grid_shape) if seed_mask.ndim == 1 else seed_mask

    # Check if any feasible point is in seed region
    seed_in_feasible = feasible_shaped & seed_mask_shaped

    if not np.any(seed_in_feasible):
        if use_fallback:
            # K8.2: Use fallback - find closest feasible to neutral
            flat_feasible = feasible_shaped.flatten()
            try:
                fallback_idx = find_closest_feasible_to_neutral(flat_feasible, grid_points)
                # Create seed mask with just this point
                seed_mask_shaped = np.zeros_like(feasible_shaped)
                seed_mask_shaped.flat[fallback_idx] = True
                seed_in_feasible = feasible_shaped & seed_mask_shaped
            except ValueError:
                # No feasible points at all
                if was_flat:
                    return np.zeros(np.prod(grid_shape), dtype=bool)
                return np.zeros(grid_shape, dtype=bool)
        else:
            # No fallback - return empty
            if was_flat:
                return np.zeros(np.prod(grid_shape), dtype=bool)
            return np.zeros(grid_shape, dtype=bool)

    # Label connected components using scipy.ndimage.label
    # Use 4D connectivity structure (face-connected)
    structure = ndimage.generate_binary_structure(len(grid_shape), 1)
    labeled, n_components = ndimage.label(feasible_shaped, structure=structure)

    # Find the label(s) of seed points
    seed_labels = set(labeled[seed_in_feasible])
    seed_labels.discard(0)  # Remove background label

    if len(seed_labels) == 0:
        # This shouldn't happen if seed_in_feasible is non-empty
        if was_flat:
            return np.zeros(np.prod(grid_shape), dtype=bool)
        return np.zeros(grid_shape, dtype=bool)

    # Keep only seed-connected component(s)
    # If multiple seed points connect to different components, keep all
    connected_mask = np.zeros_like(feasible_shaped, dtype=bool)
    for label in seed_labels:
        connected_mask |= (labeled == label)

    if was_flat:
        return connected_mask.flatten()
    return connected_mask


def compute_connectivity_stats(
    original_mask: np.ndarray,
    connected_mask: np.ndarray
) -> Dict[str, Union[int, float]]:
    """
    Compute statistics about connectivity enforcement.

    Args:
        original_mask: Original feasible mask
        connected_mask: Mask after connectivity enforcement

    Returns:
        Dictionary with statistics
    """
    n_original = int(np.sum(original_mask))
    n_connected = int(np.sum(connected_mask))
    n_removed = n_original - n_connected

    return {
        'n_original_feasible': n_original,
        'n_connected_feasible': n_connected,
        'n_removed': n_removed,
        'fraction_retained': n_connected / n_original if n_original > 0 else 0.0,
        'fraction_removed': n_removed / n_original if n_original > 0 else 0.0,
    }


# =============================================================================
# 2D Connectivity for Pairwise Projections
# =============================================================================

def enforce_connectivity_2d(
    feasible_mask_2d: np.ndarray,
    grid_q1: np.ndarray,
    grid_q2: np.ndarray,
    seed_q1_range: Tuple[float, float],
    seed_q2_range: Tuple[float, float],
    use_fallback: bool = True
) -> np.ndarray:
    """
    Enforce connectivity in 2D (for pairwise joint projections).

    Args:
        feasible_mask_2d: 2D boolean mask (n_q1, n_q2)
        grid_q1: 1D array of q1 values
        grid_q2: 1D array of q2 values
        seed_q1_range: (min, max) for q1 seed region
        seed_q2_range: (min, max) for q2 seed region
        use_fallback: If True, use fallback if no seed in feasible

    Returns:
        2D boolean mask with only seed-connected region
    """
    # Create meshgrid for checking seed region
    Q1, Q2 = np.meshgrid(grid_q1, grid_q2, indexing='ij')

    # Find seed points
    seed_mask = (
        (Q1 >= seed_q1_range[0]) & (Q1 <= seed_q1_range[1]) &
        (Q2 >= seed_q2_range[0]) & (Q2 <= seed_q2_range[1])
    )

    seed_in_feasible = feasible_mask_2d & seed_mask

    if not np.any(seed_in_feasible):
        if use_fallback:
            # Find feasible point closest to center of seed region
            center_q1 = (seed_q1_range[0] + seed_q1_range[1]) / 2
            center_q2 = (seed_q2_range[0] + seed_q2_range[1]) / 2

            feasible_indices = np.where(feasible_mask_2d)
            if len(feasible_indices[0]) == 0:
                return np.zeros_like(feasible_mask_2d)

            dist_sq = (Q1[feasible_indices] - center_q1)**2 + (Q2[feasible_indices] - center_q2)**2
            closest_idx = np.argmin(dist_sq)

            seed_mask = np.zeros_like(feasible_mask_2d)
            seed_mask[feasible_indices[0][closest_idx], feasible_indices[1][closest_idx]] = True
            seed_in_feasible = feasible_mask_2d & seed_mask
        else:
            return np.zeros_like(feasible_mask_2d)

    # Label connected components
    structure = ndimage.generate_binary_structure(2, 1)  # 4-connectivity
    labeled, n_components = ndimage.label(feasible_mask_2d, structure=structure)

    # Find seed labels
    seed_labels = set(labeled[seed_in_feasible])
    seed_labels.discard(0)

    if len(seed_labels) == 0:
        return np.zeros_like(feasible_mask_2d)

    # Keep seed-connected components
    connected_mask = np.zeros_like(feasible_mask_2d, dtype=bool)
    for label in seed_labels:
        connected_mask |= (labeled == label)

    return connected_mask
