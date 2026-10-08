"""
Clinical Filtering Pipeline.

Implements the clinical filtering rules in required order:
B3 (Scapular Plane Band) -> R4 (Elevation-Rotation) -> C5 (Cross-Body) -> E6 (Elbow-Elevation)

All filters operate on joint configurations in degrees using short names
(FE, AA, ROT, EL) for clarity.
"""

import numpy as np
from typing import Dict, Union, Optional
from abc import ABC, abstractmethod


class ClinicalFilter(ABC):
    """Base class for clinical filters."""

    def __init__(self, enabled: bool = True):
        self.enabled = enabled

    @abstractmethod
    def apply(self, cfg: Dict[str, np.ndarray]) -> np.ndarray:
        """
        Apply filter to joint configurations.

        Args:
            cfg: Dictionary with keys 'FE', 'AA', 'ROT', 'EL' containing
                 numpy arrays of joint angles in degrees

        Returns:
            Boolean mask where True = configuration passes filter
        """
        pass

    @property
    @abstractmethod
    def name(self) -> str:
        """Return filter name for logging."""
        pass


class ScapularPlaneBandFilter(ClinicalFilter):
    """
    B3: Scapular Plane Band Filter.

    Maintains positions near the scapular plane:
    - Centerline: AA_center = 0.7 * FE + 10
    - Half-width varies by elevation proxy:
      - elev_proxy < 60: w = 40
      - 60 <= elev_proxy <= 100: w = 20
      - elev_proxy > 100: w = 10
    - Accept if: |AA - AA_center| <= w
    """

    def __init__(self, enabled: bool = True):
        super().__init__(enabled)
        self.slope = 0.7
        self.intercept = 10.0
        self.width_low = 40.0    # for elev < 60
        self.width_mid = 20.0    # for 60 <= elev <= 100
        self.width_high = 10.0   # for elev > 100
        self.elev_thresh_low = 60.0
        self.elev_thresh_high = 100.0

    @property
    def name(self) -> str:
        return "B3_ScapularPlaneBand"

    def compute_elev_proxy(self, FE: np.ndarray, AA: np.ndarray) -> np.ndarray:
        """Compute elevation proxy as max(FE, AA)."""
        return np.maximum(FE, AA)

    def get_half_width(self, elev_proxy: np.ndarray) -> np.ndarray:
        """Get band half-width based on elevation proxy."""
        w = np.full_like(elev_proxy, self.width_mid, dtype=float)
        w[elev_proxy < self.elev_thresh_low] = self.width_low
        w[elev_proxy > self.elev_thresh_high] = self.width_high
        return w

    def apply(self, cfg: Dict[str, np.ndarray]) -> np.ndarray:
        """Apply scapular plane band filter."""
        FE = np.asarray(cfg['FE'], dtype=float)
        AA = np.asarray(cfg['AA'], dtype=float)

        # Compute centerline
        AA_center = self.slope * FE + self.intercept

        # Get elevation-dependent half-width
        elev_proxy = self.compute_elev_proxy(FE, AA)
        w = self.get_half_width(elev_proxy)

        # Accept if within band
        return np.abs(AA - AA_center) <= w


class ElevationRotationFilter(ClinicalFilter):
    """
    R4: Elevation-Rotation Clearance Filter.

    Prevents internal rotation at high elevations (impingement risk):
    - Thresholds by elevation proxy:
      - <= 60: no constraint (T = inf)
      - 60-90: T = 20 degrees
      - 90-120: T = 10 degrees
      - > 120: T = 0 degrees
    - Reject if: ROT is internal (ROT < 0) AND |ROT| > T

    Note: Negative ROT = internal rotation (confirmed)
    """

    def __init__(self, enabled: bool = True):
        super().__init__(enabled)
        self.thresh_elev_60 = 60.0
        self.thresh_elev_90 = 90.0
        self.thresh_elev_120 = 120.0
        self.rot_thresh_60_90 = 20.0
        self.rot_thresh_90_120 = 10.0
        self.rot_thresh_above_120 = 0.0

    @property
    def name(self) -> str:
        return "R4_ElevationRotation"

    def rot_is_too_internal(self, ROT: np.ndarray) -> np.ndarray:
        """Check if rotation is internal (negative ROT)."""
        return ROT < 0

    def get_threshold(self, elev_proxy: np.ndarray) -> np.ndarray:
        """Get rotation threshold based on elevation proxy."""
        T = np.full_like(elev_proxy, np.inf, dtype=float)
        mask_60_90 = (elev_proxy > self.thresh_elev_60) & (elev_proxy <= self.thresh_elev_90)
        mask_90_120 = (elev_proxy > self.thresh_elev_90) & (elev_proxy <= self.thresh_elev_120)
        mask_above_120 = elev_proxy > self.thresh_elev_120

        T[mask_60_90] = self.rot_thresh_60_90
        T[mask_90_120] = self.rot_thresh_90_120
        T[mask_above_120] = self.rot_thresh_above_120

        return T

    def apply(self, cfg: Dict[str, np.ndarray]) -> np.ndarray:
        """Apply elevation-rotation clearance filter."""
        FE = np.asarray(cfg['FE'], dtype=float)
        AA = np.asarray(cfg['AA'], dtype=float)
        ROT = np.asarray(cfg['ROT'], dtype=float)

        elev_proxy = np.maximum(FE, AA)
        T = self.get_threshold(elev_proxy)

        # Reject if too internal AND exceeds threshold
        is_internal = self.rot_is_too_internal(ROT)
        exceeds_threshold = np.abs(ROT) > T
        too_internal = is_internal & exceeds_threshold

        return ~too_internal


class CrossBodyPruneFilter(ClinicalFilter):
    """
    C5: Cross-Body Prune Filter.

    Prevents cross-body positions at high elevations:
    - Cross-body defined as: AA <= -20
    - Reject if: cross_body AND elev_proxy >= 90
    """

    def __init__(self, enabled: bool = True):
        super().__init__(enabled)
        self.cross_body_threshold = -20.0
        self.elev_threshold = 90.0

    @property
    def name(self) -> str:
        return "C5_CrossBodyPrune"

    def apply(self, cfg: Dict[str, np.ndarray]) -> np.ndarray:
        """Apply cross-body prune filter."""
        FE = np.asarray(cfg['FE'], dtype=float)
        AA = np.asarray(cfg['AA'], dtype=float)

        elev_proxy = np.maximum(FE, AA)
        cross_body = AA <= self.cross_body_threshold

        # Reject if cross-body AND high elevation
        return ~(cross_body & (elev_proxy >= self.elev_threshold))


class ElbowElevationFilter(ClinicalFilter):
    """
    E6: Elbow-Elevation Filter.

    Prevents highly flexed elbow at very high elevations:
    - Reject if: elev_proxy >= 100 AND EL >= 120
    """

    def __init__(self, enabled: bool = True):
        super().__init__(enabled)
        self.elev_threshold = 100.0
        self.elbow_threshold = 120.0

    @property
    def name(self) -> str:
        return "E6_ElbowElevation"

    def apply(self, cfg: Dict[str, np.ndarray]) -> np.ndarray:
        """Apply elbow-elevation filter."""
        FE = np.asarray(cfg['FE'], dtype=float)
        AA = np.asarray(cfg['AA'], dtype=float)
        EL = np.asarray(cfg['EL'], dtype=float)

        elev_proxy = np.maximum(FE, AA)

        # Reject if high elevation AND high elbow flexion
        return ~((elev_proxy >= self.elev_threshold) & (EL >= self.elbow_threshold))


class ClinicalFilterPipeline:
    """
    Applies filters in required order: B3 -> R4 -> C5 -> E6

    Filters are applied sequentially, with each filter operating on
    the remaining valid configurations from previous filters.
    """

    def __init__(
        self,
        enable_B3: bool = True,
        enable_R4: bool = True,
        enable_C5: bool = True,
        enable_E6: bool = True
    ):
        """
        Initialize filter pipeline.

        Args:
            enable_B3: Enable scapular plane band filter
            enable_R4: Enable elevation-rotation filter
            enable_C5: Enable cross-body prune filter
            enable_E6: Enable elbow-elevation filter
        """
        self.filters = [
            ScapularPlaneBandFilter(enabled=enable_B3),
            ElevationRotationFilter(enabled=enable_R4),
            CrossBodyPruneFilter(enabled=enable_C5),
            ElbowElevationFilter(enabled=enable_E6),
        ]

    def apply(
        self,
        cfg: Dict[str, np.ndarray],
        return_stats: bool = False
    ) -> Union[np.ndarray, tuple]:
        """
        Apply all enabled filters in order.

        Args:
            cfg: Dictionary with keys 'FE', 'AA', 'ROT', 'EL' containing
                 numpy arrays of joint angles in degrees
            return_stats: If True, also return per-filter statistics

        Returns:
            Boolean mask of valid configurations (True = passes all filters)
            If return_stats=True, also returns dict with per-filter stats
        """
        n = len(cfg['FE'])
        mask = np.ones(n, dtype=bool)
        stats = {}

        for f in self.filters:
            if f.enabled:
                n_before = np.sum(mask)
                filter_mask = f.apply(cfg)
                mask &= filter_mask
                n_after = np.sum(mask)

                stats[f.name] = {
                    'n_before': int(n_before),
                    'n_after': int(n_after),
                    'n_rejected': int(n_before - n_after),
                    'rejection_rate': float((n_before - n_after) / n_before) if n_before > 0 else 0.0
                }

        if return_stats:
            stats['total'] = {
                'n_input': n,
                'n_output': int(np.sum(mask)),
                'overall_acceptance_rate': float(np.sum(mask) / n) if n > 0 else 0.0
            }
            return mask, stats
        return mask

    def apply_single(self, FE: float, AA: float, ROT: float, EL: float) -> bool:
        """
        Check if a single configuration passes all filters.

        Args:
            FE: Shoulder flexion (degrees)
            AA: Shoulder abduction (degrees)
            ROT: Shoulder rotation (degrees)
            EL: Elbow flexion (degrees)

        Returns:
            True if configuration passes all filters
        """
        cfg = {
            'FE': np.array([FE]),
            'AA': np.array([AA]),
            'ROT': np.array([ROT]),
            'EL': np.array([EL])
        }
        return self.apply(cfg)[0]

    def get_enabled_filter_names(self) -> list:
        """Return list of enabled filter names."""
        return [f.name for f in self.filters if f.enabled]


def convert_to_filter_format(
    joint_configs: Dict[str, np.ndarray],
    degrees: bool = True
) -> Dict[str, np.ndarray]:
    """
    Convert joint configurations to filter format with short names.

    Args:
        joint_configs: Dict with keys like 'shoulder_flexion_r', etc.
        degrees: If False, convert from radians to degrees

    Returns:
        Dict with keys 'FE', 'AA', 'ROT', 'EL' in degrees
    """
    from .joint_config import LONG_TO_SHORT

    result = {}
    name_map = {
        'shoulder_flexion_r': 'FE',
        'shoulder_abduction_r': 'AA',
        'shoulder_rotation_r': 'ROT',
        'elbow_flexion_r': 'EL'
    }

    for long_name, short_name in name_map.items():
        if long_name in joint_configs:
            val = np.asarray(joint_configs[long_name], dtype=float)
            if not degrees:
                val = np.degrees(val)
            result[short_name] = val

    return result


def apply_clinical_filters(
    joint_configs: Dict[str, np.ndarray],
    degrees: bool = True,
    return_stats: bool = False,
    **filter_kwargs
) -> Union[np.ndarray, tuple]:
    """
    Convenience function to apply clinical filters to joint configurations.

    Args:
        joint_configs: Dict with joint angle arrays (long or short names)
        degrees: If False, input is in radians
        return_stats: If True, return per-filter statistics
        **filter_kwargs: Arguments passed to ClinicalFilterPipeline

    Returns:
        Boolean mask of valid configurations
    """
    # Check if using short names already
    if 'FE' in joint_configs:
        cfg = joint_configs
        if not degrees:
            cfg = {k: np.degrees(v) for k, v in cfg.items()}
    else:
        cfg = convert_to_filter_format(joint_configs, degrees=degrees)

    pipeline = ClinicalFilterPipeline(**filter_kwargs)
    return pipeline.apply(cfg, return_stats=return_stats)
