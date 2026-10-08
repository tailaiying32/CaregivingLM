"""
Coupling Rules (G7).

Defines which joint pairs are allowed to have SIRS bumps (coupling constraints)
and which pairs should never have bumps.
"""

from typing import List, Tuple, Set, Dict


# =============================================================================
# Allowed/Disallowed Coupling Pairs
# =============================================================================

# Allowed pairs (bumps OK) - using clinical terms
# These are pairs where biomechanical coupling makes clinical sense
ALLOWED_COUPLING_PAIRS = [
    ('elev_proxy', 'ROT'),   # Strong coupling: elevation affects rotation limits
    ('FE', 'AA'),            # Scapular plane band coupling
    ('elev_proxy', 'EL'),    # Weak coupling: elbow-elevation relationship (optional)
]

# Disallowed pairs (NO bumps) - independent DOFs
# These pairs have no clinically meaningful coupling
DISALLOWED_COUPLING_PAIRS = [
    ('EL', 'ROT'),   # Elbow and shoulder rotation are independent
    ('EL', 'AA'),    # Elbow and abduction are independent
]

# Maximum number of coupling structures (bump sets) per user/sample
MAX_COUPLING_STRUCTURES = 3


# =============================================================================
# Mapping to Implementation Joint Names
# =============================================================================

# For elev_proxy-based pairs, we need to use both FE and AA in constraints
# since elev_proxy = max(FE, AA)
IMPLEMENTATION_PAIRS: Dict[Tuple[str, str], List[Tuple[str, str]]] = {
    ('elev_proxy', 'ROT'): [
        ('shoulder_flexion_r', 'shoulder_rotation_r'),
        ('shoulder_abduction_r', 'shoulder_rotation_r'),
    ],
    ('FE', 'AA'): [
        ('shoulder_flexion_r', 'shoulder_abduction_r'),
    ],
    ('elev_proxy', 'EL'): [
        ('shoulder_flexion_r', 'elbow_flexion_r'),
        ('shoulder_abduction_r', 'elbow_flexion_r'),
    ],
}

# All valid implementation-level joint pairs (expanded from clinical pairs)
ALL_VALID_PAIRS: Set[Tuple[str, str]] = set()
for impl_pairs in IMPLEMENTATION_PAIRS.values():
    ALL_VALID_PAIRS.update(impl_pairs)

# Disallowed pairs at implementation level
DISALLOWED_IMPL_PAIRS: Set[Tuple[str, str]] = {
    ('elbow_flexion_r', 'shoulder_rotation_r'),
    ('shoulder_rotation_r', 'elbow_flexion_r'),
    ('elbow_flexion_r', 'shoulder_abduction_r'),
    ('shoulder_abduction_r', 'elbow_flexion_r'),  # This overlaps with elev_proxy-EL
}

# Note: ('shoulder_abduction_r', 'elbow_flexion_r') is in elev_proxy-EL which is allowed
# but marked as weak/optional. The plan says to remove it from MAIN_JOINT_PAIRS.


# =============================================================================
# Helper Functions
# =============================================================================

def get_implementation_pairs(clinical_pair: Tuple[str, str]) -> List[Tuple[str, str]]:
    """
    Get implementation-level joint pairs for a clinical pair.

    Args:
        clinical_pair: Tuple like ('elev_proxy', 'ROT') or ('FE', 'AA')

    Returns:
        List of implementation-level joint pairs
    """
    return IMPLEMENTATION_PAIRS.get(clinical_pair, [])


def filter_to_allowed_pairs(
    joint_pairs: List[Tuple[str, str]],
    include_weak: bool = False
) -> List[Tuple[str, str]]:
    """
    Filter joint pairs to only include allowed coupling pairs.

    Args:
        joint_pairs: List of (joint1, joint2) tuples at implementation level
        include_weak: If True, include weak pairs like elev_proxy-EL

    Returns:
        Filtered list of allowed pairs
    """
    # Build allowed set based on what we include
    allowed = set()

    for clinical_pair, impl_pairs in IMPLEMENTATION_PAIRS.items():
        # Skip weak pairs if not including them
        if not include_weak and clinical_pair == ('elev_proxy', 'EL'):
            continue
        for p in impl_pairs:
            allowed.add(p)
            # Also add reversed pair
            allowed.add((p[1], p[0]))

    filtered = []
    for pair in joint_pairs:
        if pair in allowed:
            filtered.append(pair)

    return filtered


def is_pair_allowed(
    joint1: str,
    joint2: str,
    include_weak: bool = False
) -> bool:
    """
    Check if a joint pair is allowed to have coupling (SIRS bumps).

    Args:
        joint1: First joint name (implementation level)
        joint2: Second joint name (implementation level)
        include_weak: If True, allow weak pairs like elev_proxy-EL

    Returns:
        True if pair is allowed to have bumps
    """
    pair = (joint1, joint2)
    pair_rev = (joint2, joint1)

    for clinical_pair, impl_pairs in IMPLEMENTATION_PAIRS.items():
        # Skip weak pairs if not including them
        if not include_weak and clinical_pair == ('elev_proxy', 'EL'):
            continue

        for p in impl_pairs:
            if pair == p or pair_rev == p:
                return True

    return False


def is_pair_disallowed(joint1: str, joint2: str) -> bool:
    """
    Check if a joint pair is explicitly disallowed from having coupling.

    Args:
        joint1: First joint name
        joint2: Second joint name

    Returns:
        True if pair should never have bumps
    """
    pair = (joint1, joint2)
    pair_rev = (joint2, joint1)

    return pair in DISALLOWED_IMPL_PAIRS or pair_rev in DISALLOWED_IMPL_PAIRS


def get_recommended_joint_pairs(include_weak: bool = False) -> List[Tuple[str, str]]:
    """
    Get the recommended list of joint pairs for SIRS sampling.

    This is the subset of pairs that should be used by default,
    based on clinical coupling rules.

    Args:
        include_weak: If True, include weak pairs (elev_proxy-EL)

    Returns:
        List of recommended (joint1, joint2) tuples
    """
    pairs = [
        ('shoulder_flexion_r', 'shoulder_abduction_r'),   # FE-AA
        ('shoulder_flexion_r', 'shoulder_rotation_r'),    # elev_proxy-ROT
        ('shoulder_abduction_r', 'shoulder_rotation_r'),  # elev_proxy-ROT
    ]

    if include_weak:
        pairs.extend([
            ('shoulder_flexion_r', 'elbow_flexion_r'),    # elev_proxy-EL
            ('shoulder_abduction_r', 'elbow_flexion_r'),  # elev_proxy-EL
        ])

    return pairs


def validate_coupling_count(
    active_pairs: List[Tuple[str, str]],
    max_structures: int = MAX_COUPLING_STRUCTURES
) -> Tuple[bool, str]:
    """
    Validate that the number of active coupling pairs doesn't exceed limit.

    Args:
        active_pairs: List of joint pairs with bumps
        max_structures: Maximum allowed coupling structures

    Returns:
        Tuple of (is_valid, message)
    """
    # Count unique clinical-level couplings
    clinical_couplings = set()

    for pair in active_pairs:
        for clinical_pair, impl_pairs in IMPLEMENTATION_PAIRS.items():
            if pair in impl_pairs or (pair[1], pair[0]) in impl_pairs:
                clinical_couplings.add(clinical_pair)
                break

    n_couplings = len(clinical_couplings)

    if n_couplings > max_structures:
        return False, f"Too many coupling structures: {n_couplings} > {max_structures}"

    return True, f"OK ({n_couplings} coupling structures)"
