"""
Stage 07: Full-body motion imputation utilities.

Defines the body marker set from Stage 04 `*_body.csv`. Wrist markers
(RLAW, LLAW) are excluded — wrist + finger joints come from Stage 06.
"""

# Full-body marker set (from Stage 04 *_body.csv), excluding wrist markers.
BODY_MARKERS = [
    # Head
    'HEADT', 'HEADA', 'HEADR', 'HEADL',
    # Shoulders (acromion)
    'ACROR', 'ACROL',
    # Trunk
    'TRUNKAR', 'TRUNKAL', 'TRUNKPR', 'TRUNKPL',
    # Elbow (lateral)
    'RELBL', 'LELBL',
    # Pelvis
    'ASISR', 'ASISL', 'PSISR', 'PSISL', 'HIPR', 'HIPL',
    # Legs — knee
    'RKNEEL', 'LKNEEL',
    # Legs — ankle / heel
    'RHEEL', 'RANKLEL', 'RANKLEM',
    'LHEEL', 'LANKLEL', 'LANKLEM',
    # Feet
    'RTOEL', 'RTOEM', 'RTOEF',
    'LTOEL', 'LTOEM', 'LTOEF',
    # Upper arm cluster
    'RUPARM1', 'RUPARM2', 'LUPARM1', 'LUPARM2',
    # Lower arm cluster
    'RLOWARM1', 'RLOWARM2', 'LLOWARM1', 'LLOWARM2',
    # Thigh cluster
    'RTHIGH1', 'RTHIGH2', 'LTHIGH1', 'LTHIGH2',
    # Shank cluster
    'RSHANK1', 'RSHANK2', 'LSHANK1', 'LSHANK2',
]

# Markers in *_body.csv that Stage 07 drops (handled by Stage 06)
BODY_EXCLUDED = ['RLAW', 'LLAW']

# Note: the Stage 06 utils module defines a same-named `body_joint_cols`
# but with hand-imputation semantics (30 hand+arm cols). Importers must
# pick the right module -- this one is the body-imputation 47-marker set.
body_joint_cols = [f"{m}_{ax}" for m in BODY_MARKERS for ax in ("x", "y", "z")]
