"""Physical constants defined by the competition metric.

These MUST be identical in the tracker's association cost and the evaluation
harness so that offline scores track the leaderboard.
"""

from __future__ import annotations

# Physical voxel size in micrometres, ordered (z, y, x).
# Confirmed from the competition evaluation description.
VOXEL_SCALE_UM: tuple[float, float, float] = (1.625, 0.40625, 0.40625)

# Maximum scaled centroid distance (µm) for a predicted node to match a
# ground-truth node during per-timepoint bipartite assignment.
MATCH_MAX_UM: float = 7.0

# Column schema of the submission CSV, in required order.
SUBMISSION_COLUMNS: tuple[str, ...] = (
    "id",
    "dataset",
    "row_type",
    "node_id",
    "t",
    "z",
    "y",
    "x",
    "source_id",
    "target_id",
)

# Sentinel used for "not applicable" integer fields.
SENTINEL: int = -1
