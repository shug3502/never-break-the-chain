"""Nearest-neighbour baseline tracker.

Links detections between consecutive timepoints by optimal 1-1 assignment on the
same scaled centroid distance the metric uses, gated at ``MATCH_MAX_UM``. This
is the Milestone-1 baseline; the SOTA tracking + division solver (core IP) will
replace/extend it. It does NOT yet emit divisions (1-1 only).
"""

from __future__ import annotations

from collections.abc import Mapping, Sequence

import numpy as np
from scipy.optimize import linear_sum_assignment
from scipy.spatial.distance import cdist

from celltrack.constants import MATCH_MAX_UM, VOXEL_SCALE_UM
from celltrack.detect.base import Detection
from celltrack.graph import Node, TrackGraph


def _scaled(dets: Sequence[Detection]) -> np.ndarray:
    sz, sy, sx = VOXEL_SCALE_UM
    return np.array([[d.z * sz, d.y * sy, d.x * sx] for d in dets], dtype=float)


def track_frames(
    detections: Mapping[int, Sequence[Detection]],
    *,
    max_um: float = MATCH_MAX_UM,
) -> TrackGraph:
    """Build a :class:`TrackGraph` from per-timepoint detections.

    ``detections`` maps timepoint -> list of :class:`Detection`.
    """
    graph = TrackGraph()
    next_id = 1
    # Node ids assigned per (t, index); remember the previous frame's ids.
    ids_by_t: dict[int, list[int]] = {}

    for t in sorted(detections):
        dets = list(detections[t])
        ids: list[int] = []
        for d in dets:
            graph.add_node(Node(next_id, t, d.z, d.y, d.x))
            ids.append(next_id)
            next_id += 1
        ids_by_t[t] = ids

    times = sorted(detections)
    for prev_t, cur_t in zip(times, times[1:]):
        prev = list(detections[prev_t])
        cur = list(detections[cur_t])
        if not prev or not cur:
            continue
        cost = cdist(_scaled(prev), _scaled(cur))
        rows, cols = linear_sum_assignment(cost)
        for r, c in zip(rows, cols):
            if cost[r, c] <= max_um:
                graph.add_edge(ids_by_t[prev_t][r], ids_by_t[cur_t][c])
    return graph


class NearestNeighborTracker:
    """Object wrapper around :func:`track_frames`."""

    def __init__(self, max_um: float = MATCH_MAX_UM) -> None:
        self.max_um = max_um

    def track(self, detections: Mapping[int, Sequence[Detection]]) -> TrackGraph:
        return track_frames(detections, max_um=self.max_um)
