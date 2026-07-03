"""Detection / imaging noise layer — the sim-to-real bridge.

Corrupts the clean ground-truth node set the way a real detector (Cellpose-SAM)
would: false-negative dropout, false-positive clutter, anisotropic localisation
jitter (heavier in z), and occasional merges/splits of nearby nuclei. Produces
per-timepoint :class:`~celltrack.detect.base.Detection` lists (the exact format
:func:`~celltrack.track.nearest_neighbor.track_frames` consumes) together with a
parallel label array mapping each detection back to its GT node id (or ``None``
for a false positive).
"""

from __future__ import annotations

from dataclasses import dataclass

import numpy as np
from scipy.spatial import cKDTree

from celltrack.detect.base import Detection
from celltrack.pretrain.config import SimConfig
from celltrack.pretrain.motion import to_voxel


@dataclass(frozen=True)
class ObservedDetections:
    """Noisy detections + their ground-truth labels, per timepoint."""

    by_time: dict[int, list[Detection]]  # consumed by track_frames
    gt_node_of: dict[int, list[int | None]]  # parallel labels; None == false positive


def _merge_near_pairs(
    kept: list[tuple[int, np.ndarray]], config: SimConfig, rng: np.random.Generator
) -> list[tuple[int, np.ndarray]]:
    """Collapse some within-``merge_dist_um`` pairs to their midpoint."""
    if config.merge_rate <= 0 or len(kept) < 2:
        return kept
    coords = np.vstack([p for _, p in kept])
    pairs = cKDTree(coords).query_pairs(r=config.merge_dist_um)
    consumed: set[int] = set()
    for i, j in sorted(pairs):
        if i in consumed or j in consumed:
            continue
        if rng.random() < config.merge_rate:
            gid_i, pi = kept[i]
            _, pj = kept[j]
            kept[i] = (gid_i, 0.5 * (pi + pj))  # keep one label; other becomes an FN
            consumed.add(j)
    if consumed:
        kept = [kv for k, kv in enumerate(kept) if k not in consumed]
    return kept


def apply_detection_noise(
    frames_um: dict[int, list[tuple[int, np.ndarray]]],
    config: SimConfig,
    rng: np.random.Generator,
    domain_um: np.ndarray,
) -> ObservedDetections:
    """Apply the detection-noise layers to per-frame ``(gt_id, µm-pos)`` items."""
    by_time: dict[int, list[Detection]] = {}
    gt_node_of: dict[int, list[int | None]] = {}
    jitter = np.array([config.loc_jitter_z_um, config.loc_jitter_xy_um, config.loc_jitter_xy_um])

    for t, items in frames_um.items():
        n_true = len(items)
        # 1. False-negative dropout.
        kept = [(gid, pos.copy()) for gid, pos in items if rng.random() >= config.fn_rate]
        # 2. Merge near pairs.
        kept = _merge_near_pairs(kept, config, rng)
        # 3. Localisation jitter (+ 4. splits) into parallel pos/label lists.
        pos_list: list[np.ndarray] = []
        lab_list: list[int | None] = []
        for gid, pos in kept:
            jpos = pos + rng.normal(size=3) * jitter
            pos_list.append(jpos)
            lab_list.append(gid)
            if config.split_rate > 0 and rng.random() < config.split_rate:
                d = rng.normal(size=3)
                d /= np.linalg.norm(d) + 1e-12
                pos_list.append(jpos + d * rng.uniform(2.0, 3.0))
                lab_list.append(gid if config.split_label == "same" else None)
        # 5. False-positive clutter.
        for _ in range(int(round(config.fp_per_frame_frac * n_true))):
            pos_list.append(rng.uniform(np.zeros(3), domain_um))
            lab_list.append(None)

        dets: list[Detection] = []
        for pos in pos_list:
            vox = to_voxel(pos, config.shape_vox)
            dets.append(Detection(z=int(vox[0]), y=int(vox[1]), x=int(vox[2]), probability=1.0))
        by_time[t] = dets
        gt_node_of[t] = lab_list

    return ObservedDetections(by_time=by_time, gt_node_of=gt_node_of)
