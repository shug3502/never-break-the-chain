"""Tests for the Cellpose detector's mode branching and centroid reduction.

These cover the pure helpers only, so they run without a GPU or loading Cellpose
weights. The main regression risk is the eval-kwargs branching (e.g. passing both
``do_3D`` and ``stitch_threshold``, or dropping ``z_axis``).
"""

from __future__ import annotations

import numpy as np

from celltrack.detect.cellpose_sam import _build_eval_kwargs, _masks_to_detections


def _kwargs(**overrides) -> dict:
    base = dict(
        diameter=None,
        do_3d=False,
        stitch_threshold=0.3,
        anisotropy=4.0,
        flow_threshold=None,
        cellprob_threshold=None,
    )
    base.update(overrides)
    return _build_eval_kwargs(**base)


def test_stitch_is_default_mode() -> None:
    kwargs = _kwargs()
    assert kwargs["z_axis"] == 0
    assert kwargs["stitch_threshold"] == 0.3
    # Stitching is 2D per slice: no volumetric flags.
    assert "do_3D" not in kwargs
    assert "anisotropy" not in kwargs


def test_do_3d_mode() -> None:
    kwargs = _kwargs(do_3d=True, anisotropy=4.0)
    assert kwargs["z_axis"] == 0
    assert kwargs["do_3D"] is True
    assert kwargs["anisotropy"] == 4.0
    # do_3D and stitching are mutually exclusive.
    assert "stitch_threshold" not in kwargs


def test_optional_thresholds_only_when_set() -> None:
    assert "flow_threshold" not in _kwargs()
    assert "cellprob_threshold" not in _kwargs()
    with_thresh = _kwargs(flow_threshold=0.5, cellprob_threshold=-1.0)
    assert with_thresh["flow_threshold"] == 0.5
    assert with_thresh["cellprob_threshold"] == -1.0


def test_masks_to_detections_centroids() -> None:
    # Two labelled instances in a (Z, Y, X) volume, sized for clean integer centroids.
    masks = np.zeros((3, 5, 5), dtype=np.int32)
    masks[0, 0, 0] = 1  # single voxel at (0, 0, 0)
    masks[0:3, 1:4, 1:4] = 2  # 3x3x3 block centred at (1, 2, 2)

    dets = _masks_to_detections(masks)
    assert len(dets) == 2
    by_coord = {(d.z, d.y, d.x) for d in dets}
    assert (0, 0, 0) in by_coord
    assert (1, 2, 2) in by_coord
    assert all(d.probability == 1.0 for d in dets)  # no probs array supplied


def test_masks_to_detections_empty() -> None:
    assert _masks_to_detections(np.zeros((2, 2, 2), dtype=np.int32)) == []
