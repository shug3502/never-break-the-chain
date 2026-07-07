"""Cellpose-SAM detector (default backbone).

Requires the optional ``detect`` extra: ``uv pip install -e ".[detect]"``.
Cellpose is imported lazily so the core package installs and tests without
PyTorch.
"""

from __future__ import annotations

import numpy as np
from scipy import ndimage as ndi

from celltrack.constants import VOXEL_SCALE_UM
from celltrack.detect.base import Detection

# z is ~4x coarser than xy; used only by the do_3D path to scale Cellpose's 3D kernels.
_DEFAULT_ANISOTROPY = VOXEL_SCALE_UM[0] / VOXEL_SCALE_UM[1]  # 1.625 / 0.40625 ≈ 4.0


def _build_eval_kwargs(
    *,
    diameter: float | None,
    do_3d: bool,
    stitch_threshold: float,
    anisotropy: float,
    flow_threshold: float | None,
    cellprob_threshold: float | None,
) -> dict:
    """Build Cellpose ``eval`` kwargs for the chosen detection mode.

    The volume is (Z, Y, X), so ``z_axis=0`` is always required. Two modes:

    - ``do_3d=True``: volumetric flow prediction (accurate, slow). Resamples Z to
      isotropic using ``anisotropy`` and runs the network over XY/XZ/YZ planes.
    - ``do_3d=False`` (default): 2D inference per native Z-slice, stitched into 3D
      instances by IoU (``stitch_threshold``). ``anisotropy`` is irrelevant here.
    """
    if do_3d:
        kwargs: dict = dict(diameter=diameter, anisotropy=anisotropy, do_3D=True, z_axis=0)
    else:
        kwargs = dict(diameter=diameter, stitch_threshold=stitch_threshold, z_axis=0)
    if flow_threshold is not None:
        kwargs["flow_threshold"] = flow_threshold
    if cellprob_threshold is not None:
        kwargs["cellprob_threshold"] = cellprob_threshold
    return kwargs


class CellposeSamDetector:
    """Wrap Cellpose-SAM for 3D nuclei detection.

    Produces one :class:`Detection` per segmented instance (its integer voxel
    centroid). Segmentation masks are reduced to centroids because the
    competition metric operates on nodes (centroids) and edges.

    By default detection uses fast 2D+Z-stitching (``do_3d=False``); the accurate
    but much slower volumetric ``do_3D`` path is available via ``do_3d=True``.
    """

    def __init__(
        self,
        model_type: str = "cpsam",
        *,
        diameter: float | None = None,
        anisotropy: float | None = None,
        flow_threshold: float | None = None,
        cellprob_threshold: float | None = None,
        do_3d: bool = False,
        stitch_threshold: float = 0.3,
        amp: bool = False,
        gpu: bool = True,
    ) -> None:
        # Lazy import: keeps torch/cellpose out of the core install.
        from cellpose import models  # noqa: PLC0415

        self._model = models.CellposeModel(gpu=gpu, model_type=model_type)
        self._diameter = diameter
        # Default to the physical z/xy ratio so Cellpose's 3D kernels are correctly scaled.
        self._anisotropy = anisotropy if anisotropy is not None else _DEFAULT_ANISOTROPY
        self._flow_threshold = flow_threshold
        self._cellprob_threshold = cellprob_threshold
        self._do_3d = do_3d
        self._stitch_threshold = stitch_threshold
        self._amp = amp
        self._gpu = gpu

    def detect(self, volume: np.ndarray) -> list[Detection]:
        kwargs = _build_eval_kwargs(
            diameter=self._diameter,
            do_3d=self._do_3d,
            stitch_threshold=self._stitch_threshold,
            anisotropy=self._anisotropy,
            flow_threshold=self._flow_threshold,
            cellprob_threshold=self._cellprob_threshold,
        )

        import torch  # noqa: PLC0415  # lazy, consistent with the cellpose import above

        # Optional bf16 autocast (opt-in via --amp; CUDA-only, so guard on it).
        # Off by default: benchmarked ~20% SLOWER than no-amp
        # where cellpose 4.x already runs the ViT in an efficient dtype, so the
        # outer autocast just adds overhead. May still help on older archs.
        if self._amp and self._gpu and torch.cuda.is_available():
            with torch.autocast(device_type="cuda", dtype=torch.bfloat16):
                result = self._model.eval(volume, **kwargs)
        else:
            result = self._model.eval(volume, **kwargs)

        # Cellpose eval returns (masks, flows, styles) or (masks, flows, styles, probs)
        masks = result[0]
        probs: np.ndarray | None = result[3] if len(result) > 3 else None
        return _masks_to_detections(masks, probs)


def _masks_to_detections(
    masks: np.ndarray,
    probs: np.ndarray | None = None,
) -> list[Detection]:
    labels = [lbl for lbl in np.unique(masks) if lbl != 0]
    if not labels:
        return []
    ones = np.ones_like(masks, dtype=np.float32)
    centroids = ndi.center_of_mass(ones, labels=masks, index=labels)
    out: list[Detection] = []
    for lbl, (z, y, x) in zip(labels, centroids):
        prob = float(np.mean(probs[masks == lbl])) if probs is not None else 1.0
        out.append(Detection(z=int(round(z)), y=int(round(y)), x=int(round(x)), probability=prob))
    return out
