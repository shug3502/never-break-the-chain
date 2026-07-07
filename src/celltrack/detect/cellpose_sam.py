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

# z is ~4x coarser than xy; tell Cellpose so it uses correct 3D kernel scaling.
_DEFAULT_ANISOTROPY = VOXEL_SCALE_UM[0] / VOXEL_SCALE_UM[1]  # 1.625 / 0.40625 ≈ 4.0


class CellposeSamDetector:
    """Wrap Cellpose-SAM for 3D nuclei detection.

    Produces one :class:`Detection` per segmented instance (its integer voxel
    centroid). Segmentation masks are reduced to centroids because the
    competition metric operates on nodes (centroids) and edges.
    """

    def __init__(
        self,
        model_type: str = "cpsam",
        *,
        diameter: float | None = None,
        anisotropy: float | None = None,
        flow_threshold: float | None = None,
        cellprob_threshold: float | None = None,
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

    def detect(self, volume: np.ndarray) -> list[Detection]:
        # volume is (Z, Y, X); z_axis=0 is required by newer Cellpose when do_3D=True.
        kwargs: dict = dict(
            diameter=self._diameter,
            anisotropy=self._anisotropy,
            do_3D=True,
            z_axis=0,
        )
        if self._flow_threshold is not None:
            kwargs["flow_threshold"] = self._flow_threshold
        if self._cellprob_threshold is not None:
            kwargs["cellprob_threshold"] = self._cellprob_threshold

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
