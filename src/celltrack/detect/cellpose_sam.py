"""Cellpose-SAM detector (default backbone).

Requires the optional ``detect`` extra: ``uv pip install -e ".[detect]"``.
Cellpose is imported lazily so the core package installs and tests without
PyTorch.
"""

from __future__ import annotations

import numpy as np
from scipy import ndimage as ndi

from celltrack.detect.base import Detection


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
        gpu: bool = True,
    ) -> None:
        # Lazy import: keeps torch/cellpose out of the core install.
        from cellpose import models  # noqa: PLC0415

        self._model = models.CellposeModel(gpu=gpu, model_type=model_type)
        self._diameter = diameter

    def detect(self, volume: np.ndarray) -> list[Detection]:
        masks, *_ = self._model.eval(volume, diameter=self._diameter, do_3D=True)
        return _masks_to_detections(masks)


def _masks_to_detections(masks: np.ndarray) -> list[Detection]:
    labels = [lbl for lbl in np.unique(masks) if lbl != 0]
    if not labels:
        return []
    centroids = ndi.center_of_mass(np.ones_like(masks), labels=masks, index=labels)
    out: list[Detection] = []
    for z, y, x in centroids:
        out.append(Detection(z=int(round(z)), y=int(round(y)), x=int(round(x))))
    return out
