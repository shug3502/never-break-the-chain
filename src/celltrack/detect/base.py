"""Detector interface decoupling detection from tracking.

Any detector maps a 3D volume (z, y, x) to a list of :class:`Detection`
centroids. The tracker consumes only this representation, so backends
(Cellpose-SAM default, Stardist-3D, Ultrack detection, ...) are interchangeable.
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import Protocol, runtime_checkable

import numpy as np


@dataclass(frozen=True)
class Detection:
    """A detected nucleus centroid in one timepoint (integer voxel coords)."""

    z: int
    y: int
    x: int
    probability: float = 1.0


@runtime_checkable
class Detector(Protocol):
    """Detect nuclei in a single 3D volume."""

    def detect(self, volume: np.ndarray) -> list[Detection]: ...
