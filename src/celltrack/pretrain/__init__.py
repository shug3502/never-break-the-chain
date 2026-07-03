"""Pre-training data generation for the tracking / division layer.

Currently: a mechanistic lineage simulator (Idea 2 in ``docs/ideas/pre-training.md``)
that produces fully-labelled synthetic 3D+time lineages — graphs + noisy
detections, no images — for supervised pre-training and as a known-answer oracle
for the tracker and the competition metric.
"""

from __future__ import annotations

from celltrack.pretrain.config import (
    DEFAULT_PRIORS,
    Priors,
    SimConfig,
    sample_config,
)
from celltrack.pretrain.noise import ObservedDetections
from celltrack.pretrain.simulate import (
    SimulatedDataset,
    iter_datasets,
    simulate_dataset,
)

__all__ = [
    "DEFAULT_PRIORS",
    "ObservedDetections",
    "Priors",
    "SimConfig",
    "SimulatedDataset",
    "iter_datasets",
    "sample_config",
    "simulate_dataset",
]
