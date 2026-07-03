"""Orchestrator: assemble a fully-labelled synthetic dataset.

Ties the layers together — lineage forest (:mod:`lineage`) → discretise to
anisotropic integer voxels → clean :class:`~celltrack.graph.TrackGraph` →
detection-noise layer (:mod:`noise`) → :class:`SimulatedDataset`. Everything is
CPU / numpy and reproducible from a seed.
"""

from __future__ import annotations

from collections.abc import Iterator
from dataclasses import dataclass

import numpy as np

from celltrack.constants import VOXEL_SCALE_UM
from celltrack.graph import Node, TrackGraph
from celltrack.pretrain.config import DEFAULT_PRIORS, Priors, SimConfig, sample_config
from celltrack.pretrain.lineage import simulate_forest
from celltrack.pretrain.motion import to_voxel
from celltrack.pretrain.noise import ObservedDetections, apply_detection_noise


@dataclass(frozen=True)
class SimulatedDataset:
    """A clean lineage + its noisy observation + the over-prediction estimate."""

    name: str
    gt_graph: TrackGraph  # clean ground-truth lineage forest
    observed: ObservedDetections  # noisy detections + labels
    est_nodes: float  # dense true-cell count (== gt node count; for the metric)
    config: SimConfig


def simulate_dataset(
    config: SimConfig, *, name: str = "sim", rng: np.random.Generator | None = None
) -> SimulatedDataset:
    """Generate one synthetic dataset from ``config``.

    If ``rng`` is given it drives every random draw (``config.seed`` ignored);
    otherwise a fresh ``default_rng(config.seed)`` is used.
    """
    if config.n_frames < 2:
        raise ValueError(f"n_frames must be >= 2, got {config.n_frames}")
    if rng is None:
        rng = np.random.default_rng(config.seed)

    forest = simulate_forest(config, rng)

    gt = TrackGraph()
    frames_um: dict[int, list[tuple[int, np.ndarray]]] = {}
    for nid, t, pos in zip(forest.node_id, forest.t, forest.pos_um):
        vox = to_voxel(pos, config.shape_vox)
        gt.add_node(Node(int(nid), int(t), int(vox[0]), int(vox[1]), int(vox[2])))
        frames_um.setdefault(int(t), []).append((int(nid), pos))
    for source_id, target_id in forest.edges:
        gt.add_edge(int(source_id), int(target_id))

    domain_um = np.asarray(config.shape_vox, dtype=float) * np.asarray(VOXEL_SCALE_UM, dtype=float)
    observed = apply_detection_noise(frames_um, config, rng, domain_um)

    return SimulatedDataset(
        name=name,
        gt_graph=gt,
        observed=observed,
        est_nodes=float(gt.num_nodes()),
        config=config,
    )


def iter_datasets(
    n: int | None = None,
    *,
    priors: Priors = DEFAULT_PRIORS,
    seed: int | None = None,
    name_prefix: str = "sim",
    n_frames: int = 40,
    shape_vox: tuple[int, int, int] = (60, 512, 512),
) -> Iterator[SimulatedDataset]:
    """Yield domain-randomised datasets (``n=None`` → infinite, for training).

    A single parent RNG (seeded by ``seed``) spawns a child seed per dataset, so
    datasets are independent yet each individually reproducible.
    """
    parent = np.random.default_rng(seed)
    i = 0
    while n is None or i < n:
        child = np.random.default_rng(int(parent.integers(0, 2**63 - 1)))
        cfg = sample_config(child, priors, n_frames=n_frames, shape_vox=shape_vox)
        yield simulate_dataset(cfg, name=f"{name_prefix}{i:04d}", rng=child)
        i += 1
