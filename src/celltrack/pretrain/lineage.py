"""Branching birth-death lineage forest → node table + edges.

Runs the forward simulation over discrete frames: each live cell either dies,
divides into two daughters, or continues to the next frame. Positions evolve via
the :mod:`celltrack.pretrain.motion` model. The result is a flat table of nodes
(one per cell per frame it is alive) plus directed parent→child edges — the raw
material for a :class:`~celltrack.graph.TrackGraph` (out-degree ≥ 2 == division).
"""

from __future__ import annotations

from dataclasses import dataclass

import numpy as np

from celltrack.constants import VOXEL_SCALE_UM
from celltrack.pretrain.config import SimConfig
from celltrack.pretrain.motion import (
    division_offsets,
    make_drift_field,
    step_position,
)


@dataclass
class _Cell:
    """A live lineage cell, mutated in place as it continues across frames."""

    node_id: int  # node id at the current frame
    pos: np.ndarray  # current µm position (z, y, x)
    vel: np.ndarray  # current µm/frame velocity
    next_div: int  # frame at which this cell divides


@dataclass(frozen=True)
class Forest:
    """Flat node table + edges for one simulated lineage forest (µm coords)."""

    node_id: np.ndarray  # (N,) int node ids
    t: np.ndarray  # (N,) int frame indices
    pos_um: np.ndarray  # (N, 3) µm positions (z, y, x)
    edges: list[tuple[int, int]]  # directed (source_id, target_id), t → t+1
    founder_ids: set[int]  # nodes with no parent


def _cycle_frames(config: SimConfig, rng: np.random.Generator) -> int:
    """Draw a cell-cycle duration in frames (clamped to ≥ 2)."""
    if config.cycle_dist == "gamma":
        shape = 1.0 / (config.cycle_cv**2)
        scale = config.cycle_mean_frames * (config.cycle_cv**2)
        val = rng.gamma(shape, scale)
    else:  # lognormal
        sigma = np.sqrt(np.log(1.0 + config.cycle_cv**2))
        mu = np.log(config.cycle_mean_frames) - 0.5 * sigma**2
        val = rng.lognormal(mu, sigma)
    return max(2, int(np.ceil(val)))


def simulate_forest(config: SimConfig, rng: np.random.Generator) -> Forest:
    """Run the birth-death + motion forward simulation."""
    domain_um = np.asarray(config.shape_vox, dtype=float) * np.asarray(VOXEL_SCALE_UM, dtype=float)
    drift = make_drift_field(config, rng, domain_um)
    death_hazard = 1.0 - np.exp(-config.death_rate_per_frame)

    node_ids: list[int] = []
    node_t: list[int] = []
    node_pos: list[np.ndarray] = []
    edges: list[tuple[int, int]] = []
    founder_ids: set[int] = set()
    counter = {"next": 1}

    def _new_founder(t: int) -> _Cell:
        pos = rng.uniform(np.zeros(3), domain_um)
        vel = rng.normal(size=3) * config.ou_sigma_um
        nid = counter["next"]
        counter["next"] += 1
        node_ids.append(nid)
        node_t.append(t)
        node_pos.append(pos)
        founder_ids.add(nid)
        return _Cell(node_id=nid, pos=pos, vel=vel, next_div=t + _cycle_frames(config, rng))

    def _emit(t: int, pos: np.ndarray, parent_id: int) -> int:
        nid = counter["next"]
        counter["next"] += 1
        node_ids.append(nid)
        node_t.append(t)
        node_pos.append(pos)
        edges.append((parent_id, nid))
        return nid

    alive = [_new_founder(0) for _ in range(config.n_founders)]

    for t in range(1, config.n_frames):
        new_alive: list[_Cell] = []
        for cell in alive:
            if rng.random() < death_hazard:
                continue  # extrusion / death
            new_pos, new_vel, in_bounds = step_position(
                cell.pos, cell.vel, config, rng, drift, t, domain_um
            )
            if not in_bounds:
                continue  # exited the field of view
            if t >= cell.next_div:
                off1, off2 = division_offsets(config, rng)
                for off in (off1, off2):
                    dpos = np.clip(new_pos + off, 0.0, domain_um)
                    nid = _emit(t, dpos, cell.node_id)
                    new_alive.append(
                        _Cell(nid, dpos, new_vel.copy(), t + _cycle_frames(config, rng))
                    )
            else:
                cell.node_id = _emit(t, new_pos, cell.node_id)
                cell.pos = new_pos
                cell.vel = new_vel
                new_alive.append(cell)
            if len(node_ids) > config.max_nodes:
                raise RuntimeError(
                    f"simulation exceeded max_nodes={config.max_nodes}; tighten priors"
                )
        if config.influx_per_frame > 0:
            for _ in range(int(rng.poisson(config.influx_per_frame))):
                new_alive.append(_new_founder(t))
        alive = new_alive

    pos_arr = np.vstack(node_pos) if node_pos else np.zeros((0, 3), dtype=float)
    return Forest(
        node_id=np.asarray(node_ids, dtype=int),
        t=np.asarray(node_t, dtype=int),
        pos_um=pos_arr,
        edges=edges,
        founder_ids=founder_ids,
    )
