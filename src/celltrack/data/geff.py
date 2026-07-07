"""Load ground-truth lineage graphs in the ``.geff`` format.

``.geff`` (graph exchange file format) is a zarr store with:

- ``nodes/ids``                  -> (N,) int node ids
- ``nodes/props/{t,z,y,x}/values`` -> (N,) per-node fields
- ``edges/ids``                  -> (E, 2) source/target id pairs
- ``zarr.json`` attributes ``geff.extra.estimated_number_of_nodes`` -> the dense
  true-cell-count estimate used by the metric's node over-prediction penalty.

Coordinates are stored as floats; we round to integer voxels for the
:class:`TrackGraph`. The metric matches at a 7.0 um gate (~4-17 voxels), so
sub-voxel rounding is negligible.
"""

from __future__ import annotations

import json
from pathlib import Path

import numpy as np
import zarr

from celltrack.graph import Node, TrackGraph


def read_geff(geff_path: str | Path) -> tuple[TrackGraph, float | None]:
    """Read a ``.geff`` store. Returns ``(graph, estimated_number_of_nodes)``."""
    geff_path = Path(geff_path)
    g = zarr.open(str(geff_path), mode="r")

    ids = np.asarray(g["nodes/ids"][:]).astype(np.int64)
    t = np.asarray(g["nodes/props/t/values"][:]).astype(np.int64)
    z = np.asarray(g["nodes/props/z/values"][:]).astype(np.float64)
    y = np.asarray(g["nodes/props/y/values"][:]).astype(np.float64)
    x = np.asarray(g["nodes/props/x/values"][:]).astype(np.float64)
    edges = np.asarray(g["edges/ids"][:]).astype(np.int64).reshape(-1, 2)

    graph = TrackGraph()
    for nid, ti, zi, yi, xi in zip(ids, t, z, y, x):
        graph.add_node(
            Node(int(nid), int(ti), int(round(zi)), int(round(yi)), int(round(xi)))
        )
    for s, d in edges:
        graph.add_edge(int(s), int(d))

    return graph, _estimated_number_of_nodes(geff_path)


def _estimated_number_of_nodes(geff_path: Path) -> float | None:
    try:
        zj = json.loads((geff_path / "zarr.json").read_text())
    except (OSError, json.JSONDecodeError):
        return None
    extra = zj.get("attributes", {}).get("geff", {}).get("extra", {}) or {}
    val = extra.get("estimated_number_of_nodes")
    return float(val) if val is not None else None


def load_ground_truth(root: str | Path) -> tuple[dict[str, TrackGraph], dict[str, float]]:
    """Load all ``<dataset>.geff`` under ``root`` into graphs + est-node counts."""
    root = Path(root)
    graphs: dict[str, TrackGraph] = {}
    est: dict[str, float] = {}
    for path in sorted(root.glob("*.geff")):
        dataset = path.name[: -len(".geff")]
        graph, n_est = read_geff(path)
        graphs[dataset] = graph
        if n_est is not None:
            est[dataset] = n_est
    return graphs, est
