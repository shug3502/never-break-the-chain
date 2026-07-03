"""TrackGraph: the lineage graph datastructure for a single dataset.

A node is a cell detection at one timepoint. A directed edge (source -> target)
links a cell to itself (or, at a division, to its daughters) in the next
timepoint. A node with >= 2 outgoing edges is a division.

Backed by ``networkx.DiGraph`` for connected-component / lineage queries used by
the metric.
"""

from __future__ import annotations

from dataclasses import dataclass

import networkx as nx
import numpy as np

from celltrack.constants import VOXEL_SCALE_UM


@dataclass(frozen=True, slots=True)
class Node:
    """A cell detection. Coordinates are integer voxel indices."""

    node_id: int
    t: int
    z: int
    y: int
    x: int

    def scaled_coord(self) -> np.ndarray:
        """(z, y, x) in micrometres, for distance computations."""
        sz, sy, sx = VOXEL_SCALE_UM
        return np.array([self.z * sz, self.y * sy, self.x * sx], dtype=float)


class TrackGraph:
    """Lineage graph for one dataset."""

    def __init__(self) -> None:
        self._g = nx.DiGraph()

    # -- construction -----------------------------------------------------
    def add_node(self, node: Node) -> None:
        self._g.add_node(node.node_id, t=node.t, z=node.z, y=node.y, x=node.x)

    def add_edge(self, source_id: int, target_id: int) -> None:
        self._g.add_edge(source_id, target_id)

    # -- access -----------------------------------------------------------
    @property
    def digraph(self) -> nx.DiGraph:
        return self._g

    def node(self, node_id: int) -> Node:
        d = self._g.nodes[node_id]
        return Node(node_id, d["t"], d["z"], d["y"], d["x"])

    def node_ids(self) -> list[int]:
        return list(self._g.nodes)

    def edges(self) -> list[tuple[int, int]]:
        return list(self._g.edges)

    def num_nodes(self) -> int:
        return self._g.number_of_nodes()

    def num_edges(self) -> int:
        return self._g.number_of_edges()

    def nodes_at_time(self, t: int) -> list[Node]:
        return [self.node(n) for n, d in self._g.nodes(data=True) if d["t"] == t]

    def timepoints(self) -> list[int]:
        return sorted({d["t"] for _, d in self._g.nodes(data=True)})

    def divisions(self) -> list[int]:
        """Node ids with >= 2 outgoing edges (mitosis events)."""
        return [n for n in self._g.nodes if self._g.out_degree(n) >= 2]

    def successors(self, node_id: int) -> list[int]:
        return list(self._g.successors(node_id))

    def __repr__(self) -> str:  # pragma: no cover - trivial
        return f"TrackGraph(nodes={self.num_nodes()}, edges={self.num_edges()})"
