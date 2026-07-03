"""Test the .geff ground-truth loader against a minimal synthetic store."""

from __future__ import annotations

import json

import zarr

from celltrack.data.geff import read_geff


def _write_geff(path):
    g = zarr.open(str(path), mode="w")
    g.create_array("nodes/ids", shape=(3,), dtype="int64")
    g["nodes/ids"][:] = [1, 2, 3]
    for name, vals in {
        "t": [0, 1, 1],
        "z": [30.4, 31.1, 31.0],
        "y": [100.6, 102.0, 98.0],
        "x": [100.0, 100.0, 100.0],
    }.items():
        g.create_array(f"nodes/props/{name}/values", shape=(3,), dtype="float64")
        g[f"nodes/props/{name}/values"][:] = vals
    g.create_array("edges/ids", shape=(2, 2), dtype="int64")
    g["edges/ids"][:] = [[1, 2], [1, 3]]


def test_read_geff_roundtrip(tmp_path):
    geff = tmp_path / "44b6_0113de3b.geff"
    _write_geff(geff)
    # Attach estimated_number_of_nodes into the store's zarr.json attributes.
    meta_path = geff / "zarr.json"
    meta = json.loads(meta_path.read_text())
    meta.setdefault("attributes", {})["geff"] = {"extra": {"estimated_number_of_nodes": 5000}}
    meta_path.write_text(json.dumps(meta))

    graph, n_est = read_geff(geff)
    assert graph.num_nodes() == 3
    assert set(graph.edges()) == {(1, 2), (1, 3)}
    assert graph.divisions() == [1]  # node 1 has two daughters
    # Float coords rounded to integer voxels.
    n1 = graph.node(1)
    assert (n1.z, n1.y, n1.x) == (30, 101, 100)
    assert n_est == 5000.0
