"""Tests for the mechanistic lineage simulator (celltrack.pretrain)."""

from __future__ import annotations

import numpy as np

from celltrack.constants import VOXEL_SCALE_UM
from celltrack.eval.metric import edge_jaccard, match_nodes, score
from celltrack.graph import Node, TrackGraph
from celltrack.pretrain import SimConfig, iter_datasets, simulate_dataset
from celltrack.track.nearest_neighbor import track_frames


def _cfg(**kw) -> SimConfig:
    """A small, fast config; overrides applied on top."""
    base = dict(
        n_frames=8,
        shape_vox=(20, 128, 128),
        n_founders=6,
        cycle_mean_frames=4.0,
        cycle_cv=0.3,
        death_rate_per_frame=0.0,
        influx_per_frame=0.0,
        ou_sigma_um=0.8,
        drift_speed_um=0.3,
        jitter_um=0.1,
        fn_rate=0.0,
        fp_per_frame_frac=0.0,
        merge_rate=0.0,
        split_rate=0.0,
        seed=0,
    )
    base.update(kw)
    return SimConfig(**base)


def _coords(g: TrackGraph) -> list[tuple[int, int, int, int]]:
    return [(g.node(n).t, g.node(n).z, g.node(n).y, g.node(n).x) for n in g.node_ids()]


def _copy_graph(g: TrackGraph) -> TrackGraph:
    h = TrackGraph()
    for n in g.node_ids():
        nd = g.node(n)
        h.add_node(Node(nd.node_id, nd.t, nd.z, nd.y, nd.x))
    for u, v in g.edges():
        h.add_edge(u, v)
    return h


# -- determinism ---------------------------------------------------------


def test_deterministic_same_seed():
    a = simulate_dataset(_cfg(seed=7))
    b = simulate_dataset(_cfg(seed=7))
    assert a.gt_graph.node_ids() == b.gt_graph.node_ids()
    assert a.gt_graph.edges() == b.gt_graph.edges()
    assert _coords(a.gt_graph) == _coords(b.gt_graph)
    assert a.observed.by_time == b.observed.by_time
    assert a.observed.gt_node_of == b.observed.gt_node_of


def test_different_seed_differs():
    a = simulate_dataset(_cfg(seed=1))
    b = simulate_dataset(_cfg(seed=2))
    assert _coords(a.gt_graph) != _coords(b.gt_graph)


# -- structural invariants ----------------------------------------------


def test_structural_invariants():
    ds = simulate_dataset(_cfg(seed=3, n_frames=10))
    g = ds.gt_graph
    dg = g.digraph
    zmax, ymax, xmax = ds.config.shape_vox

    assert len(g.node_ids()) == len(set(g.node_ids()))  # unique ids
    for n in g.node_ids():
        assert dg.in_degree(n) <= 1  # at most one parent
        assert dg.out_degree(n) <= 2  # continuation or a single 2-way division
        if dg.in_degree(n) == 0:
            assert g.node(n).t == 0  # founders only at t=0 (no influx here)
        nd = g.node(n)
        assert 0 <= nd.z < zmax and 0 <= nd.y < ymax and 0 <= nd.x < xmax
    for d in g.divisions():
        assert dg.out_degree(d) == 2  # divisions are exactly binary
    for u, v in g.edges():
        assert g.node(v).t == g.node(u).t + 1  # edges connect consecutive frames


# -- statistical ---------------------------------------------------------


def _count_divisions(**kw) -> int:
    total = 0
    for s in range(4):
        ds = simulate_dataset(_cfg(seed=s, n_frames=12, n_founders=8, **kw))
        total += len(ds.gt_graph.divisions())
    return total


def test_more_divisions_with_shorter_cycle():
    assert _count_divisions(cycle_mean_frames=3.0) > _count_divisions(cycle_mean_frames=12.0)


def _mean_step_um(**kw) -> float:
    ds = simulate_dataset(
        _cfg(seed=0, n_frames=10, n_founders=8, drift_speed_um=0.0, jitter_um=0.0, **kw)
    )
    g = ds.gt_graph
    steps = [
        float(np.linalg.norm(g.node(v).scaled_coord() - g.node(u).scaled_coord()))
        for u, v in g.edges()
    ]
    return float(np.mean(steps))


def test_displacement_scales_with_sigma():
    assert _mean_step_um(ou_sigma_um=0.4) < _mean_step_um(ou_sigma_um=2.5)


# -- detection-noise layers ---------------------------------------------


def _n_observed(ds) -> int:
    return sum(len(v) for v in ds.observed.by_time.values())


def test_fn_dropout_reduces_count():
    clean = simulate_dataset(_cfg(seed=0))
    noisy = simulate_dataset(_cfg(seed=0, fn_rate=0.5))
    assert _n_observed(noisy) < _n_observed(clean)
    # no FPs -> every observation is labelled
    assert all(lab is not None for labs in noisy.observed.gt_node_of.values() for lab in labs)


def test_fp_clutter_adds_unlabeled():
    ds = simulate_dataset(_cfg(seed=0, fp_per_frame_frac=0.5))
    assert _n_observed(ds) > ds.gt_graph.num_nodes()
    assert any(lab is None for labs in ds.observed.gt_node_of.values() for lab in labs)


def test_jitter_heavier_in_z():
    ds = simulate_dataset(
        _cfg(
            seed=0,
            ou_sigma_um=0.5,
            drift_speed_um=0.0,
            jitter_um=0.0,
            loc_jitter_z_um=3.0,
            loc_jitter_xy_um=0.2,
        )
    )
    z_err: list[float] = []
    xy_err: list[float] = []
    for t, dets in ds.observed.by_time.items():
        for d, gid in zip(dets, ds.observed.gt_node_of[t]):
            gnode = ds.gt_graph.node(gid)
            z_err.append(abs(d.z - gnode.z) * VOXEL_SCALE_UM[0])
            xy_err.append(abs(d.y - gnode.y) * VOXEL_SCALE_UM[1])
            xy_err.append(abs(d.x - gnode.x) * VOXEL_SCALE_UM[2])
    assert np.mean(z_err) > np.mean(xy_err)


# -- metric / tracker oracle --------------------------------------------


def test_gt_self_score_is_perfect():
    ds = simulate_dataset(_cfg(seed=0, n_frames=10))
    g = ds.gt_graph
    n_div = len(g.divisions())
    assert n_div > 0
    result = score({ds.name: g}, {ds.name: g}, est_nodes={ds.name: ds.est_nodes})
    assert result.combined == 1.0
    assert result.division_counts == (n_div, 0, 0)


def test_match_nodes_is_a_perfect_coordinate_matching():
    ds = simulate_dataset(_cfg(seed=0))
    g = ds.gt_graph
    m = match_nodes(g, g)
    assert len(m) == g.num_nodes()
    for p, gg in m.items():
        assert tuple(g.node(p).scaled_coord()) == tuple(g.node(gg).scaled_coord())


def test_clean_observations_track_perfectly():
    cfg = _cfg(
        seed=0,
        n_frames=8,
        n_founders=5,
        cycle_mean_frames=1000.0,  # effectively no divisions -> 1-to-1 is exact
        ou_sigma_um=0.4,
        drift_speed_um=0.1,
        jitter_um=0.0,
        loc_jitter_xy_um=0.0,
        loc_jitter_z_um=0.0,
    )
    ds = simulate_dataset(cfg)
    assert len(ds.gt_graph.divisions()) == 0
    pred = track_frames(ds.observed.by_time)
    result = score({ds.name: pred}, {ds.name: ds.gt_graph}, est_nodes={ds.name: ds.est_nodes})
    assert result.edge_jaccard >= 0.99


def test_overprediction_penalty_from_est_nodes():
    ds = simulate_dataset(_cfg(seed=0, n_frames=6))
    g = ds.gt_graph
    pred = _copy_graph(g)
    nid = max(g.node_ids()) + 1
    for _ in range(g.num_nodes()):  # double node count with far-corner extras
        pred.add_node(Node(nid, 0, 0, 0, 0))
        nid += 1
    m = match_nodes(pred, g)
    s = edge_jaccard(pred, g, m, n_est_nodes=ds.est_nodes)
    assert s.jaccard > 0.0
    assert s.n_pred_nodes > ds.est_nodes
    assert s.adjusted < s.jaccard


# -- domain randomisation ------------------------------------------------


def test_iter_datasets_reproducible_and_named():
    a = [d.gt_graph.num_nodes() for d in iter_datasets(3, seed=0, n_frames=6)]
    b = [d.gt_graph.num_nodes() for d in iter_datasets(3, seed=0, n_frames=6)]
    assert a == b
    names = [d.name for d in iter_datasets(2, seed=0, n_frames=6)]
    assert names == ["sim0000", "sim0001"]
