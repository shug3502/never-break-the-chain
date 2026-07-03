"""Tests for the competition metric harness."""

from __future__ import annotations

from celltrack.eval.metric import edge_jaccard, match_nodes, score
from celltrack.graph import Node, TrackGraph


def _linear_pair(offset_ids: int = 0) -> TrackGraph:
    """A single link across two timepoints."""
    g = TrackGraph()
    g.add_node(Node(1 + offset_ids, 0, 32, 128, 128))
    g.add_node(Node(2 + offset_ids, 1, 33, 130, 125))
    g.add_edge(1 + offset_ids, 2 + offset_ids)
    return g


def _division() -> TrackGraph:
    """A parent at t0 dividing into two daughters at t1."""
    g = TrackGraph()
    g.add_node(Node(1, 0, 30, 100, 100))
    g.add_node(Node(2, 1, 31, 102, 100))
    g.add_node(Node(3, 1, 31, 98, 100))
    g.add_edge(1, 2)
    g.add_edge(1, 3)
    return g


def test_matching_is_coordinate_based():
    gt = _linear_pair()
    pred = _linear_pair(offset_ids=100)  # same coords, different ids
    matching = match_nodes(pred, gt)
    assert matching == {101: 1, 102: 2}


def test_perfect_edge_jaccard():
    gt = _linear_pair()
    pred = _linear_pair(offset_ids=100)
    m = match_nodes(pred, gt)
    s = edge_jaccard(pred, gt, m)
    assert (s.tp, s.fp, s.fn) == (1, 0, 0)
    assert s.jaccard == 1.0


def test_missing_edge_is_fn():
    gt = _linear_pair()
    pred = _linear_pair(offset_ids=100)
    pred.digraph.remove_edge(101, 102)  # drop the only link
    m = match_nodes(pred, gt)
    s = edge_jaccard(pred, gt, m)
    assert (s.tp, s.fp, s.fn) == (0, 0, 1)
    assert s.jaccard == 0.0


def test_division_detected():
    gt = _division()
    pred = _division()
    result = score({"a": pred}, {"a": gt})
    assert result.division_counts == (1, 0, 0)
    assert result.division_jaccard == 1.0
    assert result.edge_jaccard == 1.0
    # combined is a convex combination (div_weight=0.5), not a sum.
    assert result.combined == 1.0


def test_missed_division_is_fn():
    gt = _division()
    pred = _division()
    pred.digraph.remove_edge(1, 3)  # nodes still match, but only one daughter linked
    result = score({"a": pred}, {"a": gt})
    tp, fp, fn = result.division_counts
    assert (tp, fn) == (0, 1)


def _two_tracks() -> TrackGraph:
    """Two independent cells, each linked across t0->t1."""
    g = TrackGraph()
    g.add_node(Node(1, 0, 30, 100, 100))
    g.add_node(Node(2, 0, 30, 200, 100))
    g.add_node(Node(3, 1, 31, 100, 100))
    g.add_node(Node(4, 1, 31, 200, 100))
    g.add_edge(1, 3)
    g.add_edge(2, 4)
    return g


def test_wrong_link_between_matched_nodes_is_fp():
    gt = _two_tracks()
    pred = _two_tracks()
    pred.digraph.remove_edge(2, 4)  # drop a real link ...
    pred.add_edge(1, 4)  # ... add a spurious one between two matched cells
    m = match_nodes(pred, gt)
    s = edge_jaccard(pred, gt, m)
    # 1 TP (1->3), 1 FP (1->4 both matched, no GT edge), 1 FN (2->4 uncovered)
    assert (s.tp, s.fp, s.fn) == (1, 1, 1)


def test_edge_to_unmatched_node_is_ignored():
    gt = _two_tracks()
    pred = _two_tracks()
    pred.add_node(Node(99, 1, 31, 900, 900))  # no GT node near -> unmatched
    pred.add_edge(3, 99)  # endpoint unmatched -> must be ignored, not FP
    m = match_nodes(pred, gt)
    s = edge_jaccard(pred, gt, m)
    assert (s.tp, s.fp, s.fn) == (2, 0, 0)


def test_node_overprediction_penalty():
    gt = _linear_pair()
    pred = _linear_pair(offset_ids=100)
    pred.add_node(Node(199, 0, 30, 900, 900))  # extra unmatched node -> n_pred=3
    m = match_nodes(pred, gt)
    s = edge_jaccard(pred, gt, m)  # n_est defaults to gt.num_nodes()=2
    assert s.jaccard == 1.0
    assert s.n_pred_nodes == 3
    assert s.adjusted == 1.0 * (2 / 3)


def test_est_nodes_used_for_penalty():
    gt = _linear_pair()
    pred = _linear_pair(offset_ids=100)
    pred.add_node(Node(199, 0, 30, 900, 900))
    # With a dense estimate of 3, n_pred (3) is not over -> no penalty.
    result = score({"a": pred}, {"a": gt}, est_nodes={"a": 3})
    assert result.per_sample["a"].adjusted == 1.0
