"""Reimplementation of the competition's combined tracking metric.

    combined = (1 - div_weight) * Edge Jaccard + div_weight * Division Jaccard

Reconciled against the community metric implementation (isakatsuyoshi's
rule-based baseline, see notebooks/lb-0-835-0-842-rule-based-baseline-v3.ipynb).
Confirmed rules:

- Nodes are matched to ground truth **per timepoint** via optimal bipartite
  assignment on **scaled** centroid distance (z=1.625, y=x=0.40625 um/voxel),
  gated at 7.0 um.
- **Edge Jaccard (per sample):** a predicted edge is a TP iff both endpoints
  match GT nodes connected by a GT edge. It is an **FP only when both endpoints
  match GT nodes but there is NO GT edge** between them; predicted edges touching
  *unmatched* nodes are **ignored** (GT is sparse — unlabeled cells must not be
  penalized). FN = GT edges not covered by a TP.
  `raw = TP / (TP + FP + FN)`.
- **Node over-prediction penalty:** `adjusted = raw * (n_est / n_pred)` when
  `n_pred > n_est`, else `raw`. `n_est` is the GT's ``estimated_number_of_nodes``
  (the dense true-cell-count estimate from the .geff metadata); pass it via
  ``est_nodes``. Without it we fall back to the (sparse) GT node count, which
  over-penalizes — so supply ``est_nodes`` when scoring against real GT.
- **Division Jaccard:** a division is a node with >= 2 outgoing edges. A GT
  division is a TP when the predicted node matched to it also splits into nodes
  matched to >= 2 of its GT daughters. FP = predicted divisions not matched to a
  scored GT division. Micro-averaged across all samples.
- **Aggregation:** per-sample adjusted edge Jaccards are weight-averaged by
  (TP + FP + FN); division Jaccards are micro-averaged across samples.

Remaining calibration: the exact ``div_weight`` in the official combination is
not published. 0.5 (simple mean) matches the observed leaderboard magnitude and
is the default; expose it so it can be calibrated against the LB.
"""

from __future__ import annotations

from collections.abc import Mapping
from dataclasses import dataclass

import numpy as np
from scipy.optimize import linear_sum_assignment
from scipy.spatial.distance import cdist

from celltrack.constants import MATCH_MAX_UM
from celltrack.graph import TrackGraph

DIV_WEIGHT: float = 0.5


@dataclass(frozen=True)
class EdgeScore:
    tp: int
    fp: int
    fn: int
    n_pred_nodes: int
    n_est_nodes: float
    jaccard: float  # raw TP/(TP+FP+FN)
    adjusted: float  # after node over-prediction penalty

    @property
    def support(self) -> int:
        return self.tp + self.fp + self.fn


@dataclass(frozen=True)
class MetricResult:
    edge_jaccard: float  # weight-averaged adjusted edge jaccard across samples
    division_jaccard: float  # micro-averaged across samples
    combined: float  # convex combination
    per_sample: dict[str, EdgeScore]
    division_counts: tuple[int, int, int]  # (tp, fp, fn)


def match_nodes(
    pred: TrackGraph,
    gt: TrackGraph,
    max_um: float = MATCH_MAX_UM,
) -> dict[int, int]:
    """Match predicted -> ground-truth node ids, per timepoint.

    Optimal bipartite assignment on scaled centroid distance; pairs above
    ``max_um`` are discarded. Returns ``{pred_node_id: gt_node_id}``.
    """
    matching: dict[int, int] = {}
    timepoints = set(pred.timepoints()) & set(gt.timepoints())
    for t in timepoints:
        pnodes = pred.nodes_at_time(t)
        gnodes = gt.nodes_at_time(t)
        if not pnodes or not gnodes:
            continue
        pcoords = np.vstack([n.scaled_coord() for n in pnodes])
        gcoords = np.vstack([n.scaled_coord() for n in gnodes])
        cost = cdist(pcoords, gcoords)  # euclidean in um
        rows, cols = linear_sum_assignment(cost)
        for r, c in zip(rows, cols):
            if cost[r, c] <= max_um:
                matching[pnodes[r].node_id] = gnodes[c].node_id
    return matching


def edge_jaccard(
    pred: TrackGraph,
    gt: TrackGraph,
    matching: dict[int, int],
    n_est_nodes: float | None = None,
) -> EdgeScore:
    """Edge Jaccard for a single dataset given a node matching.

    ``n_est_nodes`` is the estimated dense true cell count used for the node
    over-prediction penalty; defaults to the (sparse) GT node count.
    """
    gt_edges = set(gt.edges())
    tp = 0
    fp = 0
    covered: set[tuple[int, int]] = set()
    for u, v in pred.edges():
        gu = matching.get(u)
        gv = matching.get(v)
        if gu is None or gv is None:
            continue  # touches an unmatched (possibly unlabeled) node -> ignored
        ge = (gu, gv)
        if ge in gt_edges:
            tp += 1
            covered.add(ge)
        else:
            fp += 1  # both endpoints are annotated cells but no GT edge
    fn = len(gt_edges - covered)

    denom = tp + fp + fn
    raw = tp / denom if denom > 0 else 0.0

    n_pred = pred.num_nodes()
    n_est = float(n_est_nodes) if n_est_nodes else float(gt.num_nodes())
    adjusted = raw
    if n_est and n_pred > n_est:
        adjusted = raw * (n_est / n_pred)

    return EdgeScore(
        tp=tp,
        fp=fp,
        fn=fn,
        n_pred_nodes=n_pred,
        n_est_nodes=n_est,
        jaccard=raw,
        adjusted=adjusted,
    )


def divisions_jaccard(
    pred: TrackGraph,
    gt: TrackGraph,
    matching: dict[int, int],
) -> tuple[int, int, int]:
    """Return (TP, FP, FN) division counts for one dataset.

    A GT division is a TP when the predicted node matched to it is itself a
    division whose successors match >= 2 of the GT daughters. FP = predicted
    divisions not matched to a scored GT division.
    """
    # Inverse match (gt_id -> pred_id), first wins.
    inv: dict[int, int] = {}
    for p, g in matching.items():
        inv.setdefault(g, p)

    gt_divs = set(gt.divisions())
    pred_divs = set(pred.divisions())

    tp = 0
    matched_pred: set[int] = set()
    for d in gt_divs:
        daughters = set(gt.successors(d))
        p = inv.get(d)
        if p is None or pred.digraph.out_degree(p) < 2:
            continue
        pred_daughter_gts = {matching.get(s) for s in pred.successors(p)}
        pred_daughter_gts.discard(None)
        if len(daughters & pred_daughter_gts) >= 2:
            tp += 1
            matched_pred.add(p)

    fn = len(gt_divs) - tp
    fp = len([u for u in pred_divs if u not in matched_pred])
    return tp, fp, fn


def score(
    pred_graphs: Mapping[str, TrackGraph],
    gt_graphs: Mapping[str, TrackGraph],
    est_nodes: Mapping[str, float] | None = None,
    *,
    max_um: float = MATCH_MAX_UM,
    div_weight: float = DIV_WEIGHT,
) -> MetricResult:
    """Compute the combined metric across all datasets present in ``gt_graphs``.

    ``est_nodes`` maps dataset -> estimated dense true cell count (from the GT
    .geff ``estimated_number_of_nodes`` metadata) for the over-prediction
    penalty. Missing entries fall back to the sparse GT node count.
    """
    per_sample: dict[str, EdgeScore] = {}
    div_tp = div_fp = div_fn = 0

    for dataset, gt in gt_graphs.items():
        pred = pred_graphs.get(dataset, TrackGraph())
        matching = match_nodes(pred, gt, max_um=max_um)
        n_est = est_nodes.get(dataset) if est_nodes else None
        per_sample[dataset] = edge_jaccard(pred, gt, matching, n_est_nodes=n_est)
        dtp, dfp, dfn = divisions_jaccard(pred, gt, matching)
        div_tp += dtp
        div_fp += dfp
        div_fn += dfn

    # Edge: weight-average adjusted jaccard by per-sample support (TP+FP+FN).
    total_support = sum(s.support for s in per_sample.values())
    edge = (
        sum(s.adjusted * s.support for s in per_sample.values()) / total_support
        if total_support > 0
        else 0.0
    )

    # Division: micro-averaged across all samples.
    ddenom = div_tp + div_fp + div_fn
    division = div_tp / ddenom if ddenom > 0 else 0.0

    combined = (1 - div_weight) * edge + div_weight * division

    return MetricResult(
        edge_jaccard=edge,
        division_jaccard=division,
        combined=combined,
        per_sample=per_sample,
        division_counts=(div_tp, div_fp, div_fn),
    )
