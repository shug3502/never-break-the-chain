"""Detection recall check: how well do detections cover the sparse GT?

Usage (from repo root, with .venv activated):

    python scripts/check_detection_recall.py \\
        --det-dir outputs/detections \\
        --gt-dir  data/train

For each dataset present in both --det-dir and --gt-dir, reports:
  - recall    = GT nodes matched by a prediction / total GT nodes
  - precision = matched predictions / total predictions (lower bound — GT is sparse)
  - mean match distance (um) for matched pairs

Recall is the critical gate: if it is poor, edge Jaccard is capped no matter
how good the linker is. Tune Cellpose diameter / anisotropy / thresholds until
recall is comfortably > 0.8 on the test datasets before spending time on tracking.
"""

from __future__ import annotations

import argparse
from pathlib import Path

import numpy as np
import pandas as pd

from celltrack.constants import MATCH_MAX_UM, VOXEL_SCALE_UM
from celltrack.data.geff import load_ground_truth
from celltrack.eval.metric import match_nodes
from celltrack.graph import Node, TrackGraph


def _detections_to_graph(csv_path: Path) -> TrackGraph:
    """Build a TrackGraph from a detection CSV (t,z,y,x,probability)."""
    df = pd.read_csv(csv_path)
    graph = TrackGraph()
    for i, row in enumerate(df.itertuples(index=False), start=1):
        graph.add_node(Node(i, int(row.t), int(row.z), int(row.y), int(row.x)))
    return graph


def _recall_stats(
    pred: TrackGraph,
    gt: TrackGraph,
    max_um: float = MATCH_MAX_UM,
) -> dict:
    matching = match_nodes(pred, gt, max_um)
    n_pred = pred.num_nodes()
    n_gt = gt.num_nodes()
    n_matched = len(matching)

    recall = n_matched / n_gt if n_gt > 0 else float("nan")
    precision = n_matched / n_pred if n_pred > 0 else float("nan")

    # Mean match distance in um for matched pairs.
    dists = []
    for pred_id, gt_id in matching.items():
        p = pred.node(pred_id).scaled_coord()
        g = gt.node(gt_id).scaled_coord()
        dists.append(float(np.linalg.norm(p - g)))
    mean_dist = float(np.mean(dists)) if dists else float("nan")

    return {
        "n_pred": n_pred,
        "n_gt": n_gt,
        "n_matched": n_matched,
        "recall": recall,
        "precision": precision,
        "mean_match_um": mean_dist,
    }


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("--det-dir", default="outputs/detections", help="Directory of <dataset>.csv detection files.")
    parser.add_argument("--gt-dir", default="data/train", help="Directory containing <dataset>.geff ground-truth stores.")
    parser.add_argument("--max-um", type=float, default=MATCH_MAX_UM, help=f"Match gate in um (default {MATCH_MAX_UM}).")
    args = parser.parse_args()

    det_dir = Path(args.det_dir)
    gt_dir = Path(args.gt_dir)

    gt_graphs, _ = load_ground_truth(gt_dir)
    det_csvs = {p.stem: p for p in sorted(det_dir.glob("*.csv"))}

    common = sorted(set(det_csvs) & set(gt_graphs))
    if not common:
        print(f"No datasets in common between {det_dir} and {gt_dir}.")
        return

    rows = []
    for dataset in common:
        pred_graph = _detections_to_graph(det_csvs[dataset])
        stats = _recall_stats(pred_graph, gt_graphs[dataset], max_um=args.max_um)
        rows.append({"dataset": dataset, **stats})
        flag = "" if stats["recall"] >= 0.8 else "  *** LOW RECALL ***"
        print(
            f"{dataset}: recall={stats['recall']:.3f}  precision={stats['precision']:.3f}"
            f"  mean_dist={stats['mean_match_um']:.2f}um"
            f"  (n_pred={stats['n_pred']}, n_gt={stats['n_gt']}){flag}"
        )

    df = pd.DataFrame(rows)
    print(
        f"\nSummary over {len(common)} datasets:"
        f"  mean recall={df['recall'].mean():.3f}"
        f"  mean precision={df['precision'].mean():.3f}"
        f"  mean match dist={df['mean_match_um'].mean():.2f}um"
    )
    sz, sy, sx = VOXEL_SCALE_UM
    print(f"(Match gate = {args.max_um} um; voxel scale z={sz}, y={sy}, x={sx} um/vox)")


if __name__ == "__main__":
    main()
