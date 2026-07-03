"""Visualise simulated lineage tracks produced by ``celltrack simulate``.

Reads the ground-truth submission CSV (nodes + parent→child edges) plus the
noisy detection CSVs and renders, per dataset:

  1. XY trajectories coloured by founder lineage, with division events marked.
  2. A lineage tree (time on the vertical axis) for the largest founder clones.
  3. GT nodes vs noisy detections at a single timepoint (the sim-to-real gap).
  4. Per-frame cell counts: live GT cells vs observed detections.

Usage::

    .venv/bin/python scripts/visualize_sim.py --sim-dir outputs/sim
"""

from __future__ import annotations

import argparse
from collections import defaultdict
from pathlib import Path

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import networkx as nx
import numpy as np
import pandas as pd

from celltrack.constants import VOXEL_SCALE_UM
from celltrack.submit.submission import read_submission


def founder_of(g: nx.DiGraph) -> dict[int, int]:
    """Map each node to the id of its lineage founder (root of its tree)."""
    founder: dict[int, int] = {}
    roots = [n for n in g.nodes if g.in_degree(n) == 0]
    for r in roots:
        for n in nx.descendants(g, r) | {r}:
            founder[n] = r
    return founder


def node_frame(g: nx.DiGraph) -> pd.DataFrame:
    """Node attribute table with founder assignment."""
    fo = founder_of(g)
    rows = [
        {"node_id": n, "t": d["t"], "z": d["z"], "y": d["y"], "x": d["x"], "founder": fo.get(n, n)}
        for n, d in g.nodes(data=True)
    ]
    return pd.DataFrame(rows)


def plot_dataset(name: str, graph, det_csv: Path, out_png: Path) -> None:
    g = graph.digraph
    nf = node_frame(g)
    sx, sy = VOXEL_SCALE_UM[2], VOXEL_SCALE_UM[1]

    # Rank founders by clone size so we colour/annotate the biggest lineages.
    clone_sizes = nf.groupby("founder").size().sort_values(ascending=False)
    top_founders = list(clone_sizes.index[:12])
    cmap = plt.get_cmap("tab20")
    colour = {f: cmap(i % 20) for i, f in enumerate(top_founders)}

    divisions = set(graph.divisions())

    fig = plt.figure(figsize=(16, 12))
    gs = fig.add_gridspec(2, 2, hspace=0.28, wspace=0.22)
    ax_traj, ax_tree = fig.add_subplot(gs[0, 0]), fig.add_subplot(gs[0, 1])
    ax_det, ax_counts = fig.add_subplot(gs[1, 0]), fig.add_subplot(gs[1, 1])

    # --- 1. XY trajectories coloured by founder -------------------------
    for src, tgt in g.edges:
        f = nf.loc[nf.node_id == src, "founder"]
        fid = int(f.iloc[0]) if len(f) else -1
        c = colour.get(fid, (0.8, 0.8, 0.8, 0.35))
        s, t = g.nodes[src], g.nodes[tgt]
        ax_traj.plot(
            [s["x"] * sx, t["x"] * sx], [s["y"] * sy, t["y"] * sy],
            "-", color=c, lw=0.6, alpha=0.7 if fid in colour else 0.25,
        )
    div_nodes = [n for n in divisions]
    if div_nodes:
        dx = [g.nodes[n]["x"] * sx for n in div_nodes]
        dy = [g.nodes[n]["y"] * sy for n in div_nodes]
        ax_traj.scatter(dx, dy, s=18, marker="*", color="black", zorder=5,
                        label=f"divisions (n={len(div_nodes)})")
    ax_traj.set(title=f"{name}: XY lineage trajectories", xlabel="x (µm)", ylabel="y (µm)")
    ax_traj.invert_yaxis()
    ax_traj.legend(loc="upper right", fontsize=8)

    # --- 2. Lineage tree for the largest clones -------------------------
    _plot_lineage_tree(ax_tree, g, nf, top_founders[:6], colour)
    ax_tree.set(title=f"{name}: lineage trees (top 6 clones)",
                xlabel="lineage layout", ylabel="frame (t)")
    ax_tree.invert_yaxis()

    # --- 3. GT nodes vs noisy detections at one timepoint ---------------
    det = pd.read_csv(det_csv)
    mid_t = int(nf["t"].median())
    gt_t = nf[nf.t == mid_t]
    det_t = det[det.t == mid_t]
    ax_det.scatter(gt_t.x * sx, gt_t.y * sy, s=40, facecolors="none",
                   edgecolors="tab:green", label=f"GT cells (n={len(gt_t)})")
    ax_det.scatter(det_t.x * sx, det_t.y * sy, s=10, color="tab:red", alpha=0.7,
                   label=f"detections (n={len(det_t)})")
    ax_det.set(title=f"{name}: GT vs noisy detections @ t={mid_t}",
               xlabel="x (µm)", ylabel="y (µm)")
    ax_det.invert_yaxis()
    ax_det.legend(loc="upper right", fontsize=8)

    # --- 4. Per-frame counts -------------------------------------------
    gt_counts = nf.groupby("t").size()
    det_counts = det.groupby("t").size()
    ax_counts.plot(gt_counts.index, gt_counts.values, "-o", ms=3,
                   color="tab:green", label="live GT cells")
    ax_counts.plot(det_counts.index, det_counts.values, "-s", ms=3,
                   color="tab:red", label="detections")
    ax_counts.set(title=f"{name}: population over time", xlabel="frame (t)", ylabel="count")
    ax_counts.legend(loc="upper left", fontsize=8)

    fig.suptitle(f"Simulated lineage dataset: {name}", fontsize=15, y=0.995)
    fig.savefig(out_png, dpi=130, bbox_inches="tight")
    plt.close(fig)


def _plot_lineage_tree(ax, g: nx.DiGraph, nf: pd.DataFrame, founders, colour) -> None:
    """Draw simple lineage dendrograms: x = leaf layout, y = frame."""
    t_of = {n: d["t"] for n, d in g.nodes(data=True)}
    gap = 1.0
    xpos: dict[int, float] = {}

    def assign(node: int) -> float:
        kids = list(g.successors(node))
        if not kids:
            xpos[node] = layout_leaf()
            return xpos[node]
        xs = [assign(k) for k in kids]
        xpos[node] = float(np.mean(xs))
        return xpos[node]

    leaf_counter = {"n": 0.0}

    def layout_leaf() -> float:
        v = leaf_counter["n"]
        leaf_counter["n"] += gap
        return v

    offset = 0.0
    for f in founders:
        leaf_counter["n"] = offset
        assign(f)
        c = colour.get(f, "gray")
        for src, tgt in nx.bfs_edges(g, f):
            ax.plot([xpos[src], xpos[tgt]], [t_of[src], t_of[tgt]], "-", color=c, lw=0.8)
        # mark divisions in this clone
        divs = [n for n in ({f} | nx.descendants(g, f)) if g.out_degree(n) >= 2]
        if divs:
            ax.scatter([xpos[n] for n in divs], [t_of[n] for n in divs],
                       s=14, marker="*", color="black", zorder=5)
        offset = leaf_counter["n"] + 2 * gap
    ax.set_xticks([])


def main() -> None:
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--sim-dir", default="outputs/sim", type=Path)
    ap.add_argument("--out-dir", default=None, type=Path)
    args = ap.parse_args()

    sim_dir: Path = args.sim_dir
    out_dir: Path = args.out_dir or (sim_dir / "figures")
    out_dir.mkdir(parents=True, exist_ok=True)

    graphs = read_submission(str(sim_dir / "gt.csv"))
    for name, graph in graphs.items():
        det_csv = sim_dir / "detections" / f"{name}.csv"
        out_png = out_dir / f"{name}.png"
        plot_dataset(name, graph, det_csv, out_png)
        print(f"{name}: {graph.num_nodes()} nodes, {graph.num_edges()} edges, "
              f"{len(graph.divisions())} divisions -> {out_png}")


if __name__ == "__main__":
    main()
