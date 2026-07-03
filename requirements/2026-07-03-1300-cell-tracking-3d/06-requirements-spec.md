# Requirements Specification — 3D+Time Cell Tracking (Kaggle: Biohub Cell Tracking During Development)

**Slug:** cell-tracking-3d
**Date:** 2026-07-03
**Status:** Requirements complete

---

## 1. Problem Statement

Tracking cells across time in 3D light-sheet microscopy is a major manual
bottleneck in developmental biology. The Kaggle competition *Biohub – Cell
Tracking During Development* asks entrants to **detect cell nuclei in 3D, link
them across time, detect cell divisions (mitosis), and reconstruct cell
lineages** on real zebrafish embryo data (Royer group / Zebrahub, CC0). It is a
**Research · Code Competition** built on the Ultrack method (Bragantini et al.,
*Nature Methods* 2025).

We will build a state-of-the-art, reproducible pipeline to produce a
leaderboard-scorable submission, using a **hybrid strategy**: adapt a pretrained
detector for segmentation, and build our own tracking + division/lineage layer
where the competition metric is won or lost.

## 2. Solution Overview

A modular, `uv`-managed Python package (`src/celltrack/`) exposing a staged CLI:
`download → detect → track → eval → submit`.

- **Detection (adapt):** pretrained **Cellpose-SAM** (PyTorch) as default 3D
  nuclei detector, behind a swappable `detect/` interface (Stardist-3D / Ultrack
  detection as alternatives).
- **Tracking + division (own — core IP):** a tunable association + lineage
  solver with an explicit division model, consuming a generic per-frame
  detection representation.
- **Evaluation (offline):** a harness reimplementing the **competition's custom
  metric** (Edge Jaccard + Division Jaccard, scaled-centroid matching gated at
  7.0 µm — see FR4) against provided ground truth, plus **Ultrack as a benchmark
  baseline** our tracker must beat/match.
- **Submission:** emit the confirmed **node/edge graph CSV** (see FR5); provide a
  Kaggle notebook export path for the Code Competition.

**Sequencing (Q6):** first deliver a complete end-to-end pipeline that yields a
**valid, scorable** submission (baseline), then iterate on tracking quality.

## 3. Functional Requirements

### FR1 — Data acquisition (Q4)
- FR1.1 Reproducibly download the competition dataset via the **Kaggle API** into
  a git-ignored `data/` directory; cache and avoid re-downloading.
- FR1.2 **Resolve the corporate SSL blocker** (see TR6): the Kaggle Python client
  currently fails TLS verification under Netskope/OpenSSL 3.x. Provide a working
  download path (curl with CA bundle, cert workaround, or documented manual
  placement) so acquisition is reproducible.
- FR1.3 Load 3D+time imagery (OME-Zarr, multiscale, terabyte-scale) with chunked
  / streaming IO; never require the full dataset in RAM.

### FR2 — Detection / segmentation
- FR2.1 Default detector = **Cellpose-SAM** producing per-timepoint 3D nuclei
  instances (labels) and/or centroids + confidence.
- FR2.2 Expose a **swappable detector interface** (input: 3D volume → output:
  instances/centroids + probabilities) so Stardist-3D / Ultrack detection can be
  substituted without touching the tracking layer.
- FR2.3 Handle dense, touching, deforming, noisy nuclei; minimize
  under-segmentation of touching nuclei (matters for division scoring).

### FR3 — Tracking + division + lineage (core IP)
- FR3.1 Link detections across consecutive timepoints into tracks (own solver:
  e.g. overlap/NN + assignment, or global ILP/flow).
- FR3.2 Explicitly model and detect **divisions** (parent → two children) and
  produce a lineage graph (parent/child relationships).
- FR3.3 Expose tunable parameters (max displacement, division cost, gap closing,
  etc.) so the tracker can be optimized against the local metric.

### FR4 — Offline evaluation (Q10) — **CONFIRMED custom metric**
The competition uses a **custom combined tracking metric = Edge Jaccard +
Division Jaccard** (NOT the standard CTC/traccuracy metric). The local harness
must reimplement **this exact metric** to have leaderboard parity; `traccuracy`
may assist for graph handling but is not the scored metric.

- FR4.1 **Edge Jaccard:**
  - Match predicted nodes to ground-truth nodes **per timepoint** via **optimal
    bipartite assignment on scaled centroid distance**.
  - **Physical scale:** `z = 1.625`, `y = x = 0.40625` µm/voxel; **match gate =
    max 7.0 µm**.
  - A predicted **edge is a TP** when **both endpoints** match GT nodes that are
    **connected by a GT edge**. `EdgeJaccard = TP / (TP + FP + FN)`.
  - **Node over-prediction penalty (CONFIRMED):**
    `adjusted = raw * (n_est / n_pred)` when `n_pred > n_est`, else `raw`, where
    `n_est` = the GT's `estimated_number_of_nodes` (dense true-cell estimate from
    the `.geff` metadata) and `n_pred` = predicted node count. **An edge is an FP
    only when both endpoints match GT nodes but no GT edge exists**; edges
    touching unmatched nodes are **ignored** (sparse-GT accounting).
- FR4.2 **Division Jaccard:**
  - A division = a node with **≥2 outgoing edges**.
  - For each GT division, check the predicted graph for a **connected component
    that covers the pre-split stage and touches both daughter lineages**.
  - Compute division TP/FP/FN → **micro-averaged Jaccard**.
- FR4.3 **Aggregation:** per-sample adjusted edge Jaccards **weight-averaged by
  (TP + FP + FN)**; division Jaccards **micro-averaged across all samples**.
  **Combined = `(1 - div_weight)*edge + div_weight*division`** (convex
  combination, NOT a sum). `div_weight = 0.5` matches the observed LB magnitude;
  exact value is calibratable.
- FR4.4 **Sparse ground truth:** GT is **sparsely labeled**; the metric accounts
  for this, so the tracker must not be penalized for unlabeled cells but MUST
  avoid gratuitous node over-prediction (see the node penalty). **Scores can
  exceed 1.0.**
- FR4.5 Run **Ultrack** as a baseline and report our tracker relative to it (Q9).
- FR4.6 Harness runs on a data subset for fast iteration.

### FR5 — Submission — **CONFIRMED schema**
The submission is a single CSV encoding a **graph** (nodes + edges), grouped by
dataset. **Columns (exact order):**
`id, dataset, row_type, node_id, t, z, y, x, source_id, target_id`

- FR5.1 **Node rows:** `row_type = node` with `node_id, t, z, y, x` set. `t,z,y,x`
  are **integer centroid coordinates in voxels**. `source_id` and `target_id`
  set to `-1`.
- FR5.2 **Edge rows:** `row_type = edge` with `source_id` and `target_id`
  referencing `node_id`s. `node_id, t, z, y, x` set to `-1`.
- FR5.3 `id` is a **required throwaway index** (consecutive integers, 0-based).
- FR5.4 `dataset` must match the **test-set folder names without the `.zarr`
  extension** (e.g. `44b6`, `6bba`).
- FR5.5 **Every dataset in the test set must appear** in the submission.
- FR5.6 A **validator** must enforce: column names/order, per-row-type field
  rules (`-1` sentinels), integer voxel coords, edge endpoints referencing
  existing node_ids within the same dataset, and full test-set dataset coverage
  — before writing.
- FR5.7 Provide a **Kaggle notebook export** path for the Code Competition
  (offline/runtime-limited environment).

### FR6 — Milestone gating (Q6)
- FR6.1 Milestone 1 = valid scorable submission end-to-end (baseline tracker OK).
- FR6.2 Milestone 2+ = optimize the custom tracking/lineage layer to beat Ultrack.

## 4. Technical Requirements

- **TR1 — Environment:** `uv`-managed, `pyproject.toml` with pinned deps.
  PyTorch-side stack (Cellpose-SAM, Ultrack). GPU-enabled (Q2). Python 3.11+.
  Use the repo `.venv`.
- **TR2 — Package layout (to create; greenfield):**
  ```
  biohub/
    pyproject.toml
    README.md
    .gitignore                 # ignore data/, models/, .venv/, outputs/
    data/                      # Kaggle download target (git-ignored)
    src/celltrack/
      data/     # zarr/tiff IO, chunked/streaming datasets
      detect/   # Cellpose-SAM default + swappable backends
      track/    # OWN linking + division/lineage solver (core IP)
      eval/     # traccuracy CTC/division metrics + Ultrack baseline
      submit/   # format + validate competition submission
      cli.py    # download / detect / track / eval / submit entrypoints
    notebooks/  # Quarto .qmd EDA + Kaggle notebook export
    tests/
  ```
- **TR3 — IO/scale:** dask + zarr for chunked 3D+time access; per-timepoint
  processing; mind GPU memory for 3D volumes (tiling if needed).
- **TR4 — Detector interface:** a stable contract decoupling `detect/` from
  `track/` (per-frame instances/centroids + probabilities).
- **TR5 — Evaluation:** `traccuracy` for CTC metrics; `ultrack` as baseline.
- **TR6 — Corporate SSL:** account for Netskope CA
  (`~/.config/netskope/nscacert_combined.pem`) that OpenSSL 3.x rejects
  (`Basic Constraints ... not marked critical`); provide a reliable Kaggle-API
  download workaround.
- **TR7 — Reproducibility:** deterministic seeds where feasible; config-driven
  runs; cache intermediate artifacts (detections, tracks) between CLI stages.
- **TR8 — Notebooks:** Quarto `.qmd` for EDA (per user preference); provide/export
  the scored Kaggle notebook artifact.

## 5. Implementation Hints & Patterns

- Start `detect/` with Cellpose-SAM pretrained weights (no training) to reach a
  scorable baseline fast; defer fine-tuning.
- Keep the tracking solver **metric-aware**: the score rewards correct **edges**
  and **divisions** — prioritize link correctness and mitosis detection over
  segmentation IoU. Watch node count (over-prediction penalty).
- Node matching uses **scaled** distance (`z×1.625`, `y,x×0.40625` µm) gated at
  **7.0 µm** — bake this scale into both the tracker's association cost and the
  eval harness so they are consistent.
- Wire the CLI as cached stages so `track`/`eval` can iterate on saved detections
  without re-running detection.
- Coordinates in the submission are **integer voxel centroids** (`t,z,y,x`);
  round/cast centroids on write.
- Use a small data subset for the eval loop; only run full-scale for submission.

## 6. Acceptance Criteria

- **AC1:** `celltrack download` fetches the competition data reproducibly
  (SSL blocker resolved), or a documented manual fallback works.
- **AC2:** `celltrack detect` produces per-timepoint 3D nuclei instances with
  Cellpose-SAM; detector is swappable.
- **AC3:** `celltrack track` produces lineages with divisions from detections.
- **AC4:** `celltrack eval` reports TRA/LNK/DET + division F1 via `traccuracy`
  and shows our tracker vs the Ultrack baseline on a subset.
- **AC5:** `celltrack submit` writes a submission validated against the official
  sample format.
- **AC6 (Milestone 1):** the full pipeline produces a submission accepted by
  Kaggle and appears on the leaderboard.
- **AC7 (Milestone 2):** our custom tracking layer meets or beats the Ultrack
  baseline on the local metric.

## 7. Assumptions (for unconfirmed items)

- **A1:** ~~Evaluation uses CTC-style metrics.~~ **CONFIRMED:** custom metric =
  Edge Jaccard + Division Jaccard (see FR4). Remaining unknown: the exact
  **node over-prediction penalty** formula and how the two Jaccards are combined
  into the final leaderboard score (sum vs mean) — reconcile against the
  competition's linked metric-details page/implementation.
- **A2:** ~~Submission is a track_id/parent lineage table.~~ **CONFIRMED:**
  node/edge graph CSV with columns
  `id, dataset, row_type, node_id, t, z, y, x, source_id, target_id` (see FR5).
- **A3:** Ground-truth tracks (and possibly segmentation masks) are provided for
  a training split, enabling offline evaluation.
- **A4:** A CUDA GPU is available for Cellpose-SAM / any training (Q2).
- **A5:** Primary development is local; a Kaggle notebook export satisfies the
  Code Competition submission mechanism (Q1).
- **A6:** Data is CC0 and usable as provided; no extra external data required for
  Milestone 1.

## 8. Out of Scope (for now)

- Fine-tuning/custom-training a detector (Milestone 1 uses pretrained Cellpose-SAM).
- Ensembling multiple detectors/trackers.
- A user-facing GUI/visualization tool (napari viz optional, not required).

## 9. Notes for the implementer

- When committing with git, **do not add Claude as a co-author** of commits.
- Follow user global conventions: imports at top of files; use the repo `.venv`;
  prefer Quarto `.qmd` for notebooks; use `gh` CLI for PRs.
