# Context Findings

## Nature of the codebase
`biohub/` is **greenfield** (empty). There are no existing files, patterns, or
conventions to follow — so "files to modify" below is really a **proposed
project structure to create**. Tooling present locally: `python3`, `uv`;
`~/.kaggle/kaggle.json` credentials exist. No `.venv` yet.

## Competition facts (from research)
- **Name:** Biohub – Cell Tracking During Development (Kaggle). Launched
  2026-06-29. Listed as a **Research · Code Competition** (~73 teams at time of
  research).
- **Task:** Detect zebrafish embryo **cell nuclei** in 3D, link them across
  time, detect **divisions (mitosis)**, and reconstruct **cell lineages**.
- **Data provenance:** Royer Group / **Zebrahub** light-sheet microscopy of
  zebrafish embryos (e.g. histone-mCherry labelled nuclei). Described as the
  largest publicly-available cell-tracking dataset by annotation count, released
  **CC0**. Underlying Zebrahub imaging is **OME-Zarr**, multiscale, and
  **terabyte-scale** 3D+time (12h+ time-lapses). Exact voxel resolution /
  timepoint counts live in each dataset's OME-Zarr metadata — to confirm from
  the actual competition download.
- **Prior art the challenge builds on:** **Ultrack** (Bragantini et al., *Nature
  Methods* 2025) — the Royer group's segmentation-agnostic tracking method. It is
  the bar to beat and a strong reference for the tracking layer.

## ⚠️ UPDATE (confirmed by user): the metric below was WRONG
The competition does **not** use the CTC/traccuracy metric. It uses a **custom
combined metric = Edge Jaccard + Division Jaccard** with scaled-centroid node
matching (z=1.625, y=x=0.40625 µm/voxel; 7.0 µm gate), a node over-prediction
penalty, sparse GT, and scores that can exceed 1.0. Submission is a **node/edge
graph CSV** (`id, dataset, row_type, node_id, t, z, y, x, source_id, target_id`).
See 06-requirements-spec.md FR4/FR5 for the authoritative, confirmed spec. The
two inferred sections below are retained only for history.

## RECONCILED metric + data format (from competitor notebook)
Source: `notebooks/lb-0-835-0-842-rule-based-baseline-v3.ipynb` (credit
@isakatsuyoshi / @pilkwang). This gave the authoritative metric implementation,
now mirrored in `src/celltrack/eval/metric.py`:
- Edge FP counts **only** predicted edges whose both endpoints match GT nodes but
  have no GT edge; edges touching unmatched nodes are **ignored** (sparse GT).
- Node over-prediction penalty: `adjusted = raw * (n_est / n_pred)` when
  `n_pred > n_est`; `n_est` = `estimated_number_of_nodes` from GT `.geff` metadata.
- Combined = convex combination `(1-w)*edge + w*div`, `w=0.5` (calibratable).

Confirmed data/GT layout:
- **Images:** zarr v3, full-res array at `<dataset>.zarr/0`, shape `(T,Z,Y,X)`,
  blosc2-compressed chunks at `<dataset>.zarr/0/c/{t}/0/0/0`.
- **Ground truth:** `.geff` zarr graphs — `nodes/ids`, `nodes/props/{t,z,y,x}/values`,
  `edges/ids`, and `attributes.geff.extra.estimated_number_of_nodes`. Loaded by
  `src/celltrack/data/geff.py`.
- **Dataset names:** `{embryo_id}_{fov}` (e.g. `44b6_0113de3b`). Test set = 4
  datasets: `44b6_0113de3b`, `44b6_0b24845f`, `6bba_05b6850b`, `6bba_05db0fb1`.
- Baseline scale reference: rule-based DoG-detect + Hungarian/motion linking
  scores ~0.826–0.842 LB; node counts per dataset ~6k–59k.

## Evaluation metric (SUPERSEDED — inferred, now known to be incorrect)
The competition almost certainly scores with **Cell Tracking Challenge (CTC)**-
style metrics, computed via **`traccuracy`** (the Python library that
reimplements CTC metrics and adds division metrics):
- **TRA** — Tracking Accuracy: normalized AOGM over the full lineage graph
  (nodes + edges). Primary tracking score.
- **LNK** — Linking Accuracy: edge-only errors.
- **DET** — Detection Accuracy: node errors.
- **BIO / division metrics** — Complete Tracks (CT), Track Fractions (TF), and
  a **division F1 (BC)** within a frame tolerance — directly rewards correct
  mitosis detection and lineage reconstruction.
- Composite likely resembles OP_CTB = 0.5·(SEG + TRA) or a linking/BIO blend.

**Implication for strategy:** score is dominated by **association + division**
correctness, not raw segmentation IoU — this validates the chosen hybrid (Path C)
approach: rent detection, own the tracking/lineage/division layer.

## Submission format (inference — VERIFY against `sample_submission` in the data)
A Code Competition on lineage data typically requires a per-detection lineage
table. Most likely one of:
1. **Flat detection+lineage table:** columns approximately
   `id, t, z, y, x, track_id, parent_track_id` (CTC/Ultrack-style, where
   `parent_track_id` encodes divisions), or
2. **CTC `man_track.txt`-style** track table `L B E P` (label, begin frame, end
   frame, parent) plus per-frame label masks.
Exact schema (column names, coordinate units voxels-vs-microns, 0-vs-1 indexing)
**must be read from the competition's `sample_submission` / Data tab.**

## Recommended technical approach (Path C: adapt detection, own tracking)
- **Detection/segmentation (adapt, pretrained):** candidates —
  **Cellpose (incl. Cellpose-SAM) 3D**, **Stardist-3D**, or Ultrack's built-in
  detection (foreground + contour/boundary). Produce per-frame 3D nuclei
  instances or centroid+probability maps.
- **Tracking + division layer (own/build):** a tunable association solver over
  detections — e.g. overlap/nearest-neighbour + Hungarian or a global
  ILP/flow formulation with an explicit **division (parent→2 children) model**;
  benchmark against Ultrack as a baseline. This is where competition edge lives.
- **Evaluation harness:** `traccuracy` locally to compute CTC/division metrics
  against provided ground truth for fast offline iteration before submitting.

## Key technical constraints identified
- **Corporate SSL (Netskope):** the `kaggle` Python client fails TLS verification
  (`Basic Constraints of CA cert not marked critical`) under OpenSSL 3.x, even
  though `REQUESTS_CA_BUNDLE`/`SSL_CERT_FILE` point at
  `~/.config/netskope/nscacert_combined.pem`. Data download step must handle this
  (e.g. curl with the CA bundle, a patched cert, or manual placement). **This is
  a real blocker to resolve before automated Kaggle-API download works.**
- **Data scale:** terabyte-scale OME-Zarr 3D+time → cannot fit in RAM; the
  pipeline must stream/chunk (dask/zarr), process per-timepoint, and be mindful
  of GPU memory for 3D volumes.
- **Code Competition mechanics:** final scored artifact is likely a Kaggle
  notebook run in Kaggle's environment (offline, package/runtime limits), even
  though primary dev is local. Reproducibility + a notebook export path matter.

## Proposed project structure to create (greenfield)
```
biohub/
  pyproject.toml                # uv-managed env, pinned deps
  README.md
  data/                         # git-ignored; Kaggle download target
  src/celltrack/
    data/        # zarr/tiff loading, chunked IO, dataset abstractions
    detect/      # detection/segmentation backbone wrappers (Cellpose/Stardist)
    track/       # OWN linking + division/lineage solver (core IP)
    eval/        # traccuracy-based CTC/division metrics harness
    submit/      # format predictions into competition submission
    cli.py       # entrypoints: download / detect / track / eval / submit
  notebooks/     # .qmd (Quarto) EDA + Kaggle notebook export
  tests/
```

## Open items to confirm during implementation
- Exact evaluation metric + composite weighting (Kaggle Evaluation tab).
- Exact submission schema and coordinate conventions (`sample_submission`).
- Dataset specifics: #datasets, #timepoints, voxel size, channels, GT coverage.
- Whether GT segmentation masks are provided (enables SEG) or only tracks.
- Code Competition runtime limits (GPU type, time, internet-off).
