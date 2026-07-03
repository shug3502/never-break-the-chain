# AGENTS.md — working in the `celltrack` repo

Guidance for AI agents (and humans) contributing to this project. Read this
before making changes.

## What this project is

`celltrack` is a 3D+time cell-tracking pipeline for the Kaggle competition
[*Biohub – Cell Tracking During Development*](https://www.kaggle.com/competitions/biohub-cell-tracking-during-development).
It detects zebrafish nuclei in 3D light-sheet microscopy, links them across
time, and reconstructs lineages including **cell divisions**.

**Strategy (hybrid):** adapt a pretrained detector (**Cellpose-SAM**) for
detection; **build our own tracking + division/lineage layer** — that is where
the competition metric is won. Benchmark against **Ultrack**; iterate offline
with a harness that reimplements the competition metric.

Authoritative spec: `requirements/2026-07-03-1300-cell-tracking-3d/06-requirements-spec.md`.
Start-here for a new agent: `docs/handoff.md`.

## Environment & commands

- Python **3.13**, managed with **uv**. A `.venv` exists at the repo root.
- Setup:
  ```bash
  uv venv
  uv pip install -e ".[dev]"          # core + tests
  uv pip install -e ".[dev,detect]"   # + Cellpose-SAM (PyTorch) for detection
  ```
- Always activate the venv before running things: `source .venv/bin/activate`.
- Use **`just`** for the common tasks (recipes wrap the `.venv` tools; run
  `just` to list them):
  - `just test` — run the suite (`python -m pytest -q`, 15 tests, currently all
    green); pass through args, e.g. `just test -k metric`.
  - `just lint` / `just lint-fix` — `ruff check src tests` (must stay clean).
  - `just format` / `just format-check` — `ruff format src tests`.
  - `just check` — lint + format-check + test (the full pre-commit gate).
- CLI: `celltrack {download,detect,track,eval,submit} --help`

## Repository layout

```
src/celltrack/
  constants.py   # VOXEL_SCALE_UM (z=1.625, y=x=0.40625), MATCH_MAX_UM=7.0, CSV schema
  graph.py       # TrackGraph (networkx-backed): nodes, edges, divisions
  data/
    io.py        # zarr-v3 image IO (full-res array at <ds>.zarr/0), dataset listing
    geff.py      # .geff ground-truth loader (+ estimated_number_of_nodes)
    download.py  # Kaggle API download via curl (corporate-SSL workaround)
  detect/
    base.py      # Detection + Detector Protocol (swappable interface)
    cellpose_sam.py  # default detector; lazy torch/cellpose import
  track/
    nearest_neighbor.py  # Milestone-1 baseline tracker (core-IP placeholder)
  eval/
    metric.py    # competition metric: edge + division Jaccard (RECONCILED)
  submit/
    submission.py  # node/edge CSV writer / reader / validator
  cli.py         # download -> detect -> track -> eval -> submit
tests/           # pytest: submission I/O, metric, geff loader
requirements/    # spec-driven-development artifacts (the "why")
notebooks/       # reference notebooks (e.g. the community metric implementation)
docs/handoff.md  # onboarding for the next agent
```

## Domain facts you MUST respect (confirmed)

- **Physical scale** is anisotropic: `z=1.625`, `y=x=0.40625` µm/voxel. Always
  compute distances in **scaled** µm, never raw voxels. Use
  `celltrack.constants.VOXEL_SCALE_UM` — do not hardcode.
- **Node match gate** = **7.0 µm** (`MATCH_MAX_UM`). The tracker's association
  cost and the metric must use the *same* scale + gate so offline scores track
  the leaderboard.
- **Submission** is a single CSV encoding a graph, columns in exact order:
  `id, dataset, row_type, node_id, t, z, y, x, source_id, target_id`.
  - `node` rows set `node_id,t,z,y,x` (integer voxel centroids); links = `-1`.
  - `edge` rows set `source_id,target_id`; other fields = `-1`.
  - `dataset` = test folder name without `.zarr`; **every** test dataset must appear.
  - Never write this by hand — use `celltrack.submit.write_submission` + `validate_submission`.
- **Metric** = `(1 - div_weight)*EdgeJaccard + div_weight*DivisionJaccard`,
  `div_weight=0.5`. Subtleties that are easy to get wrong (all handled in
  `eval/metric.py`, keep them):
  - An edge is an **FP only when both endpoints match GT nodes but no GT edge
    exists**. Edges touching **unmatched** nodes are **ignored** (GT is sparse).
  - Node over-prediction penalty: `adjusted = raw * (n_est / n_pred)` when
    `n_pred > n_est`, where `n_est` = GT `.geff` `estimated_number_of_nodes`.
  - Edge Jaccard is weight-averaged by support `(TP+FP+FN)`; division Jaccard is
    micro-averaged across samples. **Scores can exceed 1.0.**
- **GT is `.geff`** (zarr graph), not CSV. Use `celltrack.data.geff`.
- **Images are zarr v3**, shape `(T, Z, Y, X)`, full-res at `<ds>.zarr/0`,
  terabyte-scale → stream per timepoint, never load whole datasets into RAM.

## Known blocker

The Kaggle Python client fails TLS under corporate Netskope + OpenSSL 3.x
(`Basic Constraints of CA cert not marked critical`). `celltrack download` works
around it with `curl` + the corporate CA bundle (`src/celltrack/data/download.py`).
`uv`/`pip` against PyPI are unaffected. Competition rules must be accepted on the
Kaggle website once before downloads authorize (else HTTP 403).

## Conventions

- Imports at the **top** of files (project-wide)
- Keep detection decoupled from tracking via the `Detector` Protocol; new
  detectors must not require changes in `track/` or `eval/`.
- Notebooks: prefer **Quarto `.qmd`**.
- Match the style of surrounding code; run `just check` (lint + format + tests)
  before finishing.
- When adding a metric/tracker change, add or update a test that pins the
  behavior; the metric is subtle and regressions are silent.

## Git

- When committing, **do NOT add Claude/AI as a commit co-author.**
- Use the `gh` CLI for GitHub operations / PRs.

## Guardrails

- `data/`, `models/`, `outputs/`, `*.zarr/`, `.venv/` are git-ignored — never
  commit data or model artifacts.
- Never commit `kaggle.json` or any credentials.
- Don't silently change `VOXEL_SCALE_UM`, `MATCH_MAX_UM`, or the submission
  schema — they are competition-defined.
