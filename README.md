# celltrack — Biohub Cell Tracking During Development

State-of-the-art 3D+time cell tracking for the Kaggle competition
[*Biohub – Cell Tracking During Development*](https://www.kaggle.com/competitions/biohub-cell-tracking-during-development).

**Strategy (hybrid):** adapt a pretrained detector (**Cellpose-SAM**) for 3D
nuclei detection, and build our **own tracking + division/lineage layer** — the
part of the pipeline the competition metric actually rewards. Benchmarked
against **Ultrack** with an offline harness reimplementing the competition's
custom metric.

Full requirements: `requirements/2026-07-03-1300-cell-tracking-3d/06-requirements-spec.md`.

## The task & metric (confirmed)

- **Predict** cell-nuclei detections and their links over 3D+time, including
  **divisions** (a node with ≥2 outgoing edges).
- **Metric** = **Edge Jaccard** + **Division Jaccard**. Nodes are matched to
  ground truth per timepoint by optimal bipartite assignment on **scaled**
  centroid distance (`z=1.625`, `y=x=0.40625` µm/voxel), gated at **7.0 µm**.
  Ground truth is **sparse**; scores can exceed 1.0.

## Submission format (confirmed)

A single CSV encoding a graph (nodes + edges), grouped by dataset:

```
id,dataset,row_type,node_id,t,z,y,x,source_id,target_id
0,44b6,node,1,0,32,128,128,-1,-1
1,44b6,node,2,1,33,130,125,-1,-1
2,6bba,edge,-1,-1,-1,-1,-1,1,2
```

- **node** rows set `node_id,t,z,y,x` (integer voxel centroids); `source_id/target_id = -1`.
- **edge** rows set `source_id,target_id` (referencing node_ids); other fields `-1`.
- `dataset` = test-set folder name without `.zarr`. Every test dataset must appear.

## Setup

```bash
uv venv
uv pip install -e ".[dev]"          # core + tests
uv pip install -e ".[dev,detect]"   # add Cellpose-SAM (PyTorch) when running detection
```

## CLI

```bash
celltrack download            # fetch competition data into ./data (see SSL note)
celltrack detect  ...         # per-timepoint 3D nuclei detection (Cellpose-SAM)
celltrack track   ...         # link detections into a lineage graph (+divisions)
celltrack eval    ...         # competition metric vs ground truth (+Ultrack baseline)
celltrack submit  ...         # write & validate the submission CSV
```

## Known blocker: corporate SSL (Netskope)

The Kaggle Python client fails TLS verification under Netskope + OpenSSL 3.x
(`Basic Constraints of CA cert not marked critical`). `celltrack download` works
around this via `curl` with the corporate CA bundle; see `src/celltrack/data/download.py`.

## Layout

```
src/celltrack/
  constants.py   # physical voxel scale + match gate (shared by track & eval)
  graph.py       # TrackGraph datastructure
  data/          # zarr IO + Kaggle download
  detect/        # Cellpose-SAM (default) behind a swappable interface
  track/         # OWN linking + division/lineage solver (core IP)
  eval/          # competition metric harness
  submit/        # submission CSV writer/reader/validator
  cli.py
```

## Conventions

- `uv`-managed `.venv`; imports at top of files; Quarto `.qmd` for notebooks.
- Do **not** add Claude as a git commit co-author.
