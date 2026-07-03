# Handoff — celltrack (Biohub Cell Tracking During Development)

Onboarding for the next agent picking up this project. Read `AGENTS.md` first
for conventions and hard constraints; this doc is the *state of play* and *what
to do next*.

Last updated: 2026-07-03.

## 1. Goal in one paragraph

Build a state-of-the-art 3D+time cell-tracking pipeline for the Kaggle
competition. Detect zebrafish nuclei per timepoint (rent a pretrained
detector), then link them across time and reconstruct lineages **including cell
divisions** (our own tracking layer — the part that wins the metric). Produce a
valid, scorable submission first (Milestone 1), then push quality past the
Ultrack baseline (Milestone 2).

Full requirements & the decision trail: `requirements/2026-07-03-1300-cell-tracking-3d/`
(read `06-requirements-spec.md`; the numbered files show how we got there).

## 2. What already exists (done)

A working, tested Python package scaffold. **15 tests pass, ruff clean.**

- **Submission I/O** (`submit/submission.py`) — writes/reads/validates the exact
  competition CSV (node/edge graph). Fully done.
- **Competition metric** (`eval/metric.py`) — Edge Jaccard + Division Jaccard,
  **reconciled against the community implementation** in
  `notebooks/lb-0-835-0-842-rule-based-baseline-v3.ipynb`. Handles the sparse-GT
  FP rule, the `n_est/n_pred` over-prediction penalty, weight/micro aggregation,
  and the convex combination. Considered correct pending LB calibration of
  `div_weight`.
- **GT loader** (`data/geff.py`) — reads `.geff` zarr graphs +
  `estimated_number_of_nodes`.
- **Image IO** (`data/io.py`) — opens zarr-v3 full-res arrays `<ds>.zarr/0`.
- **Baseline tracker** (`track/nearest_neighbor.py`) — per-frame Hungarian
  linking on scaled distance. 1-to-1 only; **emits no divisions yet.**
- **Detector interface** (`detect/base.py`) + **Cellpose-SAM wrapper**
  (`detect/cellpose_sam.py`, lazy torch import).
- **Kaggle download** (`data/download.py`) — curl + corporate-CA workaround.
- **CLI** (`cli.py`) — `download / detect / track / eval / submit`.

Verified end-to-end on synthetic data: `track → submit → eval` produces a CSV
byte-matching the competition's example and scores correctly.

## 3. What is NOT done / open items

1. **Real data has never been downloaded.** The Kaggle SSL blocker (Netskope +
   OpenSSL 3.x) is unresolved end-to-end. `celltrack download` is written but
   untested against the live API; competition rules must also be accepted on the
   website first. **This is the critical path** — nothing runs on real data
   until it's fixed.
2. **No divisions in the tracker.** The baseline linker is 1-to-1. The core IP —
   a division-aware tracking/lineage solver — is the main value to build.
3. **Detection not run for real.** Cellpose-SAM wrapper exists but has never
   been exercised on a real volume (needs the `detect` extra + data + GPU).
4. **Metric calibration.** `div_weight=0.5` and the exact use of
   `estimated_number_of_nodes` match the community notebook and LB magnitude but
   are not officially confirmed. Calibrate against a real LB submission.
5. **No EDA / notebook.** No Quarto `.qmd` exploring the actual data yet.
6. **Repo not under git.** No `git init`, no commits.

## 4. Recommended next steps (in order)

1. **Unblock data access.** Get `celltrack download` working through Netskope
   (curl + `$REQUESTS_CA_BUNDLE`; `insecure=True` only as a last resort), accept
   the competition rules, and land the real train + test data in `data/`
   (git-ignored). Confirm the 4 test datasets appear:
   `44b6_0113de3b, 44b6_0b24845f, 6bba_05b6850b, 6bba_05db0fb1`.
2. **EDA (Quarto `.qmd`).** Inspect one `.zarr` image and one `.geff` GT:
   shapes, timepoint counts, intensity ranges, node/edge/division counts, and
   how sparse the GT is. Sanity-check `data/io.py` and `data/geff.py` against
   real files.
3. **Milestone 1 — a real scorable submission.** Wire Cellpose-SAM detection on
   a subset → `track` (baseline) → `submit`; validate with dataset coverage;
   score locally with `celltrack eval <pred.csv> <gt_dir>/`. Then submit once to
   anchor the local-vs-LB relationship (calibrate `div_weight`).
4. **Milestone 2 — the tracking layer (core IP).** Add division detection
   (parent → 2 daughters), gap closing, and motion-aware association. The
   community notebook (`notebooks/`) shows useful patterns (two-pass linking,
   gap bridging, short-track filtering, division proposals). Stand up **Ultrack**
   (`.[baseline]`) as the bar to beat; iterate on the local metric.

## 5. Fast path to productivity

```bash
source .venv/bin/activate
uv pip install -e ".[dev]"
python -m pytest -q            # confirm green baseline (15 tests)
celltrack --help              # tour the pipeline stages

# once data is in ./data:
celltrack detect --data-dir data/train --det-dir outputs/detections
celltrack track  --det-dir outputs/detections --out submission.csv
celltrack eval   submission.csv data/train_gt/     # gt dir of .geff files
celltrack submit submission.csv --data-dir data/test
```

## 6. Landmines (things that bite)

- **Anisotropic scale.** Distances must be in scaled µm (`VOXEL_SCALE_UM`), not
  voxels. z is 4× coarser than xy.
- **Sparse GT.** Predicting more nodes than GT is normal and mostly not
  penalized — except by the explicit over-prediction factor. Edges to unmatched
  nodes are *ignored*, not FPs. Do not "fix" this in the metric.
- **Submission schema is exact** (column order, `-1` sentinels, per-dataset node
  id references). Always go through `submit/`.
- **Terabyte-scale images.** Stream per timepoint; never `np.asarray` a whole
  dataset.
- **Metric regressions are silent.** Any change to `eval/metric.py` or the
  tracker's scoring must come with a test.

## 7. Key references

- Spec: `requirements/2026-07-03-1300-cell-tracking-3d/06-requirements-spec.md`
- Community metric + baseline: `notebooks/lb-0-835-0-842-rule-based-baseline-v3.ipynb`
- Ultrack (the SOTA this competition builds on): Bragantini et al., *Nature
  Methods* 2025.
- Competition: https://www.kaggle.com/competitions/biohub-cell-tracking-during-development
