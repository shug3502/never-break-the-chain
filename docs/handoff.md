# Handoff — celltrack (Biohub Cell Tracking During Development)

Onboarding for the next agent picking up this project. Read `AGENTS.md` first
for conventions and hard constraints; this doc is the *state of play* and *what
to do next*.

Last updated: 2026-07-05.

> **2026-07-05 update — real data is in `data/`.** All 199 train pairs
> (`<ds>.zarr` + `<ds>.geff`) and the 4 test images are present. The loaders
> (`data/io.py`, `data/geff.py`) were verified against them and the metric oracle
> scores GT-vs-GT = 1.0000 on real files. Several earlier assumptions were off —
> see the new **§8 "Verified data facts (2026-07-05)"**. Open items 1, 5 and 6 in
> §3 are now resolved.

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

1. ~~**Real data has never been downloaded.**~~ **Done (2026-07-05).** All 199
   train pairs and 4 test images are in `data/` and the loaders are verified
   against them (§8).
2. **No divisions in the tracker.** The baseline linker is 1-to-1. The core IP —
   a division-aware tracking/lineage solver — is the main value to build.
   Divisions are real and common in the data: 87/199 train datasets contain at
   least one (151 divisions total), so this directly moves the metric.
3. **Detection not run for real.** Cellpose-SAM wrapper exists but has never
   been exercised on a real volume (needs the `detect` extra + data + GPU).
4. **Metric calibration.** `div_weight=0.5` and the exact use of
   `estimated_number_of_nodes` match the community notebook and LB magnitude but
   are not officially confirmed. Calibrate against a real LB submission.
5. ~~**No EDA / notebook.**~~ Partly done — the data facts in §8 are verified;
   a full Quarto `.qmd` (image intensity, per-embryo breakdown, motion stats)
   would still be worthwhile.
6. ~~**Repo not under git.**~~ **Done** — repo is under git with commits and a
   remote.

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

# data is already in ./data (train/ = 199 .zarr+.geff pairs, test/ = 4 .zarr):
celltrack detect --data-dir data/test --det-dir outputs/detections
celltrack track  --det-dir outputs/detections --out submission.csv
celltrack submit submission.csv --data-dir data/test

# Local scoring: the 4 test images are byte-identical copies that ALSO live in
# data/train WITH ground truth, so you can score the real test set offline.
# GT is the .geff dir; --pred-only restricts scoring to the datasets you
# predicted (else the other 195 train GTs score 0 and dilute the result).
celltrack eval submission.csv data/train --pred-only --per-dataset
```

## 6. Landmines (things that bite)

- **Anisotropic scale.** Distances must be in scaled µm (`VOXEL_SCALE_UM`), not
  voxels. z is 4× coarser than xy.
- **Sparse GT.** Predicting more nodes than GT is normal and mostly not
  penalized — except by the explicit over-prediction factor. Edges to unmatched
  nodes are *ignored*, not FPs. Do not "fix" this in the metric.
- **Submission schema is exact** (column order, `-1` sentinels, per-dataset node
  id references). Always go through `submit/`.
- **`eval` against a `.geff` dir scores every dataset in that dir.** A 4-dataset
  submission scored against all of `data/train` is diluted to ~0 (the 195
  un-predicted GTs count as zeros). Use `--pred-only` for local validation.
- **Images are modest, not terabyte-scale.** Each is `(100, 64, 256, 256)`
  `uint16` (~800 MB). Streaming per timepoint is still the default, but you *can*
  hold a whole volume in RAM if needed.
- **Metric regressions are silent.** Any change to `eval/metric.py` or the
  tracker's scoring must come with a test.

## 7. Key references

- Spec: `requirements/2026-07-03-1300-cell-tracking-3d/06-requirements-spec.md`
- Community metric + baseline: `notebooks/lb-0-835-0-842-rule-based-baseline-v3.ipynb`
- Ultrack (the SOTA this competition builds on): Bragantini et al., *Nature
  Methods* 2025.
- Competition: https://www.kaggle.com/competitions/biohub-cell-tracking-during-development

## 8. Verified data facts (2026-07-05)

Measured directly from the downloaded `data/` (loaders confirmed correct).

**Layout.** `data/train/` = 199 `<ds>.zarr` (image) + `<ds>.geff` (GT) pairs;
`data/test/` = 4 `<ds>.zarr` images (no GT). Two embryos by name prefix:
`44b6_*` (71 train) and `6bba_*` (128 train).

**Images (all identical geometry).** Every `.zarr` is a single-scale OME-Zarr v3
array at `<ds>.zarr/0`, shape **`(T=100, Z=64, Y=256, X=256)`**, dtype
`uint16`, chunked `(1, 64, 256, 256)` (one timepoint per chunk), ~800 MB each.
Voxel scale from `zarr.json` = `(1.0, 1.625, 0.40625, 0.40625)` (t,z,y,x) →
confirms `VOXEL_SCALE_UM = (1.625, 0.40625, 0.40625)`. Intensity ~12–4300.

**Ground truth (`.geff` v1.1, directed).** `nodes/props/{t,z,y,x}/values`,
`edges/ids`, and `attributes.geff.extra.estimated_number_of_nodes`. Node coords
are **integer voxel indices** (int64), matching the image grid. Every dataset
has `estimated_number_of_nodes`.

**GT is extremely sparse** — it is a hand-traced subset, not dense segmentation:
- GT nodes per dataset: min 50, median 659, max 1950.
- `estimated_number_of_nodes`: min 3783, median 17909, max 78644.
- → median GT covers only ~3–4% of the true cell population. This is exactly why
  the metric ignores edges to unmatched nodes and applies the `n_est/n_pred`
  over-prediction penalty. **Do not treat missing GT as false positives.**

**Divisions exist and matter.** 87/199 train datasets have ≥1 division; 151
total. Building the division-aware tracker (open item 2) is where the metric is
won.

**GT temporal coverage** runs `t=0` to a max between 50 and 99 (median 99), while
images always have 100 frames — GT may stop before the last frame.

**The test set has offline ground truth.** The 4 test images
(`44b6_0113de3b, 44b6_0b24845f, 6bba_05b6850b, 6bba_05db0fb1`) are **byte-identical
copies of same-named datasets in `data/train`, which include `.geff` GT.** So you
can compute the exact local metric on the real test images:
`celltrack eval <pred.csv> data/train --pred-only`. (Note the Kaggle leaderboard
still scores against its own held-out sparse GT; treat this as a strong local
proxy, and beware overfitting to just these 4.)
