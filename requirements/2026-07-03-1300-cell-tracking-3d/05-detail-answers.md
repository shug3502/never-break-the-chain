# Expert / Detail Answers

## Q6: End-to-end valid submission as the first milestone, before optimizing tracking?
**Answer:** Yes — baseline first. Ship a leaderboard-scorable end-to-end pipeline
early to de-risk metric/format, then iterate on quality.

## Q7: Modular uv-managed Python package (`src/celltrack/` + staged CLI) vs monolithic notebook?
**Answer:** Yes — modular package + CLI (download → detect → track → eval → submit),
with a Kaggle notebook export path for the Code Competition.

## Q8: Cellpose(-SAM) as the default 3D nuclei detector?
**Answer:** Yes — default to **Cellpose-SAM** (PyTorch-only, lean env). Stardist-3D
and Ultrack detection remain swappable alternatives behind the `detect/`
interface. (All candidates confirmed Python/pip-installable.)

## Q9: Ultrack as a benchmark baseline our own tracker must beat/match?
**Answer:** Yes — run Ultrack as the reference; our custom tracking/lineage layer
must beat or match it locally before submission is worthwhile.

## Q10: Local traccuracy harness for offline CTC/division metrics?
**Answer:** Yes — build an offline `traccuracy`-based harness (TRA/LNK/DET/division
F1) for fast iteration without spending submission attempts.
