# Discovery Questions

High-level yes/no questions to understand the problem space for the 3D+time cell
tracking Kaggle competition entry. Each has a smart default for when the answer
is unknown.

## Q1: Will development and execution happen primarily on your own/local or self-managed compute (rather than exclusively inside Kaggle's hosted notebook environment)?
**Default if unknown:** Yes (the repo lives locally and `uv`/`python3` are set up here; Kaggle notebooks can be a secondary export target)

## Q2: Is a CUDA-capable GPU available for training and running deep-learning models?
**Default if unknown:** Yes (state-of-the-art 3D cell segmentation/tracking is effectively GPU-bound)

## Q3: Should the solution build on existing open-source cell-tracking/segmentation frameworks (e.g. Cellpose, Stardist, ultrack, TrackMate, Trackastra) rather than implementing everything from scratch?
**Default if unknown:** Yes (SOTA competition entries almost always leverage and adapt established tools)

**Clarified → Hybrid (Path C):** Adapt an existing pretrained backbone for the
DETECTION/segmentation half, but build/own the LINKING + division-detection
(tracking/lineage) half, since competition edge in cell tracking comes from
association and division quality rather than raw segmentation IoU. This resolves
into two sub-decisions:
- Q3a: Adapt existing tools for detection/segmentation? → Yes
- Q3b: Build/own the tracking + division/lineage layer (vs. using an off-the-shelf tracker as-is)? → Yes

## Q4: Should the pipeline automatically download the competition dataset via the Kaggle API into this repo?
**Default if unknown:** Yes (`~/.kaggle/kaggle.json` is already present; automated, reproducible data pulls are standard)

## Q5: Is producing a valid, competition-format submission file for the Kaggle leaderboard a required deliverable of this work?
**Default if unknown:** Yes (it is a Kaggle competition; a scorable submission is the primary output)
