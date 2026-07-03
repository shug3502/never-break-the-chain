# Discovery Answers

## Q1: Will development and execution happen primarily on your own/local or self-managed compute?
**Answer:** Yes — local/self-managed. Kaggle notebooks at most a secondary export target.

## Q2: Is a CUDA-capable GPU available for training and running deep-learning models?
**Answer:** Yes — GPU available. Deep 3D segmentation + learned components are on the table.

## Q3: Should the solution build on existing OSS frameworks rather than implementing from scratch?
**Answer:** Hybrid (Path C).
- Q3a — Adapt existing pretrained tools for DETECTION/segmentation: **Yes**
- Q3b — Build/own the LINKING + division/lineage (tracking) layer: **Yes**

Rationale (from discussion): the competition metric is dominated by association
and division errors, not segmentation IoU. Detection can be "rented" from a
strong pretrained backbone; the differentiating effort goes into a tunable
tracking + division-detection layer, while always retaining a working detection
stage as a fallback.

## Q4: Should the pipeline automatically download the dataset via the Kaggle API?
**Answer:** Yes — Kaggle API. Reproducible, cached locally, git-ignored.

## Q5: Is producing a valid competition-format submission file a required deliverable?
**Answer:** Yes — the pipeline must output a scorable, leaderboard-ready submission.
