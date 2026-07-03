# Expert / Detail Questions

Detailed yes/no questions to pin down expected system behaviour, now grounded in
the competition and tooling research (see 03-context-findings.md). Each has a
smart default with rationale.

## Q6: Should the first milestone be a complete end-to-end pipeline that produces a VALID (leaderboard-scorable) submission — even if not yet state-of-the-art — before investing in optimizing the tracking layer?
**Default if unknown:** Yes (a working baseline de-risks the metric/submission format early and gives a score to iterate against; this is how competitive Kaggle work is sequenced)

## Q7: Should the deliverable be structured as a modular, uv-managed Python package (`src/celltrack/` with a CLI: download → detect → track → eval → submit) rather than a single monolithic notebook?
**Default if unknown:** Yes (matches the local-dev choice from Q1, supports reproducibility and testing, and can still export to a Kaggle notebook for the Code Competition)

## Q8: For the detection/segmentation backbone, should we default to Cellpose (including Cellpose-SAM) for 3D nuclei detection as the initial pretrained detector (with Stardist-3D / Ultrack detection kept as swappable alternatives)?
**Default if unknown:** Yes (Cellpose is the most widely-used, robust, permissively-licensed 3D nuclei detector; a swappable interface avoids lock-in)

**Clarified answer:** Yes — specifically default to **Cellpose-SAM** (PyTorch).
Confirmed all candidate detectors are Python/pip-installable. Choosing
Cellpose-SAM keeps the environment lean (PyTorch-only; avoids adding TensorFlow
for Stardist up front). Stardist-3D / Ultrack detection remain swappable
alternatives behind the `detect/` interface.

## Q9: Should we include Ultrack as a benchmark baseline that our OWN tracking/lineage layer is measured against (i.e. we must beat or match Ultrack locally before it is worth submitting)?
**Default if unknown:** Yes (Ultrack is the state of the art this competition is built on; using it as the bar keeps our custom tracker honest and quantifies improvement)

## Q10: Should we build a local evaluation harness using `traccuracy` to compute CTC-style tracking + division metrics offline against provided ground truth, so we can iterate without spending Kaggle submission attempts?
**Default if unknown:** Yes (offline metric parity is essential for fast iteration; `traccuracy` is the standard reimplementation of the likely competition metrics)
