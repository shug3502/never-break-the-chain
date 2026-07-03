"""Generative configuration for the mechanistic lineage simulator.

``SimConfig`` holds every parameter of the layered generative model (lineage,
motion, detection noise). ``Priors`` + :func:`sample_config` implement
per-dataset *domain randomisation*: drawing a config from broad priors so a
solver pre-trained on the output is robust to nuisances it cannot predict.

Distances/scales are in micrometres (µm); rates are per-frame probabilities or
Poisson means. Nothing here imports torch — pure numpy.
"""

from __future__ import annotations

from dataclasses import dataclass, field

import numpy as np


@dataclass(frozen=True)
class SimConfig:
    """All parameters of one synthetic dataset."""

    # -- volume / time ----------------------------------------------------
    n_frames: int = 40
    shape_vox: tuple[int, int, int] = (60, 512, 512)  # (z, y, x) voxel bounds
    # -- lineage ----------------------------------------------------------
    n_founders: int = 30
    cycle_mean_frames: float = 14.0
    cycle_cv: float = 0.30  # coefficient of variation of the cell-cycle time
    cycle_dist: str = "gamma"  # "gamma" | "lognormal"
    death_rate_per_frame: float = 0.005  # per-cell extrusion/death hazard
    influx_per_frame: float = 0.0  # expected new founders entering FOV / frame
    # -- motion (µm) ------------------------------------------------------
    ou_tau_frames: float = 3.0  # velocity autocorrelation time
    ou_sigma_um: float = 1.2  # per-step velocity noise scale
    drift_speed_um: float = 0.8  # amplitude of the smooth Fourier flow
    drift_n_modes: int = 3  # number of low-frequency drift modes
    jitter_um: float = 0.3  # independent per-step positional jitter
    division_push_um: float = 3.0  # anti-correlated daughter separation impulse
    boundary: str = "reflect"  # "reflect" | "exit"
    # -- detection / imaging noise ---------------------------------------
    fn_rate: float = 0.10  # false-negative dropout probability
    fp_per_frame_frac: float = 0.05  # false positives as a fraction of true nodes
    loc_jitter_xy_um: float = 0.5  # localisation jitter in x/y
    loc_jitter_z_um: float = 1.5  # localisation jitter in z (heavier)
    merge_rate: float = 0.02  # probability a near pair merges
    split_rate: float = 0.01  # probability a detection spuriously splits
    merge_dist_um: float = 4.0  # max separation for a merge candidate
    split_label: str = "none"  # "none" | "same" (label of the extra split det.)
    # -- guards -----------------------------------------------------------
    max_nodes: int = 200_000  # blow-up backstop
    seed: int | None = None


# Broad domain-randomisation priors: (low, high) per randomisable field.
DEFAULT_RANGES: dict[str, tuple[float, float]] = {
    "cycle_mean_frames": (8.0, 24.0),
    "cycle_cv": (0.15, 0.5),
    "death_rate_per_frame": (1e-3, 3e-2),
    "n_founders": (10, 80),
    "influx_per_frame": (0.0, 2.0),
    "ou_tau_frames": (1.0, 8.0),
    "ou_sigma_um": (0.4, 3.0),
    "drift_speed_um": (0.1, 2.5),
    "jitter_um": (0.05, 0.8),
    "division_push_um": (1.5, 5.0),
    "fn_rate": (0.0, 0.35),
    "fp_per_frame_frac": (0.0, 0.25),
    "loc_jitter_xy_um": (0.2, 1.2),
    "loc_jitter_z_um": (0.5, 3.0),
    "merge_rate": (0.0, 0.10),
    "split_rate": (0.0, 0.06),
}

# Fields drawn log-uniformly (rates/scales spanning orders of magnitude).
_LOG_FIELDS = frozenset({"death_rate_per_frame", "ou_sigma_um", "drift_speed_um", "jitter_um"})
# Fields rounded to integers after drawing.
_INT_FIELDS = frozenset({"n_founders"})


@dataclass(frozen=True)
class Priors:
    """Ranges + draw semantics for :func:`sample_config`."""

    ranges: dict[str, tuple[float, float]] = field(default_factory=lambda: dict(DEFAULT_RANGES))
    log_fields: frozenset[str] = _LOG_FIELDS
    int_fields: frozenset[str] = _INT_FIELDS


DEFAULT_PRIORS = Priors()


def sample_config(
    rng: np.random.Generator,
    priors: Priors = DEFAULT_PRIORS,
    *,
    n_frames: int = 40,
    shape_vox: tuple[int, int, int] = (60, 512, 512),
    seed: int | None = None,
    **overrides: object,
) -> SimConfig:
    """Draw a domain-randomised :class:`SimConfig` from ``priors``.

    ``n_frames``, ``shape_vox`` and ``seed`` are fixed (not randomised); any
    keyword in ``overrides`` pins that field to a given value instead of drawing
    it.
    """
    values: dict[str, object] = {}
    for name, (lo, hi) in priors.ranges.items():
        if name in priors.log_fields:
            lo_eff = max(lo, 1e-9)
            val: float = float(np.exp(rng.uniform(np.log(lo_eff), np.log(hi))))
        else:
            val = float(rng.uniform(lo, hi))
        values[name] = int(round(val)) if name in priors.int_fields else val
    values.update(overrides)
    return SimConfig(n_frames=n_frames, shape_vox=shape_vox, seed=seed, **values)
