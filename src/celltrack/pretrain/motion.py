"""Motion model for the lineage simulator (all in micrometres).

Cells follow a correlated random walk (Ornstein-Uhlenbeck velocity) plus a
smooth low-frequency drift field (a few Fourier modes emulating morphogenetic
divergence/curl flow) plus per-step jitter. At division the two daughters get an
anti-correlated separation impulse — the geometric cue a division head must
learn. Coordinates are ordered ``(z, y, x)`` throughout.
"""

from __future__ import annotations

from collections.abc import Callable

import numpy as np

from celltrack.constants import VOXEL_SCALE_UM

DriftField = Callable[[np.ndarray, int], np.ndarray]


def to_voxel(pos_um: np.ndarray, shape_vox: tuple[int, int, int]) -> np.ndarray:
    """Discretise a µm position to clipped integer voxel indices ``(z, y, x)``."""
    scale = np.asarray(VOXEL_SCALE_UM, dtype=float)
    vox = np.round(np.asarray(pos_um, dtype=float) / scale).astype(int)
    hi = np.asarray(shape_vox, dtype=int) - 1
    return np.clip(vox, 0, hi)


def make_drift_field(config, rng: np.random.Generator, domain_um: np.ndarray) -> DriftField:
    """Build a smooth, time-varying drift velocity field fixed for one dataset.

    ``u(p, t) = drift_speed * mean_m a_m * sin(2π k_m·(p/L) + φ_m + ω_m t)`` with
    ``m`` low-frequency modes (random integer wavevectors, phases, temporal
    frequencies and unit amplitude directions).
    """
    m = max(int(config.drift_n_modes), 1)
    length = np.asarray(domain_um, dtype=float)
    length = np.where(length <= 0, 1.0, length)

    wavevec = rng.integers(0, 3, size=(m, 3)).astype(float)  # low spatial freqs
    zero = np.all(wavevec == 0, axis=1)
    wavevec[zero, 0] = 1.0  # avoid a constant (all-zero) mode
    phase = rng.uniform(0.0, 2.0 * np.pi, size=m)
    omega = rng.uniform(0.0, 0.3, size=m)  # slow temporal drift
    amp = rng.normal(size=(m, 3))
    amp /= np.linalg.norm(amp, axis=1, keepdims=True) + 1e-12
    speed = float(config.drift_speed_um)

    def field(pos_um: np.ndarray, t: int) -> np.ndarray:
        arg = 2.0 * np.pi * (wavevec @ (np.asarray(pos_um) / length)) + phase + omega * t
        return speed * (amp * np.sin(arg)[:, None]).sum(axis=0) / m

    return field


def ou_velocity(vel: np.ndarray, config, rng: np.random.Generator) -> np.ndarray:
    """Advance an Ornstein-Uhlenbeck velocity by one frame."""
    a = float(np.exp(-1.0 / config.ou_tau_frames))
    noise = rng.normal(size=3) * config.ou_sigma_um
    return a * vel + np.sqrt(max(1.0 - a * a, 0.0)) * noise


def step_position(
    pos: np.ndarray,
    vel: np.ndarray,
    config,
    rng: np.random.Generator,
    drift: DriftField,
    t: int,
    domain_um: np.ndarray,
) -> tuple[np.ndarray, np.ndarray, bool]:
    """Advance one cell one frame; return ``(new_pos, new_vel, in_bounds)``.

    With ``boundary="reflect"`` positions are reflected back into the volume and
    ``in_bounds`` is always ``True``; with ``boundary="exit"`` an out-of-volume
    cell is reported (``in_bounds=False``) so the caller can terminate it.
    """
    new_vel = ou_velocity(vel, config, rng)
    jitter = rng.normal(size=3) * config.jitter_um
    new_pos = np.asarray(pos, dtype=float) + new_vel + drift(pos, t) + jitter
    length = np.asarray(domain_um, dtype=float)

    if config.boundary == "exit":
        in_bounds = bool(np.all(new_pos >= 0.0) and np.all(new_pos <= length))
        return new_pos, new_vel, in_bounds

    # Reflect: fold back over each violated face and flip that velocity axis.
    for i in range(3):
        if new_pos[i] < 0.0:
            new_pos[i] = -new_pos[i]
            new_vel[i] = -new_vel[i]
        elif new_pos[i] > length[i]:
            new_pos[i] = 2.0 * length[i] - new_pos[i]
            new_vel[i] = -new_vel[i]
    new_pos = np.clip(new_pos, 0.0, length)  # safety for large overshoots
    return new_pos, new_vel, True


def division_offsets(config, rng: np.random.Generator) -> tuple[np.ndarray, np.ndarray]:
    """Anti-correlated ``(+d, -d)`` daughter separation impulses (µm)."""
    d = rng.normal(size=3)
    d /= np.linalg.norm(d) + 1e-12
    half = 0.5 * float(config.division_push_um)
    return half * d, -half * d
