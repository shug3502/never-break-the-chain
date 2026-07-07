"""Benchmark detect() throughput on one dataset, three ways.

Modes:
  do3d      -- volumetric do_3D path (accurate, slow baseline)
  stitch    -- 2D + Z-stitch, bf16 autocast (new default)
  stitch-noamp -- 2D + Z-stitch, no autocast

Usage: python scripts/bench_detect.py <dataset> [n_timepoints]
"""

from __future__ import annotations

import sys
import time

import torch

from celltrack.data.io import open_dataset, read_volume
from celltrack.detect.cellpose_sam import CellposeSamDetector

DATA_DIR = "data/test"

MODES = {
    "do3d": dict(do_3d=True, amp=True),
    "stitch": dict(do_3d=False, amp=True),
    "stitch-noamp": dict(do_3d=False, amp=False),
}


def bench(mode: str, kwargs: dict, volumes: list) -> tuple[float, int]:
    det = CellposeSamDetector(gpu=True, **kwargs)
    # warm-up on first volume (kernel autotune, model transfer) — not timed
    _ = det.detect(volumes[0])
    torch.cuda.synchronize()
    t0 = time.perf_counter()
    n = 0
    for vol in volumes:
        n += len(det.detect(vol))
    torch.cuda.synchronize()
    dt = time.perf_counter() - t0
    return dt, n


def main() -> None:
    dataset = sys.argv[1]
    n_t = int(sys.argv[2]) if len(sys.argv) > 2 else 100
    array = open_dataset(DATA_DIR, dataset)
    n_t = min(n_t, array.shape[0])
    print(f"dataset={dataset} shape={array.shape} timing {n_t} timepoints", flush=True)
    volumes = [read_volume(array, t) for t in range(n_t)]

    results = {}
    for mode, kwargs in MODES.items():
        dt, n = bench(mode, kwargs, volumes)
        per = dt / n_t
        results[mode] = (dt, per, n)
        print(f"{mode:14s} total={dt:8.2f}s  per-tp={per:6.3f}s  dets={n}", flush=True)

    base = results["do3d"][0]
    print("\n--- speedup vs do3d baseline ---", flush=True)
    for mode, (dt, per, n) in results.items():
        print(f"{mode:14s} {base / dt:5.2f}x", flush=True)


if __name__ == "__main__":
    main()
