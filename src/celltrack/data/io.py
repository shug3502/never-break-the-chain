"""Chunked IO for OME-Zarr 3D+time datasets.

The competition imagery is OME-Zarr v3. As downloaded, each dataset is a single
full-resolution array at ``<ds>.zarr/0`` with shape ``(T=100, Z=64, Y=256,
X=256)`` and dtype ``uint16`` (~800 MB, chunked one timepoint per chunk). That
fits in RAM, but these helpers still open arrays lazily and read one timepoint
at a time so the pipeline stays memory-bounded if larger volumes appear.
"""

from __future__ import annotations

from pathlib import Path

import zarr


def list_datasets(root: str | Path) -> list[str]:
    """Return dataset names (``.zarr`` folder stems) under ``root``.

    The name (without ``.zarr``) is exactly what the submission's ``dataset``
    column must contain.
    """
    root = Path(root)
    return sorted(p.name[: -len(".zarr")] for p in root.glob("*.zarr"))


def open_dataset(root: str | Path, dataset: str, level: str = "0") -> zarr.Array:
    """Open a dataset's highest-resolution array lazily (no data read).

    The competition data is multiscale zarr v3; the full-resolution array lives
    at ``<dataset>.zarr/<level>`` with shape (T, Z, Y, X).
    """
    root = Path(root)
    store_path = root / f"{dataset}.zarr"
    if not store_path.exists():
        raise FileNotFoundError(store_path)
    return zarr.open(str(store_path / level), mode="r")


def read_volume(array, t: int):
    """Read a single timepoint's 3D volume (z, y, x) from a (t, z, y, x) array."""
    return array[t]
