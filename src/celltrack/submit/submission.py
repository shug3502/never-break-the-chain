"""Read/write/validate the competition submission CSV.

Format (confirmed from the competition Data tab):

    id,dataset,row_type,node_id,t,z,y,x,source_id,target_id
    0,44b6,node,1,0,32,128,128,-1,-1
    2,6bba,edge,-1,-1,-1,-1,-1,1,2

- node rows: row_type=node with node_id,t,z,y,x set; source_id,target_id = -1.
- edge rows: row_type=edge with source_id,target_id set; node_id,t,z,y,x = -1.
- id is a throwaway consecutive integer index.
- dataset = test folder name without the .zarr extension; every test dataset
  must appear in the submission.
"""

from __future__ import annotations

from collections.abc import Iterable, Mapping
from pathlib import Path

import pandas as pd

from celltrack.constants import SENTINEL, SUBMISSION_COLUMNS
from celltrack.graph import Node, TrackGraph


class SubmissionError(ValueError):
    """Raised when a submission fails validation."""


def write_submission(
    graphs: Mapping[str, TrackGraph],
    path: str | Path,
) -> Path:
    """Write per-dataset :class:`TrackGraph`\\ s to a submission CSV.

    Rows are grouped by dataset: all node rows for a dataset, then its edge
    rows. ``id`` is a global consecutive counter.
    """
    path = Path(path)
    rows: list[dict] = []
    idx = 0
    for dataset in sorted(graphs):
        graph = graphs[dataset]
        for node_id in graph.node_ids():
            n = graph.node(node_id)
            rows.append(
                {
                    "id": idx,
                    "dataset": dataset,
                    "row_type": "node",
                    "node_id": n.node_id,
                    "t": n.t,
                    "z": n.z,
                    "y": n.y,
                    "x": n.x,
                    "source_id": SENTINEL,
                    "target_id": SENTINEL,
                }
            )
            idx += 1
        for source_id, target_id in graph.edges():
            rows.append(
                {
                    "id": idx,
                    "dataset": dataset,
                    "row_type": "edge",
                    "node_id": SENTINEL,
                    "t": SENTINEL,
                    "z": SENTINEL,
                    "y": SENTINEL,
                    "x": SENTINEL,
                    "source_id": source_id,
                    "target_id": target_id,
                }
            )
            idx += 1

    df = pd.DataFrame(rows, columns=list(SUBMISSION_COLUMNS))
    if df.empty:
        # Preserve dtypes/columns even for an empty submission.
        df = pd.DataFrame({c: pd.Series(dtype="int64") for c in SUBMISSION_COLUMNS})
        df["dataset"] = pd.Series(dtype="object")
        df["row_type"] = pd.Series(dtype="object")
    path.parent.mkdir(parents=True, exist_ok=True)
    df.to_csv(path, index=False)
    return path


def read_submission(path: str | Path) -> dict[str, TrackGraph]:
    """Read a submission CSV into per-dataset :class:`TrackGraph`\\ s."""
    df = pd.read_csv(path)
    validate_submission(df)
    return _df_to_graphs(df)


def _df_to_graphs(df: pd.DataFrame) -> dict[str, TrackGraph]:
    graphs: dict[str, TrackGraph] = {}
    for dataset, sub in df.groupby("dataset", sort=True):
        g = TrackGraph()
        nodes = sub[sub["row_type"] == "node"]
        for r in nodes.itertuples(index=False):
            g.add_node(Node(int(r.node_id), int(r.t), int(r.z), int(r.y), int(r.x)))
        edges = sub[sub["row_type"] == "edge"]
        for r in edges.itertuples(index=False):
            g.add_edge(int(r.source_id), int(r.target_id))
        graphs[str(dataset)] = g
    return graphs


def validate_submission(
    df: pd.DataFrame,
    required_datasets: Iterable[str] | None = None,
) -> None:
    """Validate a submission dataframe, raising :class:`SubmissionError`.

    Checks: column names/order, row_type values, per-row-type sentinel rules,
    edge endpoints referencing existing node_ids within the same dataset, and
    (optionally) full coverage of ``required_datasets``.
    """
    if list(df.columns) != list(SUBMISSION_COLUMNS):
        raise SubmissionError(
            f"columns must be exactly {list(SUBMISSION_COLUMNS)}, got {list(df.columns)}"
        )

    bad_types = set(df["row_type"].unique()) - {"node", "edge"}
    if bad_types:
        raise SubmissionError(f"row_type must be 'node' or 'edge', found {bad_types}")

    nodes = df[df["row_type"] == "node"]
    edges = df[df["row_type"] == "edge"]

    # Node rows: coordinate/id fields must be non-sentinel integers; links = -1.
    coord_cols = ["node_id", "t", "z", "y", "x"]
    for col in coord_cols:
        if (nodes[col] == SENTINEL).any():
            raise SubmissionError(f"node rows must set '{col}' (found sentinel -1)")
    if not ((edges[["node_id", "t", "z", "y", "x"]] == SENTINEL).all().all()):
        raise SubmissionError("edge rows must set node_id,t,z,y,x to -1")
    if not ((nodes[["source_id", "target_id"]] == SENTINEL).all().all()):
        raise SubmissionError("node rows must set source_id,target_id to -1")

    # Coordinates must be integers.
    for col in coord_cols:
        if not pd.api.types.is_integer_dtype(nodes[col].dtype):
            if not (nodes[col] == nodes[col].round()).all():
                raise SubmissionError(f"node '{col}' must be integer voxel values")

    # Edge endpoints must reference existing node_ids within the same dataset.
    for dataset, sub in df.groupby("dataset"):
        node_ids = set(sub[sub["row_type"] == "node"]["node_id"].astype(int))
        sub_edges = sub[sub["row_type"] == "edge"]
        for r in sub_edges.itertuples(index=False):
            for endpoint in (int(r.source_id), int(r.target_id)):
                if endpoint not in node_ids:
                    raise SubmissionError(
                        f"dataset {dataset!r}: edge references unknown node_id {endpoint}"
                    )

    if required_datasets is not None:
        present = set(df["dataset"].astype(str).unique())
        missing = set(map(str, required_datasets)) - present
        if missing:
            raise SubmissionError(f"missing required datasets: {sorted(missing)}")
