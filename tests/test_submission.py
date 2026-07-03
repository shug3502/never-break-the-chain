"""Tests for submission CSV I/O and validation."""

from __future__ import annotations

import pandas as pd
import pytest

from celltrack.constants import SUBMISSION_COLUMNS
from celltrack.graph import Node, TrackGraph
from celltrack.submit import (
    SubmissionError,
    read_submission,
    validate_submission,
    write_submission,
)


def _sample_graphs() -> dict[str, TrackGraph]:
    g = TrackGraph()
    g.add_node(Node(1, 0, 32, 128, 128))
    g.add_node(Node(2, 1, 33, 130, 125))
    g.add_edge(1, 2)
    h = TrackGraph()
    h.add_node(Node(1, 0, 10, 10, 10))
    return {"44b6": g, "6bba": h}


def test_write_read_roundtrip(tmp_path):
    graphs = _sample_graphs()
    path = write_submission(graphs, tmp_path / "submission.csv")
    df = pd.read_csv(path)
    assert list(df.columns) == list(SUBMISSION_COLUMNS)

    back = read_submission(path)
    assert set(back) == {"44b6", "6bba"}
    assert back["44b6"].num_nodes() == 2
    assert back["44b6"].edges() == [(1, 2)]
    assert back["6bba"].num_nodes() == 1


def test_id_is_consecutive(tmp_path):
    path = write_submission(_sample_graphs(), tmp_path / "s.csv")
    df = pd.read_csv(path)
    assert df["id"].tolist() == list(range(len(df)))


def test_validate_requires_coverage(tmp_path):
    path = write_submission(_sample_graphs(), tmp_path / "s.csv")
    df = pd.read_csv(path)
    validate_submission(df, required_datasets=["44b6", "6bba"])
    with pytest.raises(SubmissionError, match="missing required datasets"):
        validate_submission(df, required_datasets=["44b6", "6bba", "zzzz"])


def test_validate_rejects_dangling_edge(tmp_path):
    path = write_submission(_sample_graphs(), tmp_path / "s.csv")
    df = pd.read_csv(path)
    # Point an edge at a non-existent node.
    df.loc[df["row_type"] == "edge", "target_id"] = 999
    with pytest.raises(SubmissionError, match="unknown node_id"):
        validate_submission(df)


def test_validate_rejects_bad_columns():
    df = pd.DataFrame({"foo": [1]})
    with pytest.raises(SubmissionError, match="columns must be exactly"):
        validate_submission(df)
