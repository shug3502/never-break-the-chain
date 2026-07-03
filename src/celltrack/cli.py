"""celltrack command-line interface: download -> detect -> track -> eval -> submit.

Intermediate artifacts are plain CSVs so stages can be cached and re-run
independently:

- detect: writes ``<det_dir>/<dataset>.csv`` with columns t,z,y,x,probability
- track:  reads those, writes a submission CSV (nodes + edges)
- eval:   scores a prediction submission CSV against a ground-truth CSV
- submit: validates a submission CSV (optionally enforcing dataset coverage)
"""

from __future__ import annotations

from pathlib import Path

import pandas as pd
import typer

from celltrack.data.io import list_datasets
from celltrack.detect.base import Detection
from celltrack.eval.metric import score
from celltrack.submit.submission import read_submission, validate_submission, write_submission
from celltrack.track.nearest_neighbor import track_frames
from celltrack.data.download import download_competition_data

app = typer.Typer(help="3D+time cell tracking for the Biohub Kaggle competition.")


@app.command()
def download(
    dest: str = typer.Option("data", help="Destination directory."),
    insecure: bool = typer.Option(False, help="curl -k (last resort for SSL)."),
) -> None:
    """Download competition data via the Kaggle API (SSL-workaround)."""

    archive = download_competition_data(dest=dest, insecure=insecure)
    typer.echo(f"Downloaded {archive}")


@app.command()
def detect(
    data_dir: str = typer.Option("data", help="Directory of <dataset>.zarr stores."),
    det_dir: str = typer.Option("outputs/detections", help="Output directory."),
    model: str = typer.Option("cpsam", help="Cellpose model type."),
    gpu: bool = typer.Option(True, help="Use GPU."),
) -> None:
    """Detect nuclei per timepoint with Cellpose-SAM (requires the 'detect' extra)."""
    from celltrack.data.io import open_dataset, read_volume
    from celltrack.detect.cellpose_sam import CellposeSamDetector

    detector = CellposeSamDetector(model_type=model, gpu=gpu)
    out = Path(det_dir)
    out.mkdir(parents=True, exist_ok=True)
    for dataset in list_datasets(data_dir):
        array = open_dataset(data_dir, dataset)
        rows = []
        n_t = array.shape[0]
        for t in range(n_t):
            for d in detector.detect(read_volume(array, t)):
                rows.append({"t": t, "z": d.z, "y": d.y, "x": d.x, "probability": d.probability})
        pd.DataFrame(rows).to_csv(out / f"{dataset}.csv", index=False)
        typer.echo(f"{dataset}: {len(rows)} detections")


@app.command()
def track(
    det_dir: str = typer.Option("outputs/detections", help="Directory of detection CSVs."),
    out: str = typer.Option("submission.csv", help="Output submission CSV."),
) -> None:
    """Link detections into a lineage graph and write a submission CSV."""
    graphs = {}
    for csv in sorted(Path(det_dir).glob("*.csv")):
        dataset = csv.stem
        df = pd.read_csv(csv)
        detections: dict[int, list[Detection]] = {}
        for r in df.itertuples(index=False):
            detections.setdefault(int(r.t), []).append(
                Detection(z=int(r.z), y=int(r.y), x=int(r.x), probability=float(r.probability))
            )
        graphs[dataset] = track_frames(detections)
    path = write_submission(graphs, out)
    typer.echo(f"Wrote submission with {len(graphs)} datasets to {path}")


@app.command("eval")
def eval_cmd(
    pred: str = typer.Argument(..., help="Prediction submission CSV."),
    gt: str = typer.Argument(..., help="Ground truth: a submission CSV, or a directory of .geff stores."),
) -> None:
    """Score a prediction against ground truth using the competition metric.

    If ``gt`` is a directory it is loaded as ``.geff`` ground truth (using each
    dataset's estimated_number_of_nodes for the over-prediction penalty);
    otherwise it is read as a ground-truth submission CSV.
    """
    pred_graphs = read_submission(pred)
    gt_path = Path(gt)
    if gt_path.is_dir():
        from celltrack.data.geff import load_ground_truth

        gt_graphs, est_nodes = load_ground_truth(gt_path)
    else:
        gt_graphs, est_nodes = read_submission(gt), None

    result = score(pred_graphs, gt_graphs, est_nodes=est_nodes)
    typer.echo(f"Edge Jaccard:     {result.edge_jaccard:.4f}")
    typer.echo(f"Division Jaccard: {result.division_jaccard:.4f}")
    typer.echo(f"Combined:         {result.combined:.4f}")


@app.command()
def submit(
    path: str = typer.Argument(..., help="Submission CSV to validate."),
    data_dir: str = typer.Option(
        None, help="If given, enforce coverage of all <dataset>.zarr in this dir."
    ),
) -> None:
    """Validate a submission CSV (schema + optional dataset coverage)."""
    df = pd.read_csv(path)
    required = list_datasets(data_dir) if data_dir else None
    validate_submission(df, required_datasets=required)
    typer.echo(f"OK: {path} is a valid submission.")


if __name__ == "__main__":  # pragma: no cover
    app()
