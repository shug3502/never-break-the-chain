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
    diameter: float = typer.Option(None, help="Expected nucleus diameter in voxels (None = auto)."),
    anisotropy: float = typer.Option(
        None,
        help="Z/XY voxel ratio for 3D kernel scaling (only used with --do-3d; None = auto ~4.0).",
    ),
    flow_threshold: float = typer.Option(None, help="Cellpose flow_threshold (None = default)."),
    cellprob_threshold: float = typer.Option(
        None, help="Cellpose cellprob_threshold (None = default)."
    ),
    stitch_threshold: float = typer.Option(0.3, help="2D→3D mask stitch IoU (default path)."),
    do_3d: bool = typer.Option(
        False, "--do-3d", help="Use slower volumetric do_3D path instead of 2D+stitch."
    ),
    amp: bool = typer.Option(
        True, "--amp/--no-amp", help="bf16 autocast on GPU (default on; no-op on CPU)."
    ),
) -> None:
    """Detect nuclei per timepoint with Cellpose-SAM (requires the 'detect' extra)."""
    from celltrack.data.io import open_dataset, read_volume
    from celltrack.detect.cellpose_sam import CellposeSamDetector

    detector = CellposeSamDetector(
        model_type=model,
        gpu=gpu,
        diameter=diameter,
        anisotropy=anisotropy,
        flow_threshold=flow_threshold,
        cellprob_threshold=cellprob_threshold,
        stitch_threshold=stitch_threshold,
        do_3d=do_3d,
        amp=amp,
    )
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


@app.command()
def simulate(
    out_dir: str = typer.Option("outputs/sim", help="Output root."),
    n_datasets: int = typer.Option(3, help="Number of synthetic datasets."),
    n_frames: int = typer.Option(40, help="Frames per dataset."),
    seed: int = typer.Option(0, help="Master RNG seed."),
    randomize: bool = typer.Option(True, help="Domain-randomize config per dataset."),
) -> None:
    """Generate synthetic labelled lineages: noisy detection CSVs + GT submission CSV.

    Writes ``<out_dir>/detections/<name>.csv`` (same schema as ``detect``, so
    ``track`` consumes them unchanged), ``<out_dir>/gt.csv`` (a GT submission
    CSV for ``eval``), and ``<out_dir>/est_nodes.csv``. Oracle workflow::

        celltrack simulate --out-dir outputs/sim
        celltrack track --det-dir outputs/sim/detections --out outputs/sim/pred.csv
        celltrack eval  outputs/sim/pred.csv outputs/sim/gt.csv
    """
    from celltrack.pretrain import SimConfig, iter_datasets, simulate_dataset

    out = Path(out_dir)
    det_dir = out / "detections"
    det_dir.mkdir(parents=True, exist_ok=True)

    if randomize:
        datasets = iter_datasets(n_datasets, seed=seed, n_frames=n_frames)
    else:
        datasets = (
            simulate_dataset(SimConfig(n_frames=n_frames, seed=seed + i), name=f"sim{i:04d}")
            for i in range(n_datasets)
        )

    graphs = {}
    est_rows = []
    for ds in datasets:
        rows = [
            {"t": t, "z": d.z, "y": d.y, "x": d.x, "probability": d.probability}
            for t, dets in sorted(ds.observed.by_time.items())
            for d in dets
        ]
        pd.DataFrame(rows, columns=["t", "z", "y", "x", "probability"]).to_csv(
            det_dir / f"{ds.name}.csv", index=False
        )
        graphs[ds.name] = ds.gt_graph
        est_rows.append({"name": ds.name, "est_nodes": ds.est_nodes})
        typer.echo(f"{ds.name}: {ds.gt_graph.num_nodes()} gt nodes, {len(rows)} detections")

    gt_path = write_submission(graphs, out / "gt.csv")
    pd.DataFrame(est_rows).to_csv(out / "est_nodes.csv", index=False)
    typer.echo(f"Wrote {len(graphs)} datasets to {out} (gt: {gt_path})")


@app.command("eval")
def eval_cmd(
    pred: str = typer.Argument(..., help="Prediction submission CSV."),
    gt: str = typer.Argument(
        ..., help="Ground truth: a submission CSV, or a directory of .geff stores."
    ),
    pred_only: bool = typer.Option(
        False,
        help=(
            "Score only the datasets present in the prediction. Use this for local "
            "validation against a GT directory that holds more datasets than you "
            "predicted (e.g. eval a 4-dataset submission against all of data/train) — "
            "otherwise the un-predicted GT datasets score 0 and dilute the result."
        ),
    ),
    per_dataset: bool = typer.Option(False, help="Print per-dataset edge scores."),
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

    if pred_only:
        keep = set(pred_graphs)
        missing = keep - set(gt_graphs)
        if missing:
            raise typer.BadParameter(f"--pred-only: no ground truth for {sorted(missing)}")
        gt_graphs = {d: g for d, g in gt_graphs.items() if d in keep}
        if est_nodes is not None:
            est_nodes = {d: n for d, n in est_nodes.items() if d in keep}

    result = score(pred_graphs, gt_graphs, est_nodes=est_nodes)
    if per_dataset:
        for dataset, s in sorted(result.per_sample.items()):
            typer.echo(
                f"  {dataset}: edge={s.adjusted:.4f} "
                f"(tp={s.tp} fp={s.fp} fn={s.fn}, n_pred={s.n_pred_nodes})"
            )
    typer.echo(f"Datasets scored:  {len(gt_graphs)}")
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
