"""Export cross-mission prediction buckets and reports.

Usage example::

    conda activate exoplanets
    python -m src.export_predictions \
      --data-dir ~/nasa/data \
      --artifacts-dir ~/nasa/artifacts \
      --runs cross-mission \
      --threshold-planet 0.95 \
      --threshold-candidate 0.50
"""

from __future__ import annotations

import argparse
import logging
from pathlib import Path
from typing import Dict, Iterable, List

import pandas as pd

if __package__ in (None, ""):
    from dataio import nasa_bucket
    from exo_tabular import predict_cross_mission, train_cross_mission
else:
    from .dataio import nasa_bucket
    from .exo_tabular import predict_cross_mission, train_cross_mission


def parse_args(argv: Iterable[str]) -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Export exoplanet prediction buckets")
    parser.add_argument("--data-dir", type=Path, required=True)
    parser.add_argument("--artifacts-dir", type=Path, required=True)
    parser.add_argument("--runs", choices=["cross-mission"], required=True)
    parser.add_argument("--threshold-planet", type=float, default=0.95)
    parser.add_argument("--threshold-candidate", type=float, default=0.5)
    parser.add_argument("--device", choices=["cpu", "gpu"], default="cpu")
    parser.add_argument("--ensemble", action="store_true", help="Enable ensemble training")
    parser.add_argument("--random-state", type=int, default=42)
    return parser.parse_args(list(argv))


def configure_logging() -> None:
    logging.basicConfig(level=logging.INFO, format="%(asctime)s - %(levelname)s - %(message)s")


def bucketize(probability: float, th_planet: float = 0.95, th_candidate: float = 0.50) -> str:
    if probability >= th_planet:
        return "planet"
    if probability >= th_candidate:
        return "candidate"
    return "non-planet"


def _ensure_columns(df: pd.DataFrame, desired: List[str]) -> pd.DataFrame:
    existing = [col for col in desired if col in df.columns]
    return df.loc[:, existing]


def _write_csv(df: pd.DataFrame, path: Path) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    df.to_csv(path, index=False)


def _write_summary(
    df: pd.DataFrame,
    thresholds: Dict[str, float],
    path: Path,
) -> None:
    lines: List[str] = []
    lines.append("Category counts:")
    counts = df["category"].value_counts().sort_index()
    for label, value in counts.items():
        lines.append(f"  {label}: {int(value)}")
    lines.append("")
    lines.append("NASA category counts:")
    nasa_counts = df["nasa_category"].value_counts().sort_index()
    for label, value in nasa_counts.items():
        lines.append(f"  {label}: {int(value)}")
    lines.append("")
    lines.append("Category vs NASA category:")
    matrix = pd.crosstab(df["category"], df["nasa_category"], dropna=False)
    matrix = matrix.reindex(index=sorted(matrix.index), columns=sorted(matrix.columns))
    lines.append(matrix.to_string())
    lines.append("")
    lines.append("Thresholds:")
    for name, value in thresholds.items():
        lines.append(f"  {name}: {value}")
    path.write_text("\n".join(lines))


def _prepare_predictions(
    mission: str,
    predictions: pd.DataFrame,
    thresholds: Dict[str, float],
) -> pd.DataFrame:
    df = predictions.copy()
    df["category"] = df["proba_planet"].apply(
        bucketize,
        th_planet=thresholds["planet"],
        th_candidate=thresholds["candidate"],
    )
    df["confidence_pct"] = df["proba_planet"] * 100.0
    label_series = df.get("label_text", pd.Series("", index=df.index, dtype=str))
    mission_series = df.get("mission", pd.Series(mission, index=df.index, dtype=str))
    df["nasa_category"] = [
        nasa_bucket(label, mission_value)
        for label, mission_value in zip(label_series, mission_series)
    ]
    return df


def _log_join_gaps(df: pd.DataFrame, mission: str, logger: logging.Logger) -> None:
    label_series = df.get("label_text")
    if label_series is None:
        logger.warning("label_text metadata missing for mission %s", mission)
        return
    missing = label_series.isna().sum()
    if missing:
        logger.warning("%d predictions missing metadata join for mission %s", missing, mission)


def run_cross_mission(args: argparse.Namespace) -> None:
    logger = logging.getLogger("export_predictions")
    data_dir = args.data_dir
    artifacts_dir = args.artifacts_dir
    thresholds = {"planet": args.threshold_planet, "candidate": args.threshold_candidate}
    missions = ["tess", "k2", "kepler"]

    for mission in missions:
        logger.info("Training cross-mission model (test=%s)", mission)
        model_path = train_cross_mission(
            mission,
            data_dir,
            artifacts_dir,
            device=args.device,
            ensemble=args.ensemble,
            random_state=args.random_state,
            logger=logger,
        )
        logger.info("Predicting mission %s", mission)
        predictions = predict_cross_mission(
            mission,
            data_dir,
            artifacts_dir,
            model_path=model_path,
            logger=logger,
        )
        _log_join_gaps(predictions, mission, logger)
        prepared = _prepare_predictions(mission, predictions, thresholds)
        export_dir = artifacts_dir / "exports" / mission
        export_dir.mkdir(parents=True, exist_ok=True)
        ordered_columns = [
            "object_id",
            "mission",
            "proba_planet",
            "confidence_pct",
            "category",
            "period",
            "t0",
            "duration",
            "depth",
            "rp",
            "teq",
            "insolation",
            "eccentricity",
            "sma",
            "snr",
            "impact",
            "mes",
        ]
        predictions_out = _ensure_columns(prepared, ordered_columns)
        _write_csv(predictions_out, export_dir / f"predictions_{mission}.csv")

        for label, filename in (
            ("planet", f"planets_{mission}.csv"),
            ("candidate", f"candidates_{mission}.csv"),
            ("non-planet", f"non_planets_{mission}.csv"),
        ):
            subset = prepared.loc[prepared["category"] == label]
            subset_out = _ensure_columns(subset, ordered_columns + ["nasa_category"])
            _write_csv(subset_out, export_dir / filename)

        discrepancy_specs = [
            (
                (prepared["category"] == "planet") & (prepared["nasa_category"] == "non-planet"),
                f"pred_planet_nasa_nonplanet_{mission}.csv",
            ),
            (
                (prepared["category"] == "planet") & (prepared["nasa_category"] == "candidate"),
                f"pred_planet_nasa_candidate_{mission}.csv",
            ),
            (
                (prepared["category"] == "candidate") & (prepared["nasa_category"] == "non-planet"),
                f"pred_candidate_nasa_nonplanet_{mission}.csv",
            ),
            (
                (prepared["category"] == "non-planet") & (prepared["nasa_category"] == "planet"),
                f"pred_nonplanet_nasa_planet_{mission}.csv",
            ),
            (
                (prepared["category"] == "non-planet") & (prepared["nasa_category"] == "candidate"),
                f"pred_nonplanet_nasa_candidate_{mission}.csv",
            ),
        ]
        discrepancy_columns = ordered_columns + ["nasa_category", "label_text"]
        for mask, filename in discrepancy_specs:
            subset = prepared.loc[mask]
            subset_out = _ensure_columns(subset, discrepancy_columns)
            _write_csv(subset_out, export_dir / filename)

        summary_path = export_dir / f"summary_{mission}.txt"
        _write_summary(prepared, thresholds, summary_path)
        logger.info("Finished exports for mission %s", mission)


def main(argv: Iterable[str]) -> None:
    configure_logging()
    args = parse_args(argv)
    if args.runs != "cross-mission":
        raise NotImplementedError("Only cross-mission run is currently supported.")
    run_cross_mission(args)


if __name__ == "__main__":
    import sys

    main(sys.argv[1:])
