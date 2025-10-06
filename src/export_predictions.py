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
import json
import logging
from pathlib import Path
from typing import Dict, Iterable, List, Optional

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
    parser.add_argument("--oversample", action="store_true", help="Apply RandomOverSampler during training")
    parser.add_argument(
        "--recall-target",
        type=float,
        default=0.6,
        help="Recall target used to calibrate thresholds",
    )
    parser.add_argument(
        "--use-calibrated-thresholds",
        action="store_true",
        help="Override candidate threshold with calibrated value saved in metrics",
    )
    return parser.parse_args(list(argv))


def configure_logging() -> None:
    logging.basicConfig(level=logging.INFO, format="%(asctime)s - %(levelname)s - %(message)s")


def _normalize_threshold(value: float) -> float:
    """Convert threshold percentages into probabilities.

    The CLI historically accepted inputs such as ``50`` or ``95`` to express
    50% and 95%.  Interpreting those literally leads to impossible decision
    rules (probabilities are in ``[0, 1]``), which in turn collapses the
    candidate bucket to zero elements.  To make the interface resilient, any
    value greater than one is treated as a percentage and scaled back to the
    ``[0, 1]`` interval.
    """

    if value > 1:
        value = value / 100.0
    if value < 0:
        value = 0.0
    if value > 1:
        value = 1.0
    return float(value)


def _normalize_probability(value: float) -> float:
    """Convert probability percentages into probabilities.

    Some historical prediction exports stored the model score in the ``0-100``
    range instead of ``0-1``.  When those values are consumed by the newer
    bucket logic the candidate range collapses because every value ends up
    greater than the planet threshold.  Detect these legacy percentages and
    coerce them back into a standard probability."""

    if value > 1:
        value = value / 100.0
    if value < 0:
        value = 0.0
    if value > 1:
        value = 1.0
    return float(value)


def bucketize(probability: float, th_planet: float = 0.95, th_candidate: float = 0.50) -> str:
    th_planet = _normalize_threshold(th_planet)
    th_candidate = _normalize_threshold(th_candidate)
    probability = _normalize_probability(probability)
    if th_candidate >= th_planet:
        th_candidate = max(0.0, min(th_candidate, th_planet - 1e-6))
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


def _load_calibrated_threshold(metrics_path: Path) -> Optional[float]:
    if not metrics_path.exists():
        return None
    try:
        data = json.loads(metrics_path.read_text())
    except json.JSONDecodeError:
        return None
    recommended = data.get("recommended_threshold")
    if not isinstance(recommended, dict):
        return None
    value = recommended.get("threshold")
    if value is None:
        return None
    try:
        return float(value)
    except (TypeError, ValueError):
        return None


def _prepare_predictions(
    mission: str,
    predictions: pd.DataFrame,
    thresholds: Dict[str, float],
) -> pd.DataFrame:
    df = predictions.copy()
    df["proba_planet"] = df["proba_planet"].apply(_normalize_probability)
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
    base_thresholds = {
        "planet": _normalize_threshold(args.threshold_planet),
        "candidate": _normalize_threshold(args.threshold_candidate),
    }
    missions = ["tess", "k2", "kepler"]

    for mission in missions:
        logger.info("Training cross-mission model (test=%s)", mission)
        model_path = train_cross_mission(
            mission,
            data_dir,
            artifacts_dir,
            device=args.device,
            ensemble=args.ensemble,
            oversample=args.oversample,
            random_state=args.random_state,
            recall_target=args.recall_target,
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
        mission_thresholds = dict(base_thresholds)
        if args.use_calibrated_thresholds:
            metrics_path = artifacts_dir / f"metrics_{mission}.json"
            calibrated = _load_calibrated_threshold(metrics_path)
            if calibrated is not None:
                if calibrated >= mission_thresholds["planet"]:
                    logger.warning(
                        (
                            "Calibrated candidate threshold %.4f for mission %s is >= planet "
                            "threshold %.2f; keeping candidate threshold %.2f"
                        ),
                        calibrated,
                        mission,
                        mission_thresholds["planet"],
                        mission_thresholds["candidate"],
                    )
                else:
                    mission_thresholds["candidate"] = calibrated
                    logger.info(
                        "Using calibrated candidate threshold %.4f for mission %s",
                        calibrated,
                        mission,
                    )
            else:
                logger.warning(
                    "Calibrated threshold requested but metrics file missing or invalid at %s",
                    metrics_path,
                )
        prepared = _prepare_predictions(mission, predictions, mission_thresholds)
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
        _write_summary(prepared, mission_thresholds, summary_path)
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
