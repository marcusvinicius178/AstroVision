"""Export within-mission prediction buckets and discrepancy reports.

Usage example::

    conda activate exoplanets
    python -m src.export_within_mission \
      --data-dir ~/nasa/data \
      --artifacts-dir ~/nasa/artifacts \
      --missions kepler k2 tess \
      --test-size 0.30 \
      --seed 42 \
      --threshold-planet 0.95 \
      --threshold-candidate 0.50
"""

from __future__ import annotations

import argparse
import logging
import sys
from pathlib import Path
from typing import Iterable, List, Sequence

import joblib
import pandas as pd
from sklearn.model_selection import StratifiedShuffleSplit

if __package__ in (None, ""):
    from dataio import (
        infer_feature_types,
        load_mission_dataset,
        nasa_bucket,
        save_feature_schema,
    )
    from modeling import (
        build_pipeline,
        evaluate_binary_classification,
        fit_pipeline_with_fallback,
        save_metrics,
    )
else:
    from .dataio import (
        infer_feature_types,
        load_mission_dataset,
        nasa_bucket,
        save_feature_schema,
    )
    from .modeling import (
        build_pipeline,
        evaluate_binary_classification,
        fit_pipeline_with_fallback,
        save_metrics,
    )


PHYSICAL_EXPORT_COLUMNS: Sequence[str] = (
    "period",
    "t0",
    "duration",
    "depth",
    "rp",
    "teq",
    "insolation",
    "eccentricity",
    "eccentrictity",
    "sma",
    "sy_snum",
    "sy_pnum",
    "stellar_teff",
    "stellar_logg",
    "stellar_radius",
    "stellar_mass",
    "ra",
    "dec",
    "sy_dist",
    "sy_vmag",
    "sy_kmag",
    "sy_gaiamag",
    "koi_model_snr",
    "koi_kepmag",
)


def parse_args(argv: Iterable[str]) -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Export within-mission predictions")
    parser.add_argument("--data-dir", type=Path, required=True)
    parser.add_argument("--artifacts-dir", type=Path, required=True)
    parser.add_argument(
        "--missions",
        nargs="*",
        default=None,
        help="Subset of missions to process (kepler, k2, tess)",
    )
    parser.add_argument("--test-size", type=float, default=0.30)
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--threshold-planet", type=float, default=0.95)
    parser.add_argument("--threshold-candidate", type=float, default=0.50)
    parser.add_argument("--device", choices=["cpu", "gpu"], default="cpu")
    parser.add_argument("--ensemble", action="store_true", help="Enable stacking ensemble")
    parser.add_argument(
        "--recall-target",
        type=float,
        default=0.6,
        help="Recall target used to calibrate candidate threshold",
    )
    parser.add_argument(
        "--keep-default-thresholds",
        action="store_true",
        help="Use provided thresholds even if calibrated values are available",
    )
    parser.add_argument(
        "--no-oversample",
        dest="oversample",
        action="store_false",
        help="Disable RandomOverSampler augmentation (enabled by default)",
    )
    parser.set_defaults(oversample=True)
    return parser.parse_args(list(argv))


def configure_logging() -> None:
    logging.basicConfig(level=logging.INFO, format="%(asctime)s - %(levelname)s - %(message)s")


def _normalize_threshold(value: float) -> float:
    if value > 1:
        value = value / 100.0
    if value < 0:
        value = 0.0
    if value > 1:
        value = 1.0
    return float(value)


def _normalize_probability(value: float) -> float:
    if value > 1:
        value = value / 100.0
    if value < 0:
        value = 0.0
    if value > 1:
        value = 1.0
    return float(value)


def bucketize(probability: float, th_planet: float, th_candidate: float) -> str:
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


def _select_missions(raw: Sequence[str] | None) -> List[str]:
    default = ["kepler", "k2", "tess"]
    if raw is None or len(raw) == 0:
        return default
    normalized: List[str] = []
    allowed = set(default)
    for mission in raw:
        key = mission.lower()
        if key not in allowed:
            raise ValueError(f"Unknown mission '{mission}'. Allowed: {sorted(allowed)}")
        normalized.append(key)
    return normalized


def _format_metrics_text(metrics: dict[str, object], test_size: float, seed: int) -> str:
    lines: List[str] = []
    roc_auc = metrics.get("roc_auc")
    pr_auc = metrics.get("pr_auc")
    positive_rate = metrics.get("positive_rate")
    if roc_auc is not None:
        lines.append(f"ROC-AUC: {roc_auc:.4f}")
    if pr_auc is not None:
        lines.append(f"PR-AUC: {pr_auc:.4f}")
    if positive_rate is not None:
        lines.append(f"Positive rate: {positive_rate:.4f}")
    lines.append("")
    lines.append("Threshold metrics:")
    threshold_metrics = metrics.get("thresholds", {})
    ordered_keys: List[str] = []
    numeric_keys = [k for k in threshold_metrics.keys() if k not in {"calibrated"}]
    ordered_keys.extend(sorted(numeric_keys, key=float))
    if "calibrated" in threshold_metrics:
        ordered_keys.append("calibrated")
    def _fmt_float(value: object) -> str:
        if value is None:
            return "nan"
        try:
            return f"{float(value):.4f}"
        except (TypeError, ValueError):
            return "nan"

    def _fmt_int(value: object) -> str:
        if value is None:
            return "n/a"
        try:
            return str(int(value))
        except (TypeError, ValueError):
            return "n/a"

    for key in ordered_keys:
        values = threshold_metrics.get(key, {})
        label = "calibrated" if key == "calibrated" else f"threshold={float(key):.2f}"
        precision = _fmt_float(values.get("precision"))
        recall = _fmt_float(values.get("recall"))
        f1 = _fmt_float(values.get("f1"))
        predicted = _fmt_int(values.get("predicted_positive"))
        lines.append(
            f"  {label}: precision={precision} recall={recall} f1={f1} predicted_positive={predicted}"
        )
    lines.append("")
    confusion = metrics.get("confusion_matrix", {})
    tn = confusion.get("tn", 0)
    fp = confusion.get("fp", 0)
    fn = confusion.get("fn", 0)
    tp = confusion.get("tp", 0)
    lines.append("Confusion matrix (threshold=0.5):")
    lines.append(f"  [[TN={tn}, FP={fp}], [FN={fn}, TP={tp}]]")
    lines.append("")
    lines.append(f"Test size: {test_size}")
    lines.append(f"Seed: {seed}")
    recommended = metrics.get("recommended_threshold")
    if isinstance(recommended, dict):
        th = _fmt_float(recommended.get("threshold"))
        rec = _fmt_float(recommended.get("recall"))
        prec = _fmt_float(recommended.get("precision"))
        f1 = _fmt_float(recommended.get("f1"))
        lines.append("")
        lines.append(
            "Recommended threshold: "
            f"threshold={th} precision={prec} recall={rec} f1={f1}"
        )
        target = recommended.get("recall_target")
        if target is not None:
            lines.append(f"Recall target: {target:.2f}")
    return "\n".join(lines)


def _prepare_metadata(metadata: pd.DataFrame, logger: logging.Logger) -> pd.DataFrame:
    required_cols = {"object_id", "mission"}
    missing = required_cols - set(metadata.columns)
    if missing:
        raise KeyError(f"Metadata missing required columns: {sorted(missing)}")
    duplicates = metadata.duplicated(subset=["object_id", "mission"], keep=False)
    if duplicates.any():
        logger.warning("%d duplicated metadata rows detected for join keys", int(duplicates.sum()))
    return metadata.copy()


def _join_predictions(
    predictions: pd.DataFrame,
    metadata: pd.DataFrame,
    mission: str,
    logger: logging.Logger,
) -> pd.DataFrame:
    metadata_copy = _prepare_metadata(metadata, logger)
    metadata_copy = metadata_copy.copy()
    metadata_copy["nasa_category"] = [
        nasa_bucket(label, mission_value)
        for label, mission_value in zip(
            metadata_copy.get("label_text", pd.Series(index=metadata_copy.index, dtype=str)),
            metadata_copy.get("mission", pd.Series(mission, index=metadata_copy.index, dtype=str)),
        )
    ]
    join_cols = ["object_id", "mission"]
    merged = predictions.merge(metadata_copy, on=join_cols, how="left")
    missing = merged["label_text"].isna().sum()
    if missing:
        logger.warning("%d rows missing metadata join for mission %s", int(missing), mission)
    else:
        logger.info("All predictions joined with NASA metadata for mission %s", mission)
    return merged


def _ensure_order(df: pd.DataFrame, columns: Sequence[str]) -> pd.DataFrame:
    seen = []
    for column in columns:
        if column in df.columns and column not in seen:
            seen.append(column)
    return df.loc[:, seen]


def _write_csv(df: pd.DataFrame, path: Path) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    df.to_csv(path, index=False)


def _write_text(content: str, path: Path) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(content)


def _fit_with_optional_oversample(
    numeric_cols: Sequence[str],
    categorical_cols: Sequence[str],
    X: pd.DataFrame,
    y: pd.Series,
    *,
    args: argparse.Namespace,
    logger: logging.Logger,
) -> tuple[object, bool]:
    try:
        pipeline = fit_pipeline_with_fallback(
            build_pipeline,
            numeric_cols,
            categorical_cols,
            X,
            y,
            device=args.device,
            ensemble=args.ensemble,
            logger=logger,
            random_state=args.seed,
            builder_kwargs={"oversample": args.oversample},
        )
        return pipeline, args.oversample
    except ImportError as exc:
        if args.oversample:
            logger.warning(
                "Oversampling requested but unavailable (%s). Continuing without RandomOverSampler.",
                exc,
            )
            pipeline = fit_pipeline_with_fallback(
                build_pipeline,
                numeric_cols,
                categorical_cols,
                X,
                y,
                device=args.device,
                ensemble=args.ensemble,
                logger=logger,
                random_state=args.seed,
                builder_kwargs={"oversample": False},
            )
            return pipeline, False
        raise


def _build_summary(
    df: pd.DataFrame,
    mission: str,
    thresholds: dict[str, float],
    base_thresholds: dict[str, float],
    metrics: dict[str, object],
    test_size: float,
    seed: int,
    join_missing: int,
    join_total: int,
) -> str:
    total_rows = len(df)
    joined_rows = join_total - join_missing
    lines: List[str] = [
        f"Mission: {mission}",
        f"Total rows: {total_rows}",
        f"Rows with NASA metadata: {joined_rows}",
        f"Rows missing NASA metadata: {join_missing}",
        "",
        "Prediction category counts:",
    ]
    category_counts = df["category"].value_counts().sort_index()
    for label, value in category_counts.items():
        lines.append(f"  {label}: {int(value)}")
    lines.append("")
    nasa_counts = df["nasa_category"].fillna("unknown").value_counts().sort_index()
    lines.append("NASA category counts:")
    for label, value in nasa_counts.items():
        lines.append(f"  {label}: {int(value)}")
    lines.append("")
    matrix = pd.crosstab(df["category"], df["nasa_category"], dropna=False)
    matrix = matrix.reindex(index=sorted(matrix.index), columns=sorted(matrix.columns))
    lines.append("Prediction vs NASA category matrix:")
    lines.append(matrix.fillna(0).astype(int).to_string())
    lines.append("")
    lines.append("Thresholds used for export:")
    for key, value in thresholds.items():
        lines.append(f"  {key}: {value}")
    lines.append("")
    lines.append("Base thresholds requested:")
    for key, value in base_thresholds.items():
        lines.append(f"  {key}: {value}")
    recommended = metrics.get("recommended_threshold")
    if isinstance(recommended, dict):
        lines.append("")
        lines.append("Calibrated candidate threshold (recall target):")
        lines.append(
            "  threshold={threshold:.4f} precision={precision:.4f} recall={recall:.4f} f1={f1:.4f} target={target}".format(
                threshold=float(recommended.get("threshold", float("nan"))),
                precision=float(recommended.get("precision", float("nan"))),
                recall=float(recommended.get("recall", float("nan"))),
                f1=float(recommended.get("f1", float("nan"))),
                target=recommended.get("recall_target", "n/a"),
            )
        )
    lines.append("")
    lines.append("Holdout metrics (threshold=0.50):")
    threshold_metrics = metrics.get("thresholds", {}).get("0.5", {})
    lines.append(
        "  precision={precision:.4f} recall={recall:.4f} f1={f1:.4f} predicted_positive={predicted}".format(
            precision=float(threshold_metrics.get("precision", float("nan"))),
            recall=float(threshold_metrics.get("recall", float("nan"))),
            f1=float(threshold_metrics.get("f1", float("nan"))),
            predicted=int(threshold_metrics.get("predicted_positive", 0)),
        )
    )
    lines.append("  confusion_matrix=[[{tn}, {fp}], [{fn}, {tp}]]".format(**{
        "tn": metrics.get("confusion_matrix", {}).get("tn", 0),
        "fp": metrics.get("confusion_matrix", {}).get("fp", 0),
        "fn": metrics.get("confusion_matrix", {}).get("fn", 0),
        "tp": metrics.get("confusion_matrix", {}).get("tp", 0),
    }))
    lines.append("")
    lines.append(
        "ROC-AUC={roc:.4f} PR-AUC={pr:.4f} positive_rate={pos:.4f}".format(
            roc=float(metrics.get("roc_auc", float("nan"))),
            pr=float(metrics.get("pr_auc", float("nan"))),
            pos=float(metrics.get("positive_rate", float("nan"))),
        )
    )
    lines.append("")
    lines.append(f"Test size: {test_size}")
    lines.append(f"Seed: {seed}")
    return "\n".join(lines)


def _export_diffs(df: pd.DataFrame, export_dir: Path, mission: str) -> None:
    mapping = [
        ("planet", "non-planet", f"pred_planet_nasa_nonplanet_{mission}.csv"),
        ("planet", "candidate", f"pred_planet_nasa_candidate_{mission}.csv"),
        ("candidate", "non-planet", f"pred_candidate_nasa_nonplanet_{mission}.csv"),
        ("non-planet", "planet", f"pred_nonplanet_nasa_planet_{mission}.csv"),
        ("non-planet", "candidate", f"pred_nonplanet_nasa_candidate_{mission}.csv"),
    ]
    for category, nasa_category, filename in mapping:
        subset = df.loc[
            (df["category"] == category)
            & (df["nasa_category"].fillna("non-planet") == nasa_category)
        ]
        _write_csv(subset, export_dir / filename)


def process_mission(
    mission: str,
    args: argparse.Namespace,
    artifacts_dir: Path,
    base_thresholds: dict[str, float],
    logger: logging.Logger,
) -> None:
    logger.info("Processing mission %s", mission)
    dataset = load_mission_dataset(mission, args.data_dir, logger)
    features = dataset.features
    labels = dataset.labels
    metadata = dataset.metadata

    splitter = StratifiedShuffleSplit(n_splits=1, test_size=args.test_size, random_state=args.seed)
    train_indices, test_indices = next(splitter.split(features, labels))
    X_train = features.iloc[train_indices].copy()
    y_train = labels.iloc[train_indices].copy()
    X_test = features.iloc[test_indices].copy()
    y_test = labels.iloc[test_indices].copy()
    logger.info(
        "Mission %s: %d train samples, %d test samples", mission, X_train.shape[0], X_test.shape[0]
    )

    numeric_cols, categorical_cols = infer_feature_types(features)
    logger.info(
        "Mission %s: %d numeric features, %d categorical features",
        mission,
        len(numeric_cols),
        len(categorical_cols),
    )

    pipeline, oversample_used = _fit_with_optional_oversample(
        numeric_cols,
        categorical_cols,
        X_train,
        y_train,
        args=args,
        logger=logger,
    )

    proba_test = pipeline.predict_proba(X_test)[:, 1]
    metrics = evaluate_binary_classification(
        y_test,
        proba_test,
        thresholds=(0.5, 0.95),
        recall_target=args.recall_target,
    )
    metrics["configuration"] = {
        "mission": mission,
        "split": "within-mission",
        "test_size": args.test_size,
        "seed": args.seed,
        "ensemble": args.ensemble,
        "device": args.device,
        "oversample": oversample_used,
        "recall_target": args.recall_target,
    }

    export_dir = artifacts_dir / "exports_within" / mission
    export_dir.mkdir(parents=True, exist_ok=True)
    metrics_json_path = export_dir / f"metrics_within-{mission}.json"
    save_metrics(metrics, metrics_json_path)
    metrics_text_path = export_dir / f"metrics_within-{mission}.txt"
    _write_text(_format_metrics_text(metrics, args.test_size, args.seed), metrics_text_path)
    logger.info("Wrote metrics to %s and %s", metrics_json_path, metrics_text_path)

    recommended = metrics.get("recommended_threshold")
    thresholds = dict(base_thresholds)
    if (
        recommended is not None
        and not args.keep_default_thresholds
        and isinstance(recommended, dict)
        and recommended.get("threshold") is not None
    ):
        calibrated_value = float(recommended["threshold"])
        planet_threshold = thresholds.get("planet", 0.95)
        if calibrated_value >= planet_threshold:
            logger.warning(
                (
                    "Calibrated candidate threshold %.4f for mission %s is >= planet threshold %.2f; "
                    "keeping candidate threshold %.2f"
                ),
                calibrated_value,
                mission,
                planet_threshold,
                thresholds["candidate"],
            )
        else:
            thresholds["candidate"] = calibrated_value
            logger.info(
                "Using calibrated candidate threshold %.4f for mission %s (recall target %.2f)",
                thresholds["candidate"],
                mission,
                args.recall_target,
            )
    elif recommended is None or not isinstance(recommended, dict):
        logger.warning("No calibrated threshold available for mission %s", mission)
    elif args.keep_default_thresholds:
        logger.info(
            "Calibrated threshold %.4f ignored due to --keep-default-thresholds", recommended.get("threshold")
        )

    final_pipeline, _ = _fit_with_optional_oversample(
        numeric_cols,
        categorical_cols,
        features,
        labels,
        args=args,
        logger=logger,
    )

    models_dir = artifacts_dir / "models_within"
    models_dir.mkdir(parents=True, exist_ok=True)
    model_path = models_dir / f"model_within-{mission}.pkl"
    joblib.dump(final_pipeline, model_path)
    logger.info("Saved model to %s", model_path)

    schema_path = models_dir / f"feature_columns_within-{mission}.json"
    save_feature_schema(features.columns, schema_path)
    logger.info("Saved feature schema to %s", schema_path)

    proba_full = final_pipeline.predict_proba(features)[:, 1]
    base_predictions = pd.DataFrame(
        {
            "object_id": metadata["object_id"].values,
            "mission": metadata["mission"].values,
            "proba_planet": proba_full,
        },
        index=metadata.index,
    )
    base_predictions["proba_planet"] = base_predictions["proba_planet"].apply(_normalize_probability)
    base_predictions["confidence_pct"] = (base_predictions["proba_planet"] * 100).round(2)
    base_predictions["category"] = base_predictions["proba_planet"].apply(
        bucketize,
        th_planet=thresholds["planet"],
        th_candidate=thresholds["candidate"],
    )

    combined = _join_predictions(base_predictions, metadata, mission, logger)
    ordered_columns = [
        "object_id",
        "mission",
        "proba_planet",
        "confidence_pct",
        "category",
        "nasa_category",
        "label_text",
        *PHYSICAL_EXPORT_COLUMNS,
    ]
    predictions_out = _ensure_order(combined, ordered_columns)
    predictions_path = export_dir / f"predictions_within-{mission}.csv"
    _write_csv(predictions_out, predictions_path)
    logger.info("Saved predictions to %s", predictions_path)

    for label, filename in (
        ("planet", f"planets_within-{mission}.csv"),
        ("candidate", f"candidates_within-{mission}.csv"),
        ("non-planet", f"non_planets_within-{mission}.csv"),
    ):
        subset = combined.loc[combined["category"] == label]
        subset_out = _ensure_order(subset, ordered_columns)
        _write_csv(subset_out, export_dir / filename)

    _export_diffs(combined, export_dir, mission)

    join_missing = combined["label_text"].isna().sum()
    summary_text = _build_summary(
        combined,
        mission,
        thresholds,
        base_thresholds,
        metrics,
        args.test_size,
        args.seed,
        int(join_missing),
        len(combined),
    )
    summary_path = export_dir / f"summary_within-{mission}.txt"
    _write_text(summary_text, summary_path)
    logger.info("Wrote summary to %s", summary_path)


def main(argv: Iterable[str] | None = None) -> None:
    if argv is None:
        argv = sys.argv[1:]
    args = parse_args(argv)
    configure_logging()
    logger = logging.getLogger("export_within_mission")
    missions = _select_missions(args.missions)
    logger.info("Running within-mission export for missions: %s", missions)
    thresholds = {
        "planet": _normalize_threshold(args.threshold_planet),
        "candidate": _normalize_threshold(args.threshold_candidate),
    }

    for mission in missions:
        process_mission(
            mission,
            args,
            args.artifacts_dir,
            thresholds,
            logger,
        )


if __name__ == "__main__":
    main()
