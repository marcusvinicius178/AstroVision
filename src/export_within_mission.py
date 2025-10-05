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
from pathlib import Path
from typing import Iterable, List, Sequence

import joblib
import numpy as np
import pandas as pd
from sklearn.metrics import (
    average_precision_score,
    confusion_matrix,
    precision_recall_fscore_support,
    roc_auc_score,
)
from sklearn.model_selection import StratifiedShuffleSplit

if __package__ in (None, ""):
    from dataio import (
        infer_feature_types,
        load_mission_dataset,
        nasa_bucket,
        save_feature_schema,
    )
    from modeling import build_lightgbm_pipeline
else:
    from .dataio import (
        infer_feature_types,
        load_mission_dataset,
        nasa_bucket,
        save_feature_schema,
    )
    from .modeling import build_lightgbm_pipeline


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
    return parser.parse_args(list(argv))


def configure_logging() -> None:
    logging.basicConfig(level=logging.INFO, format="%(asctime)s - %(levelname)s - %(message)s")


def bucketize(probability: float, th_planet: float, th_candidate: float) -> str:
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


def _format_metrics_text(
    roc_auc: float,
    pr_auc: float,
    precision: float,
    recall: float,
    f1: float,
    support: float,
    confusion: np.ndarray,
    test_size: float,
    seed: int,
) -> str:
    lines = [
        f"ROC-AUC: {roc_auc:.4f}",
        f"PR-AUC: {pr_auc:.4f}",
        f"Precision (threshold=0.5): {precision:.4f}",
        f"Recall (threshold=0.5): {recall:.4f}",
        f"F1 (threshold=0.5): {f1:.4f}",
        f"Support (positive class): {int(support)}",
        "",
        "Confusion matrix (threshold=0.5):",
        str(confusion),
        "",
        f"Test size: {test_size}",
        f"Seed: {seed}",
    ]
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


def _build_summary(
    df: pd.DataFrame,
    mission: str,
    thresholds: dict[str, float],
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
    lines.append("Thresholds:")
    for key, value in thresholds.items():
        lines.append(f"  {key}: {value}")
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
    data_dir: Path,
    artifacts_dir: Path,
    test_size: float,
    seed: int,
    thresholds: dict[str, float],
    logger: logging.Logger,
) -> None:
    logger.info("Processing mission %s", mission)
    dataset = load_mission_dataset(mission, data_dir, logger)
    features = dataset.features
    labels = dataset.labels
    metadata = dataset.metadata

    splitter = StratifiedShuffleSplit(n_splits=1, test_size=test_size, random_state=seed)
    train_indices, test_indices = next(splitter.split(features, labels))
    X_train = features.iloc[train_indices].copy()
    y_train = labels.iloc[train_indices].copy()
    X_test = features.iloc[test_indices].copy()
    y_test = labels.iloc[test_indices].copy()
    logger.info(
        "Mission %s: %d train samples, %d test samples", mission, X_train.shape[0], X_test.shape[0]
    )

    numeric_cols, categorical_cols = infer_feature_types(X_train)
    logger.info(
        "Mission %s: %d numeric features, %d categorical features",
        mission,
        len(numeric_cols),
        len(categorical_cols),
    )
    pipeline = build_lightgbm_pipeline(numeric_cols, categorical_cols, random_state=seed)
    logger.info("Training LightGBM pipeline on CPU for mission %s", mission)
    pipeline.fit(X_train, y_train)

    proba_test = pipeline.predict_proba(X_test)[:, 1]
    predictions_test = (proba_test >= 0.5).astype(int)
    roc_auc = roc_auc_score(y_test, proba_test)
    pr_auc = average_precision_score(y_test, proba_test)
    precision, recall, f1, support = precision_recall_fscore_support(
        y_test,
        predictions_test,
        average="binary",
        zero_division=0,
    )
    confusion = confusion_matrix(y_test, predictions_test)

    metrics_text = _format_metrics_text(
        roc_auc,
        pr_auc,
        precision,
        recall,
        f1,
        support,
        confusion,
        test_size,
        seed,
    )

    models_dir = artifacts_dir / "models_within"
    models_dir.mkdir(parents=True, exist_ok=True)
    model_path = models_dir / f"model_within-{mission}.pkl"
    joblib.dump(pipeline, model_path)
    logger.info("Saved model to %s", model_path)

    schema_path = models_dir / f"feature_columns_within-{mission}.json"
    save_feature_schema(X_train.columns, schema_path)
    logger.info("Saved feature schema to %s", schema_path)

    export_dir = artifacts_dir / "exports_within" / mission
    export_dir.mkdir(parents=True, exist_ok=True)
    metrics_path = export_dir / f"metrics_within-{mission}.txt"
    _write_text(metrics_text, metrics_path)
    logger.info("Wrote metrics to %s", metrics_path)

    proba_full = pipeline.predict_proba(features)[:, 1]
    base_predictions = pd.DataFrame(
        {
            "object_id": metadata["object_id"].values,
            "mission": metadata["mission"].values,
            "proba_planet": proba_full,
        },
        index=metadata.index,
    )
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
        test_size,
        seed,
        int(join_missing),
        len(combined),
    )
    summary_path = export_dir / f"summary_within-{mission}.txt"
    _write_text(summary_text, summary_path)
    logger.info("Wrote summary to %s", summary_path)


def main(argv: Iterable[str] | None = None) -> None:
    args = parse_args(argv or [])
    configure_logging()
    logger = logging.getLogger("export_within_mission")
    missions = _select_missions(args.missions)
    logger.info("Running within-mission export for missions: %s", missions)
    thresholds = {"planet": args.threshold_planet, "candidate": args.threshold_candidate}

    for mission in missions:
        process_mission(
            mission,
            args.data_dir,
            args.artifacts_dir,
            args.test_size,
            args.seed,
            thresholds,
            logger,
        )


if __name__ == "__main__":
    main()
