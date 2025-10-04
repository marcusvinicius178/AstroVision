# INSERIR este arquivo como completo em ~/nasa/src/exo_tabular.py
"""Unified CLI for training and predicting exoplanet tabular models."""

from __future__ import annotations

import argparse
import logging
import sys
from pathlib import Path
from typing import Dict, Iterable, List, Tuple

import joblib
import numpy as np
import pandas as pd
from sklearn.model_selection import GroupKFold

import dataio
from dataio import (
    MissionDataset,
    align_to_schema,
    infer_feature_types,
    load_cross_mission_split,
    load_feature_schema,
    load_mission_dataset,
    save_feature_schema,
)
from modeling import (
    build_pipeline,
    evaluate_binary_classification,
    fit_pipeline_with_fallback,
    plot_confusion_matrix,
    plot_pr_curve,
    plot_roc_curve,
    save_metrics,
)

DEFAULT_RANDOM_STATE = 42
GROUP_KFOLD_SPLITS = 5


def configure_logging() -> None:
    logging.basicConfig(
        level=logging.INFO,
        format="%(asctime)s - %(levelname)s - %(message)s",
    )


def parse_args(argv: Iterable[str]) -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Exoplanet tabular modeling pipeline")
    parser.add_argument("--mode", choices=["train", "predict"], required=True)
    parser.add_argument("--split", choices=["cross-mission", "group-kfold"], required=True)
    parser.add_argument("--test-mission", choices=["tess", "kepler", "k2"], default="tess")
    parser.add_argument("--mission", choices=["kepler", "k2", "tess"], help="Mission for group-kfold mode")
    parser.add_argument("--ensemble", action="store_true", help="Enable stacking ensemble")
    parser.add_argument("--device", choices=["cpu", "gpu"], default="cpu")
    parser.add_argument("--random-state", type=int, default=DEFAULT_RANDOM_STATE)
    return parser.parse_args(list(argv))


def get_project_paths() -> Tuple[Path, Path, Path]:
    root = Path(__file__).resolve().parents[1]
    data_dir = root / "data"
    artifacts_dir = root / "artifacts"
    artifacts_dir.mkdir(parents=True, exist_ok=True)
    return root, data_dir, artifacts_dir


def log_label_distribution(name: str, dataset: MissionDataset, logger: logging.Logger) -> None:
    counts = dataset.labels.value_counts().to_dict()
    total = int(dataset.labels.shape[0])
    normalized = {k: round(v / total, 3) for k, v in counts.items()}
    logger.info("Label distribution for %s: %s (normalized: %s)", name, counts, normalized)


def log_feature_set(features: pd.DataFrame, logger: logging.Logger) -> None:
    columns = list(features.columns)
    logger.info("Final feature count: %d", len(columns))
    logger.info("Feature columns: %s", columns)


def assign_bucket(probabilities: np.ndarray) -> List[str]:
    buckets: List[str] = []
    for value in probabilities:
        if value >= 0.95:
            buckets.append("planet")
        elif value >= 0.5:
            buckets.append("candidate")
        else:
            buckets.append("non-planet")
    return buckets


def build_artifact_paths(artifacts_dir: Path, mode_tag: str) -> Dict[str, Path]:
    return {
        "model": artifacts_dir / f"model_{mode_tag}.pkl",
        "metrics": artifacts_dir / f"metrics_{mode_tag}.json",
        "roc": artifacts_dir / f"roc_{mode_tag}.png",
        "pr": artifacts_dir / f"pr_{mode_tag}.png",
        "confusion": artifacts_dir / f"confusion_{mode_tag}.png",
        "schema": artifacts_dir / f"{mode_tag}_feature_columns.json",
        "predictions": artifacts_dir / f"predictions_{mode_tag}.csv",
    }


def train_cross_mission(
    args: argparse.Namespace,
    data_dir: Path,
    artifacts: Dict[str, Path],
    logger: logging.Logger,
) -> None:
    train_dataset, test_dataset = load_cross_mission_split(args.test_mission, data_dir, logger)
    log_label_distribution("train", train_dataset, logger)
    log_label_distribution("test", test_dataset, logger)
    X_train, y_train = train_dataset.features, train_dataset.labels
    X_test, y_test = test_dataset.features, test_dataset.labels
    log_feature_set(X_train, logger)
    numeric_cols, categorical_cols = infer_feature_types(X_train)
    pipeline = fit_pipeline_with_fallback(
        build_pipeline,
        numeric_cols,
        categorical_cols,
        X_train,
        y_train,
        device=args.device,
        ensemble=args.ensemble,
        logger=logger,
        random_state=args.random_state,
    )
    proba = pipeline.predict_proba(X_test)[:, 1]
    metrics = evaluate_binary_classification(y_test, proba, thresholds=(0.5, 0.95))
    metrics["configuration"] = {
        "mode": "train",
        "split": args.split,
        "test_mission": args.test_mission,
        "ensemble": args.ensemble,
        "device": args.device,
    }
    save_metrics(metrics, artifacts["metrics"])
    plot_roc_curve(y_test, proba, artifacts["roc"])
    plot_pr_curve(y_test, proba, artifacts["pr"])
    plot_confusion_matrix(y_test, proba, threshold=0.5, output_path=artifacts["confusion"])
    joblib.dump(pipeline, artifacts["model"])
    save_feature_schema(X_train.columns, artifacts["schema"])
    logger.info("Saved model to %s", artifacts["model"])


def train_group_kfold(
    args: argparse.Namespace,
    data_dir: Path,
    artifacts: Dict[str, Path],
    logger: logging.Logger,
) -> None:
    if not args.mission:
        raise ValueError("--mission must be provided for group-kfold split")
    dataset = load_mission_dataset(args.mission, data_dir, logger)
    log_label_distribution(args.mission, dataset, logger)
    X, y = dataset.features, dataset.labels
    groups = dataset.metadata["group_id"]
    log_feature_set(X, logger)
    numeric_cols, categorical_cols = infer_feature_types(X)
    gkf = GroupKFold(n_splits=GROUP_KFOLD_SPLITS)
    proba = np.zeros(len(y), dtype=float)
    for fold, (train_idx, val_idx) in enumerate(gkf.split(X, y, groups=groups), start=1):
        logger.info("Training fold %d/%d", fold, GROUP_KFOLD_SPLITS)
        fold_pipeline = fit_pipeline_with_fallback(
            build_pipeline,
            numeric_cols,
            categorical_cols,
            X.iloc[train_idx],
            y.iloc[train_idx],
            device=args.device,
            ensemble=args.ensemble,
            logger=logger,
            random_state=args.random_state + fold,
        )
        proba[val_idx] = fold_pipeline.predict_proba(X.iloc[val_idx])[:, 1]
    metrics = evaluate_binary_classification(y, proba, thresholds=(0.5, 0.95))
    metrics["configuration"] = {
        "mode": "train",
        "split": args.split,
        "mission": args.mission,
        "ensemble": args.ensemble,
        "device": args.device,
        "folds": GROUP_KFOLD_SPLITS,
    }
    save_metrics(metrics, artifacts["metrics"])
    plot_roc_curve(y, proba, artifacts["roc"])
    plot_pr_curve(y, proba, artifacts["pr"])
    plot_confusion_matrix(y, proba, threshold=0.5, output_path=artifacts["confusion"])
    final_pipeline = fit_pipeline_with_fallback(
        build_pipeline,
        numeric_cols,
        categorical_cols,
        X,
        y,
        device=args.device,
        ensemble=args.ensemble,
        logger=logger,
        random_state=args.random_state,
    )
    joblib.dump(final_pipeline, artifacts["model"])
    save_feature_schema(X.columns, artifacts["schema"])
    logger.info("Saved model to %s", artifacts["model"])


def predict_dataset(
    args: argparse.Namespace,
    data_dir: Path,
    artifacts: Dict[str, Path],
    logger: logging.Logger,
) -> None:
    model_path = artifacts["model"]
    schema_path = artifacts["schema"]
    if not model_path.exists():
        raise FileNotFoundError(f"Trained model not found at {model_path}")
    if not schema_path.exists():
        raise FileNotFoundError(f"Feature schema not found at {schema_path}")
    pipeline = joblib.load(model_path)
    schema = load_feature_schema(schema_path)
    if args.split == "cross-mission":
        _, dataset = load_cross_mission_split(args.test_mission, data_dir, logger)
        target_dataset = dataset
    else:
        if not args.mission:
            raise ValueError("--mission must be provided for group-kfold split")
        target_dataset = load_mission_dataset(args.mission, data_dir, logger)
    log_label_distribution("prediction_target", target_dataset, logger)
    features = align_to_schema(target_dataset.features, schema)
    log_feature_set(features, logger)
    proba = pipeline.predict_proba(features)[:, 1]
    buckets = assign_bucket(proba)
    predictions = pd.DataFrame(
        {
            "object_id": target_dataset.metadata["object_id"].values,
            "mission": target_dataset.metadata["mission"].values,
            "proba_planet": proba,
            "bucket": buckets,
        }
    )
    for column in dataio.PHYSICAL_OUTPUT_COLUMNS:
        if column in target_dataset.metadata:
            predictions[column] = target_dataset.metadata[column].values
    predictions.to_csv(artifacts["predictions"], index=False)
    logger.info("Saved predictions to %s", artifacts["predictions"])


def main(argv: Iterable[str]) -> None:
    configure_logging()
    args = parse_args(argv)
    logger = logging.getLogger("exo_tabular")
    _, data_dir, artifacts_dir = get_project_paths()
    mode_tag = args.split
    artifacts = build_artifact_paths(artifacts_dir, mode_tag)
    logger.info("Running mode=%s split=%s", args.mode, args.split)
    if args.mode == "train":
        if args.split == "cross-mission":
            train_cross_mission(args, data_dir, artifacts, logger)
        else:
            train_group_kfold(args, data_dir, artifacts, logger)
    else:
        predict_dataset(args, data_dir, artifacts, logger)


if __name__ == "__main__":
    main(sys.argv[1:])
