# INSERIR este arquivo como completo em ~/nasa/src/exo_tabular.py
"""Unified CLI for training and predicting exoplanet tabular models."""

from __future__ import annotations

import argparse
import json
import logging
import sys
from pathlib import Path
from typing import Dict, Iterable, List, Optional, Tuple

import joblib
import numpy as np
import pandas as pd
from sklearn.model_selection import GroupKFold

if __package__ in (None, ""):
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
else:
    from . import dataio
    from .dataio import (
        MissionDataset,
        align_to_schema,
        infer_feature_types,
        load_cross_mission_split,
        load_feature_schema,
        load_mission_dataset,
        save_feature_schema,
    )
    from .modeling import (
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
    parser.add_argument("--oversample", action="store_true", help="Apply RandomOverSampler during training")
    parser.add_argument("--device", choices=["cpu", "gpu"], default="cpu")
    parser.add_argument("--random-state", type=int, default=DEFAULT_RANDOM_STATE)
    parser.add_argument(
        "--recall-target",
        type=float,
        default=0.6,
        help="Recall target used to calibrate decision threshold",
    )
    parser.add_argument(
        "--use-calibrated-buckets",
        action="store_true",
        help="When predicting, use the calibrated threshold for candidate bucket",
    )
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


def assign_bucket(probabilities: np.ndarray, *, thresholds: Optional[Dict[str, float]] = None) -> List[str]:
    if thresholds is None:
        thresholds = {"planet": 0.95, "candidate": 0.5}
    planet_th = _normalize_threshold(thresholds.get("planet", 0.95))
    candidate_th = _normalize_threshold(thresholds.get("candidate", 0.5))
    if candidate_th >= planet_th:
        candidate_th = max(0.0, min(candidate_th, planet_th - 1e-6))
    buckets: List[str] = []
    for value in probabilities:
        value = _normalize_probability(float(value))
        if value >= planet_th:
            buckets.append("planet")
        elif value >= candidate_th:
            buckets.append("candidate")
        else:
            buckets.append("non-planet")
    return buckets


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


def _prepare_cross_mission_artifacts(
    test_mission: str,
    artifacts_dir: Path,
    prefix: Optional[str] = None,
) -> Dict[str, Path]:
    tag = prefix if prefix is not None else test_mission
    return {
        "model": artifacts_dir / f"model_{tag}.pkl",
        "metrics": artifacts_dir / f"metrics_{tag}.json",
        "roc": artifacts_dir / f"roc_{tag}.png",
        "pr": artifacts_dir / f"pr_{tag}.png",
        "confusion": artifacts_dir / f"confusion_{tag}.png",
        "schema": artifacts_dir / f"{tag}_feature_columns.json",
    }


def _train_cross_mission_core(
    test_mission: str,
    data_dir: Path,
    artifacts: Dict[str, Path],
    *,
    device: str,
    ensemble: bool,
    oversample: bool,
    random_state: int,
    recall_target: float,
    logger: logging.Logger,
) -> Path:
    train_dataset, test_dataset = load_cross_mission_split(test_mission, data_dir, logger)
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
        device=device,
        ensemble=ensemble,
        logger=logger,
        random_state=random_state,
        builder_kwargs={"oversample": oversample},
    )
    proba = pipeline.predict_proba(X_test)[:, 1]
    metrics = evaluate_binary_classification(
        y_test,
        proba,
        thresholds=(0.5, 0.95),
        recall_target=recall_target,
    )
    metrics["configuration"] = {
        "mode": "train",
        "split": "cross-mission",
        "test_mission": test_mission,
        "ensemble": ensemble,
        "device": device,
        "oversample": oversample,
        "recall_target": recall_target,
    }
    if "metrics" in artifacts:
        save_metrics(metrics, artifacts["metrics"])
    if "roc" in artifacts:
        plot_roc_curve(y_test, proba, artifacts["roc"])
    if "pr" in artifacts:
        plot_pr_curve(y_test, proba, artifacts["pr"])
    if "confusion" in artifacts:
        plot_confusion_matrix(y_test, proba, threshold=0.5, output_path=artifacts["confusion"])
    model_path = artifacts.get("model")
    if model_path is not None:
        joblib.dump(pipeline, model_path)
        logger.info("Saved model to %s", model_path)
    else:
        raise KeyError("Cross-mission training requires a 'model' artifact path.")
    if "schema" in artifacts:
        save_feature_schema(X_train.columns, artifacts["schema"])
        logger.info("Saved feature schema to %s", artifacts["schema"])
    return model_path


def train_cross_mission(
    test_mission: str,
    data_dir: Path,
    artifacts_dir: Path,
    *,
    device: str = "cpu",
    ensemble: bool = False,
    oversample: bool = False,
    random_state: int = DEFAULT_RANDOM_STATE,
    recall_target: float = 0.6,
    logger: Optional[logging.Logger] = None,
) -> Path:
    logger = logger or logging.getLogger("exo_tabular")
    artifacts_dir.mkdir(parents=True, exist_ok=True)
    artifacts = _prepare_cross_mission_artifacts(test_mission, artifacts_dir)
    return _train_cross_mission_core(
        test_mission,
        data_dir,
        artifacts,
        device=device,
        ensemble=ensemble,
        oversample=oversample,
        random_state=random_state,
        recall_target=recall_target,
        logger=logger,
    )


def _load_cross_mission_predictions(
    test_mission: str,
    data_dir: Path,
    *,
    model_path: Path,
    schema_path: Path,
    logger: logging.Logger,
) -> pd.DataFrame:
    if not model_path.exists():
        raise FileNotFoundError(f"Trained model not found at {model_path}")
    if not schema_path.exists():
        raise FileNotFoundError(f"Feature schema not found at {schema_path}")
    pipeline = joblib.load(model_path)
    schema = load_feature_schema(schema_path)
    _, test_dataset = load_cross_mission_split(test_mission, data_dir, logger)
    log_label_distribution("prediction_target", test_dataset, logger)
    features = align_to_schema(test_dataset.features, schema)
    log_feature_set(features, logger)
    proba = pipeline.predict_proba(features)[:, 1]
    base = pd.DataFrame(
        {
            "object_id": test_dataset.metadata["object_id"].values,
            "mission": test_dataset.metadata["mission"].values,
            "proba_planet": proba,
        }
    )
    metadata_columns = ["object_id", "mission", "label_text"]
    for column in dataio.PHYSICAL_OUTPUT_COLUMNS:
        if column in test_dataset.metadata:
            metadata_columns.append(column)
    metadata = test_dataset.metadata.loc[:, metadata_columns].copy()
    merged = base.merge(metadata, on=["object_id", "mission"], how="left")
    missing = merged["label_text"].isna().sum()
    if missing:
        logger.warning(
            "Metadata join missing for %d samples (mission=%s)",
            missing,
            test_mission,
        )
    keep_order = ["object_id", "mission", "proba_planet"]
    for column in dataio.PHYSICAL_OUTPUT_COLUMNS:
        if column in merged.columns:
            keep_order.append(column)
    if "label_text" in merged.columns:
        keep_order.append("label_text")
    return merged.loc[:, keep_order]


def predict_cross_mission(
    test_mission: str,
    data_dir: Path,
    artifacts_dir: Path,
    model_path: Optional[Path] = None,
    *,
    logger: Optional[logging.Logger] = None,
) -> pd.DataFrame:
    logger = logger or logging.getLogger("exo_tabular")
    effective_model_path = model_path or artifacts_dir / f"model_{test_mission}.pkl"
    schema_path = artifacts_dir / f"{test_mission}_feature_columns.json"
    return _load_cross_mission_predictions(
        test_mission,
        data_dir,
        model_path=effective_model_path,
        schema_path=schema_path,
        logger=logger,
    )


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
            builder_kwargs={"oversample": args.oversample},
        )
        proba[val_idx] = fold_pipeline.predict_proba(X.iloc[val_idx])[:, 1]
    metrics = evaluate_binary_classification(
        y,
        proba,
        thresholds=(0.5, 0.95),
        recall_target=args.recall_target,
    )
    metrics["configuration"] = {
        "mode": "train",
        "split": args.split,
        "mission": args.mission,
        "ensemble": args.ensemble,
        "device": args.device,
        "folds": GROUP_KFOLD_SPLITS,
        "oversample": args.oversample,
        "recall_target": args.recall_target,
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
        builder_kwargs={"oversample": args.oversample},
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
    if args.split == "cross-mission":
        predictions = _load_cross_mission_predictions(
            args.test_mission,
            data_dir,
            model_path=model_path,
            schema_path=schema_path,
            logger=logger,
        )
        thresholds = {"planet": 0.95, "candidate": 0.5}
        if args.use_calibrated_buckets:
            calibrated = _load_calibrated_threshold(artifacts["metrics"])
            if calibrated is not None:
                if calibrated >= thresholds["planet"]:
                    logger.warning(
                        (
                            "Calibrated candidate threshold %.4f for mission %s is >= planet "
                            "threshold %.2f; keeping candidate threshold %.2f"
                        ),
                        calibrated,
                        args.test_mission,
                        thresholds["planet"],
                        thresholds["candidate"],
                    )
                else:
                    thresholds["candidate"] = calibrated
                    logger.info(
                        "Using calibrated candidate threshold %.4f for mission %s",
                        calibrated,
                        args.test_mission,
                    )
            else:
                logger.warning(
                    "Calibrated threshold requested but not found at %s",
                    artifacts["metrics"],
                )
        predictions["bucket"] = assign_bucket(
            predictions["proba_planet"].to_numpy(),
            thresholds=thresholds,
        )
    else:
        if not model_path.exists():
            raise FileNotFoundError(f"Trained model not found at {model_path}")
        if not schema_path.exists():
            raise FileNotFoundError(f"Feature schema not found at {schema_path}")
        pipeline = joblib.load(model_path)
        schema = load_feature_schema(schema_path)
        if not args.mission:
            raise ValueError("--mission must be provided for group-kfold split")
        target_dataset = load_mission_dataset(args.mission, data_dir, logger)
        log_label_distribution("prediction_target", target_dataset, logger)
        features = align_to_schema(target_dataset.features, schema)
        log_feature_set(features, logger)
        proba = pipeline.predict_proba(features)[:, 1]
        thresholds = {"planet": 0.95, "candidate": 0.5}
        if args.use_calibrated_buckets:
            calibrated = _load_calibrated_threshold(artifacts["metrics"])
            if calibrated is not None:
                if calibrated >= thresholds["planet"]:
                    logger.warning(
                        (
                            "Calibrated candidate threshold %.4f for mission %s is >= planet "
                            "threshold %.2f; keeping candidate threshold %.2f"
                        ),
                        calibrated,
                        args.mission,
                        thresholds["planet"],
                        thresholds["candidate"],
                    )
                else:
                    thresholds["candidate"] = calibrated
                    logger.info(
                        "Using calibrated candidate threshold %.4f for mission %s",
                        calibrated,
                        args.mission,
                    )
            else:
                logger.warning(
                    "Calibrated threshold requested but not found at %s",
                    artifacts["metrics"],
                )
        predictions = pd.DataFrame(
            {
                "object_id": target_dataset.metadata["object_id"].values,
                "mission": target_dataset.metadata["mission"].values,
                "proba_planet": proba,
                "bucket": assign_bucket(proba, thresholds=thresholds),
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
    if args.split == "cross-mission":
        mode_tag = f"{args.split}_{args.test_mission}"
    elif args.split == "group-kfold" and args.mission:
        mode_tag = f"{args.split}_{args.mission}"
    else:
        mode_tag = args.split
    artifacts = build_artifact_paths(artifacts_dir, mode_tag)
    logger.info("Running mode=%s split=%s", args.mode, args.split)
    if args.mode == "train":
        if args.split == "cross-mission":
            _train_cross_mission_core(
                args.test_mission,
                data_dir,
                artifacts,
                device=args.device,
                ensemble=args.ensemble,
                oversample=args.oversample,
                random_state=args.random_state,
                recall_target=args.recall_target,
                logger=logger,
            )
        else:
            train_group_kfold(args, data_dir, artifacts, logger)
    else:
        predict_dataset(args, data_dir, artifacts, logger)


if __name__ == "__main__":
    main(sys.argv[1:])
