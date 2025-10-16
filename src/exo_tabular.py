# INSERIR este arquivo como completo em ~/nasa/src/exo_tabular.py
"""Unified CLI for training and predicting exoplanet tabular models."""

from __future__ import annotations

import argparse
import json
import logging
import sys
from pathlib import Path
from typing import Dict, Iterable, List, Optional, Sequence, Tuple

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
        evaluate_multiclass_classification,
        fit_pipeline_with_fallback,
        plot_confusion_matrix,
        plot_multiclass_confusion_matrix,
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
        evaluate_multiclass_classification,
        fit_pipeline_with_fallback,
        plot_confusion_matrix,
        plot_multiclass_confusion_matrix,
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
    parser.add_argument(
        "--label-mode",
        choices=["binary", "nasa"],
        default="nasa",
        help="Label encoding to use during training and prediction",
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


def _resolve_problem_type(label_mode: str) -> str:
    return "multiclass" if label_mode == "nasa" else "binary"


def _get_pipeline_model(pipeline) -> object:
    if hasattr(pipeline, "named_steps") and "model" in pipeline.named_steps:
        return pipeline.named_steps["model"]
    raise AttributeError("Pipeline does not expose a 'model' step with classes_.")


def _predict_proba_with_classes(pipeline, features: pd.DataFrame) -> Tuple[np.ndarray, np.ndarray]:
    proba = pipeline.predict_proba(features)
    model = _get_pipeline_model(pipeline)
    if not hasattr(model, "classes_"):
        raise AttributeError("Model does not provide classes_ attribute for probability alignment.")
    classes = np.asarray(model.classes_)
    return proba, classes


def _find_class_index(classes: Sequence[int], target: int) -> int:
    for idx, value in enumerate(classes):
        if int(value) == target:
            return idx
    raise ValueError(f"Target class {target} not found in model classes {classes!r}")


def _sanitize_class_name(name: str) -> str:
    return name.replace("-", "_")


def _build_probability_frame(proba: np.ndarray, class_names: Sequence[str]) -> pd.DataFrame:
    sanitized = [_sanitize_class_name(name) for name in class_names]
    columns = [f"proba_{name}" for name in sanitized]
    return pd.DataFrame(proba, columns=columns)


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
    label_mode: str,
    logger: logging.Logger,
) -> Path:
    train_dataset, test_dataset = load_cross_mission_split(
        test_mission,
        data_dir,
        logger,
        label_mode=label_mode,
    )
    log_label_distribution("train", train_dataset, logger)
    log_label_distribution("test", test_dataset, logger)
    X_train, y_train = train_dataset.features, train_dataset.labels
    X_test, y_test = test_dataset.features, test_dataset.labels
    log_feature_set(X_train, logger)
    numeric_cols, categorical_cols = infer_feature_types(X_train)
    problem_type = _resolve_problem_type(label_mode)
    class_names = train_dataset.class_names or ["non-planet", "planet"]
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
        builder_kwargs={
            "oversample": oversample,
            "problem_type": problem_type,
            "class_names": class_names,
        },
    )
    proba, classes = _predict_proba_with_classes(pipeline, X_test)
    metrics: Dict[str, object]
    preds: np.ndarray
    if problem_type == "binary":
        positive_index = _find_class_index(classes, 1)
        proba_planet = proba[:, positive_index]
        metrics = evaluate_binary_classification(
            y_test,
            proba_planet,
            thresholds=(0.5, 0.95),
            recall_target=recall_target,
        )
        preds = (proba_planet >= 0.5).astype(int)
    else:
        metrics = evaluate_multiclass_classification(y_test, proba, class_names)
        preds = np.argmax(proba, axis=1)
    metrics["configuration"] = {
        "mode": "train",
        "split": "cross-mission",
        "test_mission": test_mission,
        "ensemble": ensemble,
        "device": device,
        "oversample": oversample,
        "recall_target": recall_target,
        "label_mode": label_mode,
        "problem_type": problem_type,
    }
    metrics["class_names"] = list(class_names)
    if "metrics" in artifacts:
        save_metrics(metrics, artifacts["metrics"])
    if problem_type == "binary":
        if "roc" in artifacts:
            plot_roc_curve(y_test, proba_planet, artifacts["roc"])
        if "pr" in artifacts:
            plot_pr_curve(y_test, proba_planet, artifacts["pr"])
        if "confusion" in artifacts:
            plot_confusion_matrix(
                y_test,
                proba_planet,
                threshold=0.5,
                output_path=artifacts["confusion"],
            )
    else:
        if "confusion" in artifacts:
            plot_multiclass_confusion_matrix(
                y_test,
                preds,
                class_names,
                artifacts["confusion"],
            )
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
    label_mode: str = "nasa",
    logger: Optional[logging.Logger] = None,
) -> Path:
    logger = logger or logging.getLogger("exo_tabular")
    artifacts_dir.mkdir(parents=True, exist_ok=True)
    tag = f"{test_mission}_{label_mode}" if label_mode else test_mission
    artifacts = _prepare_cross_mission_artifacts(test_mission, artifacts_dir, prefix=tag)
    return _train_cross_mission_core(
        test_mission,
        data_dir,
        artifacts,
        device=device,
        ensemble=ensemble,
        oversample=oversample,
        random_state=random_state,
        recall_target=recall_target,
        label_mode=label_mode,
        logger=logger,
    )


def _load_cross_mission_predictions(
    test_mission: str,
    data_dir: Path,
    *,
    model_path: Path,
    schema_path: Path,
    label_mode: str,
    logger: logging.Logger,
) -> pd.DataFrame:
    if not model_path.exists():
        raise FileNotFoundError(f"Trained model not found at {model_path}")
    if not schema_path.exists():
        raise FileNotFoundError(f"Feature schema not found at {schema_path}")
    pipeline = joblib.load(model_path)
    schema = load_feature_schema(schema_path)
    _, test_dataset = load_cross_mission_split(
        test_mission,
        data_dir,
        logger,
        label_mode=label_mode,
    )
    log_label_distribution("prediction_target", test_dataset, logger)
    features = align_to_schema(test_dataset.features, schema)
    log_feature_set(features, logger)
    proba, classes = _predict_proba_with_classes(pipeline, features)
    problem_type = _resolve_problem_type(label_mode)
    class_names = test_dataset.class_names or ["non-planet", "planet"]
    base = pd.DataFrame(
        {
            "object_id": test_dataset.metadata["object_id"].values,
            "mission": test_dataset.metadata["mission"].values,
        }
    )
    if problem_type == "binary":
        positive_index = _find_class_index(classes, 1)
        proba_planet = proba[:, positive_index]
        base["proba_planet"] = proba_planet
        pred_indices = (proba_planet >= 0.5).astype(int)
    else:
        probability_frame = _build_probability_frame(proba, class_names)
        base = pd.concat([base, probability_frame], axis=1)
        if "proba_planet" not in base.columns:
            base["proba_planet"] = probability_frame.get("proba_planet", 0.0)
        pred_indices = np.argmax(proba, axis=1)
    base["predicted_class"] = [class_names[int(idx)] for idx in pred_indices]
    metadata_columns = ["object_id", "mission", "label_text"]
    if "nasa_bucket" in test_dataset.metadata.columns:
        metadata_columns.append("nasa_bucket")
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
    prob_columns = [col for col in base.columns if col.startswith("proba_")]
    keep_order = ["object_id", "mission"]
    keep_order.extend(prob_columns)
    if "predicted_class" in base.columns:
        keep_order.append("predicted_class")
    for column in dataio.PHYSICAL_OUTPUT_COLUMNS:
        if column in merged.columns:
            keep_order.append(column)
    if "label_text" in merged.columns:
        keep_order.append("label_text")
    if "nasa_bucket" in merged.columns:
        keep_order.append("nasa_bucket")
    return merged.loc[:, keep_order]


def predict_cross_mission(
    test_mission: str,
    data_dir: Path,
    artifacts_dir: Path,
    model_path: Optional[Path] = None,
    schema_path: Optional[Path] = None,
    *,
    label_mode: str = "nasa",
    logger: Optional[logging.Logger] = None,
) -> pd.DataFrame:
    logger = logger or logging.getLogger("exo_tabular")
    tag = f"{test_mission}_{label_mode}"
    effective_model_path = model_path or artifacts_dir / f"model_{tag}.pkl"
    effective_schema_path = schema_path or artifacts_dir / f"{tag}_feature_columns.json"
    return _load_cross_mission_predictions(
        test_mission,
        data_dir,
        model_path=effective_model_path,
        schema_path=effective_schema_path,
        label_mode=label_mode,
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
    dataset = load_mission_dataset(args.mission, data_dir, logger, label_mode=args.label_mode)
    log_label_distribution(args.mission, dataset, logger)
    X, y = dataset.features, dataset.labels
    groups = dataset.metadata["group_id"]
    log_feature_set(X, logger)
    numeric_cols, categorical_cols = infer_feature_types(X)
    problem_type = _resolve_problem_type(args.label_mode)
    class_names = dataset.class_names or ["non-planet", "planet"]
    gkf = GroupKFold(n_splits=GROUP_KFOLD_SPLITS)
    if problem_type == "binary":
        proba = np.zeros(len(y), dtype=float)
    else:
        proba = np.zeros((len(y), len(class_names)), dtype=float)
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
            builder_kwargs={
                "oversample": args.oversample,
                "problem_type": problem_type,
                "class_names": class_names,
            },
        )
        fold_proba, fold_classes = _predict_proba_with_classes(fold_pipeline, X.iloc[val_idx])
        if problem_type == "binary":
            positive_index = _find_class_index(fold_classes, 1)
            proba[val_idx] = fold_proba[:, positive_index]
        else:
            proba[val_idx, :] = fold_proba
    if problem_type == "binary":
        metrics = evaluate_binary_classification(
            y,
            proba,
            thresholds=(0.5, 0.95),
            recall_target=args.recall_target,
        )
    else:
        metrics = evaluate_multiclass_classification(y, proba, class_names)
    metrics["configuration"] = {
        "mode": "train",
        "split": args.split,
        "mission": args.mission,
        "ensemble": args.ensemble,
        "device": args.device,
        "folds": GROUP_KFOLD_SPLITS,
        "oversample": args.oversample,
        "recall_target": args.recall_target,
        "label_mode": args.label_mode,
        "problem_type": problem_type,
    }
    metrics["class_names"] = list(class_names)
    save_metrics(metrics, artifacts["metrics"])
    if problem_type == "binary":
        plot_roc_curve(y, proba, artifacts["roc"])
        plot_pr_curve(y, proba, artifacts["pr"])
        plot_confusion_matrix(y, proba, threshold=0.5, output_path=artifacts["confusion"])
    else:
        preds = np.argmax(proba, axis=1)
        plot_multiclass_confusion_matrix(y, preds, class_names, artifacts["confusion"])
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
        builder_kwargs={
            "oversample": args.oversample,
            "problem_type": problem_type,
            "class_names": class_names,
        },
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
            label_mode=args.label_mode,
            logger=logger,
        )
        if args.label_mode == "binary":
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
            predictions["bucket"] = predictions["predicted_class"]
    else:
        if not model_path.exists():
            raise FileNotFoundError(f"Trained model not found at {model_path}")
        if not schema_path.exists():
            raise FileNotFoundError(f"Feature schema not found at {schema_path}")
        pipeline = joblib.load(model_path)
        schema = load_feature_schema(schema_path)
        if not args.mission:
            raise ValueError("--mission must be provided for group-kfold split")
        target_dataset = load_mission_dataset(
            args.mission,
            data_dir,
            logger,
            label_mode=args.label_mode,
        )
        log_label_distribution("prediction_target", target_dataset, logger)
        features = align_to_schema(target_dataset.features, schema)
        log_feature_set(features, logger)
        proba, classes = _predict_proba_with_classes(pipeline, features)
        problem_type = _resolve_problem_type(args.label_mode)
        class_names = target_dataset.class_names or ["non-planet", "planet"]
        if problem_type == "binary":
            positive_index = _find_class_index(classes, 1)
            proba_planet = proba[:, positive_index]
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
            buckets = assign_bucket(proba_planet, thresholds=thresholds)
            pred_indices = (proba_planet >= 0.5).astype(int)
            predictions = pd.DataFrame(
                {
                    "object_id": target_dataset.metadata["object_id"].values,
                    "mission": target_dataset.metadata["mission"].values,
                    "proba_planet": proba_planet,
                    "bucket": buckets,
                    "predicted_class": [class_names[int(idx)] for idx in pred_indices],
                }
            )
        else:
            probability_frame = _build_probability_frame(proba, class_names)
            predictions = pd.concat(
                [
                    pd.DataFrame(
                        {
                            "object_id": target_dataset.metadata["object_id"].values,
                            "mission": target_dataset.metadata["mission"].values,
                        }
                    ),
                    probability_frame,
                ],
                axis=1,
            )
            if "proba_planet" not in predictions.columns:
                predictions["proba_planet"] = probability_frame.get("proba_planet", 0.0)
            pred_indices = np.argmax(proba, axis=1)
            predictions["predicted_class"] = [class_names[int(idx)] for idx in pred_indices]
            predictions["bucket"] = predictions["predicted_class"]
        predictions["label_text"] = target_dataset.metadata["label_text"].values
        if "nasa_bucket" in target_dataset.metadata.columns:
            predictions["nasa_bucket"] = target_dataset.metadata["nasa_bucket"].values
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
    mode_tag = f"{mode_tag}_{args.label_mode}"
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
                label_mode=args.label_mode,
                logger=logger,
            )
        else:
            train_group_kfold(args, data_dir, artifacts, logger)
    else:
        predict_dataset(args, data_dir, artifacts, logger)


if __name__ == "__main__":
    main(sys.argv[1:])
