# INSERIR este arquivo como completo em ~/nasa/src/modeling.py
"""Model construction, training helpers and evaluation utilities."""

from __future__ import annotations

import json
import logging
from pathlib import Path
from typing import Dict, Iterable, List, Sequence, Tuple

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from lightgbm import LGBMClassifier
from sklearn.compose import ColumnTransformer
from sklearn.ensemble import ExtraTreesClassifier, RandomForestClassifier, StackingClassifier
from sklearn.impute import SimpleImputer
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import (
    ConfusionMatrixDisplay,
    PrecisionRecallDisplay,
    RocCurveDisplay,
    average_precision_score,
    confusion_matrix,
    precision_recall_fscore_support,
    roc_auc_score,
)
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import OneHotEncoder, StandardScaler

from utils_transforms import ColumnAligner


def build_preprocessor(numeric_cols: Sequence[str], categorical_cols: Sequence[str]) -> ColumnTransformer:
    transformers: List[Tuple[str, Pipeline, Sequence[str]]] = []
    if numeric_cols:
        numeric_pipeline = Pipeline(
            steps=[
                ("imputer", SimpleImputer(strategy="median")),
                ("scaler", StandardScaler(with_mean=False)),
            ]
        )
        transformers.append(("numeric", numeric_pipeline, list(numeric_cols)))
    if categorical_cols:
        categorical_pipeline = Pipeline(
            steps=[
                ("imputer", SimpleImputer(strategy="most_frequent")),
                (
                    "encoder",
                    OneHotEncoder(handle_unknown="ignore", sparse_output=False),
                ),
            ]
        )
        transformers.append(("categorical", categorical_pipeline, list(categorical_cols)))
    if not transformers:
        raise ValueError("No feature columns available to build a preprocessing pipeline.")
    return ColumnTransformer(transformers=transformers)


def _instantiate_lgbm(device: str, random_state: int = 42) -> LGBMClassifier:
    params = {
        "n_estimators": 600,
        "learning_rate": 0.05,
        "num_leaves": 64,
        "subsample": 0.7,
        "colsample_bytree": 0.7,
        "objective": "binary",
        "random_state": random_state,
        "n_jobs": -1,
    }
    if device == "gpu":
        params.update(
            {
                "device_type": "gpu",
                "tree_learner": "gpu",
                "gpu_platform_id": 0,
                "gpu_device_id": 0,
            }
        )
    else:
        params.update({"device_type": "cpu"})
    return LGBMClassifier(**params)


def build_pipeline(
    numeric_cols: Sequence[str],
    categorical_cols: Sequence[str],
    device: str = "cpu",
    ensemble: bool = False,
    random_state: int = 42,
) -> Pipeline:
    preprocessor = build_preprocessor(numeric_cols, categorical_cols)
    aligner = ColumnAligner()
    if ensemble:
        lgbm = _instantiate_lgbm(device=device, random_state=random_state)
        rf = RandomForestClassifier(
            n_estimators=400,
            max_depth=None,
            n_jobs=-1,
            class_weight="balanced_subsample",
            random_state=random_state,
        )
        extra = ExtraTreesClassifier(
            n_estimators=400,
            n_jobs=-1,
            random_state=random_state,
            class_weight="balanced_subsample",
        )
        meta = LogisticRegression(
            max_iter=1000,
            solver="lbfgs",
            class_weight="balanced",
        )
        stacking = StackingClassifier(
            estimators=[
                ("lgbm", lgbm),
                ("rf", rf),
                ("extra", extra),
            ],
            final_estimator=meta,
            cv=3,
            stack_method="predict_proba",
            passthrough=False,
            n_jobs=-1,
        )
        model = stacking
    else:
        model = _instantiate_lgbm(device=device, random_state=random_state)
    pipeline = Pipeline(
        steps=[
            ("align", aligner),
            ("preprocess", preprocessor),
            ("model", model),
        ]
    )
    return pipeline


def fit_pipeline_with_fallback(
    pipeline_builder,
    numeric_cols: Sequence[str],
    categorical_cols: Sequence[str],
    X_train: pd.DataFrame,
    y_train: pd.Series,
    device: str,
    ensemble: bool,
    logger: logging.Logger,
    random_state: int = 42,
) -> Pipeline:
    try:
        pipeline = pipeline_builder(numeric_cols, categorical_cols, device=device, ensemble=ensemble, random_state=random_state)
        pipeline.fit(X_train, y_train)
        return pipeline
    except Exception as exc:  # noqa: BLE001
        if device == "gpu" and "gpu" in str(exc).lower():
            logger.warning("Falling back to CPU LightGBM due to GPU error: %s", exc)
            pipeline = pipeline_builder(
                numeric_cols,
                categorical_cols,
                device="cpu",
                ensemble=ensemble,
                random_state=random_state,
            )
            pipeline.fit(X_train, y_train)
            return pipeline
        raise


def evaluate_binary_classification(
    y_true: Sequence[int],
    proba: Sequence[float],
    thresholds: Sequence[float] = (0.5, 0.95),
) -> Dict[str, object]:
    y_true_arr = np.asarray(y_true)
    proba_arr = np.asarray(proba)
    metrics: Dict[str, object] = {}
    metrics["roc_auc"] = float(roc_auc_score(y_true_arr, proba_arr))
    metrics["pr_auc"] = float(average_precision_score(y_true_arr, proba_arr))
    metrics["num_samples"] = int(len(y_true_arr))
    metrics["positive_rate"] = float(np.mean(y_true_arr))
    threshold_metrics: Dict[str, Dict[str, float]] = {}
    for threshold in thresholds:
        preds = (proba_arr >= threshold).astype(int)
        precision, recall, f1, _ = precision_recall_fscore_support(
            y_true_arr,
            preds,
            beta=1.0,
            average="binary",
            zero_division=0,
        )
        threshold_metrics[str(threshold)] = {
            "precision": float(precision),
            "recall": float(recall),
            "f1": float(f1),
            "predicted_positive": int(preds.sum()),
        }
    metrics["thresholds"] = threshold_metrics
    cm = confusion_matrix(y_true_arr, (proba_arr >= 0.5).astype(int))
    metrics["confusion_matrix"] = {
        "tn": int(cm[0, 0]) if cm.shape == (2, 2) else 0,
        "fp": int(cm[0, 1]) if cm.shape == (2, 2) else 0,
        "fn": int(cm[1, 0]) if cm.shape == (2, 2) else 0,
        "tp": int(cm[1, 1]) if cm.shape == (2, 2) else 0,
    }
    bucket_masks = {
        "planet": proba_arr >= 0.95,
        "candidate": (proba_arr >= 0.5) & (proba_arr < 0.95),
        "non-planet": proba_arr < 0.5,
    }
    metrics["bucket_counts"] = {bucket: int(mask.sum()) for bucket, mask in bucket_masks.items()}
    bucket_positive_rate: Dict[str, float] = {}
    for bucket, mask in bucket_masks.items():
        if mask.sum() == 0:
            bucket_positive_rate[bucket] = float("nan")
        else:
            bucket_positive_rate[bucket] = float(np.mean(y_true_arr[mask]))
    metrics["bucket_positive_rate"] = bucket_positive_rate
    return metrics


def save_metrics(metrics: Dict[str, object], path: Path) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(metrics, indent=2))


def plot_roc_curve(y_true: Sequence[int], proba: Sequence[float], output_path: Path) -> None:
    output_path.parent.mkdir(parents=True, exist_ok=True)
    fig, ax = plt.subplots(figsize=(6, 6))
    RocCurveDisplay.from_predictions(y_true, proba, ax=ax)
    ax.set_title("ROC Curve")
    fig.tight_layout()
    fig.savefig(output_path, dpi=150)
    plt.close(fig)


def plot_pr_curve(y_true: Sequence[int], proba: Sequence[float], output_path: Path) -> None:
    output_path.parent.mkdir(parents=True, exist_ok=True)
    fig, ax = plt.subplots(figsize=(6, 6))
    PrecisionRecallDisplay.from_predictions(y_true, proba, ax=ax)
    ax.set_title("Precision-Recall Curve")
    fig.tight_layout()
    fig.savefig(output_path, dpi=150)
    plt.close(fig)


def plot_confusion_matrix(y_true: Sequence[int], proba: Sequence[float], threshold: float, output_path: Path) -> None:
    output_path.parent.mkdir(parents=True, exist_ok=True)
    preds = (np.asarray(proba) >= threshold).astype(int)
    cm = confusion_matrix(y_true, preds)
    fig, ax = plt.subplots(figsize=(5, 5))
    disp = ConfusionMatrixDisplay(confusion_matrix=cm)
    disp.plot(ax=ax, cmap="Blues", colorbar=False)
    ax.set_title(f"Confusion Matrix (threshold={threshold:.2f})")
    fig.tight_layout()
    fig.savefig(output_path, dpi=150)
    plt.close(fig)
