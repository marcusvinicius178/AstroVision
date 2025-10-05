# INSERIR este arquivo como completo em ~/nasa/src/dataio.py
"""Data loading and preprocessing utilities for the exoplanet tabular pipeline."""

from __future__ import annotations

import json
import logging
from dataclasses import dataclass
from pathlib import Path
from typing import Dict, Iterable, List, Optional, Sequence, Tuple

import numpy as np
import pandas as pd

LEAK_KEYWORDS = [
    "disposition",
    "pdisposition",
    "koi_disposition",
    "koi_pdisposition",
    "is_planet",
    "planet",
    "confirmed",
    "candidate",
    "status",
    "label",
    "class",
    "vetting",
    "fp",
    "false",
    "not_planet",
    "non_planet",
    "gaia_class",
    "tfop",
    "followup",
    "not_transit",
]

EXTRA_EXCLUDED_SUBSTRINGS = ["name", "source", "notes"]

ID_PRIORITY = [
    "kepid",
    "kepoi_name",
    "epic",
    "epic_id",
    "tic",
    "tic_id",
    "object_id",
    "id",
]

EXTENDED_ID_CANDIDATES = ID_PRIORITY + ["tid", "toi", "pl_name", "hostname"]

FEATURE_SYNONYMS: Dict[str, Sequence[str]] = {
    "period": ("period", "koi_period", "pl_orbper", "orbital_period"),
    "period_err1": ("koi_period_err1", "pl_orbpererr1"),
    "period_err2": ("koi_period_err2", "pl_orbpererr2"),
    "t0": ("koi_time0bk", "epoch", "pl_tranmid"),
    "t0_err1": ("koi_time0bk_err1", "pl_tranmiderr1"),
    "t0_err2": ("koi_time0bk_err2", "pl_tranmiderr2"),
    "duration": ("koi_duration", "pl_trandur", "pl_trandurh"),
    "duration_err1": ("koi_duration_err1", "pl_trandurherr1"),
    "duration_err2": ("koi_duration_err2", "pl_trandurherr2"),
    "depth": ("koi_depth", "pl_trandep"),
    "depth_err1": ("koi_depth_err1", "pl_trandeperr1"),
    "depth_err2": ("koi_depth_err2", "pl_trandeperr2"),
    "radius_ratio": ("koi_ror", "radius_ratio"),
    "snr": ("koi_snr", "snr", "mes"),
    "impact": ("koi_impact", "impact"),
    "mes": ("mes",),
    "rp": ("koi_prad", "pl_rade", "pl_radj", "rp"),
    "sma": ("koi_sma", "pl_orbsmax", "sma", "a"),
    "teq": ("koi_teq", "pl_eqt", "teq"),
    "insolation": ("koi_insol", "pl_insol"),
    "eccentricity": ("koi_eccen", "pl_orbeccen"),
    "inclination": ("koi_inc", "pl_orbincl"),
    "stellar_teff": ("koi_steff", "st_teff"),
    "stellar_logg": ("koi_slogg", "st_logg"),
    "stellar_radius": ("koi_srad", "st_rad"),
    "stellar_mass": ("koi_smass", "st_mass"),
}

ADDITIONAL_NUMERIC_CANDIDATES = [
    "koi_prad_err1",
    "koi_prad_err2",
    "pl_radeerr1",
    "pl_radeerr2",
    "pl_radjerr1",
    "pl_radjerr2",
    "pl_orbsmaxerr1",
    "pl_orbsmaxerr2",
    "pl_insolerr1",
    "pl_insolerr2",
    "pl_eqterr1",
    "pl_eqterr2",
    "koi_max_mult_ev",
    "koi_max_mult_ev_err1",
    "koi_max_mult_ev_err2",
    "koi_model_snr",
    "koi_duration_err1",
    "koi_duration_err2",
    "koi_depth_err1",
    "koi_depth_err2",
    "koi_insol_err1",
    "koi_insol_err2",
    "koi_teq_err1",
    "koi_teq_err2",
    "koi_srho",
    "koi_srho_err1",
    "koi_srho_err2",
    "sy_snum",
    "sy_pnum",
]

ALLOWED_LOW_CARDINALITY_CATEGORICAL = [
    "discoverymethod",
    "disc_facility",
    "soltype",
    "pl_bmassprov",
]

MISSION_FILES = {
    "kepler": "kepler_objects_of_interest.csv",
    "k2": "k2.csv",
    "tess": "TESS.csv",
}

MISSION_LABEL_CANDIDATES = {
    "kepler": ("koi_disposition", "koi_pdisposition", "disposition"),
    "k2": ("disposition",),
    "tess": ("tfopwg_disp", "disposition"),
}

MISSION_POSITIVE_LABELS = {
    "kepler": {"confirmed"},
    "k2": {"confirmed"},
    "tess": {"cp", "kp"},
}

MAX_CATEGORICAL_CARDINALITY = 30

PHYSICAL_OUTPUT_COLUMNS = [
    "period",
    "t0",
    "duration",
    "depth",
    "radius_ratio",
    "snr",
    "impact",
    "mes",
    "rp",
    "sma",
    "teq",
    "insolation",
    "eccentricity",
    "inclination",
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
]


@dataclass
class MissionDataset:
    """Container for mission specific data."""

    mission: str
    features: pd.DataFrame
    labels: pd.Series
    metadata: pd.DataFrame


def read_mission_csv(path: Path) -> pd.DataFrame:
    df = pd.read_csv(path, sep=None, engine="python", on_bad_lines="skip", comment="#")
    df.columns = df.columns.str.lower().str.strip()
    return df


def _standardize_identifiers(df: pd.DataFrame) -> pd.DataFrame:
    df = df.copy()
    if "tid" in df.columns and "tic" not in df.columns:
        df["tic"] = df["tid"]
    return df


def _detect_label_column(mission: str, df: pd.DataFrame) -> str:
    for candidate in MISSION_LABEL_CANDIDATES.get(mission, ()):  # type: ignore[arg-type]
        if candidate in df.columns:
            return candidate
    raise ValueError(f"Could not find a label column for mission '{mission}'.")


def _map_labels(mission: str, label_series: pd.Series) -> Tuple[pd.Series, pd.Series]:
    positives = MISSION_POSITIVE_LABELS.get(mission, set())
    raw = label_series.astype("string")
    lowered = raw.str.lower().str.strip()
    mask_known = raw.notna() & ~lowered.isin({"", "nan", "none"})
    mapped = pd.Series(np.nan, index=label_series.index, dtype=float)
    mapped.loc[mask_known] = lowered.loc[mask_known].isin(positives).astype(float)
    return mapped, raw.fillna("")


def _filter_leak_columns(df: pd.DataFrame, logger: logging.Logger) -> pd.DataFrame:
    lowered = [col.lower() for col in df.columns]
    keep_mask: List[bool] = []
    for original, lower in zip(df.columns, lowered):
        has_leak = any(keyword in lower for keyword in LEAK_KEYWORDS)
        has_extra = any(extra in lower for extra in EXTRA_EXCLUDED_SUBSTRINGS)
        is_id = original in EXTENDED_ID_CANDIDATES or lower.endswith("_id")
        keep_mask.append(not (has_leak or has_extra or is_id))
    filtered = df.loc[:, keep_mask]
    removed = [col for col, keep in zip(df.columns, keep_mask) if not keep]
    if removed:
        logger.debug("Removed leak/id columns: %s", removed)
    return filtered


def _coerce_numeric(df: pd.DataFrame) -> pd.DataFrame:
    converted = df.copy()
    for col in converted.columns:
        if converted[col].dtype == object:
            coerced = pd.to_numeric(converted[col], errors="coerce")
            if coerced.notna().sum() > 0:
                converted[col] = coerced
    return converted


def _prune_categorical(df: pd.DataFrame) -> pd.DataFrame:
    pruned = df.copy()
    for col in pruned.columns:
        if pruned[col].dtype == object:
            if col not in ALLOWED_LOW_CARDINALITY_CATEGORICAL:
                pruned = pruned.drop(columns=[col])
            else:
                cardinality = pruned[col].nunique(dropna=True)
                if cardinality > MAX_CATEGORICAL_CARDINALITY:
                    pruned = pruned.drop(columns=[col])
    return pruned


def _extract_canonical_features(df: pd.DataFrame) -> Tuple[pd.DataFrame, Dict[str, pd.Series]]:
    canonical_data: Dict[str, pd.Series] = {}
    used_columns: List[str] = []
    for canonical, synonyms in FEATURE_SYNONYMS.items():
        for candidate in synonyms:
            if candidate in df.columns:
                canonical_data[canonical] = pd.to_numeric(df[candidate], errors="coerce")
                used_columns.append(candidate)
                break
    remaining = df.drop(columns=used_columns, errors="ignore")
    canonical_df = pd.DataFrame(canonical_data, index=df.index)
    combined = pd.concat([canonical_df, remaining], axis=1)
    combined = combined.loc[:, ~combined.columns.duplicated()]
    combined = combined.dropna(axis=1, how="all")
    return combined, canonical_data


def _build_metadata(
    mission: str,
    df: pd.DataFrame,
    raw_labels: pd.Series,
    canonical_features: Dict[str, pd.Series],
) -> pd.DataFrame:
    metadata = pd.DataFrame(index=df.index)
    metadata["mission"] = mission
    object_id = None
    for candidate in EXTENDED_ID_CANDIDATES:
        if candidate in df.columns:
            object_id = df[candidate]
            break
    if object_id is None:
        object_id = pd.Series(df.index.astype(str), index=df.index)
    metadata["object_id"] = object_id.astype(str).fillna("unknown")
    group_id = None
    for candidate in ID_PRIORITY:
        if candidate in df.columns:
            group_id = df[candidate]
            break
    if group_id is None:
        group_id = metadata["object_id"]
    metadata["group_id"] = group_id.astype(str).fillna("unknown")
    metadata["label_text"] = raw_labels.astype(str)
    for physical in PHYSICAL_OUTPUT_COLUMNS:
        series = None
        if physical in canonical_features:
            series = canonical_features[physical]
        elif physical in df.columns:
            series = pd.to_numeric(df[physical], errors="coerce")
        if series is not None:
            metadata[physical] = series
    return metadata


def _append_additional_numeric(features: pd.DataFrame, df: pd.DataFrame) -> pd.DataFrame:
    augmented = features.copy()
    for column in ADDITIONAL_NUMERIC_CANDIDATES:
        if column in df.columns and column not in augmented.columns:
            augmented[column] = pd.to_numeric(df[column], errors="coerce")
    return augmented


def load_mission_dataset(mission: str, data_dir: Path, logger: Optional[logging.Logger] = None) -> MissionDataset:
    logger = logger or logging.getLogger(__name__)
    mission_key = mission.lower()
    if mission_key not in MISSION_FILES:
        raise ValueError(f"Unknown mission '{mission}'.")
    path = data_dir / MISSION_FILES[mission_key]
    if not path.exists():
        raise FileNotFoundError(f"Dataset file not found for mission '{mission_key}': {path}")
    logger.info("Loading mission %s from %s", mission_key, path)
    df = read_mission_csv(path)
    df = _standardize_identifiers(df)
    label_col = _detect_label_column(mission_key, df)
    labels, raw_labels = _map_labels(mission_key, df[label_col])
    valid_mask = labels.notna()
    df = df.loc[valid_mask].copy()
    labels = labels.loc[valid_mask].astype(int)
    raw_labels = raw_labels.loc[valid_mask]
    feature_source = df.drop(columns=[label_col])
    filtered = _filter_leak_columns(feature_source, logger)
    filtered = filtered.drop(columns=[col for col in filtered.columns if col.lower() == "mission"], errors="ignore")
    filtered = _coerce_numeric(filtered)
    filtered = _prune_categorical(filtered)
    features, canonical = _extract_canonical_features(filtered)
    features = _append_additional_numeric(features, filtered)
    features = features.dropna(axis=1, how="all")
    metadata = _build_metadata(mission_key, feature_source, raw_labels, canonical)
    logger.info("Mission %s: %d samples, %d features after leak removal", mission_key, len(features), features.shape[1])
    return MissionDataset(mission=mission_key, features=features, labels=labels, metadata=metadata)


def combine_datasets(datasets: Iterable[MissionDataset]) -> MissionDataset:
    datasets = list(datasets)
    if not datasets:
        raise ValueError("At least one dataset is required to combine.")
    missions = [dataset.mission for dataset in datasets]
    combined_features = pd.concat([dataset.features for dataset in datasets], axis=0, sort=False)
    combined_labels = pd.concat([dataset.labels for dataset in datasets], axis=0)
    combined_metadata = pd.concat([dataset.metadata for dataset in datasets], axis=0, sort=False)
    combined_metadata["mission"] = combined_metadata["mission"].astype(str)
    tag = "+".join(sorted(set(missions)))
    return MissionDataset(mission=tag, features=combined_features, labels=combined_labels, metadata=combined_metadata)


def load_cross_mission_split(
    test_mission: str,
    data_dir: Path,
    logger: Optional[logging.Logger] = None,
) -> Tuple[MissionDataset, MissionDataset]:
    logger = logger or logging.getLogger(__name__)
    missions = set(MISSION_FILES.keys())
    test_key = test_mission.lower()
    if test_key not in missions:
        raise ValueError(f"Unknown test mission '{test_mission}'.")
    train_keys = sorted(missions - {test_key})
    train_datasets = [load_mission_dataset(mission, data_dir, logger) for mission in train_keys]
    test_dataset = load_mission_dataset(test_key, data_dir, logger)
    train_dataset = combine_datasets(train_datasets)
    return train_dataset, test_dataset


def infer_feature_types(df: pd.DataFrame) -> Tuple[List[str], List[str]]:
    numeric_cols: List[str] = []
    categorical_cols: List[str] = []
    for column in df.columns:
        if pd.api.types.is_numeric_dtype(df[column]):
            numeric_cols.append(column)
        else:
            categorical_cols.append(column)
    return numeric_cols, categorical_cols


def save_feature_schema(columns: Sequence[str], path: Path) -> None:
    payload = {"columns": list(columns)}
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(payload, indent=2))


def load_feature_schema(path: Path) -> List[str]:
    payload = json.loads(path.read_text())
    return list(payload.get("columns", []))


def align_to_schema(df: pd.DataFrame, columns: Sequence[str]) -> pd.DataFrame:
    aligned = df.reindex(columns=columns, fill_value=np.nan)
    return aligned


def nasa_bucket(label_text: str, mission: str) -> str:
    """Normalize NASA disposition labels into three buckets."""

    text = (label_text or "").strip().lower()
    mission_key = mission.lower()
    if mission_key in ("kepler", "k2"):
        if text == "confirmed":
            return "planet"
        if text == "candidate":
            return "candidate"
        return "non-planet"
    if mission_key == "tess":
        if text in {"cp", "kp"}:
            return "planet"
        if text == "pc":
            return "candidate"
        return "non-planet"
    return "non-planet"


def load_mission_df(
    mission: str,
    data_dir: Path,
    logger: Optional[logging.Logger] = None,
) -> Tuple[pd.DataFrame, pd.DataFrame]:
    """Return mission features and metadata data frames."""

    dataset = load_mission_dataset(mission, data_dir, logger)
    return dataset.features.copy(), dataset.metadata.copy()
